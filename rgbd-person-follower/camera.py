"""Newest synchronized Orbbec RGB-D frame, using the installed SDK's D2C filter."""

from dataclasses import dataclass
import sys
import threading
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class RGBDFrame:
    color: np.ndarray
    depth_m: np.ndarray
    fx: float
    cx: float
    captured_at: float
    sequence: int
    device_timestamp: float


class OrbbecCamera:
    def __init__(self, config):
        self.config = config.get("camera", config)
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._latest = self._error = self._thread = self._pipeline = None
        self._started = False

    def start(self):
        import pyorbbecsdk as ob

        if self._thread is not None:
            raise RuntimeError("Camera is already started")
        self._ob = ob
        serial = str(self.config.get("serial", "")).strip()
        if not serial:
            raise ValueError("camera.serial is required; automatic camera selection is disabled")
        ob.Context.set_logger_to_console(ob.OBLogLevel.ERROR)
        self._context = ob.Context()
        devices = self._context.query_devices()
        found = [devices.get_device_serial_number_by_index(i) for i in range(devices.get_count())]
        if serial not in found:
            raise RuntimeError(f"Camera {serial} is absent; connected serials: {found}")
        self._pipeline = ob.Pipeline(devices.get_device_by_serial_number(serial))
        pipeline = self._pipeline

        def profiles(sensor, formats):
            available = pipeline.get_stream_profile_list(sensor)
            result = []
            for i in range(available.get_count()):
                profile = available.get_stream_profile_by_index(i).as_video_stream_profile()
                if profile.get_format() in formats:
                    result.append(profile)
            return result

        formats = [ob.OBFormat.MJPG, ob.OBFormat.RGB, ob.OBFormat.BGR,
                   ob.OBFormat.YUYV, ob.OBFormat.UYVY]
        colors = profiles(ob.OBSensorType.COLOR_SENSOR, formats)
        depths = profiles(ob.OBSensorType.DEPTH_SENSOR, [ob.OBFormat.Y16])
        width, height, fps = (int(self.config.get(k, v)) for k, v in
                              (("width", 640), ("height", 480), ("fps", 15)))
        if min(width, height, fps) <= 0:
            raise ValueError("camera width, height, and fps must be positive")
        # ponytail: try a few advertised low-bandwidth pairs; USB3/high-res needs an explicit config.
        pairs = [(c, d) for c in colors for d in depths
                 if c.get_fps() == d.get_fps() <= fps
                 and c.get_width() <= width and c.get_height() <= height
                 and d.get_width() <= 640 and d.get_height() <= 480]
        pairs.sort(key=lambda pair: (abs(pair[0].get_fps() - fps),
                   width * height - pair[0].get_width() * pair[0].get_height(),
                   formats.index(pair[0].get_format()),
                   abs(pair[1].get_width() - pair[0].get_width())
                   + abs(pair[1].get_height() - pair[0].get_height())))
        errors = []
        pipeline.enable_frame_sync()
        for color, depth in pairs[:8]:
            stream_config = ob.Config()
            stream_config.enable_stream(color)
            stream_config.enable_stream(depth)
            stream_config.set_frame_aggregate_output_mode(ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
            try:
                pipeline.start(stream_config)
                self._started = True
                print(f"Camera {serial}: color={color}; depth={depth}; software D2C", file=sys.stderr)
                break
            except Exception as exc:
                errors.append(str(exc))
                try:
                    pipeline.stop()
                except Exception:
                    pass
        if not self._started:
            raise RuntimeError(f"No usable low-bandwidth RGB-D profiles: {errors or 'no matching pair advertised'}")
        self._align = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)
        self._thread = threading.Thread(target=self._capture, name="orbbec-latest", daemon=True)
        self._thread.start()
        return self

    def _bgr(self, frame):
        ob = self._ob
        data = np.frombuffer(frame.get_data(), dtype=np.uint8)
        height, width = frame.get_height(), frame.get_width()
        fmt = frame.get_format()
        if fmt == ob.OBFormat.MJPG:
            color = cv2.imdecode(data, cv2.IMREAD_COLOR)
        elif fmt in (ob.OBFormat.RGB, ob.OBFormat.BGR):
            color = data.reshape(height, width, 3)
            color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR) if fmt == ob.OBFormat.RGB else color.copy()
        else:
            conversion = cv2.COLOR_YUV2BGR_YUY2 if fmt == ob.OBFormat.YUYV else cv2.COLOR_YUV2BGR_UYVY
            color = cv2.cvtColor(data.reshape(height, width, 2), conversion)
        if color is None or color.shape != (height, width, 3):
            raise RuntimeError("Color frame could not be decoded")
        return color

    def _capture(self):
        previous = -1.0
        offset = float("inf")
        sequence = 0
        frame_timeout = float(self.config.get("frame_timeout", 2.0))
        skew_ms = float(self.config.get("max_pair_skew_ms", 40.0))
        last_frame = time.monotonic()
        try:
            while not self._stop.is_set():
                frames = self._pipeline.wait_for_frames(200)
                arrived = time.monotonic()
                if arrived - last_frame > frame_timeout:
                    raise TimeoutError("RGB-D camera stopped producing fresh synchronized frames")
                if not frames:
                    continue
                color, depth = frames.get_color_frame(), frames.get_depth_frame()
                if not color or not depth:
                    continue
                color_time = color.get_timestamp_us() * 1e-6
                depth_time = depth.get_timestamp_us() * 1e-6
                if color_time < previous:
                    raise RuntimeError("Camera hardware timestamp moved backwards")
                if color_time == previous or abs(color_time - depth_time) * 1000 > skew_ms:
                    continue
                aligned = self._align.process(frames)
                if not aligned:
                    raise RuntimeError("Depth-to-color alignment failed")
                aligned = aligned.as_frame_set()
                color, depth = aligned.get_color_frame(), aligned.get_depth_frame()
                if not color or not depth:
                    raise RuntimeError("Alignment returned an incomplete RGB-D pair")
                bgr = self._bgr(color)
                raw = np.frombuffer(depth.get_data(), dtype=np.uint16).reshape(depth.get_height(), depth.get_width())
                scale = float(depth.get_depth_scale()) * 0.001  # SDK scale is millimeters per raw unit.
                intrinsic = color.get_stream_profile().as_video_stream_profile().get_intrinsic()
                if raw.shape != bgr.shape[:2] or not 0 < scale <= 0.1 or intrinsic.fx <= 0:
                    raise RuntimeError("Invalid aligned dimensions, depth scale, or camera intrinsics")
                offset = min(offset, arrived - color_time)
                previous = color_time
                sequence += 1
                last_frame = arrived
                frame = RGBDFrame(bgr, raw.astype(np.float32) * scale, float(intrinsic.fx),
                                  float(intrinsic.cx), min(arrived, color_time + offset), sequence, color_time)
                with self._condition:
                    self._latest = frame
                    self._condition.notify_all()
        except Exception as exc:
            if not self._stop.is_set():
                with self._condition:
                    self._error = exc
                    self._condition.notify_all()

    def read(self, after_sequence=-1, timeout=1.0):
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                if self._error is not None:
                    raise RuntimeError(f"RGB-D capture failed: {self._error}") from self._error
                if self._stop.is_set():
                    raise RuntimeError("Camera is closed")
                if self._latest is not None and self._latest.sequence > after_sequence:
                    return self._latest
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("No new RGB-D frame")
                self._condition.wait(remaining)

    def close(self):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._pipeline is not None and self._started:
            self._pipeline.stop()
            self._started = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
