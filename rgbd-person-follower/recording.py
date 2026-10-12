"""Nonblocking terminal keys and paired RGB-D recording; no camera/robot access."""
import csv
from datetime import datetime
import json
import logging
import math
import os
from pathlib import Path
import queue
import select
import sys
import tempfile
import threading

import cv2
import numpy as np


class TerminalKeys:
    def __init__(self, stream=None):
        self.stream = sys.stdin if stream is None else stream
        self.fd = self.settings = None

    def open(self):
        if not self.stream.isatty():
            return False
        import termios
        import tty
        self.fd = self.stream.fileno()
        self.settings = termios.tcgetattr(self.fd)
        try:
            tty.setcbreak(self.fd)  # Keep ISIG: Ctrl+C still requests the normal dog cleanup.
        except Exception:
            self.close()
            raise
        return True

    def read(self):
        if self.fd is not None and select.select([self.fd], [], [], 0)[0]:
            return os.read(self.fd, 64).decode("ascii", errors="ignore")
        return ""

    def close(self):
        if self.fd is not None:
            import termios
            try:
                termios.tcsetattr(self.fd, termios.TCSANOW, self.settings)
            finally:
                self.fd = self.settings = None


class RGBDRecorder:
    def __init__(self, root, fps=15, camera_serial=None, queue_size=8):
        if not math.isfinite(fps) or fps <= 0 or queue_size < 1:
            raise ValueError("Recording fps and queue_size must be positive")
        self.root, self.fps, self.camera_serial = Path(root), fps, camera_serial
        self.queue_size = queue_size
        self.path = self._thread = None
        self.frames_written = self.dropped_pairs = 0
        self._error = None
        self.logger = logging.getLogger("follower")

    @property
    def active(self):
        return self._thread is not None and not self._stop.is_set() and self._thread.is_alive()

    def handle(self, keys, frame):
        if self._thread is not None and not self._thread.is_alive():
            self.close()
        for key in keys:
            if key == "1":
                if self._thread is None:
                    self._start(frame)
                elif not self.active:
                    self.logger.info("Still saving the previous recording; press 1 after Saved")
            elif key == "a" and self.active:
                self.stop()
                self.logger.info("Saving recording: %s", self.path)
        if self.active:
            try:
                # ponytail: bounded queue drops whole pairs on slow disks; use chunked storage for higher throughput.
                self._queue.put_nowait(frame)
            except queue.Full:
                self.dropped_pairs += 1

    def _start(self, frame):
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("%Y%m%d_%H%M%S_"), dir=self.root))
        self.frames_written = self.dropped_pairs = 0
        self._error = None
        self._queue = queue.Queue(maxsize=self.queue_size)
        self._stop = threading.Event()
        self._size = (frame.color.shape[1], frame.color.shape[0])
        self._intrinsics = (float(frame.fx), float(frame.cx))
        self._thread = threading.Thread(target=self._write, name="rgbd-recording", daemon=True)
        self._thread.start()
        self.logger.info("Recording started: %s (a stops and saves)", self.path)

    def _write(self):
        writer = None
        try:
            depth_dir = self.path / "depth"
            depth_dir.mkdir()
            writer = cv2.VideoWriter(str(self.path / "color.avi"), cv2.VideoWriter_fourcc(*"MJPG"),
                                     self.fps, self._size)
            if not writer.isOpened():
                raise RuntimeError("Cannot open MJPG video encoder")
            with (self.path / "timestamps.csv").open("w", newline="", encoding="utf-8") as stream:
                rows = csv.writer(stream)
                rows.writerow(["frame_index", "sequence", "captured_at", "device_timestamp", "fx", "cx"])
                while not self._stop.is_set() or not self._queue.empty():
                    try:
                        frame = self._queue.get(timeout=.1)
                    except queue.Empty:
                        continue
                    width, height = self._size
                    if (frame.color.shape != (height, width, 3) or frame.color.dtype != np.uint8 or
                            frame.depth_m.shape != (height, width) or frame.depth_m.dtype != np.float32):
                        raise ValueError("Expected fixed-size BGR uint8 and aligned float32 depth")
                    if not all(math.isfinite(v) for v in (frame.captured_at, frame.device_timestamp, frame.fx, frame.cx)):
                        raise ValueError("Nonfinite recording timestamp or camera intrinsics")
                    index = self.frames_written
                    temporary = depth_dir / f"{index:06d}.tmp"
                    with temporary.open("wb") as depth_file:
                        np.save(depth_file, frame.depth_m, allow_pickle=False)
                    temporary.replace(depth_dir / f"{index:06d}.npy")
                    writer.write(frame.color)
                    rows.writerow([index, frame.sequence, frame.captured_at, frame.device_timestamp, frame.fx, frame.cx])
                    stream.flush()
                    self.frames_written += 1
        except Exception as error:
            self._error = error
        finally:
            self._stop.set()
            try:
                if writer is not None:
                    writer.release()
            except Exception as error:
                if self._error is None:
                    self._error = error
            metadata = {"status": "error" if self._error else "complete", "frames_written": self.frames_written,
                        "dropped_pairs": self.dropped_pairs, "camera_serial": self.camera_serial,
                        "width": self._size[0], "height": self._size[1], "video_fps": self.fps,
                        "depth_unit": "m", "depth_dtype": "float32", "depth_aligned_to": "color",
                        "fx": self._intrinsics[0], "cx": self._intrinsics[1],
                        "timing": "AVI uses constant FPS; timestamps.csv contains actual capture times in seconds. "
                                  "captured_at is host monotonic time; device_timestamp is camera time.",
                        "error": str(self._error) if self._error else None}
            try:
                temporary = self.path / "metadata.tmp"
                temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
                temporary.replace(self.path / "metadata.json")
            except Exception as error:
                if self._error is None:
                    self._error = error

    def stop(self):
        if self._thread is not None:
            self._stop.set()

    def close(self, timeout=10):
        if self._thread is None:
            return
        self.stop()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise TimeoutError(f"Recording save did not finish: {self.path}")
        self._thread = None
        if self._error is not None:
            raise RuntimeError(f"Recording failed at {self.path}: {self._error}") from self._error
        self.logger.info("Saved recording: %s (%s frames, %s dropped pairs)",
                         self.path, self.frames_written, self.dropped_pairs)
