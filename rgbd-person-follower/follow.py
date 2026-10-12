#!/usr/bin/env python3
"""Orbbec RGB-D -> fixed person ID -> Genisom walking velocities."""
import argparse
from collections import deque
import fcntl
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import time

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent


def load_config(path):
    config = json.loads(Path(path).read_text())
    for section in ("camera", "vision", "control", "robot"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")
        for key, value in config[section].items():
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{section}.{key} must be finite and positive")
    c, r, v = config["control"], config["robot"], config["vision"]
    if not 0 < c["min_distance"] < c["target_distance"] < c["max_distance"] <= 6:
        raise ValueError("Expected min_distance < target_distance < max_distance <= 6m")
    if c["max_vx"] > 3.0 or c["max_yaw"] > 3.0:
        raise ValueError("SDK move limits: max_vx <= 3.0 m/s and max_yaw <= 3.0 rad/s")
    if not .2 <= r["observation_timeout"] <= 1 or not 10 <= r["rate_hz"] <= 100:
        raise ValueError("Invalid observation_timeout or rate_hz")
    if not isinstance(r["auto_stand"], bool) or not 1 <= r["port"] <= 65535:
        raise ValueError("Invalid auto_stand or UDP port")
    if r["auto_stand"] and r["stand_delay"] < 10:
        raise ValueError("Allow at least 10 seconds for the real dog to stand and settle")
    if r["lie_down_delay"] < 3:
        raise ValueError("Allow at least 3 seconds for the lieDown command, as in the SDK demo")
    if not 0 < v["confidence"] < 1 or not 128 <= v["imgsz"] <= 640:
        raise ValueError("Invalid detector confidence or image size")
    if not 0 < v["acquire_center_fraction"] <= .5 or not 0 < v["min_depth_fraction"] <= 1:
        raise ValueError("Invalid acquisition/depth fractions")
    if not config["camera"]["serial"].strip():
        raise ValueError("An explicit head-camera serial is required")
    return config


def put_latest(channel, value):
    try:
        channel.put_nowait(value)
    except queue.Full:
        try:
            channel.get_nowait()
        except queue.Empty:
            pass
        try:
            channel.put_nowait(value)
        except queue.Full:
            pass  # A skipped message cannot extend the worker's observation deadline.


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--dry-run", action="store_true", help="Real vision and control, NEVER initialize robot SDK")
    parser.add_argument("--seconds", type=float, default=0, help="Stop after N seconds; 0 runs until Ctrl+C")
    parser.add_argument("--motion-seconds", type=float, default=0, help="Stop N seconds after following first starts")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds < 0:
        parser.error("seconds must be finite and nonnegative")
    if not math.isfinite(args.motion_seconds) or args.motion_seconds < 0:
        parser.error("motion-seconds must be finite and nonnegative")
    config = load_config(args.config)
    lock = (ROOT / ".run.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("A follower/check is already running in this project")
    lock.write(str(os.getpid()))
    lock.flush()
    output = ROOT / "output"
    output.mkdir(exist_ok=True)
    logger = logging.getLogger("follower")
    logger.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(), RotatingFileHandler(output / "follow.log", maxBytes=1_000_000, backupCount=2)):
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.info("%s: target stands centrally within %.1fm; Ctrl+C %s",
                "DRY RUN (no robot SDK)" if args.dry_run else "LIVE ROBOT", config["vision"]["acquire_max_distance"],
                "stops" if args.dry_run else "stops and lies down")
    from camera import OrbbecCamera
    from recording import RGBDRecorder, TerminalKeys
    from robot import robot_worker
    from vision import PersonTracker
    import cv2
    import numpy as np
    import torch
    torch.set_num_threads(int(config["vision"]["threads"]))
    cv2.setNumThreads(1)
    tracker = PersonTracker(ROOT / config["vision"]["model"], config)
    tracker.model.predict(np.zeros((480, 640, 3), dtype=np.uint8), device="cpu",
                          imgsz=config["vision"]["imgsz"], verbose=False)
    ctx = mp.get_context("spawn")
    config["_parent_pid"] = os.getpid()
    config["_motion_seconds"] = args.motion_seconds
    stop = ctx.Event()
    observations, statuses = ctx.Queue(maxsize=1), ctx.Queue(maxsize=4)
    worker = ctx.Process(target=robot_worker, args=(config, observations, stop, statuses, args.dry_run), name="follow-control")
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    keys = TerminalKeys()
    recorder = RGBDRecorder(output / "recordings", fps=config["camera"]["fps"], camera_serial=config["camera"]["serial"])
    camera = OrbbecCamera(config, frame_sink=recorder.submit)
    robot_status = {"state": "starting", "vx": 0.0, "yaw": 0.0, "battery": None}
    frames = visible = 0
    latencies = []
    history = deque(maxlen=500)
    started = time.monotonic()
    last_write = 0.0
    last_sequence = -1
    exit_code = 0
    battery_start_logged = False
    try:
        logger.info("Keys: 1 starts RGB-D recording, a stops and saves (no Enter)" if keys.open()
                    else "Noninteractive input: recording hotkeys disabled; run in a terminal to record")
        camera.start()
        frame = camera.read(timeout=5)
        logger.info("Aligned %sx%s, fx=%.2f cx=%.2f", frame.color.shape[1], frame.color.shape[0], frame.fx, frame.cx)
        worker.start()
        ready_deadline = time.monotonic() + config["robot"]["connect_timeout"] + 3
        if not args.dry_run and config["robot"]["auto_stand"]:
            ready_deadline += config["robot"]["stand_delay"] + config["robot"]["lie_down_delay"] + 3
        while not stop.is_set():
            now = time.monotonic()
            if args.seconds and now - started >= args.seconds:
                break
            try:
                while True:
                    robot_status = statuses.get_nowait()
                    if not battery_start_logged and robot_status.get("battery_start_percent") is not None:
                        logger.info("启动电量：%d%%", robot_status["battery_start_percent"])
                        battery_start_logged = True
                    if robot_status.get("state") in ("error", "disconnected"):
                        raise RuntimeError(robot_status["error"])
            except queue.Empty:
                pass
            if not worker.is_alive():
                raise RuntimeError(f"Control process exited ({worker.exitcode}): {robot_status}")
            if not robot_status.get("ready") and now > ready_deadline:
                raise TimeoutError("Robot initialization exceeded its deadline; stopping control process")
            try:
                frame = camera.read(after_sequence=last_sequence, timeout=.25)
            except TimeoutError:
                put_latest(observations, {"valid": False, "reason": "camera_timeout", "sequence": last_sequence,
                                          "captured_at": 0.0, "distance": 0.0, "bearing": 0.0})
                continue
            last_sequence = frame.sequence
            recorder.handle(keys.read(), frame)
            if not robot_status.get("ready") or robot_status.get("state") in ("stopping", "lying_down", "standing"):
                observation = dict(valid=False, reason="waiting_for_stand", track_id=None,
                                   distance=0.0, bearing=0.0, box=None, confidence=0.0,
                                   captured_at=frame.captured_at, sequence=frame.sequence)
            else:
                observation = tracker.update(frame)
            age = time.monotonic() - frame.captured_at
            if age > config["robot"]["observation_timeout"]:
                observation.update(valid=False, reason="inference_too_slow")
            put_latest(observations, observation)
            history.append({"observation": observation, "detections": tracker.detections,
                            "age_ms": round(age * 1000, 1), "robot": dict(robot_status)})
            frames += 1
            visible += int(observation["valid"])
            latencies.append(age)
            if len(latencies) > 500:
                del latencies[:250]
            if now - last_write >= 1:
                last_write = now
                status = {"dry_run": args.dry_run, "observation": observation, "robot": robot_status,
                          "detections": tracker.detections,
                          "recording": {"active": recorder.active, "path": str(recorder.path) if recorder.path else None,
                                        "frames_written": recorder.frames_written, "dropped_pairs": recorder.dropped_pairs,
                                        "frames_received": recorder.frames_received,
                                        "source_dropped_frames": recorder.source_dropped_frames},
                          "fps": round(frames / max(.01, now - started), 2), "frame_age_ms": round(age * 1000, 1),
                          "time": time.strftime("%Y-%m-%d %H:%M:%S")}
                save_json(output / "status.json", status)
                image = frame.color.copy()
                for detection in tracker.detections:
                    coords = detection["box"]
                    cv2.rectangle(image, tuple(coords[:2]), tuple(coords[2:]), (180, 180, 180), 1)
                    cv2.putText(image, f"ID {detection['track_id']} {detection['confidence']:.2f}",
                                (coords[0], max(45, coords[1])), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
                box = observation.get("box")
                if box:
                    cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), (0, 255, 255), 2)
                cv2.putText(image, f"ID {observation['track_id']} {observation['reason']} {observation['distance']:.2f}m",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
                ok, encoded = cv2.imencode(".jpg", image)
                if ok:
                    temporary = output / "latest.tmp"
                    temporary.write_bytes(encoded.tobytes())
                    temporary.replace(output / "latest.jpg")
                logger.info("%s id=%s d=%.2fm angle=%.2f vx=%.2f yaw=%.2f battery=%s age=%.0fms",
                            observation["reason"], observation["track_id"], observation["distance"], observation["bearing"],
                            robot_status.get("vx", 0), robot_status.get("yaw", 0), robot_status.get("battery"), age * 1000)
    except Exception:
        logger.exception("Follower stopped")
        exit_code = 1
    finally:
        stop.set()
        recorder.stop()
        try:
            keys.close()
        except Exception:
            logger.exception("Terminal cleanup failed")
            exit_code = 1
        if worker.pid is not None:
            logger.info("Stopping control%s", "" if args.dry_run else "; waiting for dog to lie down")
            worker.join(timeout=1 if args.dry_run else config["robot"]["lie_down_delay"] + 6)
            if worker.is_alive():
                logger.error("Control process unresponsive; terminating SDK sender")
                worker.terminate()
                worker.join(timeout=2)
                if worker.is_alive():
                    worker.kill()
                    worker.join(timeout=2)
                exit_code = 1
            try:
                while True:
                    final_status = statuses.get_nowait()
                    robot_status = final_status
                    if final_status.get("state") in ("error", "disconnected"):
                        logger.error("Control failure: %s", final_status.get("error"))
                        exit_code = 1
            except queue.Empty:
                pass
            if worker.exitcode not in (None, 0):
                logger.error("Control process exit code: %s", worker.exitcode)
                exit_code = 1
        try:
            camera.close()
        except Exception:
            logger.exception("Camera cleanup failed")
            exit_code = 1
        try:
            recorder.close()
        except Exception:
            logger.exception("Recording save failed")
            exit_code = 1
        report = {"dry_run": args.dry_run, "frames": frames, "valid_target_frames": visible,
                  "seconds": round(time.monotonic() - started, 2), "exit_code": exit_code,
                  "latency_p50_ms": round(float(np.percentile(latencies, 50)) * 1000, 1) if latencies else None,
                  "latency_p95_ms": round(float(np.percentile(latencies, 95)) * 1000, 1) if latencies else None,
                  "target_id": tracker.target_id, "stopped": True,
                  "lie_down_sent": robot_status.get("lie_down_sent", False)}
        report.update({key: robot_status.get(key) for key in
                       ("battery_start_percent", "battery_end_percent", "battery_used_percent_points")})
        battery_start, battery_end, battery_used = (report[key] for key in
                                                  ("battery_start_percent", "battery_end_percent", "battery_used_percent_points"))
        if battery_used is None:
            logger.info("本次耗电：无法统计（启动电量=%s，结束电量=%s）",
                        f"{battery_start}%" if battery_start is not None else "未知",
                        f"{battery_end}%" if battery_end is not None else "未知")
        elif battery_used >= 0:
            logger.info("本次耗电：%d 个百分点（启动 %d%% → 结束 %d%%）", battery_used, battery_start, battery_end)
        else:
            logger.info("本次电量上升：%d 个百分点（启动 %d%% → 结束 %d%%，可能充电或电量估计波动）",
                        -battery_used, battery_start, battery_end)
        save_json(output / "last_run.json", report)
        save_json(output / "last_trace.json", list(history))
        save_json(output / "status.json", {"state": "stopped", "vx": 0, "yaw": 0, "last_run": report})
        logger.info("Stopped: %s", report)
        observations.close()
        statuses.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
