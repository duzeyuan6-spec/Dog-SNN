#!/usr/bin/env python3
"""Hardware-free recording checks: python check_recording.py."""
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

from recording import RGBDRecorder, TerminalKeys


def frame(index):
    depth = np.arange(32 * 48, dtype=np.float32).reshape(32, 48) / 37 + index
    depth[0, :2] = [0, np.nan]
    return SimpleNamespace(color=np.full((32, 48, 3), [20 + index * 10, 70, 160], np.uint8),
                           depth_m=depth, sequence=index + 1, captured_at=100 + index * .08,
                           device_timestamp=200 + index * .08, fx=500.0, cx=24.0)


def raises(kind, action):
    try:
        action()
    except kind as error:
        return error
    raise AssertionError(f"Expected {kind.__name__}")


def saved(path, expected):
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "complete", metadata
    assert metadata["frames_written"] == len(expected), metadata
    assert metadata["depth_unit"] == "m" and metadata["camera_serial"] == "test-camera", metadata
    with (path / "timestamps.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(expected)
    depths = sorted((path / "depth").glob("*.npy"))
    assert len(depths) == len(expected)
    video = cv2.VideoCapture(str(path / "color.avi"))
    try:
        assert video.isOpened(), "Saved video cannot be opened"
        for index, (row, source, depth_path) in enumerate(zip(rows, expected, depths)):
            assert int(row["frame_index"]) == index and int(row["sequence"]) == source.sequence
            for name in ("captured_at", "device_timestamp", "fx", "cx"):
                assert float(row[name]) == getattr(source, name), (name, row)
            depth = np.load(depth_path, allow_pickle=False)
            assert depth.dtype == np.float32
            assert np.array_equal(depth, source.depth_m, equal_nan=True), "Depth was changed or mispaired"
            ok, color = video.read()
            assert ok and color.shape == source.color.shape
            assert np.max(np.abs(color.mean(axis=(0, 1)) - source.color.mean(axis=(0, 1)))) < 10
        assert not video.read()[0], "Video contains unpaired extra frames"
    finally:
        video.release()
    return metadata


def check_clips(root):
    recorder = RGBDRecorder(root, fps=15, camera_serial="test-camera")
    first, second, third = frame(0), frame(1), frame(2)
    try:
        recorder.handle("11", first)
        assert recorder.active
        first_path = Path(recorder.path)
        recorder.handle("", second)
        recorder.handle("aa", third)
        assert not recorder.active
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            info = first_path / "metadata.json"
            if info.exists() and json.loads(info.read_text(encoding="utf-8")).get("status") == "complete":
                break
            time.sleep(.01)
        else:
            raise AssertionError("Pressing a did not finalize the clip")
        # Reap the completed writer before opening another clip.
        recorder.handle("", third)
        saved(first_path, [first, second])
        recorder.handle("1", third)
        second_path = Path(recorder.path)
        assert second_path != first_path
        recorder.close()  # Ctrl+C/normal exit uses this same cleanup.
        saved(second_path, [third])
    finally:
        recorder.close()


def check_backpressure(root):
    entered, release = threading.Event(), threading.Event()
    original_save = np.save

    def slow_save(*args, **kwargs):
        entered.set()
        assert release.wait(3), "Test writer was never released"
        return original_save(*args, **kwargs)

    recorder = RGBDRecorder(root, fps=15, camera_serial="test-camera", queue_size=1)
    timer = threading.Timer(2, release.set)  # Bound a failure if enqueue/stop accidentally blocks.
    timer.start()
    try:
        with mock.patch("recording.np.save", side_effect=slow_save):
            recorder.handle("1", frame(0))
            assert entered.wait(2), "Writer did not receive the first frame"
            started = time.monotonic()
            recorder.handle("", frame(1))
            recorder.handle("", frame(2))
            recorder.handle("", frame(3))
            recorder.handle("a", frame(4))
            assert time.monotonic() - started < .3, "Recording blocked the follower"
            assert recorder.dropped_pairs == 2
            path = Path(recorder.path)
            release.set()
            recorder.close()
        metadata = saved(path, [frame(0), frame(1)])
        assert metadata["dropped_pairs"] == 2
    finally:
        release.set()
        timer.cancel()
        recorder.close()


def check_errors(root):
    recorder = RGBDRecorder(root, fps=15, camera_serial="test-camera")
    with mock.patch("recording.np.save", side_effect=OSError("synthetic disk failure")):
        error = raises(RuntimeError, lambda: (recorder.handle("1", frame(0)), recorder.close()))
        assert "synthetic disk failure" in str(error), error
    failed = SimpleNamespace(isOpened=lambda: False, release=lambda: None)
    recorder = RGBDRecorder(root, fps=15, camera_serial="test-camera")
    with mock.patch("recording.cv2.VideoWriter", return_value=failed):
        raises(RuntimeError, lambda: (recorder.handle("1", frame(0)), recorder.close()))


def check_keys():
    keys = TerminalKeys(io.StringIO("1a"))
    assert keys.open() is False
    assert keys.read() == ""
    keys.close()
    if os.name != "posix":
        print("PTY check skipped on Windows; run on the robot for the terminal check.")
        return
    import pty
    import termios

    master, slave = pty.openpty()
    with os.fdopen(slave, "r") as stream:
        original = termios.tcgetattr(stream)
        keys = TerminalKeys(stream)
        try:
            assert keys.open() is True
            settings = termios.tcgetattr(stream)
            assert settings[3] & termios.ISIG, "Ctrl+C signal handling was disabled"
            assert not settings[3] & termios.ICANON, "Keys still require Enter"
            before = time.monotonic()
            assert keys.read() == "" and time.monotonic() - before < .1
            os.write(master, b"1a")
            received, deadline = "", time.monotonic() + 1
            while len(received) < 2 and time.monotonic() < deadline:
                received += keys.read()
                time.sleep(.01)
            assert received == "1a", received
        finally:
            try:
                keys.close()
                assert termios.tcgetattr(stream) == original, "Terminal settings were not restored"
            finally:
                os.close(master)


if __name__ == "__main__":
    check_keys()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        check_clips(root)
        check_backpressure(root)
        check_errors(root)
    print("Recording checks passed; no camera or robot SDK opened.")
