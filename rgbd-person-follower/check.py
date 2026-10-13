#!/usr/bin/env python3
"""Hardware-free assertions: python check.py (no camera start or SDK import)."""
import copy
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from ultralytics.engine.results import Results

from camera import OrbbecCamera, RGBDFrame
from control import FollowController
from follow import load_config
import robot
import vision

ROOT = Path(__file__).resolve().parent
CONFIG = load_config(ROOT / "config.json")


def raises(kind, action):
    try:
        action()
    except kind as exc:
        return exc
    raise AssertionError(f"Expected {kind.__name__}")


def check_control():
    c = FollowController(CONFIG["control"])
    far = CONFIG["control"]["target_distance"] + 1.0
    for _ in range(100):
        vx, yaw = c.update(far, 0.0, .02)
        assert 0 <= vx <= CONFIG["control"]["max_vx"] and yaw == 0
    assert vx > 0
    assert c.update(2.0, .2, .1)[1] > 0
    c.reset()
    assert c.update(2.0, -.2, .1)[1] < 0
    assert c.update(.5, .1, .1)[0] == 0
    assert c.update(1.0, 0, .1)[0] == 0  # No reversing inside follow distance.
    for values in ((float("nan"), 0, .1), (2, float("nan"), .1),
                   (2, 0, float("nan")), (7, 0, .1), (2, 0, 0)):
        assert c.update(*values) == (0, 0)
    c.update(3, .2, .1)
    c.reset()
    assert (c.vx, c.yaw) == (0, 0)


def check_vision_and_camera():
    color = np.zeros((480, 640, 3), np.uint8)
    depth = np.full((480, 640), 2, np.float32)
    with mock.patch.object(vision, "YOLO"), mock.patch.object(torch.jit, "load"):
        tracker = vision.PersonTracker(ROOT / "yolo26n.pt", CONFIG)
    tracker.model.add_callback.assert_called_once_with(
        "on_predict_postprocess_end", tracker._suppress_overlaps)
    # Regression: the real one-person duplicate had IoU 0.637 and split its ID.
    boxes = torch.tensor([[276, 81, 336, 290, .501, 0],
                          [277, 81, 370, 286, .381, 0],
                          [440, 80, 520, 290, .9, 0]])
    result = Results(color, path="synthetic.jpg", names={0: "person"}, boxes=boxes)
    empty = Results(color, path="empty.jpg", names={0: "person"}, boxes=boxes[:0])
    none = Results(color, path="none.jpg", names={0: "person"})
    tracker._suppress_overlaps(SimpleNamespace(results=[result, empty, none]))
    assert sorted(result.boxes.data.tolist()) == sorted(boxes[[0, 2]].tolist())
    assert empty.boxes.data.shape == (0, 6) and none.boxes is None
    person = dict(track_id=7, distance=2.0, bearing=0.01,
                  box=(240, 80, 400, 400), confidence=.9, center=320,
                  test_feature=(1.0, 0.0))
    current = [person]
    tracker._observations = lambda _: [dict(o) for o in current]
    tracker._features = lambda _, observations: [o["test_feature"] for o in observations]
    sequence, timestamp = 0, 100.0

    def step(people=None, delta=.31, repeated=False, invalid=False):
        nonlocal sequence, timestamp, current
        if people is not None:
            current = people
        sequence += int(not repeated)
        timestamp += delta
        frame = RGBDFrame(color, depth, 0 if invalid else 600, 320,
                          timestamp, sequence, timestamp)
        return tracker.update(frame)

    def confirm(people):
        assert not step(people)["valid"]
        assert not step()["valid"]
        output = step()
        assert output["valid"] and output["track_id"] == people[0]["track_id"]
        return output

    confirm([person])
    anchor, anchor_copy = tracker.anchor, tracker.anchor.copy()
    assert tracker.target_id == 7 and tracker.gallery.maxlen == 8
    borderline = dict(person, test_feature=(.67, (1 - .67 ** 2) ** .5))
    assert step([borderline])["valid"] and not tracker.gallery
    assert step([dict(borderline, track_id=8)])["reason"] == "target_unverified"
    confirm([person])  # The lower keep threshold must never authorize a new ID.
    assert step([dict(person, confidence=.2)], delta=.05)["valid"]
    assert step([], delta=.05)["reason"] == "target_missing"
    assert not step([person], delta=.05)["valid"]
    assert not step(delta=.05)["valid"]
    assert step(delta=.05)["valid"]  # A short same-ID gap needs three frames only.
    # The original person can return far from the last location with a new raw ID.
    returned = dict(person, track_id=81, distance=4.5,
                    box=(480, 70, 620, 420), center=550)
    assert step([])["reason"] == "target_missing"
    assert not step([returned], delta=.1)["valid"]
    assert not step(delta=.1)["valid"] and not step(delta=.1)["valid"]
    assert not step(delta=.31)["valid"]  # Three frames alone cannot authorize a new ID.
    assert step(delta=.11)["valid"] and tracker.target_id == 81
    assert tracker.last_good["distance"] == 4.5
    assert step([dict(returned, track_id=82, confidence=.2)])["reason"] == "target_unverified"
    stranger = dict(returned, track_id=82, test_feature=(0.0, 1.0))
    saved = tracker.last_good
    assert step([stranger])["reason"] == "target_unverified"
    assert step([dict(stranger, track_id=81)])["reason"] == "target_unverified"
    assert tracker.target_id == 81 and tracker.last_good is saved
    lookalike = dict(returned, track_id=82, test_feature=(.98, .2))
    assert step([returned, lookalike])["reason"] == "target_ambiguous"
    assert tracker.target_id == 81 and tracker.last_good is saved

    # Any interruption breaks the consecutive-frame and elapsed-time evidence.
    for reason, interrupt in (
            ("target_missing", lambda: step([])),
            ("depth_invalid", lambda: step([dict(returned, distance=None)])),
            ("depth_invalid", lambda: step([dict(returned, distance=float("nan"))])),
            ("invalid_frame", lambda: step(delta=0, invalid=True)),
            ("repeated_frame", lambda: step(delta=0, repeated=True))):
        assert not step([returned])["valid"]
        assert not step()["valid"]
        assert interrupt()["reason"] == reason
        assert tracker.candidate is None and tracker.target_id == 81
        confirm([returned])
        step([])
    assert not step([returned])["valid"] and not step()["valid"]
    assert step(delta=.6)["reason"] == "confirming_target"
    assert not step()["valid"] and step()["valid"]
    step([])
    assert not step([dict(returned, track_id=90)])["valid"] and not step()["valid"]
    assert tracker.target_id == 81
    confirm([dict(returned, track_id=91)])

    # Gallery updates must not drift the initial identity anchor.
    view = dict(returned, track_id=91, test_feature=(.8, .6))
    assert step([view])["valid"] and len(tracker.gallery) == 1
    assert tracker.anchor is anchor and np.array_equal(tracker.anchor, anchor_copy)
    for feature in ((float("nan"), 1.0), (0.0, 0.0)):
        saved = tracker.last_good
        assert step([dict(view, test_feature=feature)])["reason"] == "appearance_invalid"
        assert tracker.target_id == 91 and tracker.last_good is saved
    # A candidate jump resets confirmation, then a stable candidate can recover.
    candidate = dict(person, track_id=92)
    assert not step([candidate])["valid"] and not step()["valid"]
    candidate = dict(returned, track_id=92)
    assert step([candidate])["reason"] == "confirming_target"
    assert tracker.target_id == 91
    assert not step()["valid"] and step()["valid"] and tracker.target_id == 92
    assert tracker.anchor is anchor and np.array_equal(tracker.anchor, anchor_copy)
    # Expiry during ReID must not teach the gallery or advance target recovery.
    saved, gallery = tracker.last_good, list(tracker.gallery)
    for clock_values in ((2.0,), (0.0, 2.0)):
        sequence += 1
        timestamp += .1
        frame = RGBDFrame(color, depth, 600, 320, timestamp, sequence, timestamp)
        with mock.patch.object(vision, "monotonic", side_effect=clock_values):
            assert tracker.update(frame, deadline=1.0)["reason"] == "inference_too_slow"
        assert tracker.last_good is saved and tracker.candidate is None
        assert tracker.anchor is anchor and len(tracker.gallery) == len(gallery)
        assert all(np.array_equal(a, b) for a, b in zip(tracker.gallery, gallery))
    box = person["box"]
    assert tracker._depth(depth, box) == 2.0
    assert tracker._depth(np.zeros_like(depth), box) is None
    assert tracker._depth(np.full_like(depth, np.nan), box) is None
    sparse = np.zeros_like(depth)
    sparse[180, 290:330] = 2
    assert tracker._depth(sparse, box) is None
    frame = RGBDFrame(color, depth, 600, 320, timestamp, sequence, timestamp)
    assert tracker.update(frame)["reason"] == "repeated_frame"
    camera = OrbbecCamera(CONFIG)  # Deliberately never call start().
    camera._latest = frame
    assert camera.read() is frame
    raises(TimeoutError, lambda: camera.read(after_sequence=sequence, timeout=.01))
    camera._error = ValueError("synthetic capture failure")
    assert isinstance(raises(RuntimeError, camera.read).__cause__, ValueError)
    camera._error = None
    camera.close()
    raises(RuntimeError, camera.read)


def check_worker():
    cfg = copy.deepcopy(CONFIG)
    cfg["robot"]["record_battery"] = True
    cfg["_parent_pid"] = os.getppid()
    observations, statuses, stop = queue.Queue(), queue.Queue(), threading.Event()
    sequence, captured_at, packets = 0, 0.0, []
    far = cfg["control"]["target_distance"] + 1.0

    def fresh():
        nonlocal sequence, captured_at
        sequence += 1
        captured_at = time.monotonic()
        observations.put(dict(valid=True, sequence=sequence, captured_at=captured_at,
                              distance=far, bearing=0.0))

    def until(predicate, feed=None, timeout=1.8):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if feed:
                feed()
            try:
                while True:
                    packet = statuses.get_nowait()
                    packets.append(packet)
                    assert packet["state"] != "error", packet
                    if predicate(packet):
                        return packet
            except queue.Empty:
                pass
            time.sleep(.025)
        raise AssertionError(f"Worker did not reach expected state; latest={packets[-1:]}")

    with mock.patch.object(robot.importlib, "import_module", side_effect=AssertionError("SDK import forbidden")):
        worker = threading.Thread(target=robot.robot_worker,
                                  args=(cfg, observations, stop, statuses, True), daemon=True)
        worker.start()
        try:
            until(lambda p: p["state"] == "following" and p["vx"] > 0, fresh)
            old = dict(valid=True, sequence=sequence, captured_at=captured_at, distance=far, bearing=0)
            packet = until(lambda p: p["state"] == "stale_camera", lambda: observations.put(old))
            assert packet["vx"] == packet["yaw"] == 0
            until(lambda p: p["state"] == "following" and p["vx"] > 0, fresh)
            observations.put(dict(valid=False, sequence=sequence, captured_at=captured_at, reason="lost"))
            packet = until(lambda p: p["state"] == "target_lost", timeout=.25)
            assert packet["vx"] == packet["yaw"] == 0
        finally:
            stop.set()
            worker.join(timeout=1)
        assert not worker.is_alive()
        assert all(packet["dry_run"] for packet in packets)
        assert all(packet["battery_start_percent"] is None and packet["battery_end_percent"] is None
                   and packet["battery_used_percent_points"] is None for packet in packets)


def check_config():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.json"
        for section, key, value in (("vision", "confidence", 1), ("control", "max_vx", 3.01),
                                   ("control", "max_yaw", 3.01),
                                   *(("vision", key, value) for key in ("reid_similarity", "reid_margin")
                                     for value in (0, 1, -.1, True, False, "0.7", None, float("nan"))),
                                   ("robot", "rate_hz", float("nan")), ("robot", "auto_stand", "yes"),
                                   *(("robot", "record_battery", value) for value in ("false", 0, 1, None)),
                                   ("robot", "lie_down_delay", 2.9),
                                   ("camera", "serial", ""), ("vision", "min_depth_fraction", 0)):
            cfg = copy.deepcopy(CONFIG)
            cfg[section][key] = value
            path.write_text(json.dumps(cfg))
            raises(ValueError, lambda: load_config(path))
        cfg = copy.deepcopy(CONFIG)
        cfg["control"].update(max_vx=3.0, max_yaw=3.0)
        path.write_text(json.dumps(cfg))
        assert load_config(path)["control"]["max_vx"] == 3.0
        for enabled in (True, False, None):
            cfg = copy.deepcopy(CONFIG)
            cfg["robot"].pop("record_battery", None)
            if enabled is not None:
                cfg["robot"]["record_battery"] = enabled
            path.write_text(json.dumps(cfg))
            assert load_config(path)["robot"].get("record_battery", True) is (True if enabled is None else enabled)


def check_motion_window():
    cfg = copy.deepcopy(CONFIG)
    cfg.update(_parent_pid=os.getppid(), _motion_seconds=.15)
    observations, statuses, stop = queue.Queue(), queue.Queue(), threading.Event()
    worker = threading.Thread(target=robot.robot_worker, args=(cfg, observations, stop, statuses, True), daemon=True)
    worker.start()
    started, sequence = time.monotonic(), 0
    while worker.is_alive() and time.monotonic() - started < .5:
        sequence += 1
        observations.put(dict(valid=True, sequence=sequence, captured_at=time.monotonic(),
                              distance=CONFIG["control"]["target_distance"] + 1, bearing=0))
        time.sleep(.02)
    worker.join(timeout=.1)
    assert not worker.is_alive() and stop.is_set()
    assert .15 <= time.monotonic() - started < .5


class FakeSDK:
    def __init__(self, mode, lie_rejections=0, stand_result=0, call_queue=None):
        self.mode, self.calls = mode, []
        self.lie_rejections, self.stand_result = lie_rejections, stand_result
        self.call_queue = call_queue

    def record(self, name, *args):
        call = (name, time.monotonic(), *args)
        self.calls.append(call)
        if self.call_queue is not None:
            self.call_queue.put(call)

    def initRobot(self, *_):
        pass

    def checkConnect(self):
        return True

    def getCurrentCtrlmode(self):
        self.record("mode", self.mode)
        return self.mode

    def standUp(self):
        self.record("stand", self.stand_result)
        self.mode = 1  # A failing native call can still have replaced its cached command.
        return self.stand_result

    def lieDown(self):
        result = 0x3007 if self.lie_rejections else 0
        self.lie_rejections = max(0, self.lie_rejections - 1)
        self.record("lie", result)
        if result == 0:
            self.mode = 51
        return result

    def move(self, vx, vy, yaw):
        self.record("move", vx, yaw)
        if self.mode in (0, 10, 51):
            return 0x3007
        self.mode = 18  # Even zero speed changes the SDK's last command mode.
        return 0

    def getBatteryPower(self):
        return 77


def check_postures():
    for mode, rejects, stand_result in ((0, 0, 0), (10, 0, 0), (51, 0, 0),
                                        (1, 0, 0), (18, 0, 0), (21, 0, 0),
                                        (1, 1, 0), (1, 0, 0x3007)):
        cfg = copy.deepcopy(CONFIG)
        cfg["_parent_pid"] = os.getppid()
        cfg["robot"].update(local_ip="127.0.0.1", port=0, sdk_path=str(ROOT),
                            stand_delay=.08, lie_down_delay=.06, record_battery=True)
        sdk = FakeSDK(mode, rejects, stand_result)
        stop, statuses = threading.Event(), queue.Queue()
        with mock.patch.object(robot.importlib, "import_module", return_value=SimpleNamespace(HighLevel=lambda: sdk)):
            worker = threading.Thread(target=robot.robot_worker,
                                      args=(cfg, queue.Queue(), stop, statuses, False), daemon=True)
            worker.start()
            deadline = time.monotonic() + 2.5
            packets = []
            try:
                while time.monotonic() < deadline:
                    try:
                        packet = statuses.get(timeout=.1)
                    except queue.Empty:
                        continue
                    packets.append(packet)
                    if packet["state"] == "error":
                        assert stand_result, packet
                        break
                    if packet["state"] in ("initializing", "stopping", "lying_down", "standing"):
                        assert not packet["ready"], packet
                    if packet["state"] == "waiting_target" and packet["ready"]:
                        break
                assert packets, "No control status"
                if not stand_result:
                    assert any(p["state"] == "standing" for p in packets)
                    assert packets[-1]["state"] == "waiting_target" and packets[-1]["ready"], packets
                    assert packets[-1]["battery"] == 77
                else:
                    assert packets[-1]["state"] == "error", packets
            finally:
                stop.set()
                worker.join(timeout=2)
            assert not worker.is_alive()
            while not statuses.empty():
                packets.append(statuses.get_nowait())
        stands = [call for call in sdk.calls if call[0] == "stand"]
        lies = [call for call in sdk.calls if call[0] == "lie" and call[2] == 0]
        assert len(stands) == 1, sdk.calls
        assert len(lies) == (1 if mode in (0, 10, 51) else 2), sdk.calls
        assert sdk.calls[0] == next(call for call in sdk.calls if call[0] == "mode"), sdk.calls
        if mode not in (0, 10, 51):
            assert stands[0][1] - lies[0][1] >= cfg["robot"]["lie_down_delay"] - .005
            assert not any(c[0] == "move" and lies[0][1] < c[1] < stands[0][1] for c in sdk.calls)
        after_stand = [call for call in sdk.calls if call[0] == "move" and call[1] > stands[0][1]]
        assert after_stand, sdk.calls
        if not stand_result:
            assert after_stand[0][1] - stands[0][1] >= cfg["robot"]["stand_delay"] - .005
        assert sdk.calls[-1] == lies[-1], sdk.calls  # No move may overwrite the accepted shutdown command.
        assert time.monotonic() - lies[-1][1] >= cfg["robot"]["lie_down_delay"] - .005
        assert packets[-1]["lie_down_sent"] and packets[-1]["state"] == ("error" if stand_result else "stopped")
        if rejects:
            rejected = next(c for c in sdk.calls if c[0] == "lie" and c[2] == 0x3007)
            assert any(c[0] == "move" and rejected[1] < c[1] < lies[0][1] for c in sdk.calls)
        assert all(call[2:] == (0.0, 0.0) for call in sdk.calls if call[0] == "move")


def check_battery():
    cases = [(True, initial, final) for initial, final in
             ((80, 77), (80, 80), (0, 0), (80, 81), (-1, 77), (80, 101), (80, None))]
    for enabled, initial, final in cases + [(False, 80, 77)]:
        cfg = copy.deepcopy(CONFIG)
        cfg["_parent_pid"] = os.getppid()
        cfg["robot"].update(local_ip="127.0.0.1", port=0, sdk_path=str(ROOT),
                            stand_delay=.01, lie_down_delay=.01, record_battery=enabled)
        sdk, samples, packets = FakeSDK(0), [], []
        readings = iter((initial, final))

        def read():
            samples.append(sdk.mode)
            return next(readings)

        sdk.getBatteryPower = read
        stop, statuses = threading.Event(), queue.Queue()
        with mock.patch.object(robot.importlib, "import_module", return_value=SimpleNamespace(HighLevel=lambda: sdk)):
            worker = threading.Thread(target=robot.robot_worker,
                                      args=(cfg, queue.Queue(), stop, statuses, False), daemon=True)
            worker.start()
            try:
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    packet = statuses.get(timeout=1)
                    packets.append(packet)
                    assert packet["state"] != "error", packet
                    if packet["ready"]:
                        if not enabled:
                            time.sleep(1.1)  # Cover the periodic read as well as startup and shutdown.
                        break
                else:
                    raise AssertionError("Fake dog never became ready")
            finally:
                stop.set()
                worker.join(timeout=2)
            assert not worker.is_alive()
        while not statuses.empty():
            packet = statuses.get_nowait()
            packets.append(packet)
        assert packet["state"] == "stopped" and packet["lie_down_sent"], packet
        if not enabled:
            assert samples == [], samples
            assert all(not any(key.startswith("battery") for key in item) for item in packets), packets
            continue
        expected_start = initial if 0 <= initial <= 100 else None
        expected_end = final if final is not None and 0 <= final <= 100 else None
        expected_used = expected_start - expected_end if expected_start is not None and expected_end is not None else None
        assert packet["battery_start_percent"] == expected_start, packet
        assert packet["battery_end_percent"] == expected_end, packet
        assert packet["battery_used_percent_points"] == expected_used, packet
        assert samples == [0, 51], samples  # Start before standUp; finish after shutdown lieDown.


def check_signal_reentrancy():
    ctx = mp.get_context("fork")

    def interrupted_with_lock_held():
        stop = ctx.Event()
        requested = robot.install_stop_signals(stop)
        with stop._cond:  # Deterministically interrupt Event's non-reentrant lock.
            for signum in (signal.SIGINT, signal.SIGINT, signal.SIGTERM):
                os.kill(os.getpid(), signum)
        assert requested() and stop.is_set()

    worker = ctx.Process(target=interrupted_with_lock_held)
    worker.start()
    try:
        worker.join(timeout=2)
        assert not worker.is_alive(), "Signal handler deadlocked on the stop Event lock"
        assert worker.exitcode == 0, worker.exitcode
    finally:
        if worker.is_alive():
            worker.kill()
            worker.join(timeout=1)


def check_shutdown_signals():
    ctx = mp.get_context("fork")
    for mode, stop_state, signals in ((1, "lying_down", (signal.SIGINT,)),
                                     (1, "lying_down", (signal.SIGINT, signal.SIGINT, signal.SIGTERM)),
                                     (0, "standing", (signal.SIGINT,)),
                                     (0, "standing", (signal.SIGINT, signal.SIGINT, signal.SIGTERM))):
        cfg = copy.deepcopy(CONFIG)
        cfg["_parent_pid"] = os.getpid()
        cfg["robot"].update(local_ip="127.0.0.1", port=0, sdk_path=str(ROOT),
                            stand_delay=.5, lie_down_delay=.15)
        stop, statuses, calls = ctx.Event(), ctx.Queue(), ctx.Queue()

        def run_fake():
            sdk = FakeSDK(mode, call_queue=calls)
            with mock.patch.object(robot.importlib, "import_module", return_value=SimpleNamespace(HighLevel=lambda: sdk)):
                robot.robot_worker(cfg, queue.Queue(), stop, statuses, False)
            calls.put(("finished", time.monotonic()))

        worker = ctx.Process(target=run_fake)
        worker.start()
        try:
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                packet = statuses.get(timeout=1)
                assert packet["state"] != "error", packet
                if packet["state"] == stop_state:
                    assert packet["lie_down_sent"] == (stop_state == "lying_down") and not packet["ready"], packet
                    break
            else:
                raise AssertionError(f"Fake worker never reached {stop_state}")
            for signum in signals:
                os.kill(worker.pid, signum)
                time.sleep(.02)
            worker.join(timeout=2)
            assert not worker.is_alive() and worker.exitcode == 0 and stop.is_set()
            recorded = []
            while True:
                try:
                    recorded.append(calls.get(timeout=.05))
                except queue.Empty:
                    break
            assert len([c for c in recorded if c[0] == "stand"]) == (1 if stop_state == "standing" else 0), recorded
            lies = [c for c in recorded if c[0] == "lie"]
            assert len(lies) == 1 and lies[0][2] == 0, recorded
            assert not any(c[0] == "move" and c[1] > lies[0][1] for c in recorded), recorded
            assert recorded[-1][0] == "finished" and recorded[-1][1] - lies[0][1] >= .145, recorded
        finally:
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=1)
            for channel in (statuses, calls):
                channel.close()
                channel.join_thread()


if __name__ == "__main__":
    check_control()
    check_vision_and_camera()
    check_worker()
    check_postures()
    check_battery()
    check_signal_reentrancy()
    check_shutdown_signals()
    check_motion_window()
    check_config()
    print("PASS: controller, target lock, depth, freshness, watchdog, posture order, shutdown hold/signals/reentrancy, battery, config")
