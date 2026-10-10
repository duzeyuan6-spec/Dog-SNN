"""Single owner of the dog SDK, with a camera freshness watchdog."""

from __future__ import annotations

import importlib
import math
import os
from pathlib import Path
import queue
import signal
import socket
import sys
import threading
import time

from control import FollowController


def robot_worker(config, observations, stop_event, status_queue, dry_run=False):
    """Keep zero speed unless a fresh, valid observation exists."""
    options = config["robot"]
    parent_pid = int(config.get("_parent_pid", os.getppid()))
    period = 1.0 / float(options.get("rate_hz", 50))
    max_age = float(options.get("observation_timeout", 0.45))
    controller = FollowController(config["control"])
    app = None
    ready = False
    battery = None
    state = "initializing"
    error = None
    latest = None
    last_sequence = -1
    last_captured_at = -math.inf
    last_status = -math.inf
    last_state = None
    last_battery = -math.inf
    standing = bool(dry_run or not options.get("auto_stand", True))
    stand_requested = False
    settle_until = 0.0
    observation_age = None
    vx = yaw = 0.0
    motion_deadline = None
    lie_down_sent = False

    # Ctrl+C reaches both processes; repeated signals must not interrupt cleanup.
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, lambda *_: stop_event.set())

    def publish(force=False):
        nonlocal last_status, last_state
        now = time.monotonic()
        if not force and state == last_state and now - last_status < 1.0:
            return
        packet = dict(state=state, vx=vx, yaw=yaw, battery=battery,
                      observation_age=observation_age, ready=ready,
                      error=error, dry_run=bool(dry_run), lie_down_sent=lie_down_sent)
        try:
            status_queue.put_nowait(packet)
        except queue.Full:
            try:
                status_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                status_queue.put_nowait(packet)
            except queue.Full:
                pass
        last_status, last_state = now, state

    def send(forward, turn):
        if not math.isfinite(forward) or not math.isfinite(turn):
            raise ValueError("Non-finite velocity rejected")
        if app is not None:
            code = app.move(float(forward), 0.0, float(turn))
            # While lying down or completing standUp, zero speed can be refused
            # with state-transition error; no nonzero command is permitted here.
            waiting_for_stand = not standing or time.monotonic() < settle_until
            if code == 0x3007 and forward == 0 and turn == 0 and waiting_for_stand:
                return
            if code not in (None, 0):
                raise RuntimeError(f"Dog move rejected: 0x{int(code):04x}")

    def lower():
        nonlocal state, lie_down_sent
        state = "stopping"
        publish(True)
        # Moving -> lieDown is forbidden. Keep zero speed before switching modes.
        until = time.monotonic() + .5
        while time.monotonic() < until:
            code = app.move(0.0, 0.0, 0.0)
            if code not in (None, 0, 0x3007):
                raise RuntimeError(f"Dog stop rejected: 0x{int(code):04x}")
            time.sleep(period)
        deadline = time.monotonic() + 2.0
        while True:
            if not app.checkConnect():
                raise RuntimeError("Cannot lower dog: SDK feedback disconnected")
            code = app.lieDown()
            if code in (None, 0):
                break
            if code != 0x3007 or time.monotonic() >= deadline:
                raise RuntimeError(f"Dog lieDown rejected: 0x{int(code):04x}")
            app.move(0.0, 0.0, 0.0)
            time.sleep(.2)
        lie_down_sent = True
        state = "lying_down"
        publish(True)
        # Match the SDK demo's 3s hold. A move(0) here would replace lieDown.
        time.sleep(float(options.get("lie_down_delay", 3.0)))

    try:
        if stop_event.is_set() or os.getppid() != parent_pid:
            return
        publish(True)
        if not dry_run:
            local_ip = options["local_ip"]
            dog_ip = options["dog_ip"]
            port = int(options.get("port", 43988))
            # A bind diagnoses a missing local address or competing SDK owner.
            # Never reconfigure the host's network to make this pass.
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                try:
                    probe.bind((local_ip, port))
                except OSError as exc:
                    raise RuntimeError(
                        f"Cannot bind dog SDK to {local_ip}:{port}; check Ethernet "
                        f"address and stop other dog controllers: {exc}"
                    ) from exc
            sdk_path = Path(options["sdk_path"]).expanduser().resolve()
            if not sdk_path.is_dir():
                raise RuntimeError(f"Dog SDK directory missing: {sdk_path}")
            sys.path.insert(0, str(sdk_path))
            sdk = importlib.import_module("mc_sdk_zsl_1_py")
            app = sdk.HighLevel()
            deadline = time.monotonic() + min(
                8.0, float(options.get("connect_timeout", 8.0)))
            # ponytail: native init can hang; the parent enforces its own startup
            # deadline. This process cannot guarantee an OS/SDK hang hard stop.
            app.initRobot(local_ip, port, dog_ip)
            while not app.checkConnect():
                try:
                    app.move(0.0, 0.0, 0.0)
                except Exception:
                    pass
                if stop_event.is_set() or os.getppid() != parent_pid:
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"No dog SDK feedback from {dog_ip} within 8 s")
                stop_event.wait(0.05)
            initial_mode = int(app.getCurrentCtrlmode())
            send(0.0, 0.0)
        ready = standing
        state = "waiting_target"
        if not standing:
            # Start from a known low posture before every stand, even if already up.
            if initial_mode not in (0, 10, 51):
                lower()
            if stop_event.is_set() or os.getppid() != parent_pid:
                return
            lie_down_sent = False
            code = app.standUp()
            if code not in (None, 0):
                raise RuntimeError(f"Dog standUp rejected: 0x{int(code):04x}")
            stand_requested = True
            standing = True
            settle_until = time.monotonic() + float(options.get("stand_delay", 10.0))
            state = "standing"
        publish(True)
        previous_tick = time.monotonic()
        while not stop_event.is_set() and os.getppid() == parent_pid:
            tick = time.monotonic()
            dt = max(0.001, min(tick - previous_tick, 0.1))
            previous_tick = tick
            while True:
                try:
                    item = observations.get_nowait()
                except queue.Empty:
                    break
                try:
                    if item.get("valid") is False:
                        # A camera/vision failure must revoke the command even
                        # when no new frame (and therefore no new ID) arrived.
                        latest = item
                        continue
                    sequence = int(item["sequence"])
                    captured_at = float(item["captured_at"])
                    if not math.isfinite(captured_at) or captured_at > tick + 0.05:
                        raise ValueError("invalid camera capture timestamp")
                    if sequence <= last_sequence or captured_at <= last_captured_at:
                        continue
                    last_sequence, last_captured_at = sequence, captured_at
                    latest = item
                except (TypeError, ValueError, KeyError, AttributeError) as exc:
                    latest = {"valid": False, "reason": f"malformed observation: {exc}"}

            observation_age = (max(0.0, tick - last_captured_at)
                               if latest is not None and math.isfinite(last_captured_at)
                               else None)
            connected = app is None or bool(app.checkConnect())
            valid = (latest is not None and latest.get("valid") is True
                     and observation_age is not None and observation_age <= max_age)
            if valid:
                try:
                    distance, bearing = float(latest["distance"]), float(latest["bearing"])
                    valid = (math.isfinite(distance) and distance > 0
                             and math.isfinite(bearing) and abs(bearing) <= math.pi / 2)
                except (KeyError, ValueError, TypeError):
                    valid = False

            error = None
            if not connected:
                state = "disconnected"
                error = "Dog SDK feedback disconnected"
            elif tick < settle_until:
                state = "standing"
            elif latest is None:
                state = "waiting_target"
            elif latest.get("valid") is False:
                state = "target_lost"
                error = str(latest.get("reason", "Invalid target or depth"))
            elif observation_age is None or observation_age > max_age:
                state = "stale_camera"
                error = "No fresh camera observation"
            elif not valid:
                state = "target_lost"
                error = str(latest.get("reason", "Invalid target or depth"))
            else:
                state = "following"

            ready = connected and standing and tick >= settle_until

            if state == "following":
                if motion_deadline is None and config.get("_motion_seconds", 0) > 0:
                    motion_deadline = tick + config["_motion_seconds"]
                vx, yaw = controller.update(distance, bearing, dt)
            else:
                controller.reset()
                vx = yaw = 0.0
            if motion_deadline is not None and tick >= motion_deadline:
                stop_event.set()
                break
            # SDK rejects sub-minimum nonzero speeds. Hold zero until the ramp
            # reaches its documented minimum, rather than increasing a request.
            vx = vx if abs(vx) >= 0.05 else 0.0
            yaw = yaw if abs(yaw) >= 0.02 else 0.0
            # Stop bypasses acceleration limiting, including target/connection loss.
            if not (stand_requested and state == "standing"):
                send(vx, yaw)
            if app is not None and connected and tick - last_battery >= 1.0:
                raw_battery = int(app.getBatteryPower())
                battery = raw_battery if 0 <= raw_battery <= 100 else None
                last_battery = tick
            publish()
            stop_event.wait(max(0.0, period - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        state, error = "error", f"{type(exc).__name__}: {exc}"
        vx = yaw = 0.0
        publish(True)
        stop_event.set()
    finally:
        controller.reset()
        vx = yaw = 0.0
        ready = False
        failed = state in ("error", "disconnected")
        if not failed:
            error = None
        if app is not None:
            try:
                if not lie_down_sent:
                    lower()
            except Exception as exc:
                failed, error = True, f"Shutdown failed: {type(exc).__name__}: {exc}"
            app = None  # HighLevel destructor joins its native sending thread.
        state = "error" if failed else "stopped"
        publish(True)
