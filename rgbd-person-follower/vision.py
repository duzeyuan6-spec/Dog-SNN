"""CPU person tracking with an explicit, permanent target ID."""

from math import atan2, hypot, isfinite
from pathlib import Path

import numpy as np
from ultralytics import YOLO


class PersonTracker:
    def __init__(self, model_path, config):
        self.cfg = config.get("vision", {})
        if not Path(model_path).is_file():
            raise FileNotFoundError(model_path)
        self.model = YOLO(str(model_path))
        self.tracker_path = str(Path(__file__).with_name("botsort.yaml"))
        self.target_id = None
        self.candidate_id = None
        self.candidate_since = 0.0
        self.candidate_frames = 0
        self.recovery_frames = 0
        self.last_good = None
        self.detections = []
        self.last_sequence = -1
        self.last_frame_at = None
        self.frame_timeout = config.get("robot", {}).get("observation_timeout", .45)

    def _empty(self, frame, reason):
        return dict(valid=False, reason=reason, track_id=self.target_id,
                    distance=0.0, bearing=0.0, box=None, confidence=0.0,
                    captured_at=frame.captured_at, sequence=frame.sequence)

    def _depth(self, depth, box):
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        # The torso avoids most background, feet and foreground hands.
        roi = depth[int(y1 + .30 * height):int(y1 + .60 * height),
                    int(x1 + .30 * width):int(x1 + .70 * width)]
        good = np.isfinite(roi) & (roi >= .35) & (roi <= 6.0)
        count = int(np.count_nonzero(good))
        if (count < self.cfg.get("min_depth_pixels", 30)
                or count < roi.size * self.cfg.get("min_depth_fraction", .15)):
            return None
        return float(np.median(roi[good]))

    def _observations(self, frame):
        result = self.model.track(
            frame.color, persist=True, classes=[0],
            conf=min(.1, self.cfg.get("confidence", .35)),
            imgsz=self.cfg.get("imgsz", 416), device="cpu",
            tracker=self.tracker_path, verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []
        height, width = frame.color.shape[:2]
        observations = []
        for coords, confidence, track_id in zip(
                boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(),
                boxes.id.int().cpu().tolist()):
            if not all(isfinite(float(v)) for v in coords):
                continue
            x1, y1, x2, y2 = coords
            box = (max(0, min(width, int(x1))), max(0, min(height, int(y1))),
                   max(0, min(width, int(x2))), max(0, min(height, int(y2))))
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            u = (box[0] + box[2]) / 2
            observations.append(dict(
                track_id=int(track_id), distance=self._depth(frame.depth_m, box),
                bearing=atan2(frame.cx - u, frame.fx), box=box,
                confidence=float(confidence), center=u,
            ))
        return observations

    def _continuous(self, observation, frame):
        if self.last_good is None:
            return True
        previous = self.last_good
        dt = max(0.0, min(1.0, frame.captured_at - previous["captured_at"]))
        if abs(observation["distance"] - previous["distance"]) > .65 + .6 * dt:
            return False
        old, new = previous["box"], observation["box"]
        movement = hypot((old[0] + old[2] - new[0] - new[2]) / 2,
                         (old[1] + old[3] - new[1] - new[3]) / 2)
        limit = max(.12 * frame.color.shape[1], .8 * (old[2] - old[0]))
        return movement <= limit + .15 * frame.color.shape[1] * dt

    def update(self, frame):
        if (not isfinite(float(frame.fx)) or frame.fx <= 0
                or not isfinite(float(frame.cx))
                or not isfinite(float(frame.captured_at))
                or frame.color.ndim != 3 or frame.color.shape[2] != 3
                or frame.depth_m.shape != frame.color.shape[:2]):
            self.recovery_frames = 0
            self.candidate_id = None
            self.candidate_frames = 0
            return self._empty(frame, "invalid_frame")
        if frame.sequence <= self.last_sequence:
            return self._empty(frame, "repeated_frame")
        self.last_sequence = frame.sequence
        if (self.last_frame_at is not None
                and frame.captured_at - self.last_frame_at > self.frame_timeout):
            self.recovery_frames = 0
            self.candidate_id = None
            self.candidate_frames = 0
        self.last_frame_at = frame.captured_at
        observations = self._observations(frame)
        self.detections = observations
        if self.target_id is None:
            width = frame.color.shape[1]
            candidates = [o for o in observations
                          if o["distance"] is not None
                          and o["confidence"] >= self.cfg.get("confidence", .35)
                          and o["distance"] <= self.cfg.get("acquire_max_distance", 3.0)
                          and abs(o["center"] - width / 2)
                          <= width * self.cfg.get("acquire_center_fraction", .25)]
            if not candidates:
                self.candidate_id = None
                self.candidate_frames = 0
                return self._empty(frame, "waiting_for_person")
            observation = min(candidates, key=lambda o: abs(o["center"] - width / 2))
            if observation["track_id"] != self.candidate_id:
                self.candidate_id = observation["track_id"]
                self.candidate_since = frame.captured_at
                self.candidate_frames = 0
            self.candidate_frames += 1
            if (self.candidate_frames < self.cfg.get("acquire_frames", 3)
                    or frame.captured_at - self.candidate_since
                    < self.cfg.get("acquire_seconds", .6)):
                return self._empty(frame, "acquiring_person")
            self.target_id = self.candidate_id
            self.recovery_frames = 3
        else:
            matches = [o for o in observations if o["track_id"] == self.target_id]
            if not matches:
                self.recovery_frames = 0
                return self._empty(frame, "target_missing")
            observation = matches[0]

        # ponytail: ID + spatial/depth gates have no appearance ReID; crossing
        # people can still confuse BoT-SORT. Stop on jumps; never select a new ID.
        if observation["distance"] is None:
            self.recovery_frames = 0
            return self._empty(frame, "depth_invalid")
        if not self._continuous(observation, frame):
            self.recovery_frames = 0
            return self._empty(frame, "target_jump")
        self.recovery_frames = min(3, self.recovery_frames + 1)
        if self.recovery_frames < 3:
            return self._empty(frame, "confirming_target")
        output = dict(valid=True, reason="following", track_id=self.target_id,
                      distance=observation["distance"], bearing=observation["bearing"],
                      box=observation["box"], confidence=observation["confidence"],
                      captured_at=frame.captured_at, sequence=frame.sequence)
        self.last_good = output
        return output
