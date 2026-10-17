"""Person tracking with appearance-verified target recovery."""

from collections import deque
from math import atan2, hypot, isfinite
import os
from pathlib import Path
from time import monotonic

import cv2
import numpy as np
import torch

_settings_root = Path(__file__).resolve().parent / "output"
_settings_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_settings_root))
from ultralytics import YOLO
from ultralytics.utils.nms import TorchNMS


class PersonTracker:
    def __init__(self, model_path, config):
        self.cfg = config.get("vision", {})
        self.device = torch.device(self.cfg.get("device", "cpu"))
        if self.device.type == "cuda" and (not torch.cuda.is_available()
                or (self.device.index or 0) >= torch.cuda.device_count()):
            raise RuntimeError(f"Requested vision.device={self.device}, but that CUDA device is unavailable")
        if not Path(model_path).is_file():
            raise FileNotFoundError(model_path)
        self.model = YOLO(str(model_path))
        self.model.add_callback("on_predict_postprocess_end", self._suppress_overlaps)
        self.tracker_path = str(Path(__file__).with_name("botsort.yaml"))
        self.reid = torch.jit.load(str(Path(__file__).with_name("osnet_x0_25_msmt17.ts")),
                                   map_location=self.device).eval()
        self.anchor = None
        self.gallery = deque(maxlen=8)
        self.gallery_at = 0.0
        self.target_id = None
        self.candidate = None
        self.recovering = True
        self.last_good = None
        self.detections = []
        self.last_sequence = -1
        self.last_frame_at = None
        self.frame_timeout = config.get("robot", {}).get("observation_timeout", .45)
        self.max_distance = config.get("control", {}).get("max_distance", 5.5)

    def _suppress_overlaps(self, predictor):
        # YOLO26 skips NMS; remove duplicate person boxes before assigning IDs.
        for result in predictor.results:
            boxes = result.boxes
            if boxes is not None:
                result.update(boxes=boxes.data[TorchNMS.nms(boxes.xyxy, boxes.conf, .5)])

    def _empty(self, frame, reason):
        self.recovering = True
        if reason not in ("acquiring_person", "confirming_target"):
            self.candidate = None
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
            imgsz=self.cfg.get("imgsz", 416), device=self.device,
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

    def _features(self, frame, observations):
        crops = []
        for observation in observations:
            x1, y1, x2, y2 = observation["box"]
            crops.append(cv2.resize(frame.color[y1:y2, x1:x2], (128, 256))[:, :, ::-1])
        pixels = np.ascontiguousarray(np.stack(crops).transpose(0, 3, 1, 2))
        batch = torch.from_numpy(pixels).to(device=self.device, dtype=torch.float32) / 255
        mean = batch.new_tensor([.485, .456, .406])[None, :, None, None]
        std = batch.new_tensor([.229, .224, .225])[None, :, None, None]
        # ponytail: CUDA shape profiling stalls for >1s on new person counts.
        # Use the frozen graph directly; a compiled engine is the speed upgrade path.
        with torch.inference_mode(), torch.jit.optimized_execution(self.device.type != "cuda"):
            return self.reid((batch - mean) / std).cpu().numpy()

    def _continuous(self, observation, frame, previous):
        if previous is None:
            return True
        dt = max(0.0, min(1.0, frame.captured_at - previous["captured_at"]))
        if abs(observation["distance"] - previous["distance"]) > .65 + .6 * dt:
            return False
        old, new = previous["box"], observation["box"]
        movement = hypot((old[0] + old[2] - new[0] - new[2]) / 2,
                         (old[1] + old[3] - new[1] - new[3]) / 2)
        limit = max(.12 * frame.color.shape[1], .8 * (old[2] - old[0]))
        return movement <= limit + .15 * frame.color.shape[1] * dt

    def _confirmed(self, observation, feature, frame):
        previous = self.candidate
        if (previous is None or observation["track_id"] != previous["track_id"]
                or not self._continuous(observation, frame, previous)
                or float(feature @ previous["feature"]) < self.cfg.get("reid_similarity", .7)):
            since, count = frame.captured_at, 1
            short_gap = (self.last_good is not None and observation["track_id"] == self.target_id
                         and frame.captured_at - self.last_good["captured_at"] <= self.frame_timeout
                         and self._continuous(observation, frame, self.last_good))
        else:
            since, count = previous["since"], previous["count"] + 1
            short_gap = previous["short_gap"]
        self.candidate = dict(observation, feature=feature.copy(), since=since, count=count,
                              captured_at=frame.captured_at, short_gap=short_gap)
        return (count >= max(3, self.cfg.get("acquire_frames", 3))
                and (short_gap or frame.captured_at - since >= max(.6, self.cfg.get("acquire_seconds", .6))))

    def update(self, frame, deadline=None):
        if (not isfinite(float(frame.fx)) or frame.fx <= 0
                or not isfinite(float(frame.cx))
                or not isfinite(float(frame.captured_at))
                or frame.color.ndim != 3 or frame.color.shape[2] != 3
                or frame.depth_m.shape != frame.color.shape[:2]):
            return self._empty(frame, "invalid_frame")
        if frame.sequence <= self.last_sequence:
            return self._empty(frame, "repeated_frame")
        if self.last_frame_at is not None and frame.captured_at <= self.last_frame_at:
            return self._empty(frame, "invalid_frame")
        if deadline is not None and monotonic() > deadline:
            return self._empty(frame, "inference_too_slow")
        self.last_sequence = frame.sequence
        if (self.last_frame_at is not None
                and frame.captured_at - self.last_frame_at > self.frame_timeout):
            self.recovering = True
            self.candidate = None
        self.last_frame_at = frame.captured_at
        observations = self._observations(frame)
        self.detections = observations
        candidates = observations
        if not candidates:
            return self._empty(frame, "target_missing" if self.target_id is not None else "waiting_for_person")
        features = np.asarray(self._features(frame, candidates), dtype=np.float32)
        # Live frames have a deadline; offline recordings use their own timeline.
        # Reject slow inference before it can teach the gallery or confirm a target.
        if deadline is not None and monotonic() > deadline:
            return self._empty(frame, "inference_too_slow")
        if (features.ndim != 2 or features.shape[0] != len(candidates)
                or not np.isfinite(features).all()):
            return self._empty(frame, "appearance_invalid")
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        if (norms <= 1e-8).any() or not np.isfinite(norms).all():
            return self._empty(frame, "appearance_invalid")
        features = features / norms
        steady = False
        if self.target_id is None:
            width = frame.color.shape[1]
            choices = [i for i, o in enumerate(candidates)
                          if o["distance"] is not None and isfinite(o["distance"])
                          and o["confidence"] >= self.cfg.get("confidence", .35)
                          and .35 <= o["distance"] <= self.cfg.get("acquire_max_distance", 3.0)
                          and abs(o["center"] - width / 2)
                          <= width * self.cfg.get("acquire_center_fraction", .25)]
            if not choices:
                return self._empty(frame, "waiting_for_person")
            index = min(choices, key=lambda i: abs(candidates[i]["center"] - width / 2))
            observation, feature = candidates[index], features[index]
            if not self._confirmed(observation, feature, frame):
                return self._empty(frame, "acquiring_person")
            self.anchor = feature.copy()
        else:
            scores = (features @ np.stack([self.anchor, *self.gallery]).T).max(axis=1)
            anchors = features @ self.anchor
            for o, score, anchor in zip(candidates, scores, anchors):
                o.update(reid_similarity=float(score), anchor_similarity=float(anchor))
            order = np.argsort(scores)[::-1]
            index = int(order[0])
            score = float(scores[index])
            margin = score - float(scores[order[1]]) if len(order) > 1 else 1.0
            observation, feature = candidates[index], features[index]
            steady = (not self.recovering and observation["track_id"] == self.target_id
                      and observation["distance"] is not None and isfinite(observation["distance"])
                      and self._continuous(observation, frame, self.last_good))
            # Small hysteresis keeps an already verified, continuous track from
            # chattering near the threshold; recovery still requires the full score.
            threshold = self.cfg.get("reid_similarity", .7) - (.05 if steady else 0)
            if score < threshold or anchors[index] < .5:
                return self._empty(frame, "target_unverified")
            if margin < self.cfg.get("reid_margin", .1):
                return self._empty(frame, "target_ambiguous")
            if (observation["track_id"] != self.target_id
                    and observation["confidence"] < self.cfg.get("confidence", .35)):
                return self._empty(frame, "target_unverified")
            if observation["distance"] is None or not isfinite(observation["distance"]):
                return self._empty(frame, "depth_invalid")
            if not .35 <= observation["distance"] <= self.max_distance:
                return self._empty(frame, "target_out_of_range")
            if not steady and not self._confirmed(observation, feature, frame):
                return self._empty(frame, "confirming_target")
            # ponytail: eight trusted recent views plus a frozen initial anchor.
            # Similar clothing can still confuse OSNet; use stronger ReID or an
            # explicit target marker if this appearance-only ceiling is reached.
            box = observation["box"]
            overlaps = any(o is not observation and min(box[2], o["box"][2]) > max(box[0], o["box"][0])
                           and min(box[3], o["box"][3]) > max(box[1], o["box"][1]) for o in candidates)
            if (steady and not overlaps and observation["confidence"] >= .7
                    and max(.75, self.cfg.get("reid_similarity", .7)) <= score < .95
                    and margin >= max(.15, self.cfg.get("reid_margin", .1))
                    and frame.captured_at - self.gallery_at >= .5):
                self.gallery.append(feature.copy())
                self.gallery_at = frame.captured_at
        self.target_id = observation["track_id"]
        self.recovering = False
        self.candidate = None
        output = dict(valid=True, reason="following", track_id=self.target_id,
                      distance=observation["distance"], bearing=observation["bearing"],
                      box=observation["box"], confidence=observation["confidence"],
                      captured_at=frame.captured_at, sequence=frame.sequence)
        self.last_good = output
        return output
