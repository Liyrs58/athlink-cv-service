"""
Role Filter — classifies tracks as player vs referee/official.
Non-player tracks excluded from P1-P22 identity assignment.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class RoleFilter:
    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.yellow_h_range = cfg.get("yellow_h_range", (18, 35))
        self.yellow_s_min = cfg.get("yellow_s_min", 80)
        self.orange_h_range = cfg.get("orange_h_range", (5, 18))
        self.orange_s_min = cfg.get("orange_s_min", 100)
        self.green_h_range = cfg.get("green_h_range", (35, 75))
        self.green_s_min = cfg.get("green_s_min", 100)
        self.referee_colour_ratio = cfg.get("referee_colour_ratio", 0.12)
        self.min_referee_confidence = cfg.get("min_referee_confidence", 0.55)
        self.edge_margin_px = cfg.get("edge_margin_px", 80)
        self.log_interval = cfg.get("log_interval", 30)

    def filter(self, tracks: list, frame: np.ndarray, frame_id: int = 0) -> Tuple[list, list]:
        players, officials = [], []
        h, w = frame.shape[:2]

        for t in tracks:
            bbox = t.bbox
            role, conf = self._classify(bbox, frame, h, w)
            if role == "player" or conf < self.min_referee_confidence:
                players.append(t)
            else:
                officials.append(t)

        if frame_id % self.log_interval == 0 and len(officials) > 0:
            print(f"[RoleFilter F{frame_id}] input={len(tracks)} players={len(players)} officials={len(officials)}")

        return players, officials

    def _classify(self, bbox, frame: np.ndarray, fh: int, fw: int) -> Tuple[str, float]:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        bh, bw = y2 - y1, x2 - x1
        if bh < 10 or bw < 5:
            return ("player", 0.0)

        # Torso crop (top 15-55% of bbox)
        ty1 = max(y1 + int(bh * 0.15), 0)
        ty2 = min(y1 + int(bh * 0.55), fh)
        tx1 = max(x1, 0)
        tx2 = min(x2, fw)
        if ty2 <= ty1 or tx2 <= tx1:
            return ("player", 0.0)

        crop = frame[ty1:ty2, tx1:tx2]
        if crop.size == 0:
            return ("player", 0.0)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        total = hsv.shape[0] * hsv.shape[1]
        if total == 0:
            return ("player", 0.0)

        yellow_r = np.count_nonzero(cv2.inRange(
            hsv, (self.yellow_h_range[0], self.yellow_s_min, 80),
            (self.yellow_h_range[1], 255, 255))) / total

        orange_r = np.count_nonzero(cv2.inRange(
            hsv, (self.orange_h_range[0], self.orange_s_min, 80),
            (self.orange_h_range[1], 255, 255))) / total

        green_r = np.count_nonzero(cv2.inRange(
            hsv, (self.green_h_range[0], self.green_s_min, 80),
            (self.green_h_range[1], 255, 255))) / total

        max_ratio = max(yellow_r, orange_r, green_r)

        if max_ratio > self.referee_colour_ratio:
            conf = min(max_ratio * 3.0, 0.95)
            if max_ratio > 0.25:
                conf = max(conf, 0.85)
            return ("referee", conf)

        # Assistant referee near touchline
        cx = (x1 + x2) / 2
        if (cx < self.edge_margin_px or cx > fw - self.edge_margin_px) and max_ratio > 0.06:
            return ("assistant_referee", 0.45)

        return ("player", 0.0)
