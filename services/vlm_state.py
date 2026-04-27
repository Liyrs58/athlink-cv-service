import cv2
import numpy as np
from enum import Enum
from typing import Dict, List, Optional
import torch


class GameState(Enum):
    PLAY = "play"
    BENCH_SHOT = "bench_shot"


class PlayerRoster:
    """Stores appearance snapshots of each track before a camera cut."""

    def __init__(self):
        self.roster: Dict[int, dict] = {}  # tid -> {color_hist, last_bbox, last_frame}

    def snapshot(self, frame: np.ndarray, tracks: List, video_frame: int):
        """Save appearance of each active track."""
        self.roster = {}
        for t in tracks:
            if len(t) < 5:
                continue
            tid = int(t[4])
            x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            self.roster[tid] = {
                "color_hist": self._color_hist(crop),
                "last_bbox": [float(t[0]), float(t[1]), float(t[2]), float(t[3])],
                "last_frame": video_frame,
            }

    def remap(self, frame: np.ndarray, new_tracks: List, video_frame: int) -> Dict[int, int]:
        """
        Match new track IDs after cut back to old roster IDs.
        Returns {new_tid: old_tid}
        """
        if not self.roster:
            return {}

        mapping = {}
        used_old = set()

        for t in new_tracks:
            if len(t) < 5:
                continue
            new_tid = int(t[4])
            x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            new_hist = self._color_hist(crop)

            best_score = -1
            best_old_tid = None
            for old_tid, data in self.roster.items():
                if old_tid in used_old:
                    continue
                score = self._hist_similarity(new_hist, data["color_hist"])
                if score > best_score and score > 0.6:  # 60% similarity threshold
                    best_score = score
                    best_old_tid = old_tid

            if best_old_tid is not None:
                mapping[new_tid] = best_old_tid
                used_old.add(best_old_tid)

        print(f"[VLM] ReID remap: matched {len(mapping)}/{len(new_tracks)} tracks at frame {video_frame}")
        return mapping

    def _color_hist(self, crop: np.ndarray) -> np.ndarray:
        """HSV color histogram for kit color matching."""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0], None, [18], [0, 180]).flatten()
        s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
        hist = np.concatenate([h, s])
        norm = np.linalg.norm(hist)
        return hist / norm if norm > 0 else hist

    def _hist_similarity(self, h1: np.ndarray, h2: np.ndarray) -> float:
        """Cosine similarity between two histograms."""
        return float(np.dot(h1, h2))


class VLMStateMachine:
    """
    Detects camera cuts (bench shots, fast pans) using green field heuristics.
    Saves player roster before cut and remaps IDs after cut using kit color ReID.
    """

    def __init__(self, device="cuda"):
        self.device = device
        self.state = GameState.PLAY
        self.prev_state = GameState.PLAY
        self.roster = PlayerRoster()
        self.pending_remap = False  # True on first PLAY frame after cut
        self.frame_counter = 0
        print("[VLM] State machine ready (green-field heuristic + kit-color ReID)")

    def analyze(self, frame: np.ndarray, video_frame: int) -> GameState:
        """Classify current frame as PLAY or BENCH_SHOT every 10 frames."""
        self.frame_counter += 1
        if self.frame_counter % 10 != 0:
            return self.state

        self.prev_state = self.state
        self.state = self._classify(frame)

        if self.state != self.prev_state:
            print(f"[VLM] Frame {video_frame}: {self.prev_state.value} → {self.state.value}")

        return self.state

    def _classify(self, frame: np.ndarray) -> GameState:
        """Green field detection: <25% green = bench/cutaway."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        green_ratio = mask.sum() / (frame.shape[0] * frame.shape[1] * 255)
        return GameState.BENCH_SHOT if green_ratio < 0.25 else GameState.PLAY

    def on_cut_start(self, frame: np.ndarray, active_tracks: List, video_frame: int):
        """Call when transitioning PLAY → BENCH_SHOT. Saves roster."""
        self.roster.snapshot(frame, active_tracks, video_frame)
        self.pending_remap = True
        print(f"[VLM] Roster saved: {len(self.roster.roster)} players at frame {video_frame}")

    def on_cut_end(self, frame: np.ndarray, new_tracks: List, video_frame: int) -> Dict[int, int]:
        """Call when transitioning BENCH_SHOT → PLAY. Returns ID remap."""
        if not self.pending_remap:
            return {}
        self.pending_remap = False
        return self.roster.remap(frame, new_tracks, video_frame)
