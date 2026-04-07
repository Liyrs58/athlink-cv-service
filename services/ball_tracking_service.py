"""
Ball tracking using Tryolabs ball.pt (YOLOv5 format).
MIT licensed — safe for commercial use.

Provides: BallTracker, PossessionDetector, PassDetector
"""

import os
import cv2
import numpy as np
import logging
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

from ultralytics import YOLO as _YOLO

_ball_model = None
_ball_model_type = None

def get_ball_model():
    """Load ball detection model from model_cache (pre-loaded at startup)."""
    global _ball_model, _ball_model_type
    if _ball_model is not None:
        return _ball_model, _ball_model_type
    try:
        from services.model_cache import get_ball_model as _cached_ball
        _ball_model = _cached_ball()
        _ball_model_type = 'ultralytics'
        logger.info("Ball detector loaded from model_cache")
    except Exception as e:
        logger.error("Ball detector load failed: %s", e)
        _ball_model = None
        _ball_model_type = None
    return _ball_model, _ball_model_type


# Lazy-load torch to avoid crashing RunPod workers at import time
torch = None

def _get_torch():
    global torch
    if torch is None:
        import torch as _torch
        torch = _torch
    return torch

BALL_MODEL_PATH = os.environ.get(
    "BALL_MODEL_PATH",
    "ball.pt",
)
INFERENCE_SIZE = 1920       # must run at full resolution
MIN_CONF = 0.3              # minimum confidence to accept detection
MAX_BALL_SIZE = 40          # pixels — reject larger detections (ball is 5-18px)
INTERPOLATE_MAX = 5         # frames to interpolate when ball lost
POSSESSION_MAX_DIST = 2.0   # metres — max distance for possession
POSSESSION_MIN_FRAMES = 3   # frames ball must be near player
BALL_CONFIDENCE_THRESHOLD = 0.45  # minimum ball confidence for possession assignment


# ---------------------------------------------------------------------------
# BallTracker
# ---------------------------------------------------------------------------

class BallTracker:
    """
    Tracks ball position across video frames.
    Uses Tryolabs YOLOv5 ball.pt model at 1920px inference.
    Includes linear interpolation for short gaps.
    """

    def __init__(self):
        self.model = None
        self.model_type: Optional[str] = None
        self._positions: Dict[int, dict] = {}   # frame_idx -> {x, y, conf, interpolated}
        self._last_pos: Optional[dict] = None
        self._last_det_frame: int = -1
        self._lost_frames: int = 0

    # ------------------------------------------------------------------
    def load_model(self):
        self.model, self.model_type = get_ball_model()

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray, frame_idx: int) -> Optional[dict]:
        """
        Detect ball in a single frame.

        Returns best detection dict or None.
        Stores result in self._positions[frame_idx].
        """
        if self.model is None:
            return None

        try:
            results = self.model(frame, imgsz=INFERENCE_SIZE, verbose=False)
            boxes = results[0].boxes

            if boxes is None or len(boxes) == 0:
                return self._handle_miss(frame_idx)

            # Filter to ball-sized detections
            best_cx = best_cy = best_conf = None
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x1
                h = y2 - y1
                if w >= MAX_BALL_SIZE or h >= MAX_BALL_SIZE:
                    continue
                c = float(box.conf[0])
                if best_conf is None or c > best_conf:
                    best_cx = (x1 + x2) / 2
                    best_cy = (y1 + y2) / 2
                    best_conf = c

            if best_conf is None:
                return self._handle_miss(frame_idx)

            cx = float(best_cx)
            cy = float(best_cy)
            conf = float(best_conf)

            pos = {"x": cx, "y": cy, "confidence": conf, "interpolated": False}
            self._positions[frame_idx] = pos
            self._lost_frames = 0

            # Back-fill interpolated positions for short gaps
            if self._last_pos is not None and self._last_det_frame >= 0:
                gap = frame_idx - self._last_det_frame
                if 1 < gap <= INTERPOLATE_MAX:
                    self._interpolate(self._last_det_frame, frame_idx)

            self._last_pos = pos
            self._last_det_frame = frame_idx

            logger.debug(
                "Frame %d: ball at (%.0f, %.0f) conf=%.3f",
                frame_idx, cx, cy, conf,
            )
            return pos

        except Exception as e:
            logger.debug("Frame %d: ball detection error: %s", frame_idx, e)
            return self._handle_miss(frame_idx)

    # ------------------------------------------------------------------
    def _handle_miss(self, frame_idx: int) -> Optional[dict]:
        """Handle frame where ball was not detected."""
        self._lost_frames += 1
        # No interpolation yet — will be filled when next detection arrives
        return None

    # ------------------------------------------------------------------
    def _interpolate(self, start_idx: int, end_idx: int):
        """Linear-interpolate ball positions between two detected frames."""
        p0 = self._positions.get(start_idx)
        p1 = self._positions.get(end_idx)
        if not p0 or not p1:
            return

        for fi in range(start_idx + 1, end_idx):
            t = (fi - start_idx) / (end_idx - start_idx)
            self._positions[fi] = {
                "x": p0["x"] + t * (p1["x"] - p0["x"]),
                "y": p0["y"] + t * (p1["y"] - p0["y"]),
                "confidence": min(p0["confidence"], p1["confidence"]) * 0.5,
                "interpolated": True,
            }

    # ------------------------------------------------------------------
    def get_positions(self) -> Dict[int, dict]:
        """Return all ball positions indexed by frame_idx."""
        return dict(self._positions)

    def get_position_at(self, frame_idx: int) -> Optional[dict]:
        """Get ball position at specific frame. None if unknown."""
        return self._positions.get(frame_idx)

    def tracking_rate(self, total_frames: int) -> float:
        """Percentage of frames where ball was located (detected or interpolated)."""
        if total_frames <= 0:
            return 0.0
        return len(self._positions) / total_frames * 100.0


# ---------------------------------------------------------------------------
# PossessionDetector
# ---------------------------------------------------------------------------

class PossessionDetector:
    """
    Infers ball possession from ball position and player positions.
    Uses distance + temporal stability (inspired by Tryolabs match.py).
    """

    def __init__(self):
        self._history: List[dict] = []       # per-frame possession states
        self._current_possessor: Optional[int] = None   # track_id
        self._current_team: Optional[int] = None
        self._frames_with_current: int = 0

    # ------------------------------------------------------------------
    def update(
        self,
        ball_pos: Optional[dict],
        players: list,
        frame_idx: int,
        pixels_per_metre: float,
    ) -> dict:
        """
        Determine which player/team has possession.

        players: list of dicts with keys: track_id, cx, cy, team_id
        pixels_per_metre: from calibration
        """
        ppm = max(pixels_per_metre, 0.1)

        if ball_pos is None or not players:
            # Carry forward current possession
            state = {
                "frame_idx": frame_idx,
                "player_id": self._current_possessor,
                "team_id": self._current_team,
                "distance_metres": None,
                "confidence": 0.0,
                "frames_held": self._frames_with_current,
            }
            self._history.append(state)
            return state

        # Gate on ball confidence — only assign possession when confidence >= 0.45
        ball_conf = ball_pos.get("confidence", 0.0)
        if ball_conf < BALL_CONFIDENCE_THRESHOLD:
            # Ball confidence too low — don't update possession
            state = {
                "frame_idx": frame_idx,
                "player_id": None,
                "team_id": None,
                "distance_metres": None,
                "confidence": 0.0,
                "frames_held": 0,
                "insufficient_data": True,
            }
            self._history.append(state)
            return state

        bx, by = ball_pos["x"], ball_pos["y"]

        # Find closest player
        best_dist_px = float("inf")
        best_player = None
        for p in players:
            dx = p["cx"] - bx
            dy = p["cy"] - by
            d = (dx * dx + dy * dy) ** 0.5
            if d < best_dist_px:
                best_dist_px = d
                best_player = p

        dist_m = best_dist_px / ppm

        if dist_m > POSSESSION_MAX_DIST or best_player is None:
            # Ball is loose — no one has possession
            state = {
                "frame_idx": frame_idx,
                "player_id": None,
                "team_id": None,
                "distance_metres": round(dist_m, 2) if best_player else None,
                "confidence": 0.0,
                "frames_held": 0,
            }
            self._history.append(state)
            return state

        candidate_id = best_player["track_id"]
        candidate_team = best_player.get("team_id")

        # Temporal stability check
        if candidate_id == self._current_possessor:
            self._frames_with_current += 1
        else:
            if self._frames_with_current < POSSESSION_MIN_FRAMES:
                # Too brief — keep current possessor (ignore noise)
                pass
            else:
                # Switch
                self._current_possessor = candidate_id
                self._current_team = candidate_team
                self._frames_with_current = 1

        # If first ever possessor
        if self._current_possessor is None:
            self._current_possessor = candidate_id
            self._current_team = candidate_team
            self._frames_with_current = 1

        conf = min(1.0, self._frames_with_current / 10.0)

        state = {
            "frame_idx": frame_idx,
            "player_id": self._current_possessor,
            "team_id": self._current_team,
            "distance_metres": round(dist_m, 2),
            "confidence": round(conf, 2),
            "frames_held": self._frames_with_current,
        }
        self._history.append(state)
        return state

    # ------------------------------------------------------------------
    def get_team_possession_pct(self) -> dict:
        """
        Return possession percentage per team from history.
        {0: 62.3, 1: 37.7}
        """
        counts = {}
        total = 0
        for h in self._history:
            tid = h.get("team_id")
            if tid is not None:
                counts[tid] = counts.get(tid, 0) + 1
                total += 1

        if total == 0:
            return {"insufficient_data": True, "message": "Insufficient data"}

        return {k: round(v / total * 100, 1) for k, v in counts.items()}

    def get_possession_events(self) -> list:
        """Return list of possession change events."""
        events = []
        prev_team = None
        for h in self._history:
            tid = h.get("team_id")
            if tid != prev_team and tid is not None:
                events.append({
                    "frame_idx": h["frame_idx"],
                    "team_id": tid,
                    "player_id": h.get("player_id"),
                })
                prev_team = tid
        return events


# ---------------------------------------------------------------------------
# PassDetector
# ---------------------------------------------------------------------------

class PassDetector:
    """
    Detects passes from possession changes within same team.
    """

    def __init__(self):
        self._passes: List[dict] = []
        self._prev_possession: Optional[dict] = None
        self._prev_ball_pos: Optional[dict] = None

    # ------------------------------------------------------------------
    def update(
        self,
        current_possession: dict,
        ball_pos: Optional[dict],
        frame_idx: int,
        fps: float,
        pixels_per_metre: float,
    ) -> Optional[dict]:
        """
        Detect if a pass just occurred.

        A pass is:
        - Same team retains possession
        - Different player receives ball
        - Ball travelled at least 3 metres
        """
        ppm = max(pixels_per_metre, 0.1)

        prev = self._prev_possession
        self._prev_possession = current_possession

        if prev is None or ball_pos is None or self._prev_ball_pos is None:
            if ball_pos is not None:
                self._prev_ball_pos = ball_pos
            return None

        prev_player = prev.get("player_id")
        curr_player = current_possession.get("player_id")
        prev_team = prev.get("team_id")
        curr_team = current_possession.get("team_id")

        # Must be same team, different player, both valid
        if (prev_team is None or curr_team is None
                or prev_team != curr_team
                or prev_player == curr_player
                or prev_player is None
                or curr_player is None):
            self._prev_ball_pos = ball_pos
            return None

        # Ball must have moved at least 3 metres
        dx = ball_pos["x"] - self._prev_ball_pos["x"]
        dy = ball_pos["y"] - self._prev_ball_pos["y"]
        dist_px = (dx * dx + dy * dy) ** 0.5
        dist_m = dist_px / ppm

        if dist_m < 3.0:
            self._prev_ball_pos = ball_pos
            return None

        timestamp = frame_idx / fps if fps > 0 else 0.0

        pass_event = {
            "type": "pass",
            "from_player_id": prev_player,
            "to_player_id": curr_player,
            "team_id": curr_team,
            "frame_idx": frame_idx,
            "timestamp": round(timestamp, 2),
            "distance_metres": round(dist_m, 1),
            "confidence": current_possession.get("confidence", 0.0),
        }
        self._passes.append(pass_event)
        self._prev_ball_pos = ball_pos

        logger.debug(
            "Frame %d: pass from P%s to P%s (%.1fm)",
            frame_idx, prev_player, curr_player, dist_m,
        )
        return pass_event

    # ------------------------------------------------------------------
    def get_passes(self) -> list:
        """Return all detected passes."""
        return list(self._passes)

    def get_passes_per_player(self) -> dict:
        """Return {player_id: pass_count} (counts outgoing passes)."""
        counts: Dict[int, int] = {}
        for p in self._passes:
            pid = p["from_player_id"]
            counts[pid] = counts.get(pid, 0) + 1
        return counts
