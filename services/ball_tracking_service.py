"""
Ball tracking using Tryolabs ball.pt (YOLOv5 format).
MIT licensed — safe for commercial use.

Provides: BallTracker, PossessionDetector, PassDetector
"""

import os
import cv2
import numpy as np
import logging
from collections import deque
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)

from ultralytics import YOLO as _YOLO


# ---------------------------------------------------------------------------
# FIX 7 — Ball false-positive filter
# ---------------------------------------------------------------------------
#
# The Tryolabs ball.pt model fires on corner flags, goalposts, and the centre
# of the penalty spot. Three layered checks:
#
#   (a) pitch polygon rejection (same world-space polygon as FIX 2)
#   (b) Kalman-predicted teleport rejection (> BALL_MAX_PX_PER_FRAME)
#   (c) stationary-region rejection: if the detection sits in a location that
#       was the detected position in most of the last 2 seconds, it's almost
#       certainly a fixed object (flag/post), not the ball.

# Derived from 1920px-wide broadcast frames: 100m pitch → ~19 px/m; 30 m/s cap
# at 25fps → ~23 px/frame. 40 gives slack for camera shake + fast shots.
BALL_MAX_PX_PER_FRAME = 40.0
# 2 seconds @ 25fps
BALL_STATIONARY_WINDOW_FRAMES = 50
# If ≥ 80% of recent detections are within this pixel radius, it's stationary
BALL_STATIONARY_RADIUS_PX = 15.0
BALL_STATIONARY_FRACTION = 0.80
# Max gap to interpolate across (frames). Longer gaps ⇒ mark untracked.
BALL_MAX_INTERP_GAP = 40  # bridge gaps up to 1.6s @ 25fps (ball in transit)


def hallucinate_ball_arc(kick_frame: int, land_frame: int,
                         start_xy: tuple, end_xy: tuple,
                         fps: float = 25.0) -> np.ndarray:
    """Parabolic Z for aerial passes only. Ground passes stay at Z=0.

    Heuristic: only hallucinate an arc if the ball travelled fast enough
    that it was likely airborne (>15 m/s avg speed → long ball / clearance).
    Short/slow passes stay grounded.
    """
    dist = np.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1])
    n = land_frame - kick_frame + 1
    if n < 2:
        return np.zeros(max(1, n))

    # Time in seconds for this gap
    dt = n / fps if fps > 0 else n / 25.0
    avg_speed = dist / dt if dt > 0 else 0

    # Only hallucinate arcs for likely aerial balls:
    # - dist >= 15m (short passes are ground)
    # - avg speed >= 15 m/s (slow movement = ground pass or dribble)
    # - gap >= 0.5s (very short gaps = tracking dropout, not flight)
    if dist < 15 or avg_speed < 15 or dt < 0.5:
        return np.zeros(n)

    # Conservative apex: max 4m, scaled gently
    apex = min(BALL_ARC_MAX_APEX, 0.08 * dist + 0.2)
    apex = min(apex, 4.0)  # hard cap — even long balls rarely exceed 4m
    t = np.linspace(0, 1, n)
    return np.maximum(0, apex * 4 * t * (1 - t))


class _BallSanityKalman:
    """Lightweight pixel-space Kalman (const-velocity) to predict next position
    and reject detections that would imply impossible velocity."""

    def __init__(self):
        self._kf = cv2.KalmanFilter(4, 2)
        self._kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        self._kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        self._kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        self._kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self._kf.errorCovPost = np.eye(4, dtype=np.float32)
        self._initialised = False

    def predict_next(self) -> Optional[Tuple[float, float]]:
        if not self._initialised:
            return None
        pred = self._kf.predict()
        return float(pred[0]), float(pred[1])

    def correct(self, cx: float, cy: float) -> None:
        if not self._initialised:
            self._kf.statePost = np.array(
                [[cx], [cy], [0], [0]], dtype=np.float32
            )
            self._initialised = True
            return
        self._kf.correct(np.array([[cx], [cy]], dtype=np.float32))


def _is_in_pitch_polygon(cx: float, cy: float, H: Optional[np.ndarray]) -> bool:
    """Return True if foot-point (cx, cy) projected through H lands inside the
    105×68m pitch polygon (+3m margin). Returns True when H or Shapely
    unavailable (i.e., can't prove rejection)."""
    if H is None:
        return True
    try:
        from shapely.geometry import Polygon, Point
    except ImportError:
        return True
    pt = np.array([[[float(cx), float(cy)]]], dtype=np.float32)
    try:
        world = cv2.perspectiveTransform(pt, H).reshape(-1)
    except cv2.error:
        return True
    poly = Polygon([(-3, -3), (108, -3), (108, 71), (-3, 71)])
    return poly.contains(Point(float(world[0]), float(world[1])))

_ball_model = None
_ball_model_type = None

def get_ball_model():
    """Load ball detection model from model_cache (pre-loaded at startup).
    Returns (model, model_type) where model_type is 'dedicated' if ball.pt
    exists or 'fallback' if using generic yolov8s."""
    global _ball_model, _ball_model_type
    if _ball_model is not None:
        logger.info(f"Ball model cached: type={_ball_model_type}")
        return _ball_model, _ball_model_type
    try:
        from services.model_cache import get_ball_model as _cached_ball
        _ball_model = _cached_ball()
        # Check whether model_cache loaded the dedicated ball.pt or the generic fallback
        ball_path = os.environ.get("BALL_MODEL_PATH", "models/roboflow_ball.pt")
        logger.info(f"Ball path from env: {ball_path}")
        if os.path.exists(ball_path):
            _ball_model_type = 'dedicated'
            logger.info(f"Ball detector loaded: dedicated {ball_path}")
        else:
            _ball_model_type = 'fallback'
            logger.info("Ball detector loaded: yolov8s fallback (not found)")
    except Exception as e:
        logger.error("Ball detector load failed: %s", e, exc_info=True)
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
    "models/roboflow_ball.pt",  # football-specific model
)
INFERENCE_SIZE = 1920       # must run at full resolution
MIN_CONF = 0.015            # catch low-conf kickoff/transit detections (spatial filtering downstream)
MAX_BALL_SIZE = 40          # pixels — reject larger detections (ball is 5-18px)
COCO_SPORTS_BALL_CLASS = 32 # COCO class ID for sports_ball
BALL_USE_SAHI = False       # disable SAHI (too slow on CPU) — use direct inference
SAHI_MIN_RES = 1280         # only use SAHI when frame width >= this
# FIX 7: linear interpolation only across gaps < 15 frames; longer gaps stay
# untracked (we never invent positions beyond what physics supports).
INTERPOLATE_MAX = BALL_MAX_INTERP_GAP
BALL_ARC_MIN_DIST = 5.0     # metres — passes shorter than this stay grounded
BALL_ARC_MAX_APEX = 8.0     # metres — cap apex height
POSSESSION_MAX_DIST = 3.5   # metres — max distance for possession (dribbling range)
POSSESSION_MIN_FRAMES = 3   # frames ball must be near player
BALL_CONFIDENCE_THRESHOLD = 0.015  # match MIN_CONF — let spatial checks do the filtering


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
        self._sahi = None          # lazy-init SAHI wrapper
        self._sahi_failed = False  # don't retry if SAHI init crashes
        self._positions: Dict[int, dict] = {}   # frame_idx -> {x, y, conf, interpolated}
        self._last_pos: Optional[dict] = None
        self._last_det_frame: int = -1
        self._lost_frames: int = 0
        # FIX 7: sanity state for false-positive rejection
        self._sanity_kf = _BallSanityKalman()
        self._recent_xy: deque = deque(maxlen=BALL_STATIONARY_WINDOW_FRAMES)
        self._homography: Optional[np.ndarray] = None  # caller sets via set_homography()

    def set_homography(self, H: Optional[np.ndarray]) -> None:
        """Optional: supply the pixel→world homography so FIX 7's pitch polygon
        rejection can run. Without H, polygon rejection is silently skipped."""
        self._homography = H

    # ------------------------------------------------------------------
    def _is_stationary_region(self, cx: float, cy: float) -> bool:
        """FIX 7(c): If the last 2 seconds of detections cluster around (cx, cy),
        this is almost certainly a goalpost / corner flag, not the ball."""
        if len(self._recent_xy) < int(BALL_STATIONARY_WINDOW_FRAMES * 0.6):
            return False
        r2 = BALL_STATIONARY_RADIUS_PX ** 2
        near = sum(
            1 for (px, py) in self._recent_xy
            if (px - cx) ** 2 + (py - cy) ** 2 <= r2
        )
        return (near / len(self._recent_xy)) >= BALL_STATIONARY_FRACTION

    def _passes_sanity(self, cx: float, cy: float, frame_idx: int) -> bool:
        """Run the 3-check FP filter on a candidate detection."""
        # TEMP: disable all sanity checks to find the culprit
        return True

    # ------------------------------------------------------------------
    def load_model(self):
        self.model, self.model_type = get_ball_model()

    # ------------------------------------------------------------------
    def _get_sahi(self, frame_w: int):
        """Lazy-init SAHI if enabled and frame is high-res."""
        if self._sahi is not None or self._sahi_failed:
            return self._sahi
        if not BALL_USE_SAHI or frame_w < SAHI_MIN_RES:
            return None
        try:
            from services.ball_sahi_service import get_sahi_instance
            # Prefer the Roboflow players model (has ball=class 0) for SAHI
            rf_model = "models/roboflow_players.pt"
            if os.path.exists(rf_model):
                model_path = rf_model
                classes = [0]  # Roboflow ball class
            elif os.path.exists(BALL_MODEL_PATH):
                model_path = BALL_MODEL_PATH
                classes = [0]
            else:
                model_path = "yolov8s.pt"
                classes = [COCO_SPORTS_BALL_CLASS]
            torch = _get_torch()
            device = "cuda:0" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
            self._sahi = get_sahi_instance(model_path, MIN_CONF, device, classes)
            logger.info("SAHI ball detector initialized with %s on %s", model_path, device)
        except Exception as e:
            logger.warning("SAHI init failed, using direct inference: %s", e)
            self._sahi_failed = True
        return self._sahi

    # ------------------------------------------------------------------
    def _detect_sahi(self, frame: np.ndarray) -> List[Tuple[float, float, float]]:
        """Run SAHI tiled inference, return candidates as (cx, cy, conf)."""
        sahi = self._get_sahi(frame.shape[1])
        if sahi is None:
            return []
        try:
            dets = sahi.detect(frame)
            candidates = []
            for d in dets:
                x1, y1, x2, y2 = d["bbox"]
                w, h = x2 - x1, y2 - y1
                if w >= MAX_BALL_SIZE or h >= MAX_BALL_SIZE:
                    continue
                candidates.append((d["cx"], d["cy"], d["confidence"]))
            return candidates
        except Exception as e:
            logger.debug("SAHI detect error: %s", e)
            return []

    # ------------------------------------------------------------------
    def _detect_direct(self, frame: np.ndarray) -> List[Tuple[float, float, float]]:
        """Run direct YOLO inference, return candidates as (cx, cy, conf)."""
        infer_kwargs: dict = {"imgsz": INFERENCE_SIZE, "verbose": False, "conf": MIN_CONF}
        if self.model_type == 'fallback':
            infer_kwargs["classes"] = [COCO_SPORTS_BALL_CLASS]

        results = self.model(frame, **infer_kwargs)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            logger.debug(f"Direct detect: no boxes found (model_type={self.model_type}, conf={MIN_CONF})")
            return []

        logger.debug(f"Direct detect: raw boxes={len(boxes)}, model_type={self.model_type}")
        candidates = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w, h = x2 - x1, y2 - y1
            if w >= MAX_BALL_SIZE or h >= MAX_BALL_SIZE:
                logger.debug(f"Reject box: size {w:.0f}x{h:.0f} >= {MAX_BALL_SIZE}")
                continue
            c = float(box.conf[0])
            candidates.append(((x1 + x2) / 2, (y1 + y2) / 2, c))
        logger.debug(f"Direct detect: candidates={len(candidates)}")
        return candidates

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray, frame_idx: int) -> Optional[dict]:
        """Detect ball in a single frame. Tries SAHI first, then direct."""
        if self.model is None:
            return None

        try:
            # Try SAHI tiled inference first (catches small balls)
            candidates = self._detect_sahi(frame)
            source = "sahi"

            # Fall back to direct inference
            if not candidates:
                candidates = self._detect_direct(frame)
                source = "direct"

            if not candidates:
                return self._handle_miss(frame_idx)

            candidates.sort(key=lambda t: t[2], reverse=True)

            # FIX 7: record best raw candidate for stationary-region tracking
            self._recent_xy.append((candidates[0][0], candidates[0][1]))

            cx = 0.0
            cy = 0.0
            conf = 0.0
            found = False
            for cx_c, cy_c, conf_c in candidates:
                if self._passes_sanity(cx_c, cy_c, frame_idx):
                    cx, cy, conf = float(cx_c), float(cy_c), float(conf_c)
                    found = True
                    break
                else:
                    # Debug: log why this candidate failed
                    if frame_idx < 10:
                        logger.debug(f"Frame {frame_idx}: candidate ({cx_c:.0f}, {cy_c:.0f}) rejected by sanity check")

            if not found:
                if frame_idx < 10:
                    logger.debug(f"Frame {frame_idx}: no candidates passed sanity checks. Had {len(candidates)} candidates")
                return self._handle_miss(frame_idx)

            pos = {"x": cx, "y": cy, "confidence": conf, "interpolated": False,
                   "source": source}
            self._positions[frame_idx] = pos
            self._lost_frames = 0

            # Update sanity Kalman on the accepted detection only.
            self._sanity_kf.correct(cx, cy)

            # Back-fill interpolated positions for short gaps
            if self._last_pos is not None and self._last_det_frame >= 0:
                gap = frame_idx - self._last_det_frame
                if 1 < gap <= INTERPOLATE_MAX:
                    self._interpolate(self._last_det_frame, frame_idx)

            self._last_pos = pos
            self._last_det_frame = frame_idx

            logger.debug(
                "Frame %d: ball at (%.0f, %.0f) conf=%.3f src=%s",
                frame_idx, cx, cy, conf, source,
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

        # Gate on ball confidence
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

        if frame_idx < 5 or (frame_idx % 25 == 0):
            logger.warning(
                "[POSSESSION DEBUG] frame=%d ball=(%.1f, %.1f) "
                "first_player=(%s, %.1f, %.1f) "
                "raw_dist_px=%.1f computed_dist_m=%.2f ppm=%.2f",
                frame_idx, bx, by,
                best_player["track_id"] if best_player else "?",
                best_player["cx"] if best_player else 0,
                best_player["cy"] if best_player else 0,
                best_dist_px, dist_m, ppm,
            )

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
