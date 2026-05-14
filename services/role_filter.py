"""
Role Filter v2 — Smart referee/official detection.

Multi-signal approach (replaces broken raw-HSV v1):
  1. YOLO class gate (class 3 = referee in Roboflow model)
  2. Known-team exclusion: after team centroids stabilise, refs are those
     whose torso colour is far from BOTH team centroids
  3. Neon-colour isolation: only bright neon yellow/green (referee vest),
     NOT normal yellow jerseys like Villa/Dortmund/Brazil
  4. Spatial isolation: near touchline and away from player clusters
  5. Conservative: better to let a ref through than to kill a real player

Usage (in tracker_core.py):
    self.role_filter = RoleFilter()
    ...
    tracks, officials = self.role_filter.filter(tracks, frame, video_frame)
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict


# ── Neon colour thresholds (strict: only bright referee vests) ────────

# Neon yellow-green referee vest — STRICT: must be very saturated AND very bright
# Raised S minimum 130→180 and V minimum 150→200 to avoid hitting keeper yellow/green kits
_NEON_YELLOW_LO = np.array([22, 180, 200], dtype=np.uint8)
_NEON_YELLOW_HI = np.array([38, 255, 255], dtype=np.uint8)

# Neon orange (some leagues' referee kits)
_NEON_ORANGE_LO = np.array([8, 180, 180], dtype=np.uint8)
_NEON_ORANGE_HI = np.array([20, 255, 255], dtype=np.uint8)

# Dark referee kit — raised V threshold to catch dark navy/charcoal kits
# V_MAX 50→80 catches dark grey; S_MAX stays 60 (excludes coloured kits)
_BLACK_KIT_V_MAX = 80
_BLACK_KIT_S_MAX = 60

# Pink/magenta referee kits (used in some leagues)
_NEON_PINK_LO = np.array([145, 80, 120], dtype=np.uint8)
_NEON_PINK_HI = np.array([170, 255, 255], dtype=np.uint8)


class RoleFilter:
    """
    Conservative referee filter.

    Design principle: NEVER filter a player. The cost of removing a real
    player from P1-P22 is far worse than letting a referee slip through.
    A referee that gets through is just 1 of 22 slots mis-used; a player
    removed is a permanent identity loss.

    Confidence scoring: a track must accumulate enough referee signals
    across multiple frames before being filtered. Single-frame spikes
    (caused by grass bleed, shadow, etc.) are ignored.
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}

        # Thresholds
        self.neon_ratio_thresh = cfg.get("neon_ratio_thresh", 0.35)   # raised: fewer false keeper hits
        self.black_ratio_thresh = cfg.get("black_ratio_thresh", 0.45)  # lowered: catch dark ref kits
        self.team_exclusion_dist = cfg.get("team_exclusion_dist", 0.40)
        self.edge_margin_frac = cfg.get("edge_margin_frac", 0.08)
        self.min_frames_to_filter = cfg.get("min_frames_to_filter", 4)  # raised: need more evidence
        self.referee_confidence_thresh = cfg.get("referee_confidence_thresh", 0.75)  # stricter
        self.log_interval = cfg.get("log_interval", 30)

        # Per-track referee evidence accumulator
        # track_id -> {"ref_votes": int, "player_votes": int, "confirmed": bool}
        self._evidence: Dict[int, Dict] = {}

        # Team colour centroids (set externally by tracker_core after team clustering)
        self.team_centroid_a: Optional[np.ndarray] = None
        self.team_centroid_b: Optional[np.ndarray] = None

    def set_team_centroids(self, centroid_a: np.ndarray, centroid_b: np.ndarray):
        """Called once team colours are known (after identity freeze)."""
        self.team_centroid_a = centroid_a
        self.team_centroid_b = centroid_b

    def filter(
        self, tracks: list, frame: np.ndarray, frame_id: int = 0
    ) -> Tuple[list, list, list]:
        """
        Classify tracks as player / official / suspected_official.

        suspected_official: at least one ref-leaning frame but not yet enough votes
        to commit. Held back from BOTH the identity layer and the rendered output
        for the current frame so refs cannot get a transient P-id during ramp-up.

        Returns:
            (player_tracks, official_tracks, suspected_official_tracks)
        """
        if len(tracks) == 0:
            return tracks, [], []

        h, w = frame.shape[:2]
        players, officials, suspected = [], [], []

        for t in tracks:
            tid = t.track_id
            bbox = t.bbox
            cls = getattr(t, "cls", 0)

            # Initialise evidence for new tracks
            if tid not in self._evidence:
                self._evidence[tid] = {
                    "ref_votes": 0,
                    "player_votes": 0,
                    "confirmed_ref": False,
                    "confirmed_player": False,
                }

            ev = self._evidence[tid]

            # If already confirmed as player, fast-path
            if ev["confirmed_player"]:
                players.append(t)
                continue

            # If already confirmed as referee, fast-path
            if ev["confirmed_ref"]:
                officials.append(t)
                continue

            # ── Signal 1: YOLO class ──
            yolo_ref = int(cls) == 3

            # ── Signal 2: Neon colour analysis (strict) ──
            neon_ref, neon_conf = self._neon_colour_check(bbox, frame, h, w)

            # ── Signal 3: Black kit check ──
            black_ref = self._black_kit_check(bbox, frame, h, w)

            # ── Signal 4: Team exclusion (only if centroids are set) ──
            team_excluded = False
            if self.team_centroid_a is not None:
                team_excluded = self._team_exclusion_check(bbox, frame, h, w)

            # ── Signal 5: Spatial isolation (touchline) ──
            near_touchline = self._touchline_check(bbox, w, h)

            # ── Combine signals ──
            ref_score = 0.0
            if yolo_ref:
                ref_score += 0.40
            if neon_ref:
                ref_score += 0.30 * neon_conf
            if black_ref:
                ref_score += 0.15
            if team_excluded:
                ref_score += 0.20
            if near_touchline and (neon_ref or yolo_ref):
                ref_score += 0.10

            # Accumulate evidence over time
            if ref_score >= 0.35:
                ev["ref_votes"] += 1
            else:
                ev["player_votes"] += 1

            total_votes = ev["ref_votes"] + ev["player_votes"]

            # Only classify after enough observations
            if total_votes >= self.min_frames_to_filter:
                ref_ratio = ev["ref_votes"] / total_votes
                if ref_ratio >= self.referee_confidence_thresh:
                    ev["confirmed_ref"] = True
                    officials.append(t)
                    continue
                elif ref_ratio < 0.30:
                    ev["confirmed_player"] = True
                    players.append(t)
                    continue

            # Not enough evidence yet. If we've seen ANY ref-leaning frame,
            # hold back as "suspected" so the identity layer doesn't issue
            # a transient P-id to a referee during the first 1-2 frames.
            if ev["ref_votes"] >= 1:
                suspected.append(t)
            else:
                players.append(t)

        # Periodic logging
        n_officials = len(officials)
        n_suspected = len(suspected)
        if frame_id % self.log_interval == 0 and (n_officials > 0 or n_suspected > 0):
            ref_ids = [t.track_id for t in officials]
            sus_ids = [t.track_id for t in suspected]
            print(
                f"[RoleFilter F{frame_id}] input={len(tracks)} "
                f"players={len(players)} officials={n_officials} "
                f"suspected={n_suspected} ref_ids={ref_ids} sus_ids={sus_ids}"
            )

        return players, officials, suspected

    # ──────────────────────────────────────────────────────────────────
    # Signal implementations
    # ──────────────────────────────────────────────────────────────────

    def _get_torso_hsv(
        self, bbox, frame: np.ndarray, fh: int, fw: int
    ) -> Optional[np.ndarray]:
        """Extract torso HSV (top 15-55% of bbox, middle 60% width)."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        bh, bw = y2 - y1, x2 - x1
        if bh < 15 or bw < 8:
            return None

        ty1 = max(y1 + int(bh * 0.15), 0)
        ty2 = min(y1 + int(bh * 0.55), fh)
        tx1 = max(x1 + int(bw * 0.20), 0)
        tx2 = min(x1 + int(bw * 0.80), fw)
        if ty2 <= ty1 + 3 or tx2 <= tx1 + 3:
            return None

        crop = frame[ty1:ty2, tx1:tx2]
        if crop.size == 0:
            return None

        return cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    def _neon_colour_check(
        self, bbox, frame: np.ndarray, fh: int, fw: int
    ) -> Tuple[bool, float]:
        """
        Check for neon referee vest colours (strict thresholds).
        Returns (is_neon, confidence).
        """
        hsv = self._get_torso_hsv(bbox, frame, fh, fw)
        if hsv is None:
            return False, 0.0

        total = hsv.shape[0] * hsv.shape[1]
        if total == 0:
            return False, 0.0

        # Neon yellow-green
        neon_y = np.count_nonzero(cv2.inRange(hsv, _NEON_YELLOW_LO, _NEON_YELLOW_HI))
        # Neon orange
        neon_o = np.count_nonzero(cv2.inRange(hsv, _NEON_ORANGE_LO, _NEON_ORANGE_HI))
        # Neon pink
        neon_p = np.count_nonzero(cv2.inRange(hsv, _NEON_PINK_LO, _NEON_PINK_HI))

        max_neon = max(neon_y, neon_o, neon_p) / total

        if max_neon >= self.neon_ratio_thresh:
            conf = min(max_neon * 2.0, 1.0)
            return True, conf

        return False, 0.0

    def _black_kit_check(
        self, bbox, frame: np.ndarray, fh: int, fw: int
    ) -> bool:
        """Check for dark/black referee kit (low V + low S)."""
        hsv = self._get_torso_hsv(bbox, frame, fh, fw)
        if hsv is None:
            return False

        total = hsv.shape[0] * hsv.shape[1]
        if total == 0:
            return False

        black_mask = (hsv[:, :, 2] < _BLACK_KIT_V_MAX) & (
            hsv[:, :, 1] < _BLACK_KIT_S_MAX
        )
        black_ratio = np.count_nonzero(black_mask) / total

        return black_ratio >= self.black_ratio_thresh

    def _team_exclusion_check(
        self, bbox, frame: np.ndarray, fh: int, fw: int
    ) -> bool:
        """
        Check if torso colour is far from BOTH team centroids.
        If it doesn't match either team, it's likely a referee.
        """
        if self.team_centroid_a is None or self.team_centroid_b is None:
            return False

        x1, y1, x2, y2 = [int(v) for v in bbox]
        bh, bw = y2 - y1, x2 - x1
        if bh < 15 or bw < 8:
            return False

        ty1 = max(y1 + int(bh * 0.15), 0)
        ty2 = min(y1 + int(bh * 0.55), fh)
        tx1 = max(x1 + int(bw * 0.20), 0)
        tx2 = min(x1 + int(bw * 0.80), fw)
        if ty2 <= ty1 + 3 or tx2 <= tx1 + 3:
            return False

        crop = frame[ty1:ty2, tx1:tx2]
        if crop.size == 0:
            return False

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [36], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        hist = np.concatenate([h_hist, s_hist])
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist = hist / norm

        # Distance to each team centroid
        dist_a = 1.0 - float(np.dot(hist, self.team_centroid_a))
        dist_b = 1.0 - float(np.dot(hist, self.team_centroid_b))

        # Far from BOTH teams → likely referee
        return dist_a > self.team_exclusion_dist and dist_b > self.team_exclusion_dist

    def _touchline_check(self, bbox, frame_w: int, frame_h: int) -> bool:
        """Check if near left/right edge (touchline / assistant referee position)."""
        cx = (bbox[0] + bbox[2]) / 2
        margin = frame_w * self.edge_margin_frac
        return cx < margin or cx > frame_w - margin

    def reset(self):
        """Clear per-track evidence on scene boundary."""
        self._evidence.clear()
