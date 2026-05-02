"""
Crop Quality Gate — scores each track's visual evidence quality.

Prevents identity memory pollution from:
  - Heavy bbox overlap (collision clusters)
  - Frame-edge truncation
  - Scoreboard/overlay occlusion
  - Too-small bboxes
  - Low detection confidence

Usage (in tracker_core.py):
    quality = self.crop_quality.score(track, frame, all_tracks)
    if quality.score >= 0.55:
        identity.update_embedding(...)
    if quality.score >= 0.40:
        identity.assign(...)

Returns quality score 0-1 and list of reasons.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field


@dataclass
class CropQuality:
    """Result of crop quality assessment."""
    score: float           # 0-1, higher = better quality
    reasons: List[str]     # human-readable degradation reasons
    allow_memory_update: bool   # True if quality >= 0.55
    allow_assignment: bool      # True if quality >= 0.40

    @property
    def is_good(self) -> bool:
        return self.score >= 0.55


class CropQualityGate:
    """
    Scores crop quality for identity memory gating.

    Each signal contributes a penalty to the base score of 1.0.
    The final score is clamped to [0, 1].
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}

        # Thresholds
        self.min_bbox_height = cfg.get("min_bbox_height", 40)
        self.min_bbox_width = cfg.get("min_bbox_width", 15)
        self.small_bbox_area = cfg.get("small_bbox_area", 1200)
        self.edge_margin_px = cfg.get("edge_margin_px", 15)
        self.overlap_iou_thresh = cfg.get("overlap_iou_thresh", 0.30)
        self.dense_cluster_iou_thresh = cfg.get("dense_cluster_iou_thresh", 0.20)
        self.dense_cluster_min_count = cfg.get("dense_cluster_min_count", 3)
        self.scoreboard_y_max = cfg.get("scoreboard_y_max", 80)
        self.scoreboard_bottom_y = cfg.get("scoreboard_bottom_y", 620)
        self.min_confidence = cfg.get("min_confidence", 0.20)
        self.memory_update_thresh = cfg.get("memory_update_thresh", 0.55)
        self.assignment_thresh = cfg.get("assignment_thresh", 0.40)
        self.log_interval = cfg.get("log_interval", 30)

        # Stats tracking
        self._updates_allowed = 0
        self._updates_blocked = 0

    def score(
        self,
        track,
        frame: np.ndarray,
        all_tracks: list,
        frame_id: int = 0,
    ) -> CropQuality:
        """
        Score the visual quality of a track's current crop.

        Args:
            track: DETrack object with .bbox, .score, .track_id
            frame: current BGR frame
            all_tracks: all current tracks (for overlap computation)
            frame_id: for logging

        Returns:
            CropQuality with score, reasons, and gating flags
        """
        bbox = track.bbox  # [x1, y1, x2, y2]
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1

        score = 1.0
        reasons: List[str] = []

        # ── 1. Too small ──────────────────────────────────────────
        if bh < self.min_bbox_height or bw < self.min_bbox_width:
            score -= 0.40
            reasons.append(f"tiny_bbox({bw:.0f}x{bh:.0f})")
        elif bw * bh < self.small_bbox_area:
            score -= 0.20
            reasons.append(f"small_bbox({bw*bh:.0f}px²)")

        # ── 2. Frame edge truncation ──────────────────────────────
        edge_penalty = 0.0
        if x1 < self.edge_margin_px:
            edge_penalty = max(edge_penalty, 0.25)
        if y1 < self.edge_margin_px:
            edge_penalty = max(edge_penalty, 0.20)
        if x2 > fw - self.edge_margin_px:
            edge_penalty = max(edge_penalty, 0.25)
        if y2 > fh - self.edge_margin_px:
            edge_penalty = max(edge_penalty, 0.15)  # feet at bottom is less harmful

        if edge_penalty > 0:
            score -= edge_penalty
            reasons.append(f"edge_truncated(-{edge_penalty:.2f})")

        # ── 3. Scoreboard/overlay occlusion ───────────────────────
        cy = (y1 + y2) / 2
        if cy < self.scoreboard_y_max:
            score -= 0.35
            reasons.append("in_scoreboard_zone")
        elif y1 > self.scoreboard_bottom_y and bh < 60:
            score -= 0.25
            reasons.append("in_bottom_overlay")

        # ── 4. Detection confidence ──────────────────────────────
        conf = getattr(track, "score", 0.5)
        if conf < self.min_confidence:
            score -= 0.30
            reasons.append(f"low_conf({conf:.2f})")
        elif conf < 0.35:
            score -= 0.10
            reasons.append(f"med_conf({conf:.2f})")

        # ── 5. Overlap with other tracks (collision clusters) ─────
        overlap_penalty = self._overlap_penalty(track, all_tracks)
        if overlap_penalty > 0:
            score -= overlap_penalty
            reasons.append(f"overlap(-{overlap_penalty:.2f})")

        # ── 6. Aspect ratio sanity ────────────────────────────────
        if bw > 0 and bh / bw < 0.6:
            # Too wide — could be a merged detection
            score -= 0.20
            reasons.append(f"wide_aspect({bh/bw:.2f})")
        elif bw > 0 and bh / bw > 5.0:
            # Too tall — probably not a player
            score -= 0.15
            reasons.append(f"tall_aspect({bh/bw:.2f})")

        # ── 7. Track freshness ────────────────────────────────────
        tsu = getattr(track, "time_since_update", 0)
        if tsu > 5:
            # Using predicted position, not a real detection
            penalty = min(tsu / 20.0, 0.30)
            score -= penalty
            reasons.append(f"stale_pred(tsu={tsu})")

        # ── Final score ──────────────────────────────────────────
        score = max(0.0, min(1.0, score))

        allow_memory = score >= self.memory_update_thresh
        allow_assign = score >= self.assignment_thresh

        if allow_memory:
            self._updates_allowed += 1
        else:
            self._updates_blocked += 1

        return CropQuality(
            score=round(score, 3),
            reasons=reasons,
            allow_memory_update=allow_memory,
            allow_assignment=allow_assign,
        )

    def score_batch(
        self,
        tracks: list,
        frame: np.ndarray,
        frame_id: int = 0,
    ) -> Dict[int, CropQuality]:
        """Score all tracks in a single call."""
        results = {}
        for t in tracks:
            results[t.track_id] = self.score(t, frame, tracks, frame_id)
        return results

    def maybe_log(self, frame_id: int):
        """Periodic quality stats log."""
        if frame_id % self.log_interval != 0:
            return
        total = self._updates_allowed + self._updates_blocked
        if total == 0:
            return
        block_pct = self._updates_blocked / total * 100
        print(
            f"[CropQuality F{frame_id}] "
            f"allowed={self._updates_allowed} blocked={self._updates_blocked} "
            f"({block_pct:.0f}% blocked)"
        )
        # Reset counters per window
        self._updates_allowed = 0
        self._updates_blocked = 0

    # ──────────────────────────────────────────────────────────────
    # Overlap computation
    # ──────────────────────────────────────────────────────────────

    def _overlap_penalty(self, track, all_tracks: list) -> float:
        """
        Compute overlap penalty for a track.
        - If IoU > 0.30 with any other track: moderate penalty
        - If 3+ tracks overlap (dense cluster): heavy penalty
        """
        bbox = track.bbox
        tid = track.track_id
        overlapping = 0

        max_iou = 0.0
        for other in all_tracks:
            if other.track_id == tid:
                continue
            iou = self._iou(bbox, other.bbox)
            max_iou = max(max_iou, iou)
            if iou > self.dense_cluster_iou_thresh:
                overlapping += 1

        penalty = 0.0

        # High overlap with at least one other track
        if max_iou > self.overlap_iou_thresh:
            penalty += 0.20

        # Dense collision cluster
        if overlapping >= self.dense_cluster_min_count:
            penalty += 0.25
        elif overlapping >= 2:
            penalty += 0.10

        return min(penalty, 0.50)

    @staticmethod
    def _iou(b1, b2) -> float:
        ix1 = max(b1[0], b2[0])
        iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2])
        iy2 = min(b1[3], b2[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / max(a1 + a2 - inter, 1e-6)
