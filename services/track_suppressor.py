"""
Track Suppressor — sits between Deep-EIoU tracker output and IdentityCore.
Reduces raw tracks to ≤28 clean candidates via ghost/duplicate/overlay removal.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


class TrackSuppressor:
    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        self.ghost_max_age = cfg.get("ghost_max_age", 15)
        self.duplicate_iou_thresh = cfg.get("duplicate_iou_thresh", 0.50)
        self.duplicate_center_dist = cfg.get("duplicate_center_dist", 60)
        self.scoreboard_y_max = cfg.get("scoreboard_y_max", 80)
        self.scoreboard_bottom_y = cfg.get("scoreboard_bottom_y", 620)
        self.min_bbox_height = cfg.get("min_bbox_height", 30)
        self.max_clean_tracks = cfg.get("max_clean_tracks", 28)
        self.log_interval = cfg.get("log_interval", 30)

    def suppress(self, raw_tracks: list, frame: np.ndarray, frame_id: int) -> Tuple[list, dict]:
        raw_n = len(raw_tracks)

        tracks = self._remove_ghosts(raw_tracks)
        ghost_rm = raw_n - len(tracks)

        tracks = self._remove_duplicates(tracks)
        dup_rm = raw_n - ghost_rm - len(tracks)

        tracks = self._remove_overlays(tracks, frame)
        overlay_rm = raw_n - ghost_rm - dup_rm - len(tracks)

        tracks = self._cap_by_quality(tracks)
        cap_rm = raw_n - ghost_rm - dup_rm - overlay_rm - len(tracks)

        # Fix negative counts from cascading
        after_ghost = raw_n - ghost_rm
        after_dup = after_ghost - max(0, after_ghost - len(tracks) + overlay_rm + cap_rm)
        dup_rm = max(0, after_ghost - (len(tracks) + overlay_rm + cap_rm))

        stats = {
            "raw": raw_n,
            "ghost_rm": ghost_rm,
            "dup_rm": dup_rm,
            "overlay_rm": overlay_rm,
            "cap_rm": cap_rm,
            "clean": len(tracks),
        }

        if frame_id % self.log_interval == 0 or raw_n > 40:
            print(
                f"[Suppress F{frame_id}] raw={raw_n} "
                f"ghost-={ghost_rm} dup-={dup_rm} "
                f"overlay-={overlay_rm} cap-={cap_rm} "
                f"→ clean={len(tracks)}"
            )

        return tracks, stats

    def _remove_ghosts(self, tracks: list) -> list:
        return [t for t in tracks if t.time_since_update <= self.ghost_max_age]

    def _remove_duplicates(self, tracks: list) -> list:
        if len(tracks) <= 1:
            return tracks
        # Sort best-quality first
        tracks_sorted = sorted(tracks, key=self._quality, reverse=True)
        removed = set()
        for i, t1 in enumerate(tracks_sorted):
            if t1.track_id in removed:
                continue
            b1 = t1.bbox
            for j in range(i + 1, len(tracks_sorted)):
                t2 = tracks_sorted[j]
                if t2.track_id in removed:
                    continue
                b2 = t2.bbox
                iou = self._iou(b1, b2)
                cd = self._cdist(b1, b2)
                if iou > self.duplicate_iou_thresh or cd < self.duplicate_center_dist:
                    removed.add(t2.track_id)
        return [t for t in tracks_sorted if t.track_id not in removed]

    def _remove_overlays(self, tracks: list, frame: np.ndarray) -> list:
        kept = []
        for t in tracks:
            b = t.bbox
            bh = b[3] - b[1]
            bw = b[2] - b[0]
            cy = (b[1] + b[3]) / 2
            if bh < self.min_bbox_height:
                continue
            if cy < self.scoreboard_y_max:
                continue
            if b[1] > self.scoreboard_bottom_y and bh < 60:
                continue
            if bw > 0 and bh / bw < 0.5:
                continue
            kept.append(t)
        return kept

    def _cap_by_quality(self, tracks: list) -> list:
        if len(tracks) <= self.max_clean_tracks:
            return tracks
        scored = sorted(tracks, key=self._quality, reverse=True)
        return scored[:self.max_clean_tracks]

    def _quality(self, t) -> float:
        age_s = min(t.age / 30.0, 1.0)
        rec_s = max(1.0 - t.time_since_update / 15.0, 0.0)
        size_s = min((t.bbox[3] - t.bbox[1]) / 200.0, 1.0)
        return 0.40 * t.score + 0.30 * age_s + 0.20 * rec_s + 0.10 * size_s

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

    @staticmethod
    def _cdist(b1, b2) -> float:
        dx = (b1[0] + b1[2]) / 2 - (b2[0] + b2[2]) / 2
        dy = (b1[1] + b1[3]) / 2 - (b2[1] + b2[3]) / 2
        return (dx * dx + dy * dy) ** 0.5
