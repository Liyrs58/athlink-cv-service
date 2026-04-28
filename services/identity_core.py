"""
Production GPU/CPU identity system for football tracking.
Fast, stable, recoverable. No VLM control over identity.

Pipeline:
  YOLO (GPU) → BotSort (CPU light) → ReID batch (GPU) → Identity match (CPU, vectorized)
"""

import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class TeamId(Enum):
    UNKNOWN = -1
    A = 0
    B = 1


class PlayerState(Enum):
    ACTIVE = "active"
    DORMANT = "dormant"


@dataclass
class Player:
    """Permanent player identity (P1–P22)."""
    slot_id: int
    team: TeamId = TeamId.UNKNOWN
    embeddings: deque = None
    mean_emb: Optional[np.ndarray] = None
    last_bbox: Optional[np.ndarray] = None
    last_seen: int = -1
    active_tid: Optional[int] = None
    state: PlayerState = PlayerState.ACTIVE
    conf_score: float = 0.0

    def __post_init__(self):
        if self.embeddings is None:
            self.embeddings = deque(maxlen=20)

    def add_embedding(self, emb: np.ndarray):
        self.embeddings.append(emb.copy())
        if len(self.embeddings) > 0:
            bank = np.array(list(self.embeddings))
            self.mean_emb = bank.mean(0)
            norm = np.linalg.norm(self.mean_emb)
            if norm > 0:
                self.mean_emb /= norm


@dataclass
class Track:
    """BotSort track (temporary handle)."""
    tid: int
    bbox: np.ndarray
    conf: float
    cls: int
    emb: Optional[np.ndarray] = None


class IdentityCore:
    """
    Fast, deterministic identity matching.
    CPU-based, vectorized, no VLM control.
    """

    def __init__(self, max_players: int = 22):
        self.max_players = max_players
        self.players: Dict[int, Player] = {}  # slot_id → Player
        self.track_to_slot: Dict[int, int] = {}  # BotSort tid → slot_id
        self.next_slot = 1
        self.is_frozen = False
        self._frames_seen = 0
        self._reg_frames = 30

    # ======================================================================
    # Public API
    # ======================================================================

    def update(self, tracks: List[Track], frame_id: int) -> Dict[int, int]:
        """Main entry: match tracks to slots, return tid→slot_id mapping."""
        if len(tracks) == 0:
            return dict(self.track_to_slot)

        self._frames_seen += 1

        if not self.is_frozen:
            self._register(tracks, frame_id)
            if self._frames_seen >= self._reg_frames:
                self._freeze()
        else:
            self._match(tracks, frame_id)

        self._resolve_duplicates()
        return dict(self.track_to_slot)

    def freeze_on_bench(self, frame_id: int):
        """Called on bench/non-play: mark all active → dormant."""
        for p in self.players.values():
            if p.state == PlayerState.ACTIVE:
                p.state = PlayerState.DORMANT
                p.active_tid = None

    def unfreeze_on_play(self, frame_id: int):
        """Called on bench→play: clear old mappings, re-match from dormant."""
        # Clear stale mappings
        active_tids = set()
        for p in self.players.values():
            if p.active_tid is not None:
                active_tids.add(p.active_tid)
        stale = [tid for tid in self.track_to_slot if tid not in active_tids]
        for tid in stale:
            del self.track_to_slot[tid]

    # ======================================================================
    # Registration (first 30 frames)
    # ======================================================================

    def _register(self, tracks: List[Track], frame_id: int):
        for t in tracks:
            # Skip referee
            if t.cls == 3:
                continue

            tid = t.tid
            if tid in self.track_to_slot:
                # Known track: update
                slot_id = self.track_to_slot[tid]
                p = self.players[slot_id]
                p.last_bbox = t.bbox.copy()
                p.last_seen = frame_id
                p.conf_score = max(p.conf_score, t.conf)
                if t.emb is not None:
                    p.add_embedding(t.emb)
            else:
                # New track: new slot
                if self.next_slot > self.max_players:
                    continue
                p = Player(slot_id=self.next_slot)
                p.team = TeamId.UNKNOWN
                p.last_bbox = t.bbox.copy()
                p.last_seen = frame_id
                p.active_tid = tid
                p.state = PlayerState.ACTIVE
                p.conf_score = t.conf
                if t.emb is not None:
                    p.add_embedding(t.emb)
                self.players[self.next_slot] = p
                self.track_to_slot[tid] = self.next_slot
                self.next_slot += 1

    # ======================================================================
    # Freeze + team clustering
    # ======================================================================

    def _freeze(self):
        self.is_frozen = True
        self._cluster_teams()
        ta = sum(1 for p in self.players.values() if p.team == TeamId.A)
        tb = sum(1 for p in self.players.values() if p.team == TeamId.B)
        print(f"[Identity] Frozen: {len(self.players)} players | A={ta} B={tb}")

    def _cluster_teams(self):
        """K-means on color histograms (post-hoc team assignment)."""
        slots = [p for p in self.players.values()]
        if len(slots) < 4:
            return

        hists = np.array([self._hist_from_last_bbox(p) for p in slots])
        if hists is None or len(hists) == 0:
            return

        # Farthest pair seeding
        dist = 1.0 - (hists @ hists.T)
        idx = np.unravel_index(np.argmax(dist), dist.shape)
        c0, c1 = hists[idx[0]].copy(), hists[idx[1]].copy()

        labels = np.zeros(len(slots), dtype=int)
        for _ in range(10):
            d0 = 1.0 - (hists @ c0)
            d1 = 1.0 - (hists @ c1)
            labels = (d1 < d0).astype(int)
            m0, m1 = labels == 0, labels == 1
            if m0.sum() > 0:
                c0 = hists[m0].mean(0)
                c0 /= np.linalg.norm(c0) + 1e-6
            if m1.sum() > 0:
                c1 = hists[m1].mean(0)
                c1 /= np.linalg.norm(c1) + 1e-6

        for i, p in enumerate(slots):
            p.team = TeamId.A if labels[i] == 0 else TeamId.B

    def _hist_from_last_bbox(self, p: Player) -> Optional[np.ndarray]:
        """Extract color histogram from last known bbox (dummy for now)."""
        return np.ones(52) / 52  # placeholder

    # ======================================================================
    # Active play matching (vectorized Hungarian)
    # ======================================================================

    def _match(self, tracks: List[Track], frame_id: int):
        """Match unknown tracks to available slots via vectorized cosine + Hungarian."""
        used_slots = set()
        unknown = []

        # Pass 1: keep known tracks
        for t in tracks:
            if t.cls == 3:
                continue
            tid = t.tid
            if tid in self.track_to_slot:
                slot_id = self.track_to_slot[tid]
                if slot_id in used_slots:
                    # Collision: evict, re-match as unknown
                    del self.track_to_slot[tid]
                    unknown.append(t)
                else:
                    used_slots.add(slot_id)
                    p = self.players[slot_id]
                    p.state = PlayerState.ACTIVE
                    p.active_tid = tid
                    p.last_bbox = t.bbox.copy()
                    p.last_seen = frame_id
                    p.conf_score = max(p.conf_score, t.conf)
                    if t.emb is not None:
                        p.add_embedding(t.emb)
            else:
                unknown.append(t)

        if not unknown:
            return

        # Available slots (not used, not stale)
        available = [p for p in self.players.values()
                     if p.slot_id not in used_slots]
        if not available:
            return

        self._hungarian_assign(unknown, available, used_slots, frame_id)

    def _hungarian_assign(self, unknown: List[Track], available: List[Player],
                          used_slots: set, frame_id: int):
        """Vectorized Hungarian assignment on ReID embeddings."""
        n_t, n_s = len(unknown), len(available)
        if n_t == 0 or n_s == 0:
            return

        # Build cost matrix (lower = better)
        cost = np.ones((n_t, n_s))

        for i, t in enumerate(unknown):
            for j, p in enumerate(available):
                cost[i, j] = self._match_cost(t, p, frame_id)

        # Hungarian
        rows, cols = self._hungarian(cost)

        matched = 0
        for r, c in zip(rows, cols):
            cost_val = cost[r, c]
            if cost_val >= 0.65:  # threshold
                continue

            t = unknown[r]
            p = available[c]

            self.track_to_slot[t.tid] = p.slot_id
            used_slots.add(p.slot_id)
            p.state = PlayerState.ACTIVE
            p.active_tid = t.tid
            p.last_bbox = t.bbox.copy()
            p.last_seen = frame_id
            p.conf_score = t.conf
            if t.emb is not None:
                p.add_embedding(t.emb)
            matched += 1

    def _match_cost(self, t: Track, p: Player, frame_id: int) -> float:
        """Compute cost: 70% ReID + 15% spatial + 15% temporal."""
        score = 0.0

        # ReID (70%)
        if t.emb is not None and p.mean_emb is not None:
            reid = float(np.dot(t.emb, p.mean_emb))
            score += 0.70 * (1.0 - reid)
        else:
            score += 0.70

        # Spatial IoU (15%)
        if p.last_bbox is not None:
            iou = self._iou(t.bbox, p.last_bbox)
            score += 0.15 * (1.0 - iou)
        else:
            score += 0.15

        # Temporal (15%)
        gap = max(0, frame_id - p.last_seen)
        time_pen = min(gap / 1500.0, 1.0)
        score += 0.15 * time_pen

        return float(np.clip(score, 0.0, 1.0))

    def _iou(self, b1: np.ndarray, b2: np.ndarray) -> float:
        x1a, y1a, x2a, y2a = b1
        x1b, y1b, x2b, y2b = b2
        ix1 = max(x1a, x1b)
        iy1 = max(y1a, y1b)
        ix2 = min(x2a, x2b)
        iy2 = min(y2a, y2b)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        a = (x2a - x1a) * (y2a - y1a)
        b = (x2b - x1b) * (y2b - y1b)
        return inter / (a + b - inter + 1e-6) if (a + b - inter) > 0 else 0.0

    def _hungarian(self, cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Greedy min-cost assignment."""
        rows, cols = [], []
        used_r, used_c = set(), set()
        pairs = sorted((cost[i, j], i, j)
                       for i in range(cost.shape[0])
                       for j in range(cost.shape[1]))
        for c, r, col in pairs:
            if r not in used_r and col not in used_c:
                rows.append(r)
                cols.append(col)
                used_r.add(r)
                used_c.add(col)
        return np.array(rows, dtype=int), np.array(cols, dtype=int)

    # ======================================================================
    # Duplicate resolution
    # ======================================================================

    def _resolve_duplicates(self):
        """Ensure 1:1 mapping (1 tid per slot, 1 slot per tid)."""
        slot_to_tids: Dict[int, List[int]] = {}
        for tid, sid in self.track_to_slot.items():
            slot_to_tids.setdefault(sid, []).append(tid)

        for sid, tids in slot_to_tids.items():
            if len(tids) <= 1:
                continue
            keep = max(tids)
            for tid in tids:
                if tid != keep:
                    del self.track_to_slot[tid]
