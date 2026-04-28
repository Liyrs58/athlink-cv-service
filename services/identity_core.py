from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


DORMANT_TTL = 180
MAX_SLOTS = 22
COST_REJECT_THRESHOLD = 0.72
EMB_ALPHA = 0.25


@dataclass
class PlayerSlot:
    pid: str
    state: str = "lost"  # active|dormant|lost
    active_track_id: Optional[int] = None
    seen_this_frame: bool = False
    last_seen_frame: int = -10**9
    last_position: Optional[Tuple[float, float]] = None
    embedding: Optional[np.ndarray] = None

    def update_embedding(self, emb: np.ndarray) -> None:
        emb = emb.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        if self.embedding is None:
            self.embedding = emb
        else:
            self.embedding = (1.0 - EMB_ALPHA) * self.embedding + EMB_ALPHA * emb
            norm2 = np.linalg.norm(self.embedding)
            if norm2 > 0:
                self.embedding = self.embedding / norm2


class IdentityCore:
    def __init__(self, logger=None, debug_every: int = 30):
        self.logger = logger
        self.debug_every = debug_every
        self.slots: List[PlayerSlot] = [PlayerSlot(pid=f"P{i}") for i in range(1, MAX_SLOTS + 1)]

        self.frame_id: int = -1
        self.assigned_this_frame: int = 0
        self.unmatched_tracks: int = 0
        self.unmatched_slots: int = 0

    # ---------- lifecycle ----------
    def begin_frame(self, frame_id: int) -> None:
        self.frame_id = frame_id
        self.assigned_this_frame = 0
        self.unmatched_tracks = 0
        self.unmatched_slots = 0

        # CRITICAL: clear only frame-local attachment flags.
        for s in self.slots:
            s.active_track_id = None
            s.seen_this_frame = False

    def assign_tracks(
        self,
        tracks: Sequence[object],
        embeddings: Dict[int, np.ndarray],
        positions: Dict[int, Tuple[float, float]],
    ) -> Dict[int, str]:
        """
        tracks: iterable with .track_id attribute
        embeddings: track_id -> embedding vector
        positions: track_id -> (x, y)
        returns: track_id -> PID
        """
        if len(tracks) == 0:
            self.unmatched_slots = MAX_SLOTS
            return {}

        track_ids = [int(t.track_id) for t in tracks]
        n_t = len(track_ids)
        n_s = len(self.slots)

        # Build cost matrix against ALL slots (active/dormant/lost alike).
        cost = np.full((n_t, n_s), 1e3, dtype=np.float32)
        for i, tid in enumerate(track_ids):
            t_emb = embeddings.get(tid)
            t_pos = positions.get(tid)
            for j, slot in enumerate(self.slots):
                cost[i, j] = self._slot_cost(slot, t_emb, t_pos)

        r_idx, c_idx = linear_sum_assignment(cost)

        track_to_pid: Dict[int, str] = {}
        matched_tracks = set()
        matched_slots = set()

        for r, c in zip(r_idx, c_idx):
            cst = float(cost[r, c])
            if cst > COST_REJECT_THRESHOLD:
                continue

            tid = track_ids[r]
            slot = self.slots[c]

            track_to_pid[tid] = slot.pid
            slot.active_track_id = tid
            slot.seen_this_frame = True
            slot.state = "active"
            slot.last_seen_frame = self.frame_id
            slot.last_position = positions.get(tid)

            emb = embeddings.get(tid)
            if emb is not None:
                slot.update_embedding(emb)

            self.assigned_this_frame += 1
            matched_tracks.add(r)
            matched_slots.add(c)

        self.unmatched_tracks = n_t - len(matched_tracks)
        self.unmatched_slots = n_s - len(matched_slots)
        return track_to_pid

    def end_frame(self, frame_id: Optional[int] = None) -> None:
        if frame_id is not None:
            self.frame_id = frame_id

        for s in self.slots:
            if s.seen_this_frame:
                continue
            age = self.frame_id - s.last_seen_frame
            s.state = "dormant" if age <= DORMANT_TTL else "lost"
            s.active_track_id = None

    # ---------- metrics / logging ----------
    def state_counts(self) -> Tuple[int, int, int]:
        active = sum(1 for s in self.slots if s.state == "active")
        dormant = sum(1 for s in self.slots if s.state == "dormant")
        lost = sum(1 for s in self.slots if s.state == "lost")
        return active, dormant, lost

    def maybe_log(self, detections_count: int, tracks_count: int) -> None:
        if self.frame_id % self.debug_every != 0:
            return

        active, dormant, lost = self.state_counts()
        lost = min(lost, MAX_SLOTS)

        print(
            f"[Frame {self.frame_id}] det={detections_count} tracks={tracks_count} "
            f"assigned={self.assigned_this_frame} active={active} dormant={dormant} lost={lost} "
            f"unmatched_tracks={self.unmatched_tracks} unmatched_slots={self.unmatched_slots}"
        )

    # ---------- internal ----------
    def _slot_cost(
        self,
        slot: PlayerSlot,
        t_emb: Optional[np.ndarray],
        t_pos: Optional[Tuple[float, float]],
    ) -> float:
        emb_cost = 0.5
        pos_cost = 0.5

        if t_emb is not None and slot.embedding is not None:
            e = t_emb.astype(np.float32)
            n = np.linalg.norm(e)
            if n > 0:
                e = e / n
            cos = float(np.clip(np.dot(e, slot.embedding), -1.0, 1.0))
            emb_cost = 1.0 - (cos + 1.0) * 0.5

        if t_pos is not None and slot.last_position is not None:
            dx = float(t_pos[0] - slot.last_position[0])
            dy = float(t_pos[1] - slot.last_position[1])
            dist = (dx * dx + dy * dy) ** 0.5
            pos_cost = min(dist / 200.0, 1.0)

        recency = min(max(self.frame_id - slot.last_seen_frame, 0), DORMANT_TTL)
        recency_penalty = recency / float(DORMANT_TTL) * 0.15

        return 0.70 * emb_cost + 0.25 * pos_cost + recency_penalty
