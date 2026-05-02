from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from services.identity_locks import (
    IdentityLockManager,
    STABLE_PROMOTE_FRAMES,
    MEMORY_UPDATE_MIN_STABLE,
)


DORMANT_TTL = 180          # frames before dormant → lost (~7s @ 25fps)
MAX_SLOTS = 22
COST_REJECT_THRESHOLD = 0.72
LOW_CONFIDENCE_THRESHOLD = 0.55  # cost above this → identity_valid=False
EMB_ALPHA = 0.25


@dataclass
class PlayerSlot:
    pid: str
    state: str = "lost"    # active | dormant | lost
    active_track_id: Optional[int] = None
    seen_this_frame: bool = False
    last_seen_frame: int = -10**9
    last_position: Optional[Tuple[float, float]] = None
    embedding: Optional[np.ndarray] = None
    stability_counter: int = 0  # number of consecutive frames slot has been active
    pending_tid: Optional[int] = None       # candidate track during stable-promote window
    pending_streak: int = 0                 # consecutive frames same pending_tid matched here

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


@dataclass
class AssignmentMeta:
    """Per-track metadata produced by assign_tracks."""
    pid: Optional[str]                 # "P7" or None if unassigned
    source: str                        # locked | revived | hungarian | unassigned
    confidence: float                  # 0..1, higher better
    identity_valid: bool               # render as P-id (True) or grey/U (False)


class IdentityCore:
    def __init__(self, logger=None, debug_every: int = 30):
        self.logger = logger
        self.debug_every = debug_every
        self.slots: List[PlayerSlot] = [PlayerSlot(pid=f"P{i}") for i in range(1, MAX_SLOTS + 1)]

        print("[ReID] using HSV fallback only for appearance matching")

        self.frame_id: int = -1
        self.assigned_this_frame: int = 0
        self.unmatched_tracks: int = 0
        self.unmatched_slots: int = 0

        # Snapshot taken at bench/freeze entry — used for revival on return
        self._bench_snapshot: Dict[str, dict] = {}  # pid -> {embedding, position, last_seen}
        self._soft_snapshot: Dict[str, dict] = {}   # pid -> {embedding, position, last_seen}
        self._recovery_frames_left: int = 0  # countdown after scene reset
        self._last_memory_skips: int = 0

        # Persistent lock layer
        self.locks = IdentityLockManager(logger=logger)

    # ------------------------------------------------------------------
    # Scene reset
    # ------------------------------------------------------------------

    def reset_for_scene(self) -> None:
        """Clear active tracking state on scene boundary (bench_shot→play)
        but preserve dormant memory and bench snapshot to revive later.
        Locks are dropped: the new tracker IDs will not match old tids."""
        for s in self.slots:
            s.active_track_id = None
            s.seen_this_frame = False
            s.pending_tid = None
            s.pending_streak = 0
            if s.state in ("active", "dormant") and s.embedding is not None:
                s.state = "dormant"
            else:
                s.state = "lost"

        if len(self._bench_snapshot) > 0:
            print(f"[Identity] Reset: {len(self._bench_snapshot)} snapshot slots survived reset")
        else:
            print("[Identity] Reset: No snapshot exists to survive reset.")

        self.locks.reset_all()
        self._recovery_frames_left = 60  # permissive matching for 60 frames

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def begin_frame(self, frame_id: int, present_tids: Optional[Set[int]] = None) -> None:
        self.frame_id = frame_id
        self.assigned_this_frame = 0
        self.unmatched_tracks = 0
        self.unmatched_slots = 0
        if self._recovery_frames_left > 0:
            self._recovery_frames_left -= 1
        # CRITICAL: clear frame-local flags only — do NOT reset state/embedding
        for s in self.slots:
            s.active_track_id = None
            s.seen_this_frame = False
        # Decay locks for tracks that aren't here this frame
        if present_tids is not None:
            self.locks.tick(frame_id, present_tids)

    def assign_tracks(
        self,
        tracks: Sequence[object],
        embeddings: Dict[int, np.ndarray],
        positions: Dict[int, Tuple[float, float]],
        memory_ok_tids: Optional[set] = None,
    ) -> Tuple[Dict[int, str], Dict[int, AssignmentMeta]]:
        """
        Persistent identity assignment.

          Step 1: For every locked tid, emit its pid directly. Refresh lock.
                  These tracks are removed from the Hungarian pool.
          Step 2: Run Hungarian on (unlocked tracks) × (unlocked slots).
          Step 3: Hungarian matches enter a "pending" state on the slot.
                  After STABLE_PROMOTE_FRAMES of agreement, promote to lock.
          Step 4: High-cost matches (> LOW_CONFIDENCE_THRESHOLD) emit
                  identity_valid=False so render shows them as uncertain.
          Step 5: Reject anything > COST_REJECT_THRESHOLD entirely.

        Returns:
            (track_to_pid, meta_map):
              track_to_pid: {tid -> "P7"} for tracks that received an identity
              meta_map: {tid -> AssignmentMeta} for ALL input tracks
        """
        meta_map: Dict[int, AssignmentMeta] = {}

        if len(tracks) == 0:
            self.unmatched_slots = MAX_SLOTS
            return {}, {}

        track_to_pid: Dict[int, str] = {}
        memory_skips = 0
        locked_kept = 0

        # ---- STEP 1: honour existing locks ------------------------------
        unlocked_tracks: List[object] = []
        unlocked_track_ids: List[int] = []
        for tr in tracks:
            tid = int(tr.track_id)
            lk = self.locks.get_lock(tid)
            if lk is None:
                unlocked_tracks.append(tr)
                unlocked_track_ids.append(tid)
                continue

            pid = lk.pid
            slot = self._slot_by_pid(pid)
            if slot is None:
                # Defensive: lock points at unknown pid — drop it
                self.locks.release_lock(tid, reason="bad_pid")
                unlocked_tracks.append(tr)
                unlocked_track_ids.append(tid)
                continue

            track_to_pid[tid] = pid
            slot.active_track_id = tid
            slot.seen_this_frame = True
            slot.state = "active"
            slot.last_seen_frame = self.frame_id
            slot.last_position = positions.get(tid)
            slot.stability_counter += 1
            self.locks.refresh_lock(tid, self.frame_id, confidence=lk.confidence)
            locked_kept += 1

            # Memory write only after lock is reasonably stable + crop quality OK
            emb = embeddings.get(tid)
            if emb is not None and lk.stable_count >= MEMORY_UPDATE_MIN_STABLE:
                if memory_ok_tids is None or tid in memory_ok_tids:
                    slot.update_embedding(emb)
                else:
                    memory_skips += 1

            self.assigned_this_frame += 1

            meta_map[tid] = AssignmentMeta(
                pid=pid,
                source="locked",
                confidence=max(0.5, 1.0 - lk.confidence),
                identity_valid=True,
            )

        if locked_kept and self.frame_id % self.debug_every == 0:
            print(f"[IDLockKeep] frame={self.frame_id} kept={locked_kept}")

        # ---- STEP 2: Hungarian on unlocked tracks × unlocked slots ------
        unlocked_slot_idx = [
            j for j, s in enumerate(self.slots)
            if not self.locks.is_pid_locked(s.pid) and not s.seen_this_frame
        ]

        if unlocked_tracks and unlocked_slot_idx:
            n_t = len(unlocked_tracks)
            n_s = len(unlocked_slot_idx)
            cost = np.full((n_t, n_s), 1e3, dtype=np.float32)
            for i, tr in enumerate(unlocked_tracks):
                tid = int(tr.track_id)
                t_emb = embeddings.get(tid)
                t_pos = positions.get(tid)
                for jj, j in enumerate(unlocked_slot_idx):
                    cost[i, jj] = self._slot_cost(self.slots[j], t_emb, t_pos)

            r_idx, c_idx = linear_sum_assignment(cost)

            for r, c in zip(r_idx, c_idx):
                cst = float(cost[r, c])
                tid = int(unlocked_tracks[r].track_id)
                slot = self.slots[unlocked_slot_idx[c]]

                if cst > COST_REJECT_THRESHOLD:
                    # Truly bad match — leave unassigned
                    meta_map[tid] = AssignmentMeta(
                        pid=None, source="unassigned", confidence=0.0,
                        identity_valid=False,
                    )
                    continue

                # Confidence inverse of cost (higher cost = lower confidence)
                confidence = max(0.0, 1.0 - cst)
                identity_valid = cst <= LOW_CONFIDENCE_THRESHOLD

                # ---- STEP 3: stable-promote → lock --------------------
                if slot.pending_tid == tid:
                    slot.pending_streak += 1
                else:
                    slot.pending_tid = tid
                    slot.pending_streak = 1

                # If this track wants a slot whose pid is currently locked to
                # someone else (shouldn't happen because we filtered, but be safe)
                if self.locks.is_pid_locked(slot.pid):
                    self.locks.record_blocked_switch(
                        self.frame_id, slot.pid,
                        self.locks.get_tid_for_pid(slot.pid) or -1,
                        tid,
                        reason="locked",
                    )
                    meta_map[tid] = AssignmentMeta(
                        pid=None, source="unassigned", confidence=0.0,
                        identity_valid=False,
                    )
                    continue

                # Tentative assignment for this frame
                track_to_pid[tid] = slot.pid
                slot.active_track_id = tid
                slot.seen_this_frame = True
                slot.state = "active"
                slot.last_seen_frame = self.frame_id
                slot.last_position = positions.get(tid)
                slot.stability_counter += 1

                # Memory write only when we are confident
                if identity_valid:
                    emb = embeddings.get(tid)
                    if emb is not None:
                        if memory_ok_tids is None or tid in memory_ok_tids:
                            slot.update_embedding(emb)
                        else:
                            memory_skips += 1

                # Promote to lock when streak is stable enough
                if slot.pending_streak >= STABLE_PROMOTE_FRAMES and identity_valid:
                    self.locks.create_lock(
                        tid=tid,
                        pid=slot.pid,
                        source="hungarian",
                        frame_id=self.frame_id,
                        confidence=cst,
                    )
                    slot.pending_tid = None
                    slot.pending_streak = 0

                self.assigned_this_frame += 1

                meta_map[tid] = AssignmentMeta(
                    pid=slot.pid,
                    source="hungarian",
                    confidence=confidence,
                    identity_valid=identity_valid,
                )

            # Tracks not picked up by Hungarian
            picked = {int(unlocked_tracks[r].track_id) for r in r_idx}
            for tr in unlocked_tracks:
                tid = int(tr.track_id)
                if tid not in picked and tid not in meta_map:
                    meta_map[tid] = AssignmentMeta(
                        pid=None, source="unassigned", confidence=0.0,
                        identity_valid=False,
                    )

        else:
            for tr in unlocked_tracks:
                tid = int(tr.track_id)
                meta_map[tid] = AssignmentMeta(
                    pid=None, source="unassigned", confidence=0.0,
                    identity_valid=False,
                )

        self.unmatched_tracks = sum(1 for m in meta_map.values() if m.pid is None)
        self.unmatched_slots = sum(1 for s in self.slots if not s.seen_this_frame)
        self._last_memory_skips = memory_skips
        return track_to_pid, meta_map

    def end_frame(self, frame_id: Optional[int] = None) -> None:
        if frame_id is not None:
            self.frame_id = frame_id
        for s in self.slots:
            if s.seen_this_frame:
                continue
            age = self.frame_id - s.last_seen_frame
            s.state = "dormant" if age <= DORMANT_TTL else "lost"
            s.active_track_id = None
            if not s.seen_this_frame:
                s.stability_counter = 0
                # If pending_streak track wasn't here this frame, decay it
                if s.pending_tid is not None:
                    s.pending_streak = max(0, s.pending_streak - 1)
                    if s.pending_streak == 0:
                        s.pending_tid = None

    # ------------------------------------------------------------------
    # Bench / soft snapshot
    # ------------------------------------------------------------------

    def snapshot_scene(self, frame_id: int) -> int:
        """Save active+dormant slot state on freeze entry."""
        self._bench_snapshot = {}
        for s in self.slots:
            if s.state in ("active", "dormant") and s.embedding is not None:
                self._bench_snapshot[s.pid] = {
                    "embedding": s.embedding.copy(),
                    "position": s.last_position,
                    "last_seen": s.last_seen_frame,
                }
        saved = len(self._bench_snapshot)
        print(f"[Identity] SceneSnapshot: {saved} slots saved at frame {frame_id}")
        return saved

    def snapshot_soft(self, frame_id: int) -> int:
        """Save state on soft-collapse entry."""
        self._soft_snapshot = {}
        for s in self.slots:
            if s.state in ("active", "dormant") and s.embedding is not None:
                self._soft_snapshot[s.pid] = {
                    "embedding": s.embedding.copy(),
                    "position": s.last_position,
                    "last_seen": s.last_seen_frame,
                }
        saved = len(self._soft_snapshot)
        print(f"[Identity] SoftSnapshot: {saved} slots saved at frame {frame_id}")
        return saved

    def revive_cost_matrix(
        self,
        tracks: Sequence[object],
        embeddings: Dict[int, np.ndarray],
        positions: Dict[int, Tuple[float, float]],
    ) -> Tuple[Dict[int, str], Dict[int, AssignmentMeta]]:
        """Scene revival: match new tids to bench snapshot, create locks immediately."""
        if not self._bench_snapshot or len(tracks) == 0:
            return {}, {}

        track_ids = [int(t.track_id) for t in tracks]
        snap_pids = list(self._bench_snapshot.keys())
        n_t = len(track_ids)
        n_s = len(snap_pids)

        cost = np.full((n_t, n_s), 1e3, dtype=np.float32)
        for i, tid in enumerate(track_ids):
            t_emb = embeddings.get(tid)
            t_pos = positions.get(tid)
            for j, pid in enumerate(snap_pids):
                snap = self._bench_snapshot[pid]
                emb_cost = 0.5
                pos_cost = 0.5
                if t_emb is not None and snap["embedding"] is not None:
                    e = t_emb.astype(np.float32)
                    n = np.linalg.norm(e)
                    if n > 0:
                        e = e / n
                    cos = float(np.clip(np.dot(e, snap["embedding"]), -1.0, 1.0))
                    emb_cost = 1.0 - (cos + 1.0) * 0.5
                if t_pos is not None and snap["position"] is not None:
                    dx = float(t_pos[0] - snap["position"][0])
                    dy = float(t_pos[1] - snap["position"][1])
                    dist = (dx * dx + dy * dy) ** 0.5
                    pos_cost = min(dist / 300.0, 1.0)
                cost[i, j] = 0.75 * emb_cost + 0.25 * pos_cost

        r_idx, c_idx = linear_sum_assignment(cost)
        revived: Dict[int, str] = {}
        meta: Dict[int, AssignmentMeta] = {}
        for r, c in zip(r_idx, c_idx):
            cst = float(cost[r, c])
            if cst < 0.60:
                tid = track_ids[r]
                pid = snap_pids[c]
                revived[tid] = pid
                meta[tid] = AssignmentMeta(
                    pid=pid, source="revived", confidence=max(0.0, 1.0 - cst),
                    identity_valid=True,
                )
                self._apply_revival(tid, pid, embeddings, positions, source="scene", cost=cst)

        print(f"[Identity] Revival: {len(revived)}/{n_t} tracks matched from scene snapshot")
        self._bench_snapshot = {}
        return revived, meta

    def revive_from_soft_snapshot(
        self,
        tracks: Sequence[object],
        embeddings: Dict[int, np.ndarray],
        positions: Dict[int, Tuple[float, float]],
        is_first_recovery_frame: bool = False,
    ) -> Tuple[Dict[int, str], Dict[int, AssignmentMeta]]:
        """Soft revival: match tids to soft snapshot, create locks immediately."""
        if not self._soft_snapshot or len(tracks) == 0:
            return {}, {}

        track_ids = [int(t.track_id) for t in tracks]
        snap_pids = list(self._soft_snapshot.keys())
        n_t = len(track_ids)
        n_s = len(snap_pids)

        cost = np.full((n_t, n_s), 1e3, dtype=np.float32)
        debug_costs = []
        for i, tid in enumerate(track_ids):
            t_emb = embeddings.get(tid)
            t_pos = positions.get(tid)
            for j, pid in enumerate(snap_pids):
                snap = self._soft_snapshot[pid]
                emb_cost = 0.5
                pos_cost = 0.5
                if t_emb is not None and snap["embedding"] is not None:
                    e = t_emb.astype(np.float32)
                    n = np.linalg.norm(e)
                    if n > 0:
                        e = e / n
                    cos = float(np.clip(np.dot(e, snap["embedding"]), -1.0, 1.0))
                    emb_cost = 1.0 - (cos + 1.0) * 0.5
                if t_pos is not None and snap["position"] is not None:
                    dx = float(t_pos[0] - snap["position"][0])
                    dy = float(t_pos[1] - snap["position"][1])
                    dist = (dx * dx + dy * dy) ** 0.5
                    pos_cost = min(dist / 300.0, 1.0)
                cst = 0.85 * emb_cost + 0.15 * pos_cost
                cost[i, j] = cst
                if is_first_recovery_frame:
                    debug_costs.append({
                        "tid": tid, "pid": pid, "cost": float(cst),
                        "emb_cost": emb_cost, "pos_cost": pos_cost,
                    })

        r_idx, c_idx = linear_sum_assignment(cost)
        revived: Dict[int, str] = {}
        meta: Dict[int, AssignmentMeta] = {}
        for r, c in zip(r_idx, c_idx):
            cst = float(cost[r, c])
            if cst < 0.60:
                tid = track_ids[r]
                pid = snap_pids[c]
                revived[tid] = pid
                meta[tid] = AssignmentMeta(
                    pid=pid, source="revived", confidence=max(0.0, 1.0 - cst),
                    identity_valid=True,
                )
                self._apply_revival(tid, pid, embeddings, positions, source="soft", cost=cst)
                if pid in self._soft_snapshot:
                    del self._soft_snapshot[pid]

        if is_first_recovery_frame:
            debug_costs.sort(key=lambda x: x["cost"])
            for d in debug_costs[:5]:
                accepted = d["cost"] < 0.60
                print(
                    f"[SoftReviveCost] tid={d['tid']} pid={d['pid']} cost={d['cost']:.3f} "
                    f"emb={d['emb_cost']:.3f} pos={d['pos_cost']:.3f} accepted={accepted}"
                )
        return revived, meta

    def _apply_revival(
        self,
        tid: int,
        pid: str,
        embeddings: Dict[int, np.ndarray],
        positions: Dict[int, Tuple[float, float]],
        source: str,
        cost: float,
    ) -> None:
        """Mark slot as active for revival AND create a lock that protects it."""
        slot = self._slot_by_pid(pid)
        if slot is None:
            return
        slot.active_track_id = tid
        slot.seen_this_frame = True
        slot.state = "active"
        slot.last_seen_frame = self.frame_id
        if tid in positions:
            slot.last_position = positions[tid]
        slot.stability_counter += 1
        # Don't write the noisy revival embedding straight in — wait for stability
        self.assigned_this_frame += 1

        self.locks.create_lock(
            tid=tid, pid=pid, source="revived",
            frame_id=self.frame_id, confidence=cost,
        )
        print(f"[RevivalLock] frame={self.frame_id} tid={tid} pid={pid} ttl=90 source={source}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _slot_by_pid(self, pid: str) -> Optional[PlayerSlot]:
        for s in self.slots:
            if s.pid == pid:
                return s
        return None

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
            f"unmatched_tracks={self.unmatched_tracks} unmatched_slots={self.unmatched_slots} "
            f"memory_skips={self._last_memory_skips} locks={len(self.locks._tid_to_lock)}"
        )

    # ------------------------------------------------------------------
    # Cost
    # ------------------------------------------------------------------

    def _slot_cost(
        self,
        slot: PlayerSlot,
        t_emb: Optional[np.ndarray],
        t_pos: Optional[Tuple[float, float]],
    ) -> float:
        if self._recovery_frames_left > 0 and slot.state == "lost" and slot.embedding is None:
            return 0.30

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

        lock_discount = 0.0
        if slot.stability_counter >= 10:
            lock_discount = 0.15

        if self._recovery_frames_left > 0:
            final_cost = 0.90 * emb_cost + 0.05 * pos_cost + recency_penalty - lock_discount
        else:
            if recency == 0:
                final_cost = 0.50 * emb_cost + 0.35 * pos_cost - 0.10 - lock_discount
            else:
                final_cost = 0.65 * emb_cost + 0.25 * pos_cost + recency_penalty - (lock_discount * 0.5)

        return float(max(0.0, min(1.0, final_cost)))
