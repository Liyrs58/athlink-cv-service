from __future__ import annotations

from dataclasses import dataclass
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
    state: str = "lost"
    active_track_id: Optional[int] = None
    seen_this_frame: bool = False
    last_seen_frame: int = -10**9
    last_position: Optional[Tuple[float, float]] = None
    embedding: Optional[np.ndarray] = None
    stability_counter: int = 0
    pending_tid: Optional[int] = None
    pending_streak: int = 0

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
    pid: Optional[str]
    source: str          # locked | revived | hungarian | unassigned
    confidence: float
    identity_valid: bool


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

        self._bench_snapshot: Dict[str, dict] = {}
        self._soft_snapshot: Dict[str, dict] = {}
        self._recovery_frames_left: int = 0
        self._last_memory_skips: int = 0

        # Mode flags set by tracker_core
        self.in_soft_collapse: bool = False    # no new Hungarian locks during collapse
        self.in_soft_recovery: bool = False    # no rebind/takeover during recovery
        self.in_scene_recovery: bool = False   # same protection post-bench
        self._scene_reset_frame: int = -1      # frame of last scene reset

        self.locks = IdentityLockManager(logger=logger)

    # ------------------------------------------------------------------
    # Scene reset
    # ------------------------------------------------------------------

    def reset_for_scene(self, frame_id: int = -1) -> None:
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
        self._recovery_frames_left = 60
        self.in_soft_collapse = False
        self.in_soft_recovery = False
        self.in_scene_recovery = True
        self._scene_reset_frame = frame_id

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
        for s in self.slots:
            s.active_track_id = None
            s.seen_this_frame = False
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
        Step 1: Auto-emit locked pairs (no Hungarian).
        Step 2: Hungarian on unlocked tracks × unlocked slots.
                - During collapse: skip lock promotion.
                - During recovery: block any rebind/takeover.
        Step 3: Promote stable Hungarian streak → lock (only outside collapse).
        """
        meta_map: Dict[int, AssignmentMeta] = {}
        track_to_pid: Dict[int, str] = {}
        memory_skips = 0
        locked_kept = 0

        if len(tracks) == 0:
            self.unmatched_slots = MAX_SLOTS
            return {}, {}

        # ── Step 1: locked pairs emit directly ──────────────────────────
        unlocked_tracks: List[object] = []
        for tr in tracks:
            tid = int(tr.track_id)
            lk = self.locks.get_lock(tid)
            if lk is None:
                unlocked_tracks.append(tr)
                continue

            pid = lk.pid
            slot = self._slot_by_pid(pid)
            if slot is None:
                self.locks.release_lock(tid, reason="bad_pid", frame_id=self.frame_id)
                unlocked_tracks.append(tr)
                continue

            track_to_pid[tid] = pid
            self._activate_slot(slot, tid, positions)
            self.locks.refresh_lock(tid, self.frame_id, confidence=lk.confidence)
            locked_kept += 1

            emb = embeddings.get(tid)
            if emb is not None and lk.stable_count >= MEMORY_UPDATE_MIN_STABLE:
                if memory_ok_tids is None or tid in memory_ok_tids:
                    slot.update_embedding(emb)
                else:
                    memory_skips += 1

            self.assigned_this_frame += 1
            meta_map[tid] = AssignmentMeta(pid=pid, source="locked",
                                           confidence=min(1.0, lk.stable_count / 30.0),
                                           identity_valid=True)

        if locked_kept and self.frame_id % self.debug_every == 0:
            print(f"[IDLockKeep] frame={self.frame_id} kept={locked_kept}")

        # ── Step 2: Hungarian on unlocked tracks × unlocked slots ───────
        unlocked_slot_idx = [
            j for j, s in enumerate(self.slots)
            if not self.locks.is_pid_locked(s.pid) and not s.seen_this_frame
        ]

        if not unlocked_tracks or not unlocked_slot_idx:
            for tr in unlocked_tracks:
                tid = int(tr.track_id)
                if tid not in meta_map:
                    meta_map[tid] = AssignmentMeta(pid=None, source="unassigned",
                                                   confidence=0.0, identity_valid=False)
        else:
            n_t = len(unlocked_tracks)
            n_s = len(unlocked_slot_idx)
            cost = np.full((n_t, n_s), 1e3, dtype=np.float32)
            for i, tr in enumerate(unlocked_tracks):
                tid = int(tr.track_id)
                for jj, j in enumerate(unlocked_slot_idx):
                    cost[i, jj] = self._slot_cost(
                        self.slots[j], embeddings.get(tid), positions.get(tid)
                    )

            r_idx, c_idx = linear_sum_assignment(cost)

            for r, c in zip(r_idx, c_idx):
                cst = float(cost[r, c])
                tid = int(unlocked_tracks[r].track_id)
                slot = self.slots[unlocked_slot_idx[c]]

                if cst > COST_REJECT_THRESHOLD:
                    meta_map[tid] = AssignmentMeta(pid=None, source="unassigned",
                                                   confidence=0.0, identity_valid=False)
                    continue

                confidence = max(0.0, 1.0 - cst)
                identity_valid = cst <= LOW_CONFIDENCE_THRESHOLD

                # During recovery: block takeover of any pid that had a recent lock
                if (self.in_soft_recovery or self.in_scene_recovery):
                    if self.locks.is_pid_locked(slot.pid):
                        old_tid = self.locks.get_tid_for_pid(slot.pid)
                        self.locks.record_blocked_switch(
                            self.frame_id, slot.pid, old_tid or -1, tid,
                            reason="recovery_lock",
                        )
                        self.locks.soft_recovery_rebinds_blocked += 1
                        meta_map[tid] = AssignmentMeta(pid=None, source="unassigned",
                                                       confidence=0.0, identity_valid=False)
                        continue

                # Stable-promote streak tracking
                if slot.pending_tid == tid:
                    slot.pending_streak += 1
                else:
                    slot.pending_tid = tid
                    slot.pending_streak = 1

                # Tentative assignment
                track_to_pid[tid] = slot.pid
                self._activate_slot(slot, tid, positions)

                if identity_valid:
                    emb = embeddings.get(tid)
                    if emb is not None:
                        if memory_ok_tids is None or tid in memory_ok_tids:
                            slot.update_embedding(emb)
                        else:
                            memory_skips += 1

                # Promote to lock only when streak is stable AND we are NOT in collapse
                if (not self.in_soft_collapse
                        and slot.pending_streak >= STABLE_PROMOTE_FRAMES
                        and identity_valid):
                    lk_new, status = self.locks.try_create_lock(
                        tid=tid, pid=slot.pid, source="hungarian",
                        frame_id=self.frame_id, confidence=cst,
                        # Outside recovery: allow rebind/takeover (counts as switch)
                        allow_takeover=not (self.in_soft_recovery or self.in_scene_recovery),
                        allow_rebind=not (self.in_soft_recovery or self.in_scene_recovery),
                    )
                    if status in ("created", "refreshed"):
                        slot.pending_tid = None
                        slot.pending_streak = 0
                    elif status in ("blocked_takeover", "blocked_rebind"):
                        # Should not happen since we already checked above, but be safe
                        self.locks.record_blocked_switch(
                            self.frame_id, slot.pid,
                            self.locks.get_tid_for_pid(slot.pid) or -1, tid,
                            reason=status,
                        )
                elif self.in_soft_collapse and slot.pending_streak >= STABLE_PROMOTE_FRAMES:
                    self.locks.collapse_lock_creations += 1

                self.assigned_this_frame += 1
                meta_map[tid] = AssignmentMeta(
                    pid=slot.pid, source="hungarian",
                    confidence=confidence, identity_valid=identity_valid,
                )

            for tr in unlocked_tracks:
                tid = int(tr.track_id)
                if tid not in meta_map:
                    meta_map[tid] = AssignmentMeta(pid=None, source="unassigned",
                                                   confidence=0.0, identity_valid=False)

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
                if s.pending_tid is not None:
                    s.pending_streak = max(0, s.pending_streak - 1)
                    if s.pending_streak == 0:
                        s.pending_tid = None

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def snapshot_scene(self, frame_id: int) -> int:
        """Save active+dormant slot state at freeze entry. Never overwrite
        with empty — if no slots have embeddings, keep the old snapshot."""
        candidate = {}
        for s in self.slots:
            if s.state in ("active", "dormant") and s.embedding is not None:
                candidate[s.pid] = {
                    "embedding": s.embedding.copy(),
                    "position": s.last_position,
                    "last_seen": s.last_seen_frame,
                }
        if len(candidate) == 0 and len(self._bench_snapshot) > 0:
            print(
                f"[Identity] SceneSnapshot: skip overwrite with 0 slots at frame {frame_id} "
                f"— keeping {len(self._bench_snapshot)} from earlier snapshot"
            )
            return len(self._bench_snapshot)
        self._bench_snapshot = candidate
        saved = len(self._bench_snapshot)
        print(f"[Identity] SceneSnapshot: {saved} slots saved at frame {frame_id}")
        return saved

    def snapshot_soft(self, frame_id: int) -> int:
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

    # ------------------------------------------------------------------
    # Revival
    # ------------------------------------------------------------------

    def revive_cost_matrix(
        self,
        tracks: Sequence[object],
        embeddings: Dict[int, np.ndarray],
        positions: Dict[int, Tuple[float, float]],
    ) -> Tuple[Dict[int, str], Dict[int, AssignmentMeta]]:
        """Scene revival — called once after bench→play. Logs exact failure reason."""
        revived: Dict[int, str] = {}
        meta: Dict[int, AssignmentMeta] = {}

        snap = self._bench_snapshot
        if not snap:
            print(f"[SceneReviveFail] frame={self.frame_id} reason=snapshot_empty")
            return {}, {}
        if len(tracks) == 0:
            print(f"[SceneReviveFail] frame={self.frame_id} reason=no_candidate_tracks")
            return {}, {}

        embed_missing = sum(1 for tr in tracks if embeddings.get(int(tr.track_id)) is None)
        if embed_missing == len(tracks):
            print(f"[SceneReviveFail] frame={self.frame_id} reason=all_embeddings_missing")
            return {}, {}

        track_ids = [int(t.track_id) for t in tracks]
        snap_pids = list(snap.keys())
        n_t, n_s = len(track_ids), len(snap_pids)
        cost = np.full((n_t, n_s), 1e3, dtype=np.float32)

        for i, tid in enumerate(track_ids):
            t_emb = embeddings.get(tid)
            t_pos = positions.get(tid)
            for j, pid in enumerate(snap_pids):
                s = snap[pid]
                emb_c = 0.5
                pos_c = 0.5
                if t_emb is not None and s["embedding"] is not None:
                    e = t_emb.astype(np.float32)
                    n = np.linalg.norm(e)
                    if n > 0: e /= n
                    cos = float(np.clip(np.dot(e, s["embedding"]), -1.0, 1.0))
                    emb_c = 1.0 - (cos + 1.0) * 0.5
                if t_pos is not None and s["position"] is not None:
                    dx = float(t_pos[0] - s["position"][0])
                    dy = float(t_pos[1] - s["position"][1])
                    pos_c = min((dx*dx + dy*dy)**0.5 / 300.0, 1.0)
                cost[i, j] = 0.75 * emb_c + 0.25 * pos_c

        r_idx, c_idx = linear_sum_assignment(cost)
        accepted = 0
        min_cost = float(cost[r_idx, c_idx].min()) if len(r_idx) else 999.0

        for r, c in zip(r_idx, c_idx):
            cst = float(cost[r, c])
            if cst < 0.60:
                tid = track_ids[r]
                pid = snap_pids[c]
                revived[tid] = pid
                meta[tid] = AssignmentMeta(pid=pid, source="revived",
                                           confidence=max(0.0, 1.0 - cst),
                                           identity_valid=True)
                self._apply_revival(tid, pid, embeddings, positions, source="scene", cost=cst)
                accepted += 1

        if accepted == 0:
            print(
                f"[SceneReviveFail] frame={self.frame_id} reason=costs_too_high "
                f"n_tracks={n_t} n_snap={n_s} min_cost={min_cost:.3f} threshold=0.60"
            )
        else:
            print(f"[Identity] Revival: {accepted}/{n_t} tracks matched from scene snapshot")

        self._bench_snapshot = {}
        return revived, meta

    def revive_from_soft_snapshot(
        self,
        tracks: Sequence[object],
        embeddings: Dict[int, np.ndarray],
        positions: Dict[int, Tuple[float, float]],
        is_first_recovery_frame: bool = False,
    ) -> Tuple[Dict[int, str], Dict[int, AssignmentMeta]]:
        """Soft revival — enforces strict 1:1, blocks conflicts, no rebind."""
        revived: Dict[int, str] = {}
        meta: Dict[int, AssignmentMeta] = {}

        if not self._soft_snapshot or len(tracks) == 0:
            return {}, {}

        track_ids = [int(t.track_id) for t in tracks]
        snap_pids = list(self._soft_snapshot.keys())
        n_t, n_s = len(track_ids), len(snap_pids)
        cost = np.full((n_t, n_s), 1e3, dtype=np.float32)
        debug_costs = []

        for i, tid in enumerate(track_ids):
            t_emb = embeddings.get(tid)
            t_pos = positions.get(tid)
            for j, pid in enumerate(snap_pids):
                s = self._soft_snapshot[pid]
                emb_c = 0.5
                pos_c = 0.5
                if t_emb is not None and s["embedding"] is not None:
                    e = t_emb.astype(np.float32)
                    n = np.linalg.norm(e)
                    if n > 0: e /= n
                    cos = float(np.clip(np.dot(e, s["embedding"]), -1.0, 1.0))
                    emb_c = 1.0 - (cos + 1.0) * 0.5
                if t_pos is not None and s["position"] is not None:
                    dx = float(t_pos[0] - s["position"][0])
                    dy = float(t_pos[1] - s["position"][1])
                    pos_c = min((dx*dx + dy*dy)**0.5 / 300.0, 1.0)
                cst = 0.85 * emb_c + 0.15 * pos_c
                cost[i, j] = cst
                if is_first_recovery_frame:
                    debug_costs.append({"tid": tid, "pid": pid, "cost": float(cst),
                                        "emb": emb_c, "pos": pos_c})

        r_idx, c_idx = linear_sum_assignment(cost)

        used_tids: Set[int] = set()
        used_pids: Set[str] = set()

        for r, c in zip(r_idx, c_idx):
            cst = float(cost[r, c])
            if cst >= 0.60:
                continue
            tid = track_ids[r]
            pid = snap_pids[c]

            # Strict 1:1 — skip if either already consumed
            if tid in used_tids or pid in used_pids:
                continue

            # Block if pid already has a live lock to a different tid
            existing_tid = self.locks.get_tid_for_pid(pid)
            if existing_tid is not None and existing_tid != tid:
                self.locks.record_blocked_switch(
                    self.frame_id, pid, existing_tid, tid, reason="soft_revival_conflict"
                )
                self.locks.soft_recovery_rebinds_blocked += 1
                continue

            revived[tid] = pid
            meta[tid] = AssignmentMeta(pid=pid, source="revived",
                                       confidence=max(0.0, 1.0 - cst),
                                       identity_valid=True)
            self._apply_revival(tid, pid, embeddings, positions, source="soft", cost=cst)
            used_tids.add(tid)
            used_pids.add(pid)
            if pid in self._soft_snapshot:
                del self._soft_snapshot[pid]

        if is_first_recovery_frame:
            debug_costs.sort(key=lambda x: x["cost"])
            for d in debug_costs[:5]:
                print(
                    f"[SoftReviveCost] tid={d['tid']} pid={d['pid']} cost={d['cost']:.3f} "
                    f"emb={d['emb']:.3f} pos={d['pos']:.3f} accepted={d['cost'] < 0.60}"
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
        slot = self._slot_by_pid(pid)
        if slot is None:
            return
        self._activate_slot(slot, tid, positions)
        self.assigned_this_frame += 1
        self.locks.create_lock(tid=tid, pid=pid, source="revived",
                               frame_id=self.frame_id, confidence=cost)
        print(f"[RevivalLock] frame={self.frame_id} tid={tid} pid={pid} ttl=90 source={source}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _activate_slot(
        self, slot: PlayerSlot, tid: int,
        positions: Dict[int, Tuple[float, float]],
    ) -> None:
        slot.active_track_id = tid
        slot.seen_this_frame = True
        slot.state = "active"
        slot.last_seen_frame = self.frame_id
        slot.last_position = positions.get(tid, slot.last_position)
        slot.stability_counter += 1

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
        print(
            f"[Frame {self.frame_id}] det={detections_count} tracks={tracks_count} "
            f"assigned={self.assigned_this_frame} active={active} dormant={dormant} "
            f"lost={min(lost, MAX_SLOTS)} unmatched_tracks={self.unmatched_tracks} "
            f"unmatched_slots={self.unmatched_slots} "
            f"memory_skips={self._last_memory_skips} locks={len(self.locks._tid_to_lock)}"
        )

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
            if n > 0: e /= n
            cos = float(np.clip(np.dot(e, slot.embedding), -1.0, 1.0))
            emb_cost = 1.0 - (cos + 1.0) * 0.5

        if t_pos is not None and slot.last_position is not None:
            dx = float(t_pos[0] - slot.last_position[0])
            dy = float(t_pos[1] - slot.last_position[1])
            pos_cost = min((dx*dx + dy*dy)**0.5 / 200.0, 1.0)

        recency = min(max(self.frame_id - slot.last_seen_frame, 0), DORMANT_TTL)
        recency_penalty = recency / float(DORMANT_TTL) * 0.15
        lock_discount = 0.15 if slot.stability_counter >= 10 else 0.0

        if self._recovery_frames_left > 0:
            final_cost = 0.90 * emb_cost + 0.05 * pos_cost + recency_penalty - lock_discount
        elif recency == 0:
            final_cost = 0.50 * emb_cost + 0.35 * pos_cost - 0.10 - lock_discount
        else:
            final_cost = 0.65 * emb_cost + 0.25 * pos_cost + recency_penalty - (lock_discount * 0.5)

        return float(max(0.0, min(1.0, final_cost)))
