"""
Identity Core — stateful player identity engine.

Identity validity states (per output box):
  LOCKED     — stable lock (stable_count >= LOCK_PROMOTE_FRAMES); export as P-ID.
  REVIVED    — recovered from snapshot with good confidence; export as P-ID.
  PROVISIONAL — matched by Hungarian in normal play, not yet stable; do NOT export.
  UNKNOWN    — no match, or restricted mode, or rejected by gate; render as UNK.

Single source of truth for restricted mode: _identity_restricted() property.
This is the ONLY check used everywhere inside this class.

Key invariants (enforced in code, not just comments):
  1. P-IDs emitted ONLY for LOCKED or REVIVED.
  2. _identity_restricted() == True → allow_new_assignments=False, no PROVISIONAL,
     no Hungarian lock creation, no slot embedding updates from unknown tracks.
  3. _identity_restricted() includes ALL of: soft_collapse, soft_recovery,
     scene_recovery, _recovery_frames_left > 0.
  4. Lock tick passes restricted flag → stale locks become DORMANT not expired.
  5. Cross-team matches → hard reject.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    from services.identity_locks import (
        IdentityLockManager,
        MEMORY_UPDATE_MIN_STABLE,
    )
    from services.pitch_geometry import assignment_position_cost
except ModuleNotFoundError:
    import os as _os, sys as _sys
    _services_dir = _os.path.join(_os.path.dirname(__file__))
    if _services_dir not in _sys.path:
        _sys.path.insert(0, _services_dir)
    from identity_locks import (  # type: ignore[no-redef]
        IdentityLockManager,
        MEMORY_UPDATE_MIN_STABLE,
    )
    from pitch_geometry import assignment_position_cost  # type: ignore[no-redef]


DORMANT_TTL = 180
MAX_SLOTS = 22
COST_REJECT_THRESHOLD = 0.72
REVIVAL_COST_THRESHOLD = 0.60
REVIVAL_MARGIN_MIN = 0.05
LOW_CONFIDENCE_THRESHOLD = 0.55
EMB_ALPHA = 0.25
STABLE_PROTECT_THRESHOLD = 10
LOCK_PROMOTE_FRAMES = 5


class IdentityState(str, Enum):
    LOCKED = "locked"
    REVIVED = "revived"
    PROVISIONAL = "provisional"
    UNKNOWN = "unknown"


@dataclass
class AssignmentMeta:
    pid: Optional[str]
    source: str                 # locked | revived | provisional | unknown
    identity_state: IdentityState
    confidence: float
    identity_valid: bool        # True only for LOCKED or REVIVED


@dataclass
class PlayerSlot:
    pid: str
    state: str = "lost"         # active | dormant | lost
    active_track_id: Optional[int] = None
    seen_this_frame: bool = False
    last_seen_frame: int = -10**9
    last_position: Optional[Tuple[float, float]] = None
    last_pitch: Optional[Tuple[float, float]] = None
    embedding: Optional[np.ndarray] = None
    stability_counter: int = 0
    pending_tid: Optional[int] = None
    pending_streak: int = 0
    team_id: Optional[int] = None

    def update_embedding(self, emb: np.ndarray) -> None:
        emb = emb.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm
        if self.embedding is None:
            self.embedding = emb
        else:
            self.embedding = (1.0 - EMB_ALPHA) * self.embedding + EMB_ALPHA * emb
            norm2 = np.linalg.norm(self.embedding)
            if norm2 > 0:
                self.embedding /= norm2


class IdentityCore:
    def __init__(self, logger=None, debug_every: int = 30):
        self.logger = logger
        self.debug_every = debug_every
        self.slots: List[PlayerSlot] = [
            PlayerSlot(pid=f"P{i}") for i in range(1, MAX_SLOTS + 1)
        ]
        print("[ReID] using HSV fallback only for appearance matching")

        self.frame_id: int = -1
        self.assigned_this_frame: int = 0
        self.unmatched_tracks: int = 0
        self.unmatched_slots: int = 0

        self._bench_snapshot: Dict[str, dict] = {}
        self._soft_snapshot: Dict[str, dict] = {}
        self._recovery_frames_left: int = 0
        self._last_memory_skips: int = 0

        # Mode flags — set externally by tracker_core
        self.in_soft_collapse: bool = False
        self.in_soft_recovery: bool = False
        self.in_scene_recovery: bool = False
        self._scene_reset_frame: int = -1

        # Per-frame extras injected by tracker_core
        self.pitch_positions: Dict[int, Tuple[float, float]] = {}
        self.team_labels: Dict[int, Optional[int]] = {}
        self.reid_mode: str = "HSV-fallback"

        self.locks = IdentityLockManager(logger=logger)

        # Run-level metrics
        self.recovery_normal_assignments: int = 0
        self.scene_recovery_normal_assignments: int = 0
        self.restricted_hungarian_assignments: int = 0
        self.ambiguous_rejects: int = 0
        self.revived_count: int = 0

    # ------------------------------------------------------------------
    # Single source of truth for restricted identity mode
    # ------------------------------------------------------------------

    def _recovery_lock_protected(self, lock) -> bool:
        """True when restricted and the existing lock was freshly revived — must not be stolen."""
        return self._identity_restricted and lock is not None and lock.source == "revived"

    @property
    def _identity_restricted(self) -> bool:
        """
        True when ANY recovery/collapse condition is active.
        This is the ONE check used everywhere. No partial checks allowed.
        """
        return (
            self.in_soft_collapse
            or self.in_soft_recovery
            or self.in_scene_recovery
            or self._recovery_frames_left > 0
        )

    def _identity_restricted_reason(self) -> str:
        reasons = []
        if self.in_soft_collapse:
            reasons.append("soft_collapse")
        if self.in_soft_recovery:
            reasons.append("soft_recovery")
        if self.in_scene_recovery:
            reasons.append("scene_recovery")
        if self._recovery_frames_left > 0:
            reasons.append(f"recovery_frames_left={self._recovery_frames_left}")
        return ",".join(reasons) if reasons else "none"

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

        snap_count = len(self._bench_snapshot)
        if snap_count > 0:
            print(f"[Identity] Reset: {snap_count} snapshot slots survived reset")
        else:
            print("[Identity] Reset: No snapshot exists to survive reset.")

        self.locks.reset_all()
        self._recovery_frames_left = 60
        self.in_soft_collapse = False
        self.in_soft_recovery = False
        self.in_scene_recovery = True
        self._scene_reset_frame = frame_id
        # Sync lock manager restricted flag
        self.locks.in_restricted = True
        self.locks.in_collapse = False

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
        # Sync lock manager restricted/collapse flags from our state
        restricted = self._identity_restricted
        self.locks.in_restricted = restricted
        self.locks.in_collapse = self.in_soft_collapse
        if present_tids is not None:
            self.locks.tick(frame_id, present_tids, restricted=restricted)

    # ------------------------------------------------------------------
    # assign_tracks — single entry point every frame
    # ------------------------------------------------------------------

    def assign_tracks(
        self,
        tracks: Sequence[object],
        embeddings: Dict[int, np.ndarray],
        positions: Dict[int, Tuple[float, float]],
        memory_ok_tids: Optional[set] = None,
        allow_new_assignments: bool = True,
    ) -> Tuple[Dict[int, str], Dict[int, AssignmentMeta]]:
        """
        allow_new_assignments=False OR _identity_restricted=True:
          → only locked pairs emitted; all others UNKNOWN.
        Caller always passes allow_new_assignments=not _identity_restricted().
        The internal check here is a second safety net.
        """
        meta_map: Dict[int, AssignmentMeta] = {}
        track_to_pid: Dict[int, str] = {}
        memory_skips = 0
        locked_kept = 0

        # Internal gate — overrides caller if they passed wrong value
        restricted = self._identity_restricted
        if restricted:
            allow_new_assignments = False

        if len(tracks) == 0:
            self.unmatched_slots = MAX_SLOTS
            return {}, {}

        # ── Step 1: locked pairs always pass through ─────────────────────
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
            if (emb is not None
                    and lk.stable_count >= MEMORY_UPDATE_MIN_STABLE
                    and not restricted
                    and (memory_ok_tids is None or tid in memory_ok_tids)):
                slot.update_embedding(emb)
            elif emb is not None:
                memory_skips += 1

            self.assigned_this_frame += 1
            meta_map[tid] = AssignmentMeta(
                pid=pid,
                source="locked",
                identity_state=IdentityState.LOCKED,
                confidence=min(1.0, lk.stable_count / 30.0),
                identity_valid=True,
            )

        if locked_kept and self.frame_id % self.debug_every == 0:
            print(f"[IDLockKeep] frame={self.frame_id} kept={locked_kept}")

        # ── Step 2: gate — restricted or not allowed → all UNKNOWN ───────
        if not allow_new_assignments:
            reason = self._identity_restricted_reason()
            for tr in unlocked_tracks:
                tid = int(tr.track_id)
                meta_map[tid] = AssignmentMeta(
                    pid=None, source="unknown",
                    identity_state=IdentityState.UNKNOWN,
                    confidence=0.0, identity_valid=False,
                )
            self.unmatched_tracks = len(unlocked_tracks)
            self.unmatched_slots = sum(1 for s in self.slots if not s.seen_this_frame)
            self._last_memory_skips = memory_skips
            if unlocked_tracks and self.frame_id % self.debug_every == 0:
                print(f"[IdentityGate] frame={self.frame_id} blocked={len(unlocked_tracks)} "
                      f"reason={reason}")
            return track_to_pid, meta_map

        # ── Step 3: Normal play — Hungarian on unlocked tracks × free slots
        unlocked_slot_idx = [
            j for j, s in enumerate(self.slots)
            if not self.locks.is_pid_locked(s.pid) and not s.seen_this_frame
        ]

        if not unlocked_tracks or not unlocked_slot_idx:
            for tr in unlocked_tracks:
                tid = int(tr.track_id)
                if tid not in meta_map:
                    meta_map[tid] = AssignmentMeta(
                        pid=None, source="unknown",
                        identity_state=IdentityState.UNKNOWN,
                        confidence=0.0, identity_valid=False,
                    )
        else:
            n_t = len(unlocked_tracks)
            n_s = len(unlocked_slot_idx)
            cost = np.full((n_t, n_s), 1e3, dtype=np.float32)
            for i, tr in enumerate(unlocked_tracks):
                tid = int(tr.track_id)
                for jj, j in enumerate(unlocked_slot_idx):
                    cost[i, jj] = self._slot_cost(
                        self.slots[j], embeddings.get(tid), positions.get(tid), tid=tid
                    )

            r_idx, c_idx = linear_sum_assignment(cost)

            for r, c in zip(r_idx, c_idx):
                cst = float(cost[r, c])
                tid = int(unlocked_tracks[r].track_id)
                slot = self.slots[unlocked_slot_idx[c]]

                if cst > COST_REJECT_THRESHOLD:
                    meta_map[tid] = AssignmentMeta(
                        pid=None, source="unknown",
                        identity_state=IdentityState.UNKNOWN,
                        confidence=0.0, identity_valid=False,
                    )
                    continue

                # Normal play: stable-promote streak → lock
                confidence = max(0.0, 1.0 - cst)

                if slot.pending_tid == tid:
                    slot.pending_streak += 1
                else:
                    slot.pending_tid = tid
                    slot.pending_streak = 1

                track_to_pid[tid] = slot.pid
                self._activate_slot(slot, tid, positions)

                if cst <= LOW_CONFIDENCE_THRESHOLD:
                    emb = embeddings.get(tid)
                    if emb is not None:
                        if memory_ok_tids is None or tid in memory_ok_tids:
                            slot.update_embedding(emb)
                        else:
                            memory_skips += 1

                if slot.pending_streak >= LOCK_PROMOTE_FRAMES and cst <= LOW_CONFIDENCE_THRESHOLD:
                    lk_new, status = self.locks.try_create_lock(
                        tid=tid, pid=slot.pid, source="hungarian",
                        frame_id=self.frame_id, confidence=cst,
                        allow_takeover=True, allow_rebind=True,
                    )
                    if status in ("created", "refreshed"):
                        slot.pending_tid = None
                        slot.pending_streak = 0

                self.assigned_this_frame += 1
                is_locked = self.locks.is_tid_locked(tid)
                meta_map[tid] = AssignmentMeta(
                    pid=slot.pid,
                    source="locked" if is_locked else "provisional",
                    identity_state=IdentityState.LOCKED if is_locked else IdentityState.PROVISIONAL,
                    confidence=confidence,
                    identity_valid=is_locked,
                )

            for tr in unlocked_tracks:
                tid = int(tr.track_id)
                if tid not in meta_map:
                    meta_map[tid] = AssignmentMeta(
                        pid=None, source="unknown",
                        identity_state=IdentityState.UNKNOWN,
                        confidence=0.0, identity_valid=False,
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
            s.stability_counter = 0
            if s.pending_tid is not None:
                s.pending_streak = max(0, s.pending_streak - 1)
                if s.pending_streak == 0:
                    s.pending_tid = None

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def snapshot_scene(self, frame_id: int) -> int:
        candidate = {}
        for s in self.slots:
            if s.state in ("active", "dormant") and s.embedding is not None:
                candidate[s.pid] = {
                    "embedding": s.embedding.copy(),
                    "position": s.last_position,
                    "pitch": s.last_pitch,
                    "team_id": s.team_id,
                    "last_seen": s.last_seen_frame,
                }
        if len(candidate) == 0 and len(self._bench_snapshot) > 0:
            print(
                f"[Identity] SceneSnapshot: skip overwrite with 0 slots at frame {frame_id}"
                f" — keeping {len(self._bench_snapshot)} from earlier snapshot"
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
                    "pitch": s.last_pitch,
                    "team_id": s.team_id,
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
        """Scene revival after bench→play. Only unlocked tracks passed in."""
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
            t_pitch = self.pitch_positions.get(tid)
            t_pos = positions.get(tid)
            t_team = self.team_labels.get(tid)
            for j, pid in enumerate(snap_pids):
                s = snap[pid]
                if (t_team is not None and s.get("team_id") is not None
                        and t_team != s["team_id"]):
                    cost[i, j] = 1e3
                    continue
                emb_c, pos_c = 0.5, 0.5
                if t_emb is not None and s["embedding"] is not None:
                    e = t_emb.astype(np.float32)
                    n = np.linalg.norm(e)
                    if n > 0: e /= n
                    cos = float(np.clip(np.dot(e, s["embedding"]), -1.0, 1.0))
                    emb_c = 1.0 - (cos + 1.0) * 0.5
                if t_pitch is not None and s.get("pitch") is not None:
                    pos_c = assignment_position_cost(s["pitch"], t_pitch)
                elif t_pos is not None and s["position"] is not None:
                    dx = float(t_pos[0] - s["position"][0])
                    dy = float(t_pos[1] - s["position"][1])
                    pos_c = min((dx * dx + dy * dy) ** 0.5 / 300.0, 1.0)
                cost[i, j] = 0.75 * emb_c + 0.25 * pos_c

        r_idx, c_idx = linear_sum_assignment(cost)
        accepted = 0
        min_cost = float(cost[r_idx, c_idx].min()) if len(r_idx) else 999.0

        pairs = sorted(zip(r_idx, c_idx), key=lambda rc: cost[rc[0], rc[1]])
        used_tids: Set[int] = set()
        used_pids: Set[str] = set()

        for r, c in pairs:
            cst = float(cost[r, c])
            if cst >= REVIVAL_COST_THRESHOLD:
                continue
            tid = track_ids[r]
            pid = snap_pids[c]
            if tid in used_tids or pid in used_pids:
                continue

            row_costs = cost[r, :]
            valid_row = row_costs[row_costs < 1e2]
            if len(valid_row) >= 2:
                sorted_row = np.sort(valid_row)
                margin = float(sorted_row[1] - sorted_row[0])
                if margin < REVIVAL_MARGIN_MIN:
                    print(f"[SceneReviveReject] tid={tid} pid={pid} cost={cst:.3f} "
                          f"margin={margin:.3f} reason=ambiguous")
                    self.ambiguous_rejects += 1
                    continue

            existing_tid = self.locks.get_tid_for_pid(pid)
            if existing_tid is not None and existing_tid != tid:
                existing_lk = self.locks.get_lock(existing_tid)
                if self._recovery_lock_protected(existing_lk):
                    print(f"[SceneReviveReject] pid={pid} old_tid={existing_tid} "
                          f"new_tid={tid} reason=recovery_lock_protected")
                    self.ambiguous_rejects += 1
                    continue
                if existing_lk and existing_lk.stable_count >= STABLE_PROTECT_THRESHOLD:
                    print(f"[SceneReviveReject] pid={pid} old_tid={existing_tid} "
                          f"new_tid={tid} reason=stable_lock_protected")
                    continue

            revived[tid] = pid
            meta[tid] = AssignmentMeta(
                pid=pid, source="revived",
                identity_state=IdentityState.REVIVED,
                confidence=max(0.0, 1.0 - cst),
                identity_valid=True,
            )
            self._apply_revival(tid, pid, positions, source="scene", cost=cst)
            used_tids.add(tid)
            used_pids.add(pid)
            accepted += 1
            self.revived_count += 1

        if accepted == 0:
            print(f"[SceneReviveFail] frame={self.frame_id} reason=costs_too_high "
                  f"n_tracks={n_t} n_snap={n_s} min_cost={min_cost:.3f}")
        else:
            print(f"[Identity] Revival: {accepted}/{n_t} from scene snapshot")

        # Do NOT clear bench_snapshot here — tracker_core keeps retrying for 90 frames
        return revived, meta

    def revive_from_soft_snapshot(
        self,
        tracks: Sequence[object],
        embeddings: Dict[int, np.ndarray],
        positions: Dict[int, Tuple[float, float]],
        is_first_recovery_frame: bool = False,
    ) -> Tuple[Dict[int, str], Dict[int, AssignmentMeta]]:
        """Soft revival with stable-lock protection, margin check, team gate."""
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
            t_pitch = self.pitch_positions.get(tid)
            t_team = self.team_labels.get(tid)
            for j, pid in enumerate(snap_pids):
                s = self._soft_snapshot[pid]
                if (t_team is not None and s.get("team_id") is not None
                        and t_team != s["team_id"]):
                    cost[i, j] = 1e3
                    continue
                emb_c, pos_c = 0.5, 0.5
                if t_emb is not None and s["embedding"] is not None:
                    e = t_emb.astype(np.float32)
                    n = np.linalg.norm(e)
                    if n > 0: e /= n
                    cos = float(np.clip(np.dot(e, s["embedding"]), -1.0, 1.0))
                    emb_c = 1.0 - (cos + 1.0) * 0.5
                if t_pitch is not None and s.get("pitch") is not None:
                    pos_c = assignment_position_cost(s["pitch"], t_pitch)
                elif t_pos is not None and s["position"] is not None:
                    dx = float(t_pos[0] - s["position"][0])
                    dy = float(t_pos[1] - s["position"][1])
                    pos_c = min((dx * dx + dy * dy) ** 0.5 / 300.0, 1.0)
                cst = 0.85 * emb_c + 0.15 * pos_c
                cost[i, j] = cst
                if is_first_recovery_frame:
                    debug_costs.append(
                        {"tid": tid, "pid": pid, "cost": float(cst), "emb": emb_c, "pos": pos_c}
                    )

        r_idx, c_idx = linear_sum_assignment(cost)
        pairs = sorted(zip(r_idx, c_idx), key=lambda rc: cost[rc[0], rc[1]])
        used_tids: Set[int] = set()
        used_pids: Set[str] = set()

        for r, c in pairs:
            cst = float(cost[r, c])
            if cst >= REVIVAL_COST_THRESHOLD:
                continue
            tid = track_ids[r]
            pid = snap_pids[c]
            if tid in used_tids or pid in used_pids:
                continue

            row_costs = cost[r, :]
            valid_row = row_costs[row_costs < 1e2]
            if len(valid_row) >= 2:
                margin = float(np.sort(valid_row)[1] - np.sort(valid_row)[0])
                if margin < REVIVAL_MARGIN_MIN:
                    print(f"[SoftReviveReject] tid={tid} pid={pid} cost={cst:.3f} "
                          f"margin={margin:.3f} reason=ambiguous")
                    self.ambiguous_rejects += 1
                    continue

            existing_tid = self.locks.get_tid_for_pid(pid)
            if existing_tid is not None and existing_tid != tid:
                existing_lk = self.locks.get_lock(existing_tid)
                if self._recovery_lock_protected(existing_lk):
                    print(f"[SoftReviveReject] pid={pid} old_tid={existing_tid} "
                          f"new_tid={tid} reason=recovery_lock_protected")
                    self.locks.soft_recovery_rebinds_blocked += 1
                    self.ambiguous_rejects += 1
                    continue
                if existing_lk and existing_lk.stable_count >= STABLE_PROTECT_THRESHOLD:
                    print(f"[SoftReviveReject] pid={pid} old_tid={existing_tid} "
                          f"new_tid={tid} reason=stable_lock_protected stable={existing_lk.stable_count}")
                    self.locks.soft_recovery_rebinds_blocked += 1
                    continue

            existing_lk_for_tid = self.locks.get_lock(tid)
            if (existing_lk_for_tid is not None
                    and existing_lk_for_tid.pid != pid
                    and existing_lk_for_tid.stable_count >= STABLE_PROTECT_THRESHOLD):
                print(f"[SoftReviveReject] tid={tid} current_pid={existing_lk_for_tid.pid} "
                      f"attempted_pid={pid} reason=tid_stable_lock_protected")
                self.locks.soft_recovery_rebinds_blocked += 1
                continue

            revived[tid] = pid
            meta[tid] = AssignmentMeta(
                pid=pid, source="revived",
                identity_state=IdentityState.REVIVED,
                confidence=max(0.0, 1.0 - cst),
                identity_valid=True,
            )
            self._apply_revival(tid, pid, positions, source="soft", cost=cst)
            used_tids.add(tid)
            used_pids.add(pid)
            if pid in self._soft_snapshot:
                del self._soft_snapshot[pid]
            self.revived_count += 1

        if is_first_recovery_frame:
            debug_costs.sort(key=lambda x: x["cost"])
            for d in debug_costs[:8]:
                accepted = d["cost"] < REVIVAL_COST_THRESHOLD
                print(f"[SoftReviveCost] tid={d['tid']} pid={d['pid']} "
                      f"cost={d['cost']:.3f} emb={d['emb']:.3f} pos={d['pos']:.3f} "
                      f"accepted={accepted}")
        return revived, meta

    def _apply_revival(
        self,
        tid: int,
        pid: str,
        positions: Dict[int, Tuple[float, float]],
        source: str,
        cost: float,
    ) -> None:
        slot = self._slot_by_pid(pid)
        if slot is None:
            return
        self._activate_slot(slot, tid, positions)
        self.assigned_this_frame += 1

        existing_tid = self.locks.get_tid_for_pid(pid)
        allow_over = True
        if existing_tid is not None and existing_tid != tid:
            existing_lk = self.locks.get_lock(existing_tid)
            if existing_lk and existing_lk.stable_count >= STABLE_PROTECT_THRESHOLD:
                allow_over = False

        lk_result, lk_status = self.locks.try_create_lock(
            tid=tid, pid=pid, source="revived",
            frame_id=self.frame_id, confidence=cost,
            allow_takeover=allow_over,
            allow_rebind=allow_over,
        )
        if lk_result is not None:
            print(f"[RevivalLock] frame={self.frame_id} tid={tid} pid={pid} ttl=90 source={source}")
        else:
            print(f"[RevivalLockBlocked] frame={self.frame_id} tid={tid} pid={pid} "
                  f"source={source} status={lk_status}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _activate_slot(
        self,
        slot: PlayerSlot,
        tid: int,
        positions: Dict[int, Tuple[float, float]],
    ) -> None:
        slot.active_track_id = tid
        slot.seen_this_frame = True
        slot.state = "active"
        slot.last_seen_frame = self.frame_id
        slot.last_position = positions.get(tid, slot.last_position)
        slot.stability_counter += 1
        t_pitch = self.pitch_positions.get(tid)
        if t_pitch is not None:
            slot.last_pitch = t_pitch
        if slot.stability_counter >= 5 and slot.team_id is None:
            t_team = self.team_labels.get(tid)
            if t_team is not None:
                slot.team_id = t_team

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
        n_locks = len(self.locks._tid_to_lock)
        restricted = self._identity_restricted
        print(
            f"[Frame {self.frame_id}] det={detections_count} tracks={tracks_count} "
            f"assigned={self.assigned_this_frame} active={active} dormant={dormant} "
            f"lost={min(lost, MAX_SLOTS)} unmatched_t={self.unmatched_tracks} "
            f"unmatched_s={self.unmatched_slots} skips={self._last_memory_skips} "
            f"locks={n_locks} restricted={restricted}"
        )

    def _slot_cost(
        self,
        slot: PlayerSlot,
        t_emb: Optional[np.ndarray],
        t_pos: Optional[Tuple[float, float]],
        tid: Optional[int] = None,
    ) -> float:
        # Boot: empty lost slots eagerly accept — but NEVER during any restricted mode
        if not self._identity_restricted and slot.state == "lost" and slot.embedding is None:
            return 0.30

        # Team hard gate
        if tid is not None:
            t_team = self.team_labels.get(tid)
            if t_team is not None and slot.team_id is not None and t_team != slot.team_id:
                return float(COST_REJECT_THRESHOLD + 0.05)

        emb_cost = 0.5
        if t_emb is not None and slot.embedding is not None:
            e = t_emb.astype(np.float32)
            n = np.linalg.norm(e)
            if n > 0:
                e /= n
            cos = float(np.clip(np.dot(e, slot.embedding), -1.0, 1.0))
            emb_cost = 1.0 - (cos + 1.0) * 0.5

        t_pitch = self.pitch_positions.get(tid) if tid is not None else None
        pos_cost = assignment_position_cost(slot.last_pitch, t_pitch) if t_pitch is not None \
            else (assignment_position_cost(slot.last_position, t_pos) if t_pos is not None else 0.5)

        recency = min(max(self.frame_id - slot.last_seen_frame, 0), DORMANT_TTL)
        recency_cost = recency / float(DORMANT_TTL)
        lock_discount = 0.15 if slot.stability_counter >= 10 else 0.0

        mode = self.reid_mode
        if mode == "OSNet":
            if recency == 0:
                final_cost = (0.55 * emb_cost + 0.20 * pos_cost
                              + 0.05 * recency_cost - 0.10 - lock_discount)
            else:
                final_cost = (0.55 * emb_cost + 0.20 * pos_cost
                              + 0.10 * recency_cost - lock_discount)
        elif mode == "ResNet50":
            if recency == 0:
                final_cost = (0.45 * emb_cost + 0.25 * pos_cost
                              + 0.05 * recency_cost - 0.10 - lock_discount)
            else:
                final_cost = (0.45 * emb_cost + 0.25 * pos_cost
                              + 0.15 * recency_cost - lock_discount)
        else:
            if recency == 0:
                final_cost = (0.30 * emb_cost + 0.35 * pos_cost
                              + 0.05 * recency_cost - 0.10 - lock_discount)
            else:
                final_cost = (0.30 * emb_cost + 0.35 * pos_cost
                              + 0.20 * recency_cost - lock_discount)

        return float(max(0.0, min(1.0, final_cost)))

    def end_run_summary(self) -> Dict[str, object]:
        lock_summary = self.locks.summary()
        stable_locked = sum(
            1 for lk in self.locks._tid_to_lock.values()
            if lk.stable_count >= LOCK_PROMOTE_FRAMES
        )
        collapse_lock_creations = lock_summary.get("collapse_lock_creations", 0)
        collapse_lock_attempts = lock_summary.get("collapse_lock_attempts", 0)
        restricted_lock_attempts = lock_summary.get("restricted_lock_attempts", 0)
        locks_created = lock_summary.get("locks_created", 1)
        locks_expired = lock_summary.get("locks_expired", 0)
        retention = round((locks_created - locks_expired) / max(locks_created, 1), 3)

        ok_collapse = collapse_lock_creations == 0
        ok_recovery_normal = self.recovery_normal_assignments == 0
        ok_scene_normal = self.scene_recovery_normal_assignments == 0
        ok_restricted_hungarian = self.restricted_hungarian_assignments == 0
        ok_retention = retention >= 0.65
        ok_locks = locks_created <= 40

        print("\n[IdentityMetrics]")
        print(f"  collapse_lock_attempts            = {collapse_lock_attempts}  (blocked, not written)")
        print(f"  restricted_lock_attempts          = {restricted_lock_attempts}  (all restricted blocks)")
        print(f"  collapse_lock_creations           = {collapse_lock_creations}  {'OK' if ok_collapse else 'FAIL must=0'}")
        print(f"  recovery_normal_assignments       = {self.recovery_normal_assignments}  {'OK' if ok_recovery_normal else 'FAIL must=0'}")
        print(f"  scene_recovery_normal_assignments = {self.scene_recovery_normal_assignments}  {'OK' if ok_scene_normal else 'FAIL must=0'}")
        print(f"  restricted_hungarian_assignments  = {self.restricted_hungarian_assignments}  {'OK' if ok_restricted_hungarian else 'FAIL must=0'}")
        print(f"  locks_created                     = {locks_created}  {'OK' if ok_locks else 'WARN target<=40'}")
        print(f"  lock_retention_rate               = {retention}  {'OK' if ok_retention else 'WARN target>=0.65'}")
        print(f"  locks_dormanted                   = {lock_summary.get('locks_dormanted', 0)}")
        print(f"  ambiguous_rejects                 = {self.ambiguous_rejects}")
        print(f"  revived_count                     = {self.revived_count}")
        print(f"  stable_locked_count               = {stable_locked}")

        violations = []
        if not ok_collapse:
            violations.append(f"collapse_lock_creations={collapse_lock_creations} (must be 0)")
        if not ok_recovery_normal:
            violations.append(f"recovery_normal_assignments={self.recovery_normal_assignments} (must be 0)")
        if not ok_scene_normal:
            violations.append(f"scene_recovery_normal_assignments={self.scene_recovery_normal_assignments} (must be 0)")
        if not ok_restricted_hungarian:
            violations.append(f"restricted_hungarian_assignments={self.restricted_hungarian_assignments} (must be 0)")
        if violations:
            msg = "[IdentityInvariantFAIL] " + "; ".join(violations)
            print(msg)
            raise RuntimeError(msg)

        return {
            **lock_summary,
            "recovery_normal_assignments": self.recovery_normal_assignments,
            "scene_recovery_normal_assignments": self.scene_recovery_normal_assignments,
            "restricted_hungarian_assignments": self.restricted_hungarian_assignments,
            "ambiguous_rejects": self.ambiguous_rejects,
            "revived_count": self.revived_count,
            "stable_locked_count": stable_locked,
        }
