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

import cv2
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
SOFT_REVIVE_COST_MAX = 0.38           # was 0.30 — loosened with real OSNet embeddings
SOFT_REVIVE_AUTO_ACCEPT_COST = 0.15
SCENE_REVIVE_WINDOW = 90
SCENE_REVIVE_COST_MAX = 0.38
SCENE_REVIVE_FORCE_COST_MAX = 0.42
STRICT_COLOR_THRESHOLD = 0.45       # Threshold for hard color reject
SPATIAL_GATE_RADIUS = 150.0         # 150px safety valve
SOFT_REVIVE_MARGIN_MIN = 0.025      # Tightened from 0.003
SCENE_REVIVE_MARGIN_MIN = 0.025     # Tightened from 0.003
RECOVERY_PATCH_ID = "reid-v5-first-order-motion-gated"
LOW_CONFIDENCE_THRESHOLD = 0.55
EMB_ALPHA = 0.25
STABLE_PROTECT_THRESHOLD = 10
LOCK_PROMOTE_FRAMES = 5


def _unwrap_emb(emb) -> Optional[np.ndarray]:
    """Return the raw embedding ndarray from either a plain array or a dual-embedding dict."""
    if emb is None:
        return None
    if isinstance(emb, dict):
        return emb.get("emb")
    return emb


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
    pending_seen_seq: int = 0  # identity_frame_seq when last matched
    last_assigned_tid: Optional[int] = None  # for continuity bias in cost matrix
    last_assigned_seq: int = 0
    team_id: Optional[int] = None
    velocity_px: Optional[Tuple[float, float]] = None
    velocity_pitch: Optional[Tuple[float, float]] = None
    hsv_signature: Optional[np.ndarray] = None

    def update_embedding(self, emb) -> None:
        if isinstance(emb, dict):
            # Support dual-embedding: {"emb": ..., "hsv": ...}
            hsv = emb.get("hsv")
            if hsv is not None:
                self.update_color(hsv)
            emb = emb.get("emb")
            if emb is None: return

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

    def update_color(self, hsv: np.ndarray) -> None:
        hsv = hsv.astype(np.float32)
        norm = np.linalg.norm(hsv)
        if norm > 0: hsv /= norm
        if self.hsv_signature is None:
            self.hsv_signature = hsv
        else:
            # Slower EMA for color signature
            self.hsv_signature = 0.85 * self.hsv_signature + 0.15 * hsv
            norm2 = np.linalg.norm(self.hsv_signature)
            if norm2 > 0: self.hsv_signature /= norm2

    def predict_position(self, current_frame: int) -> Optional[Tuple[float, float]]:
        if self.last_position is None: return None
        if self.velocity_px is None: return self.last_position
        dt = current_frame - self.last_seen_frame
        if dt <= 0: return self.last_position
        # Linear prediction
        return (
            self.last_position[0] + self.velocity_px[0] * dt,
            self.last_position[1] + self.velocity_px[1] * dt
        )

    def predict_pitch(self, current_frame: int) -> Optional[Tuple[float, float]]:
        if self.last_pitch is None: return None
        if self.velocity_pitch is None: return self.last_pitch
        dt = current_frame - self.last_seen_frame
        if dt <= 0: return self.last_pitch
        return (
            self.last_pitch[0] + self.velocity_pitch[0] * dt,
            self.last_pitch[1] + self.velocity_pitch[1] * dt
        )


class IdentityCore:
    def __init__(self, logger=None, debug_every: int = 30):
        self.logger = logger
        self.debug_every = debug_every
        self.slots: List[PlayerSlot] = [
            PlayerSlot(pid=f"P{i}") for i in range(1, MAX_SLOTS + 1)
        ]
        print("[ReID] using HSV fallback only for appearance matching")

        self.frame_id: int = -1
        self.identity_frame_seq: int = 0  # increments on each assign_tracks call
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

        # Camera motion state — set via assign_tracks(camera_motion=...)
        self.camera_motion: Dict = {}
        self.camera_motion_class: str = "stable"

        # Per-frame extras injected by tracker_core
        self.pitch_positions: Dict[int, Tuple[float, float]] = {}
        self.team_labels: Dict[int, Optional[int]] = {}
        self.reid_mode: str = "HSV-fallback"
        self._present_tids: Set[int] = set()

        self.locks = IdentityLockManager(logger=logger)
        print(
            f"[ReIDPatch] id={RECOVERY_PATCH_ID} file={__file__} "
            f"soft_auto={SOFT_REVIVE_AUTO_ACCEPT_COST:.2f} "
            f"soft_margin={SOFT_REVIVE_MARGIN_MIN:.3f} "
            f"scene_window={SCENE_REVIVE_WINDOW} force_commit={SCENE_REVIVE_FORCE_COST_MAX:.2f}"
        )

        # Run-level metrics
        self.recovery_normal_assignments: int = 0
        self.scene_recovery_normal_assignments: int = 0
        self.restricted_hungarian_assignments: int = 0
        self.ambiguous_rejects: int = 0
        self.revived_count: int = 0

        # Pan-safe gate metrics
        self.pan_lock_attempts_blocked: int = 0
        self.pan_rebinds_blocked: int = 0
        self.pan_takeovers_blocked: int = 0
        self.pan_ttl_extensions: int = 0
        self.camera_motion_recovery_frames: int = 0
        self.fast_pan_frames: int = 0
        self.cut_frames: int = 0

    # ------------------------------------------------------------------
    # Single source of truth for restricted identity mode
    # ------------------------------------------------------------------

    def _recovery_lock_protected(self, lock) -> bool:
        """
        True when restricted AND the existing lock was FRESHLY revived (stable<30) —
        protects against takeover bouncing between candidates while a revival settles.
        Once stable_count >= 30 (~1s @ 30fps), the lock has earned its identity and a
        new revival decision (with cost+margin already vetted) is allowed to relink it.
        """
        if not self._identity_restricted or lock is None or lock.source != "revived":
            return False
        return lock.stable_count < 30

    def _relink_absent_existing_lock(self, pid: str, old_tid: Optional[int], new_tid: int,
                                     source: str, cost: float) -> bool:
        """Relink a reserved PID when its previous tracker id is absent from this frame."""
        if old_tid is None or old_tid == new_tid or not self._identity_restricted:
            return False
        existing_lk = self.locks.get_lock(old_tid)
        if existing_lk is None or old_tid in self._present_tids:
            return False
        lk, status = self.locks.relink_absent_lock(
            old_tid=old_tid,
            new_tid=new_tid,
            pid=pid,
            frame_id=self.frame_id,
            source="revived",
            confidence=cost,
        )
        if lk is None:
            print(
                f"[{source}ReviveRelinkBlocked] frame={self.frame_id} pid={pid} "
                f"old_tid={old_tid} new_tid={new_tid} status={status}"
            )
            return False
        print(
            f"[{source}ReviveRelink] frame={self.frame_id} pid={pid} "
            f"old_tid={old_tid} new_tid={new_tid} cost={cost:.3f}"
        )
        return True

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

    def _camera_motion_restricted(self) -> bool:
        """True when camera motion requires identity safety (no new locks/rebinds/takeovers)."""
        return self.camera_motion_class in ("fast_pan", "cut")

    def _camera_motion_reason(self) -> Optional[str]:
        """Reason why camera motion is restricting identity decisions."""
        if self.camera_motion_class == "fast_pan":
            return "CAMERA_FAST_PAN"
        if self.camera_motion_class == "cut":
            return "CAMERA_CUT_DETECTED"
        return None

    def get_scene_revive_thresholds(self, max_window: int = SCENE_REVIVE_WINDOW) -> Tuple[float, float]:
        """Progressively relax scene revival as the post-reset window approaches expiry."""
        if self._scene_reset_frame < 0:
            progress = 0.0
        else:
            progress = max(0.0, min(1.0, (self.frame_id - self._scene_reset_frame) / float(max_window)))

        if progress < 0.33:
            return 0.22, 0.010
        if progress < 0.66:
            return 0.28, 0.005
        return 0.32, 0.003

    @staticmethod
    def _revive_margin_ok(cost: float, margin: Optional[float], cost_max: float,
                          margin_min: float, auto_accept_cost: float = SOFT_REVIVE_AUTO_ACCEPT_COST) -> Tuple[bool, str]:
        if cost > cost_max:
            return False, "cost_too_high"
        if cost < auto_accept_cost:
            return True, "excellent"
        if margin is not None and margin < margin_min:
            return False, "ambiguous"
        return True, "margin_ok"

    # ------------------------------------------------------------------
    # Scene reset
    # ------------------------------------------------------------------

    def reset_for_scene(self, frame_id: int = -1) -> None:
        if frame_id >= 0:
            self.frame_id = frame_id
        reset_count = sum(1 for s in self.slots if s.pending_tid is not None)
        for s in self.slots:
            s.active_track_id = None
            s.seen_this_frame = False
            s.pending_tid = None
            s.pending_streak = 0
            s.pending_seen_seq = 0
        if reset_count > 0:
            print(f"[PendingReset] fn=reset_for_scene frame={self.frame_id} reset={reset_count} slots")
            if s.state in ("active", "dormant") and s.embedding is not None:
                s.state = "dormant"
            else:
                s.state = "lost"

        snap_count = len(self._bench_snapshot)
        if snap_count > 0:
            print(f"[Identity] Reset: {snap_count} snapshot slots survived reset")
        else:
            print("[Identity] Reset: No snapshot exists to survive reset.")

        preserved = len(self.locks.locked_tids())
        if preserved:
            print(f"[IDLockFreeze] frame={frame_id} preserving {preserved} locks for scene recovery")
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
            self._present_tids = set(present_tids)
            self.locks.tick(frame_id, present_tids, restricted=restricted)
        else:
            self._present_tids = set()

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
        camera_motion: Optional[Dict] = None,
    ) -> Tuple[Dict[int, str], Dict[int, AssignmentMeta]]:
        """
        allow_new_assignments=False OR _identity_restricted=True:
          → only locked pairs emitted; all others UNKNOWN.
        Caller always passes allow_new_assignments=not _identity_restricted().
        The internal check here is a second safety net.
        """
        # Increment identity frame sequence for streak continuity across frame_stride
        self.identity_frame_seq += 1

        # Update camera motion state for this frame
        if camera_motion is not None:
            self.camera_motion = camera_motion
            self.camera_motion_class = camera_motion.get("motion_class", "stable")
        else:
            self.camera_motion = {}
            self.camera_motion_class = "stable"

        # Track pan recovery frames and extend dormant lock TTL for protection
        if self.camera_motion_class in ("fast_pan", "cut"):
            self.camera_motion_recovery_frames += 1
            if self.camera_motion_class == "fast_pan":
                self.fast_pan_frames += 1
                # Extend TTL for dormant locks to protect them during pan
                ext_count = self.locks.extend_dormant_ttl_for_pan(self.frame_id)
                self.pan_ttl_extensions += ext_count
            elif self.camera_motion_class == "cut":
                self.cut_frames += 1
                # Also extend for cuts to preserve identity continuity across scene transitions
                ext_count = self.locks.extend_dormant_ttl_for_pan(self.frame_id)
                self.pan_ttl_extensions += ext_count

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

        if allow_new_assignments and len(unlocked_tracks) > 0:
            emb_present = sum(1 for t in unlocked_tracks if embeddings.get(int(t.track_id)) is not None)
            print(f"[LockDiag] frame={self.frame_id} unlocked={len(unlocked_tracks)} emb_present={emb_present}")

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
                    base = self._slot_cost(
                        self.slots[j], embeddings.get(tid), positions.get(tid), tid=tid,
                        camera_motion=camera_motion,
                    )
                    # Apply continuity bias: strongly prefer slots that matched this tid recently
                    slot = self.slots[j]
                    if slot.pending_tid is not None and int(slot.pending_tid) == int(tid):
                        base = max(0.0, base - 0.50)  # Very strong bias to maintain tid-pid continuity
                    elif (hasattr(slot, 'last_assigned_tid') and slot.last_assigned_tid is not None
                          and int(slot.last_assigned_tid) == int(tid)):
                        base = max(0.0, base - 0.25)  # Strong bias for recent history
                    cost[i, jj] = base

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

                # Track pending_streak using identity frame sequence, not raw frames
                # This survives frame_stride differences and late seeding
                # Check if this tid matches the slot's last assignment (from previous processed frame)
                same_tid = (hasattr(slot, 'last_assigned_tid') and slot.last_assigned_tid is not None
                           and int(slot.last_assigned_tid) == int(tid))
                seq_ok = slot.pending_seen_seq == self.identity_frame_seq - 1
                pending_streak_before = slot.pending_streak

                if same_tid and seq_ok:
                    slot.pending_streak += 1
                    branch = "increment"
                    reset_reason = None
                else:
                    branch = "reset"
                    reset_reason = []
                    if not same_tid:
                        reset_reason.append("tid_changed")
                    if not seq_ok:
                        reset_reason.append("seq_gap")
                    slot.pending_tid = int(tid)
                    slot.pending_streak = 1

                if self.frame_id % self.debug_every == 0:
                    print(f"[StreakBranch] frame={self.frame_id} tid={tid} pid={slot.pid} "
                          f"slot_pending_tid={slot.pending_tid} same_tid={same_tid} "
                          f"pending_seq={slot.pending_seen_seq} expected_prev_seq={self.identity_frame_seq-1} seq_ok={seq_ok} "
                          f"streak_before={pending_streak_before} branch={branch} reason={reset_reason}")

                slot.pending_seen_seq = self.identity_frame_seq

                track_to_pid[tid] = slot.pid
                old_pid = getattr(slot, '_last_assigned_pid', None)
                self._activate_slot(slot, tid, positions)
                slot._last_assigned_pid = slot.pid
                if self.frame_id % self.debug_every == 0 and tid == 1:
                    changed = "yes" if old_pid is not None and old_pid != slot.pid else "no"
                    print(f"[TidPidContinuity] frame={self.frame_id} tid={tid} pid={slot.pid} old_pid={old_pid} changed={changed}")

                if cst <= LOW_CONFIDENCE_THRESHOLD:
                    emb = embeddings.get(tid)
                    if emb is not None:
                        if memory_ok_tids is None or tid in memory_ok_tids:
                            slot.update_embedding(emb)
                        else:
                            memory_skips += 1

                if self.frame_id % self.debug_every == 0:
                    print(f"[StreakDiag] frame={self.frame_id} tid={tid} pid={slot.pid} cost={cst:.3f} streak={slot.pending_streak}/{LOCK_PROMOTE_FRAMES} seq={slot.pending_seen_seq}=={self.identity_frame_seq-1}? threshold={LOW_CONFIDENCE_THRESHOLD:.2f}")

                # Try lock promotion if streak is ready and cost is good
                # Congestion guard: in dense scenes (>15 active tracks), require
                # stronger evidence before promoting to a lock
                n_active_locks = len(self.locks.locked_tids())
                is_congested = n_active_locks >= 15 or len(tracks) >= 18
                
                effective_lock_frames = LOCK_PROMOTE_FRAMES + (3 if is_congested else 0)
                effective_cost_threshold = LOW_CONFIDENCE_THRESHOLD - (0.10 if is_congested else 0)
                
                lock_ready = slot.pending_streak >= effective_lock_frames
                cost_ok = cst <= effective_cost_threshold
                if self.frame_id % self.debug_every == 0 and (lock_ready or cost_ok):
                    print(f"[LockAttempt] frame={self.frame_id} tid={tid} pid={slot.pid} cost={cst:.3f} "
                          f"streak={slot.pending_streak}/{effective_lock_frames} ready={lock_ready} "
                          f"cost_ok={cost_ok} congested={is_congested} active_locks={n_active_locks}")

                if lock_ready and cost_ok:
                    # Pan-safe gate: restrict new locks/rebinds/takeovers during fast_pan/cut
                    pan_restricted = self._camera_motion_restricted()
                    reason = self._camera_motion_reason() if pan_restricted else None

                    # Block all lock attempts (new, rebind, takeover) during pan motion
                    if pan_restricted:
                        self.pan_lock_attempts_blocked += 1
                        if self.frame_id % self.debug_every == 0:
                            print(f"[PanGate] frame={self.frame_id} tid={tid} pid={slot.pid} "
                                  f"action=block_lock_creation reason={reason}")
                    else:
                        lk_new, status = self.locks.try_create_lock(
                            tid=tid, pid=slot.pid, source="hungarian",
                            frame_id=self.frame_id, confidence=cst,
                            allow_takeover=True,
                            allow_rebind=True,
                        )

                        if self.frame_id % self.debug_every == 0:
                            print(f"[LockAttemptResult] frame={self.frame_id} tid={tid} pid={slot.pid} status={status}")

                        if status in ("created", "refreshed"):
                            print(f"[LockCreate] frame={self.frame_id} tid={tid} pid={slot.pid} cost={cst:.3f} status={status}")
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
        decayed = 0
        for s in self.slots:
            if s.seen_this_frame:
                continue
            age = self.frame_id - s.last_seen_frame
            s.state = "dormant" if age <= DORMANT_TTL else "lost"
            s.active_track_id = None
            s.stability_counter = 0
            if s.pending_tid is not None:
                s.pending_streak = max(0, s.pending_streak - 1)
                decayed += 1
                if s.pending_streak == 0:
                    s.pending_tid = None
        if decayed > 0 and self.frame_id % self.debug_every == 0:
            print(f"[PendingDecay] fn=end_frame frame={self.frame_id} decayed={decayed} slots")

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def snapshot_scene(self, frame_id: int, merge_existing: bool = False) -> int:
        candidate = {}
        for s in self.slots:
            if s.embedding is None:
                continue
            if merge_existing or s.state in ("active", "dormant"):
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
        retained = 0
        if merge_existing and self._bench_snapshot:
            merged = dict(self._bench_snapshot)
            retained = len([pid for pid in merged if pid not in candidate])
            merged.update(candidate)
            candidate = merged
        self._bench_snapshot = candidate
        saved = len(self._bench_snapshot)
        if merge_existing:
            fresh = saved - retained
            print(
                f"[Identity] SceneSnapshot: {saved} slots saved at frame {frame_id} "
                f"(fresh={fresh} retained={retained})"
            )
        else:
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

    def seed_provisional_from_tracks(
        self,
        embed_map: Dict[int, np.ndarray],
        positions: Dict[int, tuple],
        pitch_positions: Dict[int, tuple],
        team_labels: Dict[int, int],
        frame_id: int,
    ) -> int:
        """Bootstrap: assign raw tracks to empty slots so snapshot_soft() can capture them."""
        empty_slots = [s for s in self.slots if s.state == "lost" and s.embedding is None]
        tids = list(embed_map.keys())
        seeded = 0
        for i, tid in enumerate(tids[:len(empty_slots)]):
            slot = empty_slots[i]
            emb = _unwrap_emb(embed_map.get(tid))
            if emb is not None:
                slot.embedding = emb.copy()
                slot.last_position = positions.get(tid, (0, 0))
                slot.last_pitch = pitch_positions.get(tid, (0, 0))
                slot.team_id = team_labels.get(tid)
                slot.last_seen_frame = frame_id
                # DON'T set pending_tid during seed.
                # Let Hungarian create the first match naturally.
                # The first match will set pending_tid=matched_tid and pending_streak=1.
                # Then subsequent frames can check continuity.
                # For now, just mark that this slot was "seeded" so pending continuity starts fresh
                slot.pending_tid = None
                slot.pending_streak = 0
                slot.pending_seen_seq = self.identity_frame_seq
                slot.active_track_id = None  # Don't set active_track_id; let assign_tracks do that
                # Do NOT set seen_this_frame=True; seed is before assign_tracks processes
                seeded += 1
        if seeded > 0:
            print(f"[SlotSeed] frame={frame_id} seeded {seeded} empty slots from {len(tids)} tracks cur_seq={self.identity_frame_seq}")
        return seeded

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

        cost, track_ids, snap_pids = self._snapshot_cost_matrix(
            snap, tracks, embeddings, positions,
            emb_weight=0.75, pos_weight=0.25,
        )
        n_t, n_s = len(track_ids), len(snap_pids)
        cost_max, margin_min = self.get_scene_revive_thresholds()

        r_idx, c_idx = linear_sum_assignment(cost)
        accepted = 0
        min_cost = float(cost[r_idx, c_idx].min()) if len(r_idx) else 999.0
        reject_reasons: Dict[str, int] = {}

        def note_reject(reason: str) -> None:
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1

        pairs = sorted(zip(r_idx, c_idx), key=lambda rc: cost[rc[0], rc[1]])
        used_tids: Set[int] = set()
        used_pids: Set[str] = set()

        for r, c in pairs:
            cst = float(cost[r, c])
            if cst >= REVIVAL_COST_THRESHOLD:
                note_reject("costs_too_high")
                continue
            tid = track_ids[r]
            pid = snap_pids[c]
            if tid in used_tids or pid in used_pids:
                continue

            row_costs = cost[r, :]
            valid_row = row_costs[row_costs < 1e2]
            margin = None
            if len(valid_row) >= 2:
                sorted_row = np.sort(valid_row)
                margin = float(sorted_row[1] - sorted_row[0])
            ok, reason = self._revive_margin_ok(cst, margin, cost_max, margin_min)
            if not ok:
                note_reject(reason)
                if reason == "ambiguous":
                    self.ambiguous_rejects += 1
                margin_s = f"{margin:.3f}" if margin is not None else "n/a"
                print(f"[SceneReviveReject] tid={tid} pid={pid} cost={cst:.3f} "
                      f"margin={margin_s} reason={reason} cost_max={cost_max:.3f}")
                continue

            existing_tid = self.locks.get_tid_for_pid(pid)
            if existing_tid is not None and existing_tid != tid:
                existing_lk = self.locks.get_lock(existing_tid)
                relinked = self._relink_absent_existing_lock(pid, existing_tid, tid, "Scene", cst)
                if relinked:
                    existing_lk = self.locks.get_lock(tid)
                elif self._recovery_lock_protected(existing_lk):
                    print(f"[SceneReviveReject] pid={pid} old_tid={existing_tid} "
                          f"new_tid={tid} reason=recovery_lock_protected")
                    note_reject("recovery_lock_protected")
                    self.ambiguous_rejects += 1
                    continue
                if (not relinked and existing_lk
                        and existing_lk.stable_count >= STABLE_PROTECT_THRESHOLD):
                    print(f"[SceneReviveReject] pid={pid} old_tid={existing_tid} "
                          f"new_tid={tid} reason=stable_lock_protected")
                    note_reject("stable_lock_protected")
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
            fail_reason = "costs_too_high"
            if reject_reasons:
                fail_reason = max(reject_reasons.items(), key=lambda kv: kv[1])[0]
            print(f"[SceneReviveFail] frame={self.frame_id} reason={fail_reason} "
                  f"n_tracks={n_t} n_snap={n_s} min_cost={min_cost:.3f}")
        else:
            print(f"[Identity] Revival: {accepted}/{n_t} from scene snapshot")

        # Do NOT clear bench_snapshot here — tracker_core keeps retrying for 90 frames
        return revived, meta

    def force_commit_remaining_scene_slots(
        self,
        tracks: Sequence[object],
        embeddings: Dict[int, np.ndarray],
        positions: Dict[int, Tuple[float, float]],
    ) -> Tuple[Dict[int, str], Dict[int, AssignmentMeta]]:
        """Last-resort scene recovery before the 90-frame window exits."""
        revived: Dict[int, str] = {}
        meta: Dict[int, AssignmentMeta] = {}
        remaining_snap = {
            pid: snap
            for pid, snap in self._bench_snapshot.items()
            if (
                not self.locks.is_pid_locked(pid)
                or self.locks.get_tid_for_pid(pid) not in self._present_tids
            )
            and (self._slot_by_pid(pid) is None or not self._slot_by_pid(pid).seen_this_frame)
        }
        if not remaining_snap or len(tracks) == 0:
            print(f"[ForceCommit] frame={self.frame_id} skipped remaining={len(remaining_snap)} tracks={len(tracks)}")
            return revived, meta

        cost, track_ids, snap_pids = self._snapshot_cost_matrix(
            remaining_snap, tracks, embeddings, positions,
            emb_weight=0.75, pos_weight=0.25,
        )
        r_idx, c_idx = linear_sum_assignment(cost)
        used_tids: Set[int] = set()
        used_pids: Set[str] = set()

        force_rejected = 0
        for r, c in sorted(zip(r_idx, c_idx), key=lambda rc: cost[rc[0], rc[1]]):
            cst = float(cost[r, c])
            # Hard cap — anything above is unreliable, leave UNKNOWN instead
            if cst > SCENE_REVIVE_FORCE_COST_MAX:
                force_rejected += 1
                print(f"[ForceCommitReject] frame={self.frame_id} cost={cst:.3f} > "
                      f"max={SCENE_REVIVE_FORCE_COST_MAX:.3f} (kept UNKNOWN)")
                continue
            tid = track_ids[r]
            pid = snap_pids[c]
            if tid in used_tids or pid in used_pids:
                continue
            # Margin check: refuse forced commits when 2nd-best is too close
            row_costs = cost[r, :]
            valid_row = row_costs[row_costs < 1e2]
            if len(valid_row) >= 2:
                sorted_row = np.sort(valid_row)
                margin = float(sorted_row[1] - sorted_row[0])
                if margin < 0.020:
                    force_rejected += 1
                    print(f"[ForceCommitReject] frame={self.frame_id} pid={pid} tid={tid} "
                          f"cost={cst:.3f} margin={margin:.3f} reason=ambiguous")
                    continue
            existing_lk = self.locks.get_lock(tid)
            if existing_lk is not None and existing_lk.pid != pid:
                continue

            revived[tid] = pid
            meta[tid] = AssignmentMeta(
                pid=pid, source="revived",
                identity_state=IdentityState.REVIVED,
                confidence=max(0.0, 1.0 - cst),
                identity_valid=True,
            )
            self._apply_revival(tid, pid, positions, source="force_scene", cost=cst)
            used_tids.add(tid)
            used_pids.add(pid)
            self.revived_count += 1
            print(f"[ForceCommit] frame={self.frame_id} pid={pid} tid={tid} cost={cst:.3f}")
        if force_rejected and self.frame_id % 5 == 0:
            print(f"[ForceCommitSummary] frame={self.frame_id} rejected={force_rejected}")

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

        # Use the centralized First-Order cost matrix logic (Prediction + Gating)
        cost, track_ids, snap_pids = self._snapshot_cost_matrix(
            self._soft_snapshot, tracks, embeddings, positions,
            emb_weight=0.85, pos_weight=0.15,
        )
        n_t, n_s = len(track_ids), len(snap_pids)
        debug_costs = []

        r_idx, c_idx = linear_sum_assignment(cost)
        pairs = sorted(zip(r_idx, c_idx), key=lambda rc: cost[rc[0], rc[1]])
        used_tids: Set[int] = set()
        used_pids: Set[str] = set()

        for r, c in pairs:
            cst = float(cost[r, c])
            if is_first_recovery_frame:
                debug_costs.append({"tid": track_ids[r], "pid": snap_pids[c], "cost": cst})

            if cst >= REVIVAL_COST_THRESHOLD:
                continue
            tid = track_ids[r]
            pid = snap_pids[c]
            if tid in used_tids or pid in used_pids:
                continue

            row_costs = cost[r, :]
            valid_row = row_costs[row_costs < 1e2]
            margin = None
            if len(valid_row) >= 2:
                margin = float(np.sort(valid_row)[1] - np.sort(valid_row)[0])

            # Stricter thresholds during fast pan/cut to prevent wrong revivals
            cost_max = SOFT_REVIVE_COST_MAX
            margin_min = SOFT_REVIVE_MARGIN_MIN
            if self._camera_motion_restricted():
                cost_max = cost_max * 0.7  # Stricter maximum cost
                margin_min = margin_min * 1.5  # Need stronger confidence margin
            
            ok, reason = self._revive_margin_ok(
                cst, margin, cost_max, margin_min
            )

            if not ok:
                if reason == "ambiguous":
                    self.ambiguous_rejects += 1
                margin_s = f"{margin:.3f}" if margin is not None else "n/a"
                print(f"[SoftReviveReject] tid={tid} pid={pid} cost={cst:.3f} "
                      f"margin={margin_s} reason={reason}")
                continue

            existing_tid = self.locks.get_tid_for_pid(pid)
            if existing_tid is not None and existing_tid != tid:
                existing_lk = self.locks.get_lock(existing_tid)
                relinked = self._relink_absent_existing_lock(pid, existing_tid, tid, "Soft", cst)
                if relinked:
                    existing_lk = self.locks.get_lock(tid)
                elif self._recovery_lock_protected(existing_lk):
                    print(f"[SoftReviveReject] pid={pid} old_tid={existing_tid} "
                          f"new_tid={tid} reason=recovery_lock_protected")
                    self.locks.soft_recovery_rebinds_blocked += 1
                    self.ambiguous_rejects += 1
                    continue
                if (not relinked and existing_lk
                        and existing_lk.stable_count >= STABLE_PROTECT_THRESHOLD):
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
                cost_ok = d["cost"] <= SOFT_REVIVE_COST_MAX
                print(f"[SoftReviveCost] tid={d['tid']} pid={d['pid']} "
                      f"cost={d['cost']:.3f} "
                      f"cost_ok={cost_ok}")
        return revived, meta

    def _snapshot_cost_matrix(
        self,
        snap: Dict[str, dict],
        tracks: Sequence[object],
        embeddings: Dict[int, np.ndarray],
        positions: Dict[int, Tuple[float, float]],
        emb_weight: float,
        pos_weight: float,
    ) -> Tuple[np.ndarray, List[int], List[str]]:
        track_ids = [int(t.track_id) for t in tracks]
        snap_pids = list(snap.keys())
        cost = np.full((len(track_ids), len(snap_pids)), 1e3, dtype=np.float32)

        for i, tid in enumerate(track_ids):
            t_data = embeddings.get(tid)
            t_emb = t_data.get("emb") if isinstance(t_data, dict) else t_data
            t_hsv = t_data.get("hsv") if isinstance(t_data, dict) else None

            t_pitch = self.pitch_positions.get(tid)
            t_pos = positions.get(tid)
            t_team = self.team_labels.get(tid)

            for j, pid in enumerate(snap_pids):
                s = snap[pid]
                slot = self._slot_by_pid(pid)

                # 1. Team Gate
                if (t_team is not None and s.get("team_id") is not None
                        and t_team != s["team_id"]):
                    continue

                # 2. Hard Color Gate (GK Protection)
                if t_hsv is not None and slot is not None and slot.hsv_signature is not None:
                    hsv_sim = float(np.clip(np.dot(t_hsv, slot.hsv_signature), 0.0, 1.0))
                    hsv_dist = 1.0 - hsv_sim
                    if hsv_dist > STRICT_COLOR_THRESHOLD:
                        continue

                emb_c, pos_c = 0.5, 0.5

                # 3. Visual Embedding Cost (OSNet)
                if t_emb is not None and s.get("embedding") is not None:
                    e = t_emb.astype(np.float32)
                    n = np.linalg.norm(e)
                    if n > 0: e /= n
                    cos = float(np.clip(np.dot(e, s["embedding"]), -1.0, 1.0))
                    emb_c = 1.0 - (cos + 1.0) * 0.5

                # 4. First-Order Spatial Cost (Prediction)
                if slot is not None:
                    pred_pos = slot.predict_position(self.frame_id)
                    pred_pitch = slot.predict_pitch(self.frame_id)
                else:
                    pred_pos = s.get("position")
                    pred_pitch = s.get("pitch")

                if t_pitch is not None and pred_pitch is not None:
                    pos_c = assignment_position_cost(pred_pitch, t_pitch)
                elif t_pos is not None and pred_pos is not None:
                    dx = float(t_pos[0] - pred_pos[0])
                    dy = float(t_pos[1] - pred_pos[1])
                    dist = (dx * dx + dy * dy) ** 0.5

                    # 5. Spatial Gating (Safety Valve)
                    if dist > SPATIAL_GATE_RADIUS:
                        continue

                    pos_c = min(dist / 300.0, 1.0)

                # 6. Dynamic Weighting
                # If visual is getting noisy (emb_c high), trust motion more
                w_emb = emb_weight
                w_pos = pos_weight
                if emb_c > 0.25:
                    w_emb = emb_weight * 0.8
                    w_pos = 1.0 - w_emb

                cost[i, j] = w_emb * emb_c + w_pos * pos_c

        return cost, track_ids, snap_pids

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
            relinked = self._relink_absent_existing_lock(
                pid, existing_tid, tid, "Apply", cost
            )
            if relinked:
                existing_lk = self.locks.get_lock(tid)
            elif existing_lk and existing_lk.stable_count >= STABLE_PROTECT_THRESHOLD:
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
        new_pos = positions.get(tid, slot.last_position)
        new_pitch = self.pitch_positions.get(tid)

        # Update velocity if we have previous history and the gap is reasonable
        if slot.last_position is not None and slot.last_seen_frame > 0:
            dt = self.frame_id - slot.last_seen_frame
            if 0 < dt < 30: # 1s max gap for velocity calc
                inv_dt = 1.0 / dt
                v_px = ((new_pos[0] - slot.last_position[0]) * inv_dt,
                        (new_pos[1] - slot.last_position[1]) * inv_dt)
                if slot.velocity_px is None:
                    slot.velocity_px = v_px
                else:
                    slot.velocity_px = (0.7 * slot.velocity_px[0] + 0.3 * v_px[0],
                                        0.7 * slot.velocity_px[1] + 0.3 * v_px[1])

                if new_pitch is not None and slot.last_pitch is not None:
                    v_pi = ((new_pitch[0] - slot.last_pitch[0]) * inv_dt,
                            (new_pitch[1] - slot.last_pitch[1]) * inv_dt)
                    if slot.velocity_pitch is None:
                        slot.velocity_pitch = v_pi
                    else:
                        slot.velocity_pitch = (0.7 * slot.velocity_pitch[0] + 0.3 * v_pi[0],
                                               0.7 * slot.velocity_pitch[1] + 0.3 * v_pi[1])

        slot.active_track_id = tid
        slot.last_assigned_tid = int(tid)
        slot.last_assigned_seq = self.identity_frame_seq
        slot.seen_this_frame = True
        slot.state = "active"
        slot.last_seen_frame = self.frame_id
        slot.last_position = new_pos
        slot.last_pitch = new_pitch if new_pitch is not None else slot.last_pitch
        slot.stability_counter += 1
        slot.team_id = self.team_labels.get(tid, slot.team_id)
        if slot.stability_counter >= 5 and slot.team_id is None:
            t_team = self.team_labels.get(tid)
            if t_team is not None:
                slot.team_id = t_team

    def get_slot(self, pid: str) -> Optional[PlayerSlot]:
        """Public PID slot lookup used by tracker/export code."""
        if pid is None:
            return None
        return self._slot_by_pid(pid)

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
        t_data: Optional[Dict | np.ndarray],
        t_pos: Optional[Tuple[float, float]],
        tid: Optional[int] = None,
        camera_motion: Optional[Dict] = None,
    ) -> float:
        t_emb = t_data.get("emb") if isinstance(t_data, dict) else t_data
        t_hsv = t_data.get("hsv") if isinstance(t_data, dict) else None

        # 1. Boot: empty lost slots eagerly accept — but NEVER during any restricted mode
        if not self._identity_restricted and slot.state == "lost" and slot.embedding is None:
            return 0.30

        # 2. Team hard gate
        if tid is not None:
            t_team = self.team_labels.get(tid)
            if t_team is not None and slot.team_id is not None and t_team != slot.team_id:
                return float(COST_REJECT_THRESHOLD + 0.05)

        # 3. Hard Color Gate (GK Protection)
        if t_hsv is not None and slot.hsv_signature is not None:
            hsv_sim = float(np.clip(np.dot(t_hsv, slot.hsv_signature), 0.0, 1.0))
            if (1.0 - hsv_sim) > STRICT_COLOR_THRESHOLD:
                return float(COST_REJECT_THRESHOLD + 0.10)

        emb_cost = 0.5
        if t_emb is not None and slot.embedding is not None:
            e = t_emb.astype(np.float32)
            n = np.linalg.norm(e)
            if n > 0: e /= n
            try:
                cos = float(np.clip(np.dot(e, slot.embedding), -1.0, 1.0))
            except Exception as ex:
                print(f"FATAL: np.dot crashed: {ex}")
                print(f"type(e)={type(e)} e.shape={getattr(e, 'shape', None)} e.dtype={getattr(e, 'dtype', None)}")
                print(f"type(slot.embedding)={type(slot.embedding)} shape={getattr(slot.embedding, 'shape', None)} dtype={getattr(slot.embedding, 'dtype', None)}")
                print(f"slot.embedding repr: {repr(slot.embedding)[:200]}")
                import sys; sys.exit(1)
            emb_cost = 1.0 - (cos + 1.0) * 0.5

        # 4. First-Order Spatial Cost (Prediction)
        pred_pos = slot.predict_position(self.frame_id)
        slot_pos_for_cost = pred_pos if pred_pos is not None else slot.last_position
        log_motion_comp = False
        if camera_motion is not None and slot.last_position is not None and t_pos is not None:
            slot_x, slot_y = slot.last_position
            if camera_motion.get("affine") is not None:
                # Use affine transform for precise motion compensation
                M = np.array(camera_motion["affine"], dtype=np.float32)
                pt = np.array([[slot_x, slot_y]], dtype=np.float32).reshape(-1, 1, 2)
                pt_comp = cv2.transform(pt, M)
                slot_pos_for_cost = tuple(pt_comp[0, 0])
            else:
                # Fallback: simple translation
                dx = camera_motion.get("dx", 0.0)
                dy = camera_motion.get("dy", 0.0)
                slot_pos_for_cost = (slot_x + dx, slot_y + dy)
            log_motion_comp = True

        t_pitch = self.pitch_positions.get(tid) if tid is not None else None
        pos_cost = assignment_position_cost(slot.last_pitch, t_pitch) if t_pitch is not None \
            else (assignment_position_cost(slot_pos_for_cost, t_pos) if t_pos is not None else 0.5)

        # Log position compensation effect if applied
        if log_motion_comp and self.frame_id % 30 == 0:
            slot_orig_cost = assignment_position_cost(slot.last_position, t_pos) if t_pos is not None else 0.5
            dist_before = np.sqrt((slot.last_position[0] - t_pos[0])**2 + (slot.last_position[1] - t_pos[1])**2) if t_pos is not None else 0.0
            dist_after = np.sqrt((slot_pos_for_cost[0] - t_pos[0])**2 + (slot_pos_for_cost[1] - t_pos[1])**2) if t_pos is not None else 0.0
            improvement = dist_before - dist_after
            if improvement > 0.5:  # Only log if improvement is significant
                print(f"[MotionComp] frame={self.frame_id} pid={slot.pid} tid={tid} "
                      f"old_pos=({slot.last_position[0]:.0f},{slot.last_position[1]:.0f}) "
                      f"comp_pos=({slot_pos_for_cost[0]:.0f},{slot_pos_for_cost[1]:.0f}) "
                      f"det_pos=({t_pos[0]:.0f},{t_pos[1]:.0f}) "
                      f"dist_before={dist_before:.1f} dist_after={dist_after:.1f} "
                      f"improvement={improvement:.1f}px motion_class={camera_motion.get('motion_class', 'unknown')}")

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
        print(f"\n[PanSafeGateMetrics]")
        print(f"  fast_pan_frames                   = {self.fast_pan_frames}")
        print(f"  cut_frames                        = {self.cut_frames}")
        print(f"  camera_motion_recovery_frames     = {self.camera_motion_recovery_frames}")
        print(f"  pan_lock_attempts_blocked         = {self.pan_lock_attempts_blocked}")
        print(f"  pan_rebinds_blocked               = {self.pan_rebinds_blocked}")
        print(f"  pan_takeovers_blocked             = {self.pan_takeovers_blocked}")
        print(f"  pan_ttl_extensions                = {self.pan_ttl_extensions}")

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
            "lock_retention_rate": retention,
            "fast_pan_frames": self.fast_pan_frames,
            "cut_frames": self.cut_frames,
            "camera_motion_recovery_frames": self.camera_motion_recovery_frames,
            "pan_lock_attempts_blocked": self.pan_lock_attempts_blocked,
            "pan_rebinds_blocked": self.pan_rebinds_blocked,
            "pan_takeovers_blocked": self.pan_takeovers_blocked,
            "pan_ttl_extensions": self.pan_ttl_extensions,
        }
