"""
Identity Lock Manager — persistent track_id <-> pid bindings.

Invariants:
  - 1:1 mapping: each tid has at most one pid, each pid at most one tid.
  - During restricted mode (collapse/recovery/bench): locks decay to DORMANT,
    never expire. Dormant locks reserve the pid — no Hungarian can steal it.
  - Hungarian lock creation is hard-blocked during restricted mode.
  - rebind/takeover during normal play are counted as switches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


STABLE_PROMOTE_FRAMES = 5      # Hungarian must agree this many consecutive frames to lock
LOCK_DEFAULT_TTL = 300         # frames a lock survives absent track in normal play
LOCK_REVIVED_TTL = 390         # revivals get a longer grace
LOCK_DORMANT_TTL = 450         # frames a dormant lock reserves pid before true expiry
PAN_LOCK_TTL_EXTENSION_FRAMES = 45  # extra frames to protect dormant locks during pan
MEMORY_UPDATE_MIN_STABLE = 5   # don't write embedding until lock is this stable
RECENT_DORMANT_REVIVE_FRAMES = 60
STABLE_AUTO_DORMANT_THRESHOLD = 10  # locks above this stable_count go DORMANT on expiry instead of hard-release


def _stability_ttl(base: int, stable_count: int) -> int:
    """Stability-aware TTL: well-established locks get up to 2x more grace.
    stable=0 -> base, stable=150 -> 1.5x, stable>=300 -> 2x. Capped at 2x."""
    factor = 1.0 + min(1.0, max(0, stable_count) / 300.0)
    return int(base * factor)


@dataclass
class IdentityLock:
    track_id: int
    pid: str                    # "P7"
    source: str                 # bootstrap | hungarian | revived | manual
    confidence: float = 0.0
    stable_count: int = 0
    last_seen_frame: int = -1
    ttl: int = LOCK_DEFAULT_TTL
    created_frame: int = -1
    dormant: bool = False       # True → pid reserved but track absent; cannot be stolen
    dormant_since_frame: int = -1


class IdentityLockManager:
    """Two-way 1:1 locking with honest switch accounting."""

    def __init__(self, logger=None):
        self.logger = logger
        self._tid_to_lock: Dict[int, IdentityLock] = {}
        self._pid_to_tid: Dict[str, int] = {}

        # Honest metrics
        self.identity_switches: int = 0
        self.id_switches_blocked: int = 0
        self.locks_created: int = 0
        self.locks_expired: int = 0
        self.locks_dormanted: int = 0          # locks converted to dormant instead of expired
        self.id_rebind_count: int = 0
        self.pid_takeover_count: int = 0
        self.collapse_lock_attempts: int = 0   # hungarian attempts blocked before write
        self.collapse_lock_creations: int = 0  # locks actually written during restricted (must be 0)
        self.restricted_lock_attempts: int = 0 # all restricted-mode block attempts
        self.soft_recovery_rebinds_blocked: int = 0
        self._switch_log: List[Tuple] = []

        # Set True by tracker whenever identity is restricted
        self.in_restricted: bool = False
        # Legacy alias kept for callers that set in_collapse
        self.in_collapse: bool = False
        # Set True during soft_recovery/soft_collapse to skip dormancy
        self.in_recovery: bool = False
        # Track last tick frame to support frame-gap-aware TTL decay
        self._last_tick_frame: int = -1

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_lock(self, tid: int) -> Optional[IdentityLock]:
        return self._tid_to_lock.get(tid)

    def get_pid(self, tid: int) -> Optional[str]:
        lk = self._tid_to_lock.get(tid)
        return lk.pid if lk else None

    def get_tid_for_pid(self, pid: str) -> Optional[int]:
        return self._pid_to_tid.get(pid)

    def is_pid_locked(self, pid: str) -> bool:
        return pid in self._pid_to_tid

    def is_tid_locked(self, tid: int) -> bool:
        return tid in self._tid_to_lock

    def is_pid_dormant(self, pid: str) -> bool:
        tid = self._pid_to_tid.get(pid)
        if tid is None:
            return False
        lk = self._tid_to_lock.get(tid)
        return lk is not None and lk.dormant

    def locked_tids(self) -> Set[int]:
        return set(self._tid_to_lock.keys())

    def locked_pids(self) -> Set[str]:
        return set(self._pid_to_tid.keys())

    def count_live_locks(self) -> int:
        """Count non-dormant locks."""
        return sum(1 for lk in self._tid_to_lock.values() if not lk.dormant)

    def lock_ttl(self, tid: int) -> int:
        lk = self._tid_to_lock.get(tid)
        return lk.ttl if lk else 0

    def relink_absent_lock(
        self,
        old_tid: int,
        new_tid: int,
        pid: str,
        frame_id: int,
        source: str = "revived",
        confidence: float = 0.0,
        ttl: Optional[int] = None,
    ) -> Tuple[Optional[IdentityLock], str]:
        """
        Move a reserved PID from an absent tracker id to the current tracker id.

        This is not a visual identity switch: the old tracker id is no longer present,
        so the lock is following the same PID across a tracker reset/fragment.
        """
        lk = self._tid_to_lock.get(old_tid)
        if lk is None or lk.pid != pid:
            return None, "missing_old_lock"

        existing_new = self._tid_to_lock.get(new_tid)
        if existing_new is not None and existing_new.pid != pid:
            return None, "blocked_new_tid_owned"

        self._tid_to_lock.pop(old_tid, None)
        lk.track_id = new_tid
        lk.source = source
        lk.confidence = confidence
        lk.stable_count = max(1, lk.stable_count)
        lk.last_seen_frame = frame_id
        lk.ttl = ttl if ttl is not None else LOCK_REVIVED_TTL
        lk.dormant = False
        lk.dormant_since_frame = -1
        self._tid_to_lock[new_tid] = lk
        self._pid_to_tid[pid] = new_tid
        print(
            f"[IDLockRelink] frame={frame_id} pid={pid} old_tid={old_tid} "
            f"new_tid={new_tid} reason=absent_tracker_fragment"
        )
        return lk, "relinked_absent"

    # ------------------------------------------------------------------
    # Create / refresh / release
    # ------------------------------------------------------------------

    def try_create_lock(
        self,
        tid: int,
        pid: str,
        source: str,
        frame_id: int,
        confidence: float = 0.0,
        ttl: Optional[int] = None,
        allow_takeover: bool = False,
        allow_rebind: bool = False,
    ) -> Tuple[Optional[IdentityLock], str]:
        """
        Returns (lock_or_None, status_string).
        Hard-blocks hungarian source during restricted/collapse mode.
        """
        restricted = self.in_restricted or self.in_collapse

        # Hard block: any Hungarian lock during restricted mode
        if restricted and source == "hungarian":
            self.collapse_lock_attempts += 1
            self.restricted_lock_attempts += 1
            print(
                f"[CollapseBlock] frame={frame_id} tid={tid} pid={pid} "
                f"source=hungarian BLOCKED restricted=True (not written)"
            )
            return None, "blocked_collapse"

        existing_tid_for_pid = self._pid_to_tid.get(pid)
        existing_pid_for_tid = self._tid_to_lock.get(tid)

        # Check pid conflict
        if existing_tid_for_pid is not None and existing_tid_for_pid != tid:
            existing_lk = self._tid_to_lock.get(existing_tid_for_pid)
            if existing_lk and existing_lk.dormant and restricted:
                # Dormant pid is reserved during restricted mode — block takeover
                self.restricted_lock_attempts += 1
                print(
                    f"[DormantBlock] frame={frame_id} pid={pid} "
                    f"dormant_tid={existing_tid_for_pid} new_tid={tid} BLOCKED (restricted)"
                )
                return None, "blocked_dormant"
            if (existing_lk and existing_lk.dormant and not restricted
                    and (existing_pid_for_tid is None or existing_pid_for_tid.pid == pid)
                    and 0 <= frame_id - existing_lk.dormant_since_frame <= RECENT_DORMANT_REVIVE_FRAMES):
                self._tid_to_lock.pop(existing_tid_for_pid, None)
                existing_lk.track_id = tid
                existing_lk.source = source
                existing_lk.confidence = confidence
                existing_lk.stable_count = 1
                existing_lk.last_seen_frame = frame_id
                existing_lk.ttl = ttl if ttl is not None else (
                    LOCK_REVIVED_TTL if source == "revived" else LOCK_DEFAULT_TTL
                )
                existing_lk.dormant = False
                existing_lk.dormant_since_frame = -1
                self._tid_to_lock[tid] = existing_lk
                self._pid_to_tid[pid] = tid
                print(f"[IDLockRevive] frame={frame_id} pid={pid} tid={tid} "
                      f"old_tid={existing_tid_for_pid} reason=recent_dormant")
                return existing_lk, "revived_dormant"
            if existing_lk and existing_lk.source == "revived" and restricted:
                # Freshly revived lock cannot be stolen while identity is restricted
                self.restricted_lock_attempts += 1
                self.id_switches_blocked += 1
                print(
                    f"[RecoveryTakeoverBlock] frame={frame_id} pid={pid} "
                    f"revived_tid={existing_tid_for_pid} new_tid={tid} BLOCKED (restricted)"
                )
                return None, "blocked_recovery_takeover"
            if not allow_takeover:
                return None, "blocked_takeover"
            self._record_switch_internal(
                frame_id, pid,
                old_tid=existing_tid_for_pid, new_tid=tid,
                reason=f"takeover:{source}",
            )
            self._release_internal(existing_tid_for_pid, reason="takeover", frame_id=frame_id)
            self.pid_takeover_count += 1

        # Check tid conflict
        if existing_pid_for_tid is not None and existing_pid_for_tid.pid != pid:
            if not allow_rebind:
                return None, "blocked_rebind"
            self._record_switch_internal(
                frame_id, existing_pid_for_tid.pid,
                old_tid=tid, new_tid=-1,
                reason=f"rebind:{source}",
            )
            self._release_internal(tid, reason="rebind", frame_id=frame_id)
            self.id_rebind_count += 1

        # Exact same lock already — just refresh
        if tid in self._tid_to_lock and self._tid_to_lock[tid].pid == pid:
            self.refresh_lock(tid, frame_id, confidence)
            return self._tid_to_lock[tid], "refreshed"

        # Create fresh
        eff_ttl = ttl if ttl is not None else (
            LOCK_REVIVED_TTL if source == "revived" else LOCK_DEFAULT_TTL
        )
        lk = IdentityLock(
            track_id=tid, pid=pid, source=source,
            confidence=confidence, stable_count=1,
            last_seen_frame=frame_id, ttl=eff_ttl, created_frame=frame_id,
        )
        self._tid_to_lock[tid] = lk
        self._pid_to_tid[pid] = tid
        self.locks_created += 1

        # Audit: should never happen after the block above, but count if it does
        if restricted and source == "hungarian":
            self.collapse_lock_creations += 1
            print(f"[IdentityInvariantFAIL] Hungarian lock written during restricted mode! "
                  f"frame={frame_id} tid={tid} pid={pid}")

        print(
            f"[IDLock] frame={frame_id} tid={tid} pid={pid} "
            f"source={source} stable={lk.stable_count} ttl={lk.ttl}"
        )
        return lk, "created"

    def create_lock(
        self,
        tid: int,
        pid: str,
        source: str,
        frame_id: int,
        confidence: float = 0.0,
        ttl: Optional[int] = None,
    ) -> IdentityLock:
        """Force-create used by revival paths — conflicts counted as switches."""
        lk, status = self.try_create_lock(
            tid, pid, source, frame_id, confidence, ttl,
            allow_takeover=True, allow_rebind=True,
        )
        return lk  # type: ignore[return-value]

    def refresh_lock(
        self,
        tid: int,
        frame_id: int,
        confidence: Optional[float] = None,
    ) -> Optional[IdentityLock]:
        lk = self._tid_to_lock.get(tid)
        if lk is None:
            return None
        lk.stable_count += 1
        lk.last_seen_frame = frame_id
        lk.dormant = False  # track seen → no longer dormant
        lk.dormant_since_frame = -1
        if confidence is not None:
            lk.confidence = confidence
        base_ttl = LOCK_REVIVED_TTL if lk.source == "revived" else LOCK_DEFAULT_TTL
        lk.ttl = _stability_ttl(base_ttl, lk.stable_count)
        return lk

    def release_lock(self, tid: int, reason: str = "manual", frame_id: int = -1) -> Optional[str]:
        return self._release_internal(tid, reason=reason, frame_id=frame_id)

    def extend_dormant_ttl_for_pan(self, frame_id: int, extension: int = PAN_LOCK_TTL_EXTENSION_FRAMES) -> int:
        """Extend TTL for all dormant locks to protect them during camera pan motion."""
        extended = 0
        for tid, lk in self._tid_to_lock.items():
            if lk.dormant:
                lk.ttl += extension
                extended += 1
                if extended <= 3:  # Log first 3 for debugging
                    print(f"[PanTTLExtend] frame={frame_id} tid={tid} pid={lk.pid} "
                          f"old_ttl={lk.ttl - extension} new_ttl={lk.ttl}")
        if extended > 3:
            print(f"[PanTTLExtend] frame={frame_id} extended {extended} dormant locks (showing first 3)")
        return extended

    def _release_internal(self, tid: int, reason: str, frame_id: int = -1) -> Optional[str]:
        lk = self._tid_to_lock.pop(tid, None)
        if lk is None:
            return None
        if self._pid_to_tid.get(lk.pid) == tid:
            self._pid_to_tid.pop(lk.pid, None)
        if reason == "stale":
            self.locks_expired += 1
        print(
            f"[IDLockExpired] frame={frame_id} tid={tid} pid={lk.pid} "
            f"reason={reason} stable={lk.stable_count}"
        )
        return lk.pid

    def _record_switch_internal(
        self, frame_id: int, pid: str, old_tid: int, new_tid: int, reason: str
    ) -> None:
        self.identity_switches += 1
        self._switch_log.append((frame_id, pid, old_tid, new_tid, reason))
        print(
            f"[IDSwitch] frame={frame_id} pid={pid} "
            f"old_tid={old_tid} new_tid={new_tid} reason={reason}"
        )

    def reset_all(self) -> None:
        if self._tid_to_lock:
            print(f"[IDLockReset] dropping {len(self._tid_to_lock)} locks")
        self._tid_to_lock.clear()
        self._pid_to_tid.clear()

    # ------------------------------------------------------------------
    # Per-frame TTL decay — dormant-aware
    # ------------------------------------------------------------------

    def tick(self, frame_id: int, present_tids: Set[int], restricted: bool = False,
             frozen: bool = False) -> None:
        """
        Decay TTL for absent tracks.
        In restricted mode: convert to DORMANT instead of expiring.
        Dormant locks reserve the pid — they decay on a longer timer.
        frozen=True: bench/cutaway shot — skip all TTL decay entirely.
        """
        self._last_tick_frame = frame_id
        if frozen:
            return
        to_expire = []
        for tid, lk in self._tid_to_lock.items():
            if tid in present_tids:
                # Track visible — ensure not dormant
                if lk.dormant:
                    lk.dormant = False
                continue

            # Decay by 1 per processed tick regardless of frame_stride.
            # TTL is in "processed-frame" units — burning it by video-frame gap
            # kills locks in ~60 ticks at stride=5 instead of 300.
            lk.ttl -= 1

            if lk.ttl <= 0:
                # Auto-dormant for sufficiently-stable locks even in normal play.
                # Otherwise a 308-frame-stable player going off-screen for 150 frames
                # gets hard-released, then recreated as a NEW pid — pure churn.
                # Don't dormant during recovery—keep locks active for revival attempts
                should_dormant = (
                    not lk.dormant and
                    not self.in_recovery and
                    (restricted or lk.stable_count >= STABLE_AUTO_DORMANT_THRESHOLD)
                )
                if should_dormant:
                    lk.dormant = True
                    lk.dormant_since_frame = frame_id
                    lk.ttl = LOCK_DORMANT_TTL
                    self.locks_dormanted += 1
                    reason_tag = "stale_during_restricted" if restricted else "stale_auto_dormant"
                    print(
                        f"[IDLockDormant] frame={frame_id} tid={tid} pid={lk.pid} "
                        f"stable={lk.stable_count} reason={reason_tag}"
                    )
                elif lk.dormant and lk.ttl <= 0:
                    to_expire.append(tid)
                elif not lk.dormant:
                    to_expire.append(tid)

        for tid in to_expire:
            self._release_internal(tid, reason="stale", frame_id=frame_id)

    # ------------------------------------------------------------------
    # Blocked-switch helpers
    # ------------------------------------------------------------------

    def record_blocked_switch(
        self, frame_id: int, pid: str, old_tid: int, new_tid: int, reason: str
    ) -> None:
        self.id_switches_blocked += 1
        print(
            f"[IDSwitchBlocked] frame={frame_id} pid={pid} "
            f"old_tid={old_tid} new_tid={new_tid} reason={reason}"
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, object]:
        live = len(self._tid_to_lock)
        dormant = sum(1 for lk in self._tid_to_lock.values() if lk.dormant)
        total_seen = self.locks_created
        retention = (
            (total_seen - self.locks_expired) / float(total_seen)
            if total_seen > 0 else 0.0
        )
        churn_warning = self.locks_created > 40
        return {
            "identity_switches": self.identity_switches,
            "id_rebind_count": self.id_rebind_count,
            "pid_takeover_count": self.pid_takeover_count,
            "switches_blocked": self.id_switches_blocked,
            "soft_recovery_rebinds_blocked": self.soft_recovery_rebinds_blocked,
            "collapse_lock_attempts": self.collapse_lock_attempts,
            "collapse_lock_creations": self.collapse_lock_creations,
            "restricted_lock_attempts": self.restricted_lock_attempts,
            "locks_created": self.locks_created,
            "locks_expired": self.locks_expired,
            "locks_dormanted": self.locks_dormanted,
            "locks_live": live,
            "locks_dormant": dormant,
            "lock_retention_rate": round(retention, 3),
            "excessive_lock_churn": churn_warning,
        }
