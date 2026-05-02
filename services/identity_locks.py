"""
Identity Lock Manager — persistent track_id <-> pid bindings.

Replaces frame-by-frame Hungarian re-assignment with sticky locks.

Core idea:
  - When a (track_id, pid) pair is established (revival, or stable Hungarian
    match), record it as a lock.
  - On subsequent frames, if the track_id is still fresh, the pid is reused
    automatically — Hungarian never sees that track or that slot.
  - Locks expire on TTL miss (track unseen for too long) or are released
    explicitly.

Lock sources:
  - "bootstrap"    : first frames, weak commitment
  - "hungarian"    : promoted after stable Hungarian match for >= STABLE_PROMOTE
  - "revived"      : created immediately by soft/scene revival
  - "manual"       : reserved
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple


STABLE_PROMOTE_FRAMES = 5      # Hungarian match must hold this long to lock
LOCK_DEFAULT_TTL = 60          # frames a lock survives without re-confirmation
LOCK_REVIVED_TTL = 90          # revivals get a longer grace period
MEMORY_UPDATE_MIN_STABLE = 5   # don't write embedding to slot until lock is this stable


@dataclass
class IdentityLock:
    track_id: int
    pid: str                       # "P7"
    source: str                    # bootstrap | hungarian | revived | manual
    confidence: float = 0.0        # last assignment confidence (0..1, higher better)
    stable_count: int = 0          # frames this lock has held
    last_seen_frame: int = -1
    ttl: int = LOCK_DEFAULT_TTL    # frames until expiry without re-confirmation
    created_frame: int = -1


class IdentityLockManager:
    """
    Two-way locking: tid <-> pid, 1:1.
    """

    def __init__(self, logger=None):
        self.logger = logger
        self._tid_to_lock: Dict[int, IdentityLock] = {}
        self._pid_to_tid: Dict[str, int] = {}

        # Switch / metric accounting
        self.identity_switches: int = 0
        self.id_switches_blocked: int = 0
        self.locks_expired: int = 0
        self.locks_created: int = 0
        self._switch_log: list = []   # (frame, pid, old_tid, new_tid, reason)

    # ------------------------------------------------------------------
    # Public API: lock lifecycle
    # ------------------------------------------------------------------

    def get_lock(self, tid: int) -> Optional[IdentityLock]:
        return self._tid_to_lock.get(tid)

    def get_pid(self, tid: int) -> Optional[str]:
        lk = self._tid_to_lock.get(tid)
        return lk.pid if lk is not None else None

    def get_tid_for_pid(self, pid: str) -> Optional[int]:
        return self._pid_to_tid.get(pid)

    def is_pid_locked(self, pid: str) -> bool:
        return pid in self._pid_to_tid

    def is_tid_locked(self, tid: int) -> bool:
        return tid in self._tid_to_lock

    def locked_tids(self) -> Set[int]:
        return set(self._tid_to_lock.keys())

    def locked_pids(self) -> Set[str]:
        return set(self._pid_to_tid.keys())

    # ------------------------------------------------------------------
    # Create / refresh / release
    # ------------------------------------------------------------------

    def create_lock(
        self,
        tid: int,
        pid: str,
        source: str,
        frame_id: int,
        confidence: float = 0.0,
        ttl: Optional[int] = None,
    ) -> IdentityLock:
        """
        Force-create a lock. Caller must have already resolved any conflict
        on pid/tid (this function will release whatever was there).
        """
        if pid in self._pid_to_tid:
            old_tid = self._pid_to_tid[pid]
            if old_tid != tid:
                # Hard takeover (e.g. the old tid is dead and we are reassigning)
                self._release_internal(old_tid, reason="takeover")
        if tid in self._tid_to_lock:
            old_pid = self._tid_to_lock[tid].pid
            if old_pid != pid:
                self._release_internal(tid, reason="rebind")

        eff_ttl = ttl if ttl is not None else (
            LOCK_REVIVED_TTL if source == "revived" else LOCK_DEFAULT_TTL
        )
        lk = IdentityLock(
            track_id=tid,
            pid=pid,
            source=source,
            confidence=confidence,
            stable_count=1,
            last_seen_frame=frame_id,
            ttl=eff_ttl,
            created_frame=frame_id,
        )
        self._tid_to_lock[tid] = lk
        self._pid_to_tid[pid] = tid
        self.locks_created += 1
        print(
            f"[IDLock] frame={frame_id} tid={tid} pid={pid} "
            f"source={source} stable={lk.stable_count} ttl={lk.ttl}"
        )
        return lk

    def refresh_lock(
        self,
        tid: int,
        frame_id: int,
        confidence: Optional[float] = None,
    ) -> Optional[IdentityLock]:
        """Mark a lock as confirmed this frame; bumps stable_count, resets TTL."""
        lk = self._tid_to_lock.get(tid)
        if lk is None:
            return None
        lk.stable_count += 1
        lk.last_seen_frame = frame_id
        if confidence is not None:
            lk.confidence = confidence
        # Lock observed → its remaining TTL resets to the source default
        lk.ttl = LOCK_REVIVED_TTL if lk.source == "revived" else LOCK_DEFAULT_TTL
        return lk

    def release_lock(self, tid: int, reason: str = "manual") -> Optional[str]:
        """Release lock for tid. Returns the freed pid (or None)."""
        return self._release_internal(tid, reason=reason)

    def _release_internal(self, tid: int, reason: str) -> Optional[str]:
        lk = self._tid_to_lock.pop(tid, None)
        if lk is None:
            return None
        if self._pid_to_tid.get(lk.pid) == tid:
            self._pid_to_tid.pop(lk.pid, None)
        if reason == "stale":
            self.locks_expired += 1
        print(
            f"[IDLockExpired] frame=? tid={tid} pid={lk.pid} reason={reason} "
            f"stable={lk.stable_count}"
        )
        return lk.pid

    def reset_all(self) -> None:
        """Drop every lock — use on scene reset."""
        if self._tid_to_lock:
            print(f"[IDLockReset] dropping {len(self._tid_to_lock)} locks")
        self._tid_to_lock.clear()
        self._pid_to_tid.clear()

    # ------------------------------------------------------------------
    # Per-frame maintenance
    # ------------------------------------------------------------------

    def tick(self, frame_id: int, present_tids: Set[int]) -> None:
        """
        Decay locks. Called once per frame BEFORE assignment.
        - Tracks present this frame: TTL untouched (will be refreshed on confirm).
        - Tracks absent this frame: TTL decremented; expire at 0.
        """
        expired = []
        for tid, lk in self._tid_to_lock.items():
            if tid in present_tids:
                continue
            lk.ttl -= 1
            if lk.ttl <= 0:
                expired.append(tid)
        for tid in expired:
            self._release_internal(tid, reason="stale")

    # ------------------------------------------------------------------
    # ID switch detection
    # ------------------------------------------------------------------

    def record_blocked_switch(
        self, frame_id: int, pid: str, old_tid: int, new_tid: int, reason: str
    ) -> None:
        self.id_switches_blocked += 1
        print(
            f"[IDSwitchBlocked] frame={frame_id} pid={pid} "
            f"old_tid={old_tid} new_tid={new_tid} reason={reason}"
        )

    def record_switch(
        self, frame_id: int, pid: str, old_tid: int, new_tid: int, reason: str
    ) -> None:
        self.identity_switches += 1
        self._switch_log.append((frame_id, pid, old_tid, new_tid, reason))
        print(
            f"[IDSwitch] frame={frame_id} pid={pid} "
            f"old_tid={old_tid} new_tid={new_tid} reason={reason}"
        )

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, float]:
        live = len(self._tid_to_lock)
        total_seen = self.locks_created
        retention = (
            (total_seen - self.locks_expired) / float(total_seen)
            if total_seen > 0 else 0.0
        )
        return {
            "identity_switches": self.identity_switches,
            "switches_blocked": self.id_switches_blocked,
            "locks_created": self.locks_created,
            "locks_expired": self.locks_expired,
            "locks_live": live,
            "lock_retention_rate": round(retention, 3),
        }
