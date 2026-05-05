"""
Identity Lock Manager — persistent track_id <-> pid bindings.

Key invariants:
  - 1:1 mapping: each tid has at most one pid, each pid at most one tid.
  - Locks survive until their TTL drains (track absent) or explicit release.
  - rebind / takeover are ALWAYS counted as ID switches (never hidden).
  - During recovery or collapse, rebind/takeover are BLOCKED, not silently done.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


STABLE_PROMOTE_FRAMES = 5      # Hungarian must agree this many consecutive frames to lock
LOCK_DEFAULT_TTL = 60          # frames a lock survives when track is absent
LOCK_REVIVED_TTL = 90          # revivals get a longer grace
MEMORY_UPDATE_MIN_STABLE = 5   # don't write embedding until lock is this stable


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


class IdentityLockManager:
    """Two-way 1:1 locking with honest switch accounting."""

    def __init__(self, logger=None):
        self.logger = logger
        self._tid_to_lock: Dict[int, IdentityLock] = {}
        self._pid_to_tid: Dict[str, int] = {}

        # Honest metrics — rebind/takeover count as switches
        self.identity_switches: int = 0       # actual switches (rebind + takeover)
        self.id_switches_blocked: int = 0     # attempts blocked during recovery/collapse
        self.locks_created: int = 0
        self.locks_expired: int = 0           # stale TTL expiry only
        self.id_rebind_count: int = 0         # tid changed its pid
        self.pid_takeover_count: int = 0      # pid stolen from a live tid
        self.collapse_lock_attempts: int = 0  # hungarian attempts while collapse (blocked before write)
        self.collapse_lock_creations: int = 0  # locks actually written during collapse (must be 0)
        self.soft_recovery_rebinds_blocked: int = 0
        self._switch_log: List[Tuple] = []    # (frame, pid, old_tid, new_tid, reason)

        # Mirror of IdentityCore mode — checked at lock-creation to audit collapse
        self.in_collapse: bool = False

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

    def locked_tids(self) -> Set[int]:
        return set(self._tid_to_lock.keys())

    def locked_pids(self) -> Set[str]:
        return set(self._pid_to_tid.keys())

    def lock_ttl(self, tid: int) -> int:
        lk = self._tid_to_lock.get(tid)
        return lk.ttl if lk else 0

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
        Attempt to create a lock. Returns (lock_or_None, status_string).

        Status: "created" | "blocked_takeover" | "blocked_rebind" | "refreshed"

        If allow_takeover=False and pid already has a different live tid → block.
        If allow_rebind=False and tid already has a different pid → block.
        Caller decides whether to count block as switch or not.
        """
        existing_tid_for_pid = self._pid_to_tid.get(pid)
        existing_pid_for_tid = self._tid_to_lock.get(tid)

        # Check pid conflict
        if existing_tid_for_pid is not None and existing_tid_for_pid != tid:
            if not allow_takeover:
                return None, "blocked_takeover"
            # Takeover allowed → count as switch
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
            # Rebind allowed → count as switch
            self._record_switch_internal(
                frame_id, existing_pid_for_tid.pid,
                old_tid=tid, new_tid=-1,
                reason=f"rebind:{source}",
            )
            self._release_internal(tid, reason="rebind", frame_id=frame_id)
            self.id_rebind_count += 1

        # If it's the exact same lock already — just refresh
        if tid in self._tid_to_lock and self._tid_to_lock[tid].pid == pid:
            self.refresh_lock(tid, frame_id, confidence)
            return self._tid_to_lock[tid], "refreshed"

        # Hard block: collapse + hungarian — count attempt but write nothing
        if self.in_collapse and source == "hungarian":
            self.collapse_lock_attempts += 1
            print(f"[CollapseBlock] frame={frame_id} tid={tid} pid={pid} BLOCKED (not written)")
            return None, "blocked_collapse"

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
        """Force-create (used by revival paths — conflicts counted as switches)."""
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
        if confidence is not None:
            lk.confidence = confidence
        lk.ttl = LOCK_REVIVED_TTL if lk.source == "revived" else LOCK_DEFAULT_TTL
        return lk

    def release_lock(self, tid: int, reason: str = "manual", frame_id: int = -1) -> Optional[str]:
        return self._release_internal(tid, reason=reason, frame_id=frame_id)

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
    # Per-frame TTL decay
    # ------------------------------------------------------------------

    def tick(self, frame_id: int, present_tids: Set[int]) -> None:
        """Decay TTL for absent tracks. Expire at 0."""
        expired = []
        for tid, lk in self._tid_to_lock.items():
            if tid in present_tids:
                continue
            lk.ttl -= 1
            if lk.ttl <= 0:
                expired.append(tid)
        for tid in expired:
            self._release_internal(tid, reason="stale", frame_id=frame_id)

    # ------------------------------------------------------------------
    # Blocked-switch helpers for callers
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
        total_seen = self.locks_created
        retention = (
            (total_seen - self.locks_expired) / float(total_seen)
            if total_seen > 0 else 0.0
        )
        churn_warning = self.locks_created > 40
        return {
            "identity_switches": self.identity_switches,    # honest: rebind+takeover
            "id_rebind_count": self.id_rebind_count,
            "pid_takeover_count": self.pid_takeover_count,
            "switches_blocked": self.id_switches_blocked,
            "soft_recovery_rebinds_blocked": self.soft_recovery_rebinds_blocked,
            "collapse_lock_attempts": self.collapse_lock_attempts,
            "collapse_lock_creations": self.collapse_lock_creations,
            "locks_created": self.locks_created,
            "locks_expired": self.locks_expired,
            "locks_live": live,
            "lock_retention_rate": round(retention, 3),
            "excessive_lock_churn": churn_warning,
        }
