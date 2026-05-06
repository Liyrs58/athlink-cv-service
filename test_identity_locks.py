"""
Focused tests for recovery-takeover protection.
Runnable with pytest or directly: python3 test_identity_locks.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from services.identity_locks import IdentityLockManager, IdentityLock
from services.identity_core import IdentityCore


# ── helpers ─────────────────────────────────────────────────────────

def make_manager(restricted=False):
    m = IdentityLockManager()
    m.in_restricted = restricted
    return m


def plant_revived_lock(m, tid, pid, frame=100):
    """Force a revived lock into the manager bypassing try_create_lock guards."""
    from services.identity_locks import LOCK_REVIVED_TTL
    lk = IdentityLock(
        track_id=tid, pid=pid, source="revived",
        confidence=0.9, stable_count=3,
        last_seen_frame=frame, ttl=LOCK_REVIVED_TTL,
        created_frame=frame,
    )
    m._tid_to_lock[tid] = lk
    m._pid_to_tid[pid] = tid
    m.locks_created += 1
    return lk


# ── Test 1: restricted revived lock cannot be taken over ────────────

def test_revived_lock_blocked_during_restricted():
    m = make_manager(restricted=True)
    plant_revived_lock(m, tid=126, pid="P2", frame=829)

    lk, status = m.try_create_lock(
        130, "P2", "revived", 832, 0.85,
        allow_takeover=True, allow_rebind=True,
    )
    assert lk is None, f"Expected no lock, got {lk}"
    assert status == "blocked_recovery_takeover", f"Expected blocked_recovery_takeover, got {status!r}"
    assert m._pid_to_tid["P2"] == 126, "Original tid should still own P2"
    assert m.restricted_lock_attempts >= 1
    assert m.id_switches_blocked >= 1
    assert m.pid_takeover_count == 0
    print("PASS test_revived_lock_blocked_during_restricted")


# ── Test 2: same takeover allowed after restricted mode ends ─────────

def test_revived_lock_takeover_allowed_after_recovery():
    m = make_manager(restricted=False)
    plant_revived_lock(m, tid=126, pid="P2", frame=829)

    lk, status = m.try_create_lock(
        130, "P2", "revived", 950, 0.85,
        allow_takeover=True, allow_rebind=True,
    )
    assert lk is not None, "Lock should be created in normal play"
    assert status in ("created", "refreshed"), f"Unexpected status {status!r}"
    assert m._pid_to_tid["P2"] == 130, "New tid should own P2 after normal-play takeover"
    assert m.pid_takeover_count == 1
    print("PASS test_revived_lock_takeover_allowed_after_recovery")


# ── Test 3: scene revival rejects candidate trying to steal restricted revived lock

def test_scene_revival_rejects_recovery_lock_protected():
    class FakeTrack:
        def __init__(self, tid): self.track_id = tid

    ic = IdentityCore()
    ic.in_scene_recovery = True
    ic.frame_id = 832

    # Plant a revived lock for P2 → tid 126
    plant_revived_lock(ic.locks, tid=126, pid="P2", frame=829)
    ic.locks.in_restricted = True

    # Build a minimal bench snapshot with P2 entry
    import numpy as np
    snap_emb = np.random.randn(512).astype(np.float32)
    snap_emb /= np.linalg.norm(snap_emb)
    ic._bench_snapshot = {
        "P2": {
            "embedding": snap_emb,
            "position": (0.5, 0.5),
            "pitch": (0.5, 0.5),
            "team_id": 0,
            "last_seen": 800,
        }
    }

    # tid=130 has a perfect embedding match for P2 — but P2 is already revived+restricted
    ic.team_labels = {130: 0}
    embeddings = {130: snap_emb.copy()}
    positions = {130: (0.5, 0.5)}

    revived, meta = ic.revive_cost_matrix(
        [FakeTrack(130)], embeddings, positions
    )

    assert 130 not in revived, "tid=130 should NOT steal P2 from restricted revived lock"
    assert ic.ambiguous_rejects >= 1, "ambiguous_rejects should increment for recovery_lock_protected"
    print("PASS test_scene_revival_rejects_recovery_lock_protected")


# ── Test 4: _recovery_lock_protected helper logic ────────────────────

def test_recovery_lock_protected_helper():
    from services.identity_locks import IdentityLock
    ic = IdentityCore()

    revived_lk = IdentityLock(track_id=1, pid="P1", source="revived",
                               confidence=0.9, stable_count=2, last_seen_frame=0)
    hungarian_lk = IdentityLock(track_id=2, pid="P2", source="hungarian",
                                 confidence=0.8, stable_count=8, last_seen_frame=0)

    ic.in_scene_recovery = True
    assert ic._recovery_lock_protected(revived_lk) is True
    assert ic._recovery_lock_protected(hungarian_lk) is False
    assert ic._recovery_lock_protected(None) is False

    ic.in_scene_recovery = False
    assert ic._recovery_lock_protected(revived_lk) is False  # not restricted anymore
    print("PASS test_recovery_lock_protected_helper")


def test_low_cost_soft_revive_bypasses_tight_margin():
    class FakeTrack:
        def __init__(self, tid): self.track_id = tid

    import numpy as np
    ic = IdentityCore()
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    ic.frame_id = 240
    ic.in_soft_recovery = True
    ic._soft_snapshot = {
        "P1": {"embedding": emb.copy(), "position": (0.5, 0.5), "pitch": None, "team_id": None, "last_seen": 150},
        "P2": {"embedding": emb.copy(), "position": (0.5, 0.5), "pitch": None, "team_id": None, "last_seen": 150},
    }

    revived, meta = ic.revive_from_soft_snapshot(
        [FakeTrack(44)], {44: emb.copy()}, {44: (0.5, 0.5)}
    )

    assert revived, "Excellent soft-revive cost should bypass tiny ambiguity margin"
    assert next(iter(meta.values())).identity_valid is True
    print("PASS test_low_cost_soft_revive_bypasses_tight_margin")


def test_force_commit_remaining_scene_slot():
    class FakeTrack:
        def __init__(self, tid): self.track_id = tid

    import numpy as np
    ic = IdentityCore()
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    ic.frame_id = 910
    ic.in_scene_recovery = True
    ic._scene_reset_frame = 819
    ic._bench_snapshot = {
        "P1": {"embedding": emb.copy(), "position": (0.5, 0.5), "pitch": None, "team_id": None, "last_seen": 818}
    }

    forced, meta = ic.force_commit_remaining_scene_slots(
        [FakeTrack(130)], {130: emb.copy()}, {130: (0.5, 0.5)}
    )

    assert forced == {130: "P1"}
    assert meta[130].identity_valid is True
    assert ic.locks.get_tid_for_pid("P1") == 130
    assert ic.locks.identity_switches == 0
    print("PASS test_force_commit_remaining_scene_slot")


def test_renderer_never_labels_raw_tracker_ids():
    from render_video import _label_for

    assert _label_for({"identity_valid": False, "trackId": 130, "rawTrackId": 130}) is None
    assert _label_for({"identity_valid": False, "displayId": "U T130"}) is None
    assert _label_for({"identity_valid": False, "displayId": "130"}) is None
    assert _label_for({"identity_valid": False, "displayId": 130}) is None
    assert _label_for({"identity_valid": False, "assignment_pending": True}) == "?"
    assert _label_for({"identity_valid": True, "playerId": "P7", "assignment_source": "locked"}) == "P7 locked"
    print("PASS test_renderer_never_labels_raw_tracker_ids")


def test_recent_dormant_lock_revives_without_new_churn():
    m = make_manager(restricted=False)
    lk = plant_revived_lock(m, tid=10, pid="P4", frame=100)
    lk.dormant = True
    lk.dormant_since_frame = 120
    before_created = m.locks_created

    revived, status = m.try_create_lock(
        130, "P4", "hungarian", 150, 0.12,
        allow_takeover=True, allow_rebind=True,
    )

    assert revived is not None
    assert status == "revived_dormant"
    assert m.get_tid_for_pid("P4") == 130
    assert m.locks_created == before_created
    assert m.identity_switches == 0
    print("PASS test_recent_dormant_lock_revives_without_new_churn")


# ── runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_revived_lock_blocked_during_restricted()
    test_revived_lock_takeover_allowed_after_recovery()
    test_scene_revival_rejects_recovery_lock_protected()
    test_recovery_lock_protected_helper()
    test_low_cost_soft_revive_bypasses_tight_margin()
    test_force_commit_remaining_scene_slot()
    test_renderer_never_labels_raw_tracker_ids()
    test_recent_dormant_lock_revives_without_new_churn()
    print("\nAll tests PASS")
