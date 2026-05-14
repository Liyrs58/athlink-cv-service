"""
Tests for Phase 1 identity continuity fixes:
- Snapshot freshness timestamp (Task 2)
- Team-gate revival (Task 3)
- Tighter revival margins (Task 4)
"""
import numpy as np


def test_snapshot_records_freshness_timestamp():
    """Each snapshot entry must include the frame_id when the embedding was last updated."""
    from services.identity_core import IdentityCore

    ic = IdentityCore()
    ic.frame_id = 100

    # Use the first slot (P1) which already exists
    slot = ic.slots[0]
    slot.state = "active"
    slot.embedding = np.ones(512, dtype=np.float32)
    slot._emb_last_updated_frame = 50  # stale: 50 frames old
    pid = slot.pid  # "P1"

    ic.snapshot_soft(frame_id=100)

    snap = ic._soft_snapshot.get(pid)
    assert snap is not None, f"{pid} should be in snapshot"
    assert "emb_age" in snap, "snapshot must record embedding age"
    assert snap["emb_age"] == 50, f"emb_age should be 50 frames, got {snap['emb_age']}"


def test_revival_blocked_for_wrong_team():
    """A snapshot slot for team 0 must never revive onto a track assigned to team 1."""
    from services.identity_core import IdentityCore

    ic = IdentityCore()
    ic.frame_id = 50

    # Plant snapshot: P1 is team 0
    ic._soft_snapshot = {
        "P1": {
            "embedding": np.ones(512, dtype=np.float32),
            "position": (200.0, 400.0),
            "pitch": None,
            "team_id": 0,
            "stable_count": 15,
            "emb_age": 5,
        }
    }

    # Incoming track tid=99 is team 1 (PSG) with low embedding cost vs P1's anchor
    ic.team_labels = {99: 1}

    class FakeTrack:
        track_id = 99
        bbox = [190, 390, 230, 450]

    embeddings = {99: {"emb": np.ones(512, dtype=np.float32), "hsv": None}}
    positions = {99: (210.0, 420.0)}

    revived, _ = ic.revive_from_soft_snapshot(
        [FakeTrack()], embeddings, positions,
    )

    assert 99 not in revived, "tid=99 (team 1) must NOT revive P1 (team 0)"


def test_ambiguous_revival_rejected_with_tight_margin():
    """When two snapshot slots have similar cost to one track, neither should be revived."""
    from services.identity_core import IdentityCore, SOFT_REVIVE_MARGIN_MIN

    assert SOFT_REVIVE_MARGIN_MIN >= 0.07, (
        f"SOFT_REVIVE_MARGIN_MIN={SOFT_REVIVE_MARGIN_MIN} is too loose; must be >= 0.07 "
        "to prevent ambiguous same-team revives"
    )
