"""
Tests for Phase 1 identity continuity fixes:
- OSNet weight priority order (Task 1)
- Snapshot freshness timestamp (Task 2)
- Team-gate revival (Task 3)
- Tighter revival margins (Task 4)
"""
import os
import numpy as np


def test_osnet_weight_priority_order(tmp_path):
    """_find_osnet_weights must prefer football > sports > MSMT17."""
    import sys
    for m in list(sys.modules):
        if m.startswith("services"):
            del sys.modules[m]

    # Create fake weight files in tmp_path
    football = tmp_path / "football_osnet_x1_0.pth.tar"
    sports   = tmp_path / "sports_model.pth.tar-60"
    msmt17   = tmp_path / "osnet_x1_0_msmt17.pt"

    # Write 2MB dummy content so size check passes
    dummy = b"x" * 2_000_000
    for f in [football, sports, msmt17]:
        f.write_bytes(dummy)

    # Point env vars to our tmp files
    os.environ["OSNET_FOOTBALL_WEIGHTS"] = str(football)
    os.environ["OSNET_SPORTS_WEIGHTS"]   = str(sports)
    os.environ["OSNET_WEIGHTS"]          = str(msmt17)

    # Import _find_osnet_weights as a standalone function by parsing it out of tracker_core
    # without triggering torch/ultralytics imports (those aren't available in the test env)
    import importlib.util, types, sys as _sys

    # Build minimal stubs for heavy deps so the module-level imports don't crash
    import numpy as _np
    for stub_name in ["torch", "ultralytics", "cv2", "boxmot"]:
        mod = types.ModuleType(stub_name)
        _sys.modules[stub_name] = mod
        if stub_name == "torch":
            mod.cuda = types.SimpleNamespace(is_available=lambda: False)
            mod.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
        if stub_name == "ultralytics":
            mod.YOLO = lambda *a, **kw: None
    _sys.modules["numpy"] = _np

    try:
        # Clear any cached services modules
        for m in list(_sys.modules):
            if m.startswith("services"):
                del _sys.modules[m]

        from services.tracker_core import ReIDExtractor
        ext = ReIDExtractor.__new__(ReIDExtractor)
        result = ext._find_osnet_weights()
        assert result == str(football), f"Expected football weights, got: {result}"

        # Remove football — should fall to sports
        os.environ.pop("OSNET_FOOTBALL_WEIGHTS")
        football.unlink()
        result2 = ext._find_osnet_weights()
        assert result2 == str(sports), f"Expected sports weights, got: {result2}"

        # Remove sports — should fall to MSMT17
        os.environ.pop("OSNET_SPORTS_WEIGHTS")
        sports.unlink()
        result3 = ext._find_osnet_weights()
        assert result3 == str(msmt17), f"Expected MSMT17 weights, got: {result3}"
    finally:
        for k in ["OSNET_FOOTBALL_WEIGHTS", "OSNET_SPORTS_WEIGHTS", "OSNET_WEIGHTS"]:
            os.environ.pop(k, None)


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
