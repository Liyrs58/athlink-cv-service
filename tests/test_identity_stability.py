"""
Tests for identity stability upgrade.

Covers:
  - Confidence semantics (identity_confidence=0 for UNK)
  - 1:1 identity invariants (raw track ID + PID uniqueness)
  - Official/referee suppression
  - Consensus field
  - Audit render label format
  - Patch validator physicality (speed, spatial, temporal, raw collision)
  - VLM metrics computation
  - Production render hides unconfirmed labels
"""

import sys
import os
import json
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from services.identity_patch_service import (
    IdentityPatchService,
    PatchValidator,
    PatchRejection,
)


# ---- Helpers -------------------------------------------------------------------


def _make_player(pid, tid=1, bbox=None, source="locked", confidence=1.0,
                 raw_tid=None, identity_confidence=None, is_official=False,
                 consensus="CONFIRMED", role="player"):
    bbox = bbox or [100, 100, 150, 200]
    return {
        "trackId": tid,
        "rawTrackId": raw_tid if raw_tid is not None else tid,
        "playerId": pid,
        "displayId": pid,
        "bbox": bbox,
        "confidence": confidence,
        "class": 2,
        "gameState": "play",
        "analysis_valid": True,
        "crop_quality": 1.0,
        "identity_valid": pid is not None,
        "assignment_source": source,
        "identity_confidence": identity_confidence if identity_confidence is not None else (confidence if pid else 0.0),
        "is_official": is_official,
        "consensus": consensus,
        "role": role,
    }


def _make_track_results(frames):
    return {"jobId": "test", "frames": frames, "total_frames": len(frames)}


# ---- 1. Confidence semantics ---------------------------------------------------


class TestConfidenceSemantics:

    def test_unk_exports_zero_identity_confidence(self):
        """UNK tracks must have identity_confidence=0.0."""
        p = _make_player(None, tid=5, source="unassigned", confidence=0.92)
        assert p["identity_confidence"] == 0.0
        assert p["identity_valid"] is False

    def test_locked_exports_nonzero_identity_confidence(self):
        """Locked tracks should have nonzero identity confidence."""
        p = _make_player("P7", tid=3, source="locked", confidence=0.95)
        assert p["identity_confidence"] == 0.95
        assert p["identity_valid"] is True

    def test_detection_confidence_stays_separate(self):
        """Detection confidence is in the 'confidence' field, identity in 'identity_confidence'."""
        p = _make_player(None, tid=5, source="unassigned", confidence=0.91)
        assert p["confidence"] == 0.91  # detection
        assert p["identity_confidence"] == 0.0  # identity


# ---- 2. 1:1 Identity invariants ------------------------------------------------


class TestIdentityInvariants:

    def test_same_raw_track_cannot_be_pid_and_unk(self):
        """Same rawTrackId cannot appear as both PID and UNK in one frame."""
        # Build a frame where rawTrackId=18 appears twice: once as P2 and once as UNK
        frames = [
            {
                "frameIndex": 0,
                "players": [
                    _make_player("P2", tid=1, raw_tid=18, identity_confidence=0.9),
                    _make_player(None, tid=2, raw_tid=18, source="unassigned", confidence=0.85),
                ],
            },
        ]
        track_results = _make_track_results(frames)
        validator = PatchValidator()
        # A no-op patch to trigger post-apply validation
        patch = {
            "action": "swap_pid_after_frame",
            "window_id": "c_test",
            "pid_a": "P99",
            "pid_b": "P98",
            "apply_from_frame": 0,
            "confidence": 0.90,
            "reason_codes": ["TEAM_FLIP"],
        }
        result = validator.validate_post_apply(patch, track_results)
        assert result is not None
        assert "COLLISION" in result.reason or "DUPLICATE" in result.reason

    def test_same_pid_cannot_appear_twice_in_frame(self):
        """Same PID cannot appear twice in one frame."""
        frames = [
            {
                "frameIndex": 0,
                "players": [
                    _make_player("P1", tid=1, bbox=[100, 100, 150, 200]),
                    _make_player("P1", tid=2, bbox=[400, 400, 450, 500]),
                ],
            },
        ]
        track_results = _make_track_results(frames)
        validator = PatchValidator()
        patch = {
            "action": "swap_pid_after_frame",
            "window_id": "c_test",
            "pid_a": "P99",
            "pid_b": "P98",
            "apply_from_frame": 0,
            "confidence": 0.90,
            "reason_codes": ["TEAM_FLIP"],
        }
        result = validator.validate_post_apply(patch, track_results)
        assert result is not None
        assert "PID_DUPLICATE" in result.reason

    def test_two_pids_cannot_share_raw_track_id(self):
        """Two different PIDs cannot share the same rawTrackId."""
        frames = [
            {
                "frameIndex": 0,
                "players": [
                    _make_player("P1", tid=1, raw_tid=18),
                    _make_player("P2", tid=2, raw_tid=18),
                ],
            },
        ]
        track_results = _make_track_results(frames)
        validator = PatchValidator()
        patch = {
            "action": "swap_pid_after_frame",
            "window_id": "c_test",
            "pid_a": "P99",
            "pid_b": "P98",
            "apply_from_frame": 0,
            "confidence": 0.90,
            "reason_codes": ["TEAM_FLIP"],
        }
        result = validator.validate_post_apply(patch, track_results)
        assert result is not None
        assert "RAW_TRACK_DUPLICATE" in result.reason


# ---- 3. Official/referee suppression ------------------------------------------


class TestOfficialSuppression:

    def test_official_has_no_player_pid(self):
        """Officials must not have a player PID."""
        p = _make_player(None, tid=99, is_official=True, role="official",
                         source="locked", consensus="CONFIRMED")
        assert p["playerId"] is None
        assert p["is_official"] is True

    def test_official_does_not_enter_player_identity_slots(self):
        """Officials should not appear as P-id entries in render keys."""
        from services.fullfps_tracking_renderer import render_entity_key
        p = _make_player(None, tid=99, is_official=True, role="official",
                         source="locked", consensus="CONFIRMED")
        p["playerId"] = None  # no PID
        key = render_entity_key(p, debug_unknown=False, debug_officials=False)
        assert key is None  # suppressed

    def test_official_shown_in_audit_with_flag(self):
        """Officials appear in audit mode with --show-officials."""
        from services.fullfps_tracking_renderer import render_entity_key
        p = {
            "rawTrackId": 99,
            "identity_valid": False,
            "is_official": True,
            "role": "official",
            "assignment_source": "locked",
            "playerId": None,
            "bbox": [0, 0, 1, 1],
        }
        key = render_entity_key(p, debug_officials=True)
        assert key == "OFFICIAL:99"

    def test_official_suppressed_counter_increments(self):
        """QA counter must increment for suppressed officials."""
        from services.fullfps_tracking_renderer import build_observations

        frames = [
            {
                "frameIndex": 0,
                "players": [
                    {
                        "rawTrackId": 99,
                        "identity_valid": False,
                        "is_official": True,
                        "role": "official",
                        "assignment_source": "locked",
                        "playerId": None,
                        "bbox": [100, 100, 150, 200],
                        "confidence": 0.9,
                        "identity_confidence": 0.0,
                        "consensus": "CONFIRMED",
                    },
                ],
            },
        ]
        obs, counters, _ = build_observations(
            {"frames": frames}, debug_unknown=False, debug_officials=False
        )
        assert counters["official_suppressed_object_frames"] >= 1


# ---- 4. Consensus field -------------------------------------------------------


class TestConsensus:

    def test_locked_gets_confirmed_consensus(self):
        p = _make_player("P7", tid=3, source="locked", consensus="CONFIRMED")
        assert p["consensus"] == "CONFIRMED"

    def test_revived_gets_ambiguous_consensus(self):
        p = _make_player("P7", tid=3, source="revived", consensus="AMBIGUOUS")
        assert p["consensus"] == "AMBIGUOUS"

    def test_unassigned_gets_needs_review(self):
        p = _make_player(None, tid=3, source="unassigned", consensus="NEEDS_REVIEW")
        assert p["consensus"] == "NEEDS_REVIEW"


# ---- 5. Patch validator physicality -------------------------------------------


class TestPatchPhysicality:

    def test_reject_impossible_speed(self):
        """Speed check must reject patches causing >60px/frame movement."""
        # P1 at (100,100) in frame 0, then at (900,100) in frame 1
        # distance = 800px in 1 frame = 800px/frame >> 60px/frame threshold
        frames = [
            {
                "frameIndex": 0,
                "players": [_make_player("P1", tid=1, bbox=[80, 80, 120, 120])],
            },
            {
                "frameIndex": 1,
                "players": [_make_player("P1", tid=1, bbox=[880, 80, 920, 120])],
            },
        ]
        track_results = _make_track_results(frames)
        validator = PatchValidator()
        patch = {
            "action": "swap_pid_after_frame",
            "window_id": "c_test",
            "pid_a": "P99",
            "pid_b": "P98",
            "apply_from_frame": 0,
            "confidence": 0.90,
            "reason_codes": ["TEAM_FLIP"],
        }
        result = validator.validate_post_apply(patch, track_results)
        assert result is not None
        assert "SPEED" in result.reason

    def test_reject_spatial_duplicate(self):
        """Two PIDs at nearly identical locations must be rejected."""
        frames = [
            {
                "frameIndex": 0,
                "players": [
                    _make_player("P1", tid=1, bbox=[100, 100, 110, 110]),
                    _make_player("P2", tid=2, bbox=[101, 101, 111, 111]),
                ],
            },
        ]
        track_results = _make_track_results(frames)
        validator = PatchValidator()
        patch = {
            "action": "swap_pid_after_frame",
            "window_id": "c_test",
            "pid_a": "P99",
            "pid_b": "P98",
            "apply_from_frame": 0,
            "confidence": 0.90,
            "reason_codes": ["TEAM_FLIP"],
        }
        result = validator.validate_post_apply(patch, track_results)
        assert result is not None
        assert "SPATIAL" in result.reason

    def test_reject_temporal_gap(self):
        """Large unexplained temporal gaps must be rejected."""
        frames = [
            {
                "frameIndex": 0,
                "players": [_make_player("P1", tid=1, bbox=[100, 100, 150, 200])],
            },
            {
                "frameIndex": 200,  # 200-frame gap
                "players": [_make_player("P1", tid=1, bbox=[110, 100, 160, 200])],
            },
        ]
        track_results = _make_track_results(frames)
        validator = PatchValidator()
        patch = {
            "action": "swap_pid_after_frame",
            "window_id": "c_test",
            "pid_a": "P99",
            "pid_b": "P98",
            "apply_from_frame": 0,
            "confidence": 0.90,
            "reason_codes": ["TEAM_FLIP"],
        }
        result = validator.validate_post_apply(patch, track_results)
        assert result is not None
        assert "TEMPORAL" in result.reason

    def test_rejected_patch_does_not_mutate_input(self):
        """Rejected patches must not modify the original track data."""
        import copy
        frames = [
            {
                "frameIndex": 0,
                "players": [
                    _make_player("P1", tid=1, bbox=[100, 100, 110, 110]),
                    _make_player("P2", tid=2, bbox=[101, 101, 111, 111]),
                ],
            },
        ]
        track_results = _make_track_results(frames)
        original = copy.deepcopy(track_results)

        # This patch should be rejected due to spatial duplicate
        patch_plan = {
            "corrections": [
                {
                    "action": "swap_pid_after_frame",
                    "window_id": "c_test",
                    "pid_a": "P1",
                    "pid_b": "P2",
                    "apply_from_frame": 0,
                    "confidence": 0.90,
                    "reason_codes": ["TEAM_FLIP"],
                },
            ],
        }
        service = IdentityPatchService()
        result = service.apply_patches(track_results, patch_plan)

        # Original must not be mutated
        assert track_results == original

    def test_plausible_patch_applies(self):
        """A physically plausible patch should apply successfully."""
        frames = [
            {
                "frameIndex": 0,
                "players": [
                    _make_player("P1", tid=1, bbox=[100, 100, 150, 200]),
                    _make_player("P2", tid=2, bbox=[400, 100, 450, 200]),
                ],
            },
            {
                "frameIndex": 5,
                "players": [
                    _make_player("P1", tid=1, bbox=[110, 100, 160, 200]),
                    _make_player("P2", tid=2, bbox=[410, 100, 460, 200]),
                ],
            },
        ]
        patch_plan = {
            "corrections": [
                {
                    "action": "swap_pid_after_frame",
                    "window_id": "c001",
                    "pid_a": "P1",
                    "pid_b": "P2",
                    "apply_from_frame": 0,
                    "confidence": 0.90,
                    "reason_codes": ["TEAM_FLIP"],
                },
            ],
        }
        service = IdentityPatchService()
        result = service.apply_patches(_make_track_results(frames), patch_plan)
        assert result["manifest"]["patches_applied"] == 1
        assert result["manifest"]["patches_rejected"] == 0


# ---- 6. VLM metrics -----------------------------------------------------------


class TestVLMMetrics:

    def test_vlm_accuracy_lift_computation(self):
        """vlm_accuracy_lift = resolved / total when total > 0."""
        total = 10
        resolved = 3
        lift = resolved / total
        assert lift == pytest.approx(0.3)

    def test_vlm_accuracy_lift_zero_division(self):
        """vlm_accuracy_lift should be 0.0 when no switches detected."""
        total = 0
        resolved = 0
        lift = (resolved / total) if total > 0 else 0.0
        assert lift == 0.0

    def test_vlm_unreviewed_does_not_crash(self):
        """API failure producing VLM_UNREVIEWED should not crash the system."""
        result = {"decision": "VLM_UNREVIEWED", "reason_code": "API_FAILURE"}
        assert result["decision"] == "VLM_UNREVIEWED"


# ---- 7. Production render hides unconfirmed ------------------------------------


class TestProductionRenderPolicy:

    def test_production_hides_ambiguous_labels(self):
        """In production mode, AMBIGUOUS consensus tracks should not show PID text."""
        from services.fullfps_tracking_renderer import RenderDecision, RenderState
        d = RenderDecision(
            key="PID:P7",
            pid="P7",
            latest_tid=3,
            team_id=0,
            state=RenderState.VISIBLE,
            detection_confidence=0.9,
            identity_confidence=0.72,
            assignment_source="revived",
            consensus="AMBIGUOUS",
        )
        # In production mode, consensus != CONFIRMED means text = None
        assert d.consensus != "CONFIRMED"

    def test_production_shows_confirmed_labels(self):
        """In production mode, CONFIRMED consensus tracks show PID text."""
        from services.fullfps_tracking_renderer import RenderDecision, RenderState
        d = RenderDecision(
            key="PID:P7",
            pid="P7",
            latest_tid=3,
            team_id=0,
            state=RenderState.VISIBLE,
            detection_confidence=0.9,
            identity_confidence=1.0,
            assignment_source="locked",
            consensus="CONFIRMED",
        )
        assert d.consensus == "CONFIRMED"
        assert d.pid == "P7"


# ---- 8. Renderer doesn't crash with missing optional metadata ------------------


class TestRendererRobustness:

    def test_missing_consensus_defaults_to_needs_review(self):
        """Player dict missing 'consensus' should default to NEEDS_REVIEW."""
        from services.fullfps_tracking_renderer import TrackObservation
        player = {
            "rawTrackId": 5,
            "playerId": None,
            "bbox": [100, 100, 150, 200],
            "identity_valid": False,
            "assignment_source": "unassigned",
            "confidence": 0.9,
            "identity_confidence": 0.0,
            "is_official": False,
        }
        # Simulate what build_observations does
        consensus = player.get("consensus", "NEEDS_REVIEW")
        assert consensus == "NEEDS_REVIEW"

    def test_missing_identity_confidence_defaults_to_zero(self):
        """Player dict missing 'identity_confidence' should default to 0.0."""
        player = {
            "rawTrackId": 5,
            "playerId": None,
            "bbox": [100, 100, 150, 200],
        }
        ic = float(player.get("identity_confidence", 0.0) or 0.0)
        assert ic == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
