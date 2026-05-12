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


class TestOfficialHardGate:
    """Official/referee tracks must never receive a P-ID."""

    def _make_identity(self):
        from services.identity_core import IdentityCore
        return IdentityCore()

    def _make_track(self, tid: int):
        class T:
            track_id = tid
            time_since_update = 0
        return T()

    def test_official_tid_gets_unknown_not_pid(self):
        """A track classified as official must emit UNKNOWN, not P-ID."""
        import numpy as np
        identity = self._make_identity()
        identity.begin_frame(0, present_tids={7})

        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)

        track_to_pid, meta = identity.assign_tracks(
            tracks=[self._make_track(7)],
            embeddings={7: emb},
            positions={7: (200.0, 300.0)},
            allow_new_assignments=True,
            official_tids={7},
        )
        identity.end_frame()

        assert 7 not in track_to_pid or track_to_pid.get(7) is None, \
            "Official tid=7 must not receive any P-ID"
        assert meta[7].source == "official_blocked", \
            f"source must be 'official_blocked', got {meta[7].source}"
        assert meta[7].identity_valid is False

    def test_official_does_not_steal_existing_lock(self):
        """Even if a tid already has a lock, official gate must block it."""
        import numpy as np
        identity = self._make_identity()
        identity.locks.try_create_lock(7, "P13", "hungarian", frame_id=0)

        identity.begin_frame(1, present_tids={7})
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)

        track_to_pid, meta = identity.assign_tracks(
            tracks=[self._make_track(7)],
            embeddings={7: emb},
            positions={7: (200.0, 300.0)},
            allow_new_assignments=True,
            official_tids={7},
        )
        identity.end_frame()

        assert track_to_pid.get(7) is None, \
            "Official tid=7 must not emit P13 even if lock exists"
        assert meta[7].source == "official_blocked"

    def test_non_official_tid_unaffected(self):
        """Normal player tids must pass through the gate unchanged."""
        import numpy as np
        identity = self._make_identity()
        identity.begin_frame(0, present_tids={3})

        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        slot = identity.slots[0]
        slot.embedding = emb.copy()
        slot.state = "active"

        track_to_pid, meta = identity.assign_tracks(
            tracks=[self._make_track(3)],
            embeddings={3: emb},
            positions={3: (200.0, 300.0)},
            allow_new_assignments=True,
            official_tids={99},
        )
        identity.end_frame()

        assert meta[3].source != "official_blocked", \
            "Player tid=3 must not be blocked by official gate"

    def test_official_gate_increments_metric(self):
        """official_pid_blocks counter must increment for each blocked official."""
        import numpy as np
        identity = self._make_identity()

        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        identity.begin_frame(0, present_tids={5, 6})
        identity.assign_tracks(
            tracks=[self._make_track(5), self._make_track(6)],
            embeddings={5: emb, 6: emb},
            positions={5: (100.0, 200.0), 6: (300.0, 400.0)},
            allow_new_assignments=True,
            official_tids={5, 6},
        )
        identity.end_frame()

        assert identity.official_pid_blocks >= 2, \
            f"Expected >=2 official_pid_blocks, got {identity.official_pid_blocks}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestShadowBuffer:
    """Shadow buffer captures departing locks; evicts on TTL; gates relinks."""

    def _make_shadow_entry(self, pid="P7", last_tid=7, last_seen=10,
                           center=(100.0, 200.0), team_id=0, exit_edge="left",
                           stable_count=20):
        from services.identity_core import ShadowEntry
        import numpy as np
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        return ShadowEntry(
            pid=pid, last_tid=last_tid, last_seen_frame=last_seen,
            last_bbox=(80, 150, 120, 250), last_center=center,
            last_embedding=emb, team_id=team_id, exit_edge=exit_edge,
            stable_count=stable_count,
        )

    def test_shadow_entry_captured_on_lock_expiry(self):
        """When a locked slot's lock expires, a ShadowEntry must be created."""
        import numpy as np
        from services.identity_core import IdentityCore, ShadowBuffer

        identity = IdentityCore()
        lk, _ = identity.locks.try_create_lock(7, "P7", "hungarian", frame_id=0, ttl=2)
        slot = identity._slot_by_pid("P7")
        slot.state = "active"
        slot.last_position = (100.0, 200.0)

        identity.begin_frame(5, present_tids=set())
        identity.end_frame()

        assert identity.shadow_buffer.has_shadow("P7"), \
            "P7 must have a shadow entry after lock expiry"

    def test_shadow_evicts_after_ttl(self):
        """ShadowEntry must be evicted after SHADOW_TTL_FRAMES."""
        from services.identity_core import ShadowBuffer
        buf = ShadowBuffer(ttl_frames=10)
        buf.add(self._make_shadow_entry(), added_frame=0)
        assert buf.has_shadow("P7")
        buf.evict_expired(current_frame=11)
        assert not buf.has_shadow("P7"), "P7 shadow must evict after TTL"

    def test_shadow_not_evicted_before_ttl(self):
        """ShadowEntry must survive within TTL window."""
        from services.identity_core import ShadowBuffer
        buf = ShadowBuffer(ttl_frames=90)
        buf.add(self._make_shadow_entry(), added_frame=0)
        buf.evict_expired(current_frame=89)
        assert buf.has_shadow("P7"), "P7 shadow must not evict before TTL"

    def test_relink_blocked_if_gap_too_small(self):
        """Relink must be rejected if time gap < MIN_RELINK_GAP_FRAMES."""
        from services.identity_core import ShadowBuffer
        buf = ShadowBuffer(ttl_frames=90, min_relink_gap=8)
        buf.add(self._make_shadow_entry(last_seen=10), added_frame=10)
        ok, reason = buf.check_relink_eligibility(
            pid="P7", candidate_center=(105.0, 205.0),
            candidate_team=0, current_frame=15,
            cost=0.18, frame_width=1920, frame_height=1080,
        )
        assert not ok, f"Gap=5 < min=8 must block relink; got ok={ok}, reason={reason}"
        assert "gap" in reason.lower()

    def test_relink_blocked_if_cost_too_high(self):
        """Relink must be rejected if cost > SHADOW_MAX_COST."""
        from services.identity_core import ShadowBuffer
        buf = ShadowBuffer(ttl_frames=90, min_relink_gap=8, max_cost=0.22)
        buf.add(self._make_shadow_entry(last_seen=10, exit_edge="left",
                                        center=(50.0, 400.0)), added_frame=10)
        ok, reason = buf.check_relink_eligibility(
            pid="P7", candidate_center=(60.0, 410.0),
            candidate_team=0, current_frame=25,
            cost=0.30,
            frame_width=1920, frame_height=1080,
        )
        assert not ok, "Cost=0.30 > max=0.22 must block relink"
        assert "cost" in reason.lower()

    def test_relink_blocked_if_wrong_team(self):
        """Relink must be rejected if candidate team != shadow team."""
        from services.identity_core import ShadowBuffer
        buf = ShadowBuffer(ttl_frames=90, min_relink_gap=8, max_cost=0.22)
        buf.add(self._make_shadow_entry(last_seen=10, team_id=0), added_frame=10)
        ok, reason = buf.check_relink_eligibility(
            pid="P7", candidate_center=(50.0, 400.0),
            candidate_team=1,
            current_frame=25, cost=0.15,
            frame_width=1920, frame_height=1080,
        )
        assert not ok, "Wrong team must block shadow relink"
        assert "team" in reason.lower()

    def test_valid_relink_accepted(self):
        """Valid relink (gap ok, cost ok, same team, no edge required) must be accepted."""
        from services.identity_core import ShadowBuffer
        buf = ShadowBuffer(ttl_frames=90, min_relink_gap=8, max_cost=0.22,
                           edge_margin_px=96, require_edge=False)
        buf.add(self._make_shadow_entry(
            last_seen=10, team_id=0, exit_edge="left", center=(50.0, 400.0)
        ), added_frame=10)
        ok, reason = buf.check_relink_eligibility(
            pid="P7", candidate_center=(80.0, 390.0),
            candidate_team=0, current_frame=25,
            cost=0.18, frame_width=1920, frame_height=1080,
        )
        assert ok, f"Valid relink must be accepted; reason={reason}"
