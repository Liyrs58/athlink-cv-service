"""
Tests for IdentityConflictDetector.

Covers:
  1. test_team_flip_creates_conflict
  2. test_goalkeeper_to_outfield_creates_conflict
  3. test_duplicate_pid_creates_conflict
  4. test_overlap_gt_30_creates_conflict_window
  5. test_camera_compensated_jump_creates_conflict
  6. test_implausible_speed_creates_conflict
  7. test_identity_source_downgrade_creates_conflict
  8. test_rev_relock_after_occlusion
  9. test_pid_teleport_creates_conflict
  10. test_empty_frames_no_crash
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from services.identity_conflict_detector import (
    IdentityConflictDetector,
    ConflictType,
    _bbox_iou,
    _bbox_center,
)


def _make_track_results(frames):
    """Helper to wrap frame data in the track_results structure."""
    return {"jobId": "test", "frames": frames, "total_frames": len(frames)}


def _make_player(
    pid, tid=1, bbox=None, team_id=None, role=None,
    source="locked", confidence=1.0, is_official=False,
    raw_tid=None,
):
    """Convenience player builder."""
    bbox = bbox or [100, 100, 150, 200]
    p = {
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
        "identity_valid": True,
        "assignment_source": source,
        "identity_confidence": confidence,
        "is_official": is_official,
    }
    if team_id is not None:
        p["team_id"] = team_id
    if role is not None:
        p["role"] = role
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConflictDetector:

    def test_team_flip_creates_conflict(self):
        """A PID with team_id 0 for early frames and team_id 1 later must emit TEAM_FLIP."""
        frames = [
            {"frameIndex": 0, "players": [_make_player("P1", tid=1, team_id=0)]},
            {"frameIndex": 5, "players": [_make_player("P1", tid=1, team_id=0)]},
            {"frameIndex": 10, "players": [_make_player("P1", tid=1, team_id=1)]},
        ]
        detector = IdentityConflictDetector()
        conflicts = detector.detect(_make_track_results(frames))

        team_flips = [c for c in conflicts if ConflictType.TEAM_FLIP.value in c.conflict_types]
        assert len(team_flips) >= 1, f"Expected TEAM_FLIP conflict, got: {[c.conflict_types for c in conflicts]}"
        assert team_flips[0].pids == ["P1"]
        assert team_flips[0].severity == "high"

    def test_goalkeeper_to_outfield_creates_conflict(self):
        """A PID changing role goalkeeper -> player must emit GOALKEEPER_TO_OUTFIELD."""
        frames = [
            {"frameIndex": 0, "players": [_make_player("P13", tid=13, role="goalkeeper")]},
            {"frameIndex": 10, "players": [_make_player("P13", tid=13, role="player")]},
        ]
        detector = IdentityConflictDetector()
        conflicts = detector.detect(_make_track_results(frames))

        gk_flips = [c for c in conflicts if ConflictType.GOALKEEPER_TO_OUTFIELD.value in c.conflict_types]
        assert len(gk_flips) >= 1, f"Expected GOALKEEPER_TO_OUTFIELD, got: {[c.conflict_types for c in conflicts]}"
        assert gk_flips[0].severity == "high"

    def test_duplicate_pid_creates_conflict(self):
        """Two boxes with same PID in one frame must emit DUPLICATE_PID."""
        frames = [
            {
                "frameIndex": 5,
                "players": [
                    _make_player("P1", tid=1, bbox=[100, 100, 150, 200]),
                    _make_player("P1", tid=2, bbox=[500, 100, 550, 200], raw_tid=2),
                ],
            },
        ]
        detector = IdentityConflictDetector()
        conflicts = detector.detect(_make_track_results(frames))

        dups = [c for c in conflicts if ConflictType.DUPLICATE_PID.value in c.conflict_types]
        assert len(dups) >= 1, f"Expected DUPLICATE_PID, got: {[c.conflict_types for c in conflicts]}"
        assert dups[0].pids == ["P1"]
        assert dups[0].severity == "medium"

    def test_overlap_gt_30_creates_conflict_window(self):
        """Two bboxes with IoU > 0.30 must create OVERLAP_GT_0_30."""
        # Overlapping bboxes
        frames = [
            {
                "frameIndex": 10,
                "players": [
                    _make_player("P1", tid=1, bbox=[100, 100, 200, 200]),
                    _make_player("P2", tid=2, bbox=[130, 100, 230, 200]),
                ],
            },
        ]
        # Verify IoU is > 0.30
        iou = _bbox_iou([100, 100, 200, 200], [130, 100, 230, 200])
        assert iou > 0.30, f"IoU={iou} should be > 0.30 for this test"

        detector = IdentityConflictDetector()
        conflicts = detector.detect(_make_track_results(frames))

        overlaps = [c for c in conflicts if ConflictType.OVERLAP_GT_0_30.value in c.conflict_types]
        assert len(overlaps) >= 1, f"Expected OVERLAP_GT_0_30, got: {[c.conflict_types for c in conflicts]}"
        assert set(overlaps[0].pids) == {"P1", "P2"}

    def test_camera_compensated_jump_creates_conflict(self):
        """Large centroid jump after camera_motion compensation must emit CAMERA_COMPENSATED_CENTROID_JUMP."""
        frames = [
            {"frameIndex": 0, "players": [_make_player("P1", tid=1, bbox=[100, 100, 150, 200])]},
            {"frameIndex": 5, "players": [_make_player("P1", tid=1, bbox=[500, 500, 550, 600])]},
        ]
        # Camera motion that would NOT explain the jump
        camera_motion = {
            "frames": [
                {"frame": 5, "dx": 10.0, "dy": 5.0, "motion_class": "fast_pan"},
            ]
        }
        detector = IdentityConflictDetector(centroid_jump_threshold=150.0)
        conflicts = detector.detect(_make_track_results(frames), camera_motion=camera_motion)

        jumps = [c for c in conflicts if ConflictType.CAMERA_COMPENSATED_CENTROID_JUMP.value in c.conflict_types]
        assert len(jumps) >= 1, f"Expected CAMERA_COMPENSATED_CENTROID_JUMP, got: {[c.conflict_types for c in conflicts]}"
        # Verify evidence has motion_class
        assert jumps[0].evidence.get("motion_class") == "fast_pan"

    def test_implausible_speed_creates_conflict(self):
        """Centroid moving too fast between frames must emit IMPLAUSIBLE_SPEED."""
        # 500px movement in 1 frame = 500 px/frame → way over threshold
        frames = [
            {"frameIndex": 0, "players": [_make_player("P1", tid=1, bbox=[100, 100, 110, 110])]},
            {"frameIndex": 1, "players": [_make_player("P1", tid=1, bbox=[600, 100, 610, 110])]},
        ]
        detector = IdentityConflictDetector(speed_threshold_px_per_frame=60.0)
        conflicts = detector.detect(_make_track_results(frames))

        speed_conflicts = [c for c in conflicts if ConflictType.IMPLAUSIBLE_SPEED.value in c.conflict_types]
        assert len(speed_conflicts) >= 1

    def test_identity_source_downgrade_creates_conflict(self):
        """Locked -> unassigned is a source downgrade."""
        frames = [
            {"frameIndex": 0, "players": [_make_player("P1", tid=1, source="locked")]},
            {"frameIndex": 5, "players": [_make_player("P1", tid=1, source="unassigned")]},
        ]
        detector = IdentityConflictDetector()
        conflicts = detector.detect(_make_track_results(frames))

        downgrades = [c for c in conflicts if ConflictType.IDENTITY_SOURCE_DOWNGRADE.value in c.conflict_types]
        assert len(downgrades) >= 1

    def test_rev_relock_after_occlusion(self):
        """REV -> LOCK within 10 frames is suspicious."""
        frames = [
            {"frameIndex": 0, "players": [_make_player("P1", tid=1, source="revived")]},
            {"frameIndex": 5, "players": [_make_player("P1", tid=1, source="locked")]},
        ]
        detector = IdentityConflictDetector()
        conflicts = detector.detect(_make_track_results(frames))

        relocks = [c for c in conflicts if ConflictType.REV_RELOCK_AFTER_OCCLUSION.value in c.conflict_types]
        assert len(relocks) >= 1

    def test_pid_teleport_creates_conflict(self):
        """rawTrackId change with large distance must emit PID_TELEPORT."""
        frames = [
            {"frameIndex": 0, "players": [_make_player("P1", tid=1, bbox=[100, 100, 150, 200], raw_tid=10)]},
            {"frameIndex": 5, "players": [_make_player("P1", tid=2, bbox=[600, 600, 650, 700], raw_tid=20)]},
        ]
        detector = IdentityConflictDetector(centroid_jump_threshold=150.0)
        conflicts = detector.detect(_make_track_results(frames))

        teleports = [c for c in conflicts if ConflictType.PID_TELEPORT.value in c.conflict_types]
        assert len(teleports) >= 1
        assert teleports[0].severity == "high"

    def test_empty_frames_no_crash(self):
        """Empty frames should not crash."""
        frames = [
            {"frameIndex": 0, "players": []},
            {"frameIndex": 5, "players": []},
        ]
        detector = IdentityConflictDetector()
        conflicts = detector.detect(_make_track_results(frames))
        assert conflicts == []

    def test_bbox_iou_basic(self):
        """Basic IoU sanity check."""
        assert _bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
        assert _bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]) == pytest.approx(0.0)
        assert _bbox_iou([0, 0, 10, 10], [5, 0, 15, 10]) > 0.0

    def test_detect_and_save(self, tmp_path):
        """detect_and_save writes valid JSON."""
        frames = [
            {
                "frameIndex": 5,
                "players": [
                    _make_player("P1", tid=1),
                    _make_player("P1", tid=2, raw_tid=2),
                ],
            },
        ]
        detector = IdentityConflictDetector()
        out = tmp_path / "conflicts.json"
        manifest = detector.detect_and_save(_make_track_results(frames), out)
        assert out.exists()
        assert manifest["conflict_windows_total"] >= 1
        assert manifest["duplicate_pid_conflicts"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
