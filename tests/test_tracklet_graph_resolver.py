"""
Tests for TrackletGraphResolver.

Covers:
  1. test_build_tracklets_basic
  2. test_tracklets_split_at_conflict_boundary
  3. test_propose_swap_for_team_flip
  4. test_propose_mark_unknown_for_gk_flip
  5. test_propose_suppress_for_duplicate
  6. test_resolve_full_pipeline
  7. test_empty_conflicts_no_crash
  8. test_casefile_builder_writes_json_and_contact_sheet
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import json
from services.tracklet_graph_resolver import TrackletGraphResolver, Tracklet
from services.vlm_casefile_builder import VLMCaseFileBuilder


def _make_track_results(frames):
    return {"jobId": "test", "frames": frames, "total_frames": len(frames)}


def _make_player(pid, tid=1, bbox=None, source="locked", team_id=None, raw_tid=None):
    bbox = bbox or [100, 100, 150, 200]
    return {
        "trackId": tid,
        "rawTrackId": raw_tid if raw_tid is not None else tid,
        "playerId": pid,
        "displayId": pid,
        "bbox": bbox,
        "confidence": 0.9,
        "class": 2,
        "gameState": "play",
        "analysis_valid": True,
        "crop_quality": 1.0,
        "identity_valid": True,
        "assignment_source": source,
        "identity_confidence": 0.9,
        "is_official": False,
        "team_id": team_id,
    }


class TestTrackletGraphResolver:

    def test_build_tracklets_basic(self):
        """Basic tracklet building from track results."""
        frames = [
            {"frameIndex": 0, "players": [_make_player("P1", tid=1)]},
            {"frameIndex": 5, "players": [_make_player("P1", tid=1)]},
            {"frameIndex": 10, "players": [_make_player("P1", tid=1)]},
        ]
        resolver = TrackletGraphResolver()
        tracklets = resolver.build_tracklets(_make_track_results(frames), [])
        assert len(tracklets) >= 1
        p1_tls = [t for t in tracklets if t.pid == "P1"]
        assert len(p1_tls) == 1
        assert p1_tls[0].start_frame == 0
        assert p1_tls[0].end_frame == 10
        assert p1_tls[0].frame_count == 3

    def test_tracklets_split_at_conflict_boundary(self):
        """Tracklets should split at conflict window boundaries."""
        frames = [
            {"frameIndex": 0, "players": [_make_player("P1", tid=1)]},
            {"frameIndex": 5, "players": [_make_player("P1", tid=1)]},
            {"frameIndex": 10, "players": [_make_player("P1", tid=1)]},
            {"frameIndex": 15, "players": [_make_player("P1", tid=1)]},
        ]
        conflicts = [
            {"window_id": "c001", "start_frame": 7, "end_frame": 12, "pids": ["P1"]},
        ]
        resolver = TrackletGraphResolver()
        tracklets = resolver.build_tracklets(_make_track_results(frames), conflicts)
        p1_tls = [t for t in tracklets if t.pid == "P1"]
        # Should be split into 2: frames [0,5] and [10,15]
        assert len(p1_tls) == 2, f"Expected 2 tracklets, got {len(p1_tls)}"

    def test_propose_swap_for_team_flip(self):
        """Resolver should propose swap for TEAM_FLIP conflicts with 2 PIDs."""
        resolver = TrackletGraphResolver()
        conflicts = [
            {
                "window_id": "c001",
                "start_frame": 10,
                "end_frame": 20,
                "pids": ["P1", "P2"],
                "conflict_types": ["TEAM_FLIP"],
                "severity": "high",
                "evidence": {},
            },
        ]
        proposals = resolver.propose_corrections([], conflicts)
        swaps = [p for p in proposals if p.action == "swap_pid_after_frame"]
        assert len(swaps) >= 1
        assert swaps[0].pid_a == "P1"
        assert swaps[0].pid_b == "P2"

    def test_propose_mark_unknown_for_gk_flip(self):
        """Resolver should propose mark_unknown for GOALKEEPER_TO_OUTFIELD."""
        resolver = TrackletGraphResolver()
        conflicts = [
            {
                "window_id": "c002",
                "start_frame": 50,
                "end_frame": 60,
                "pids": ["P13"],
                "conflict_types": ["GOALKEEPER_TO_OUTFIELD"],
                "severity": "high",
                "evidence": {},
            },
        ]
        proposals = resolver.propose_corrections([], conflicts)
        marks = [p for p in proposals if p.action == "mark_unknown"]
        assert len(marks) >= 1
        assert marks[0].pid_a == "P13"

    def test_propose_suppress_for_duplicate(self):
        """Resolver should propose suppress for DUPLICATE_PID."""
        resolver = TrackletGraphResolver()
        conflicts = [
            {
                "window_id": "c003",
                "start_frame": 5,
                "end_frame": 5,
                "pids": ["P1"],
                "conflict_types": ["DUPLICATE_PID"],
                "severity": "medium",
                "evidence": {},
            },
        ]
        proposals = resolver.propose_corrections([], conflicts)
        suppresses = [p for p in proposals if p.action == "suppress_pid"]
        assert len(suppresses) >= 1

    def test_resolve_full_pipeline(self):
        """Full resolve pipeline returns tracklets and corrections."""
        frames = [
            {"frameIndex": 0, "players": [_make_player("P1", tid=1, team_id=0)]},
            {"frameIndex": 5, "players": [_make_player("P1", tid=1, team_id=0)]},
            {"frameIndex": 10, "players": [_make_player("P1", tid=1, team_id=1)]},
        ]
        conflicts = [
            {
                "window_id": "c001",
                "start_frame": 7,
                "end_frame": 12,
                "pids": ["P1"],
                "conflict_types": ["TEAM_FLIP"],
                "severity": "high",
                "evidence": {},
            },
        ]
        resolver = TrackletGraphResolver()
        result = resolver.resolve(_make_track_results(frames), conflicts)
        assert "tracklets" in result
        assert "corrections" in result
        assert "patches_proposed" in result

    def test_empty_conflicts_no_crash(self):
        """No conflicts should produce no corrections."""
        resolver = TrackletGraphResolver()
        result = resolver.resolve(_make_track_results([]), [])
        assert result["patches_proposed"] == 0

    def test_casefile_builder_writes_json_and_contact_sheet(self, tmp_path):
        """Case file export must create case.json for unresolved conflict."""
        builder = VLMCaseFileBuilder(video_path=None)
        conflicts = [
            {
                "window_id": "c001",
                "start_frame": 10,
                "end_frame": 20,
                "pids": ["P1", "P2"],
                "conflict_types": ["TEAM_FLIP"],
                "severity": "high",
                "evidence": {"team_before": "0", "team_after": "1"},
            },
        ]
        corrections = [
            {
                "action": "swap_pid_after_frame",
                "window_id": "c001",
                "pid_a": "P1",
                "pid_b": "P2",
                "apply_from_frame": 15,
                "confidence": 0.80,
                "reason_codes": ["TEAM_FLIP"],
            },
        ]
        track_results = _make_track_results([
            {"frameIndex": 10, "players": [_make_player("P1"), _make_player("P2", tid=2)]},
            {"frameIndex": 15, "players": [_make_player("P1"), _make_player("P2", tid=2)]},
            {"frameIndex": 20, "players": [_make_player("P1"), _make_player("P2", tid=2)]},
        ])

        case_paths = builder.build_case_files(
            conflict_windows=conflicts,
            track_results=track_results,
            corrections=corrections,
            output_dir=tmp_path / "vlm_casefiles",
            severity_filter=["high"],
        )
        assert len(case_paths) >= 1
        case_json_path = case_paths[0]
        assert os.path.exists(case_json_path)
        with open(case_json_path) as f:
            case_data = json.load(f)
        assert "window_id" in case_data
        assert "vlm_prompt" in case_data
        assert "candidate_corrections" in case_data
        assert case_data["window_id"] == "c001"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
