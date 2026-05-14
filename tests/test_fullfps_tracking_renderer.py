"""Tests for services/fullfps_tracking_renderer.py."""

import pytest
import numpy as np

from services.fullfps_tracking_renderer import (
    MAX_HOLD_RAW_FRAMES,
    PAN_LABEL_FREEZE_FRAMES,
    HideReason,
    RenderState,
    _color_thread_points_for_frame,
    _draw_color_thread_raw_boxes,
    _draw_color_threads,
    build_cumulative_offset,
    build_color_thread_runtime_index,
    build_manifest,
    build_observations,
    load_color_threads,
    render_entity_key,
    render_frames,
    tracking_frame_range,
)


def _player(**overrides):
    base = {
        "trackId": 1,
        "rawTrackId": 1,
        "playerId": None,
        "displayId": None,
        "bbox": [100.0, 200.0, 140.0, 320.0],
        "confidence": 0.9,
        "identity_valid": False,
        "assignment_source": "unassigned",
        "identity_confidence": 0.0,
        "is_official": False,
    }
    base.update(overrides)
    return base


def _frame(frame_index, players=None, officials=None):
    return {
        "frameIndex": frame_index,
        "players": players or [],
        "officials": officials or [],
    }


# ---- Entity key tests ----------------------------------------------------------


def test_entity_key_prefers_pid_over_tid():
    p = _player(playerId="P7", rawTrackId=11, identity_valid=True, assignment_source="locked")
    assert render_entity_key(p) == "PID:P7"


def test_unknown_hidden_by_default():
    p = _player(identity_valid=False, rawTrackId=12)
    assert render_entity_key(p) is None
    assert render_entity_key(p, debug_unknown=True) == "TID:12"


def test_hungarian_identity_hidden_by_default():
    p = _player(playerId="P7", identity_valid=True, assignment_source="hungarian", rawTrackId=11)
    assert render_entity_key(p) is None
    # Even in debug_unknown, hungarian must not enter PID namespace.
    assert render_entity_key(p, debug_unknown=True) == "TID:11"


def test_provisional_identity_hidden_by_default():
    p = _player(playerId="P9", identity_valid=True, assignment_source="provisional", rawTrackId=5)
    assert render_entity_key(p) is None


def test_revived_renders_as_pid():
    p = _player(playerId="P3", identity_valid=True, assignment_source="revived", rawTrackId=8)
    assert render_entity_key(p) == "PID:P3"


def test_official_dropped_before_entity_key():
    p = _player(playerId="P7", identity_valid=True, assignment_source="locked", is_official=True, rawTrackId=20)
    assert render_entity_key(p) is None
    assert render_entity_key(p, debug_officials=True) == "OFFICIAL:20"


def test_tid_fallback_reads_either_field():
    # Some code paths emit trackId not rawTrackId.
    p = {"trackId": 99, "identity_valid": False, "is_official": False, "bbox": [0, 0, 1, 1]}
    assert render_entity_key(p, debug_unknown=True) == "TID:99"


# ---- Camera-motion timeline ----------------------------------------------------


def test_camera_motion_cumulative_offset_interpolation():
    # dx are deltas between samples; frame-0 dx is ignored as baseline.
    motions = [
        {"frameIndex": 0, "dx": 10.0, "dy": 0.0, "motion_class": "stable"},
        {"frameIndex": 5, "dx": 10.0, "dy": 0.0, "motion_class": "pan"},
        {"frameIndex": 10, "dx": 10.0, "dy": 0.0, "motion_class": "pan"},
    ]
    offset = build_cumulative_offset(motions, total_raw_frames=11)
    assert offset[0] == (0.0, 0.0)
    assert offset[5] == pytest.approx((10.0, 0.0))
    assert offset[10] == pytest.approx((20.0, 0.0))
    cx7, cy7 = offset[7]
    assert cx7 == pytest.approx(14.0, abs=0.01)
    assert cy7 == pytest.approx(0.0, abs=0.01)


def test_camera_motion_cut_resets_segment():
    motions = [
        {"frameIndex": 0, "dx": 0.0, "dy": 0.0, "motion_class": "stable"},
        {"frameIndex": 5, "dx": 10.0, "dy": 0.0, "motion_class": "pan"},
        {"frameIndex": 10, "dx": 200.0, "dy": 0.0, "motion_class": "cut"},
        {"frameIndex": 15, "dx": 5.0, "dy": 0.0, "motion_class": "pan"},
    ]
    offset = build_cumulative_offset(motions, total_raw_frames=16)
    # Pre-cut: cumulative at 5 = 10
    assert offset[5] == pytest.approx((10.0, 0.0))
    # Cut resets the segment at frame 10.
    assert offset[10] == (0.0, 0.0)
    # Post-cut: cumulative at 15 = 5 (relative to cut baseline)
    assert offset[15] == pytest.approx((5.0, 0.0))


def test_camera_motion_missing_falls_back_to_zero():
    offset = build_cumulative_offset([], total_raw_frames=5)
    for f in range(5):
        assert offset[f] == (0.0, 0.0)


# ---- Render loop ---------------------------------------------------------------


def _basic_tracking(frames):
    return {"frames": frames}


def test_visible_takes_precedence_over_interpolated():
    bbox = [100.0, 100.0, 140.0, 220.0]
    frames = [
        _frame(0, [_player(playerId="P1", rawTrackId=1, identity_valid=True,
                           assignment_source="locked", bbox=bbox)]),
        _frame(5, [_player(playerId="P1", rawTrackId=1, identity_valid=True,
                           assignment_source="locked", bbox=bbox)]),
        _frame(10, [_player(playerId="P1", rawTrackId=1, identity_valid=True,
                            assignment_source="locked", bbox=bbox)]),
    ]
    decisions = render_frames(
        _basic_tracking(frames),
        motions=[],
        total_raw_frames=11,
        frame_dims=(1920, 1080),
    )
    # At frame 5, a real sample exists: must be VISIBLE, not INTERPOLATED.
    f5 = [d for d in decisions[5] if d.key == "PID:P1"]
    assert f5, "expected PID:P1 at frame 5"
    assert f5[0].state == RenderState.VISIBLE


def test_interpolation_between_samples_is_interpolated():
    bbox_a = [100.0, 100.0, 140.0, 220.0]
    bbox_b = [200.0, 100.0, 240.0, 220.0]
    frames = [
        _frame(0, [_player(playerId="P1", rawTrackId=1, identity_valid=True,
                           assignment_source="locked", bbox=bbox_a)]),
        _frame(5, [_player(playerId="P1", rawTrackId=1, identity_valid=True,
                           assignment_source="locked", bbox=bbox_b)]),
    ]
    decisions = render_frames(_basic_tracking(frames), motions=[], total_raw_frames=6,
                              frame_dims=(1920, 1080))
    f2 = [d for d in decisions[2] if d.key == "PID:P1"]
    assert f2 and f2[0].state == RenderState.INTERPOLATED


def test_no_interpolation_across_cut():
    bbox = [100.0, 100.0, 140.0, 220.0]
    frames = [
        _frame(0, [_player(playerId="P1", rawTrackId=1, identity_valid=True,
                           assignment_source="locked", bbox=bbox)]),
        _frame(10, [_player(playerId="P1", rawTrackId=1, identity_valid=True,
                            assignment_source="locked", bbox=bbox)]),
    ]
    motions = [
        {"frameIndex": 0, "dx": 0.0, "dy": 0.0, "motion_class": "stable"},
        {"frameIndex": 5, "dx": 0.0, "dy": 0.0, "motion_class": "cut"},
        {"frameIndex": 10, "dx": 0.0, "dy": 0.0, "motion_class": "stable"},
    ]
    decisions = render_frames(_basic_tracking(frames), motions=motions, total_raw_frames=11,
                              frame_dims=(1920, 1080))
    # Frames 6..9 are inside the cut segment; expect no visible PID:P1.
    for f in (6, 7, 8, 9):
        visible = [d for d in decisions[f] if d.key == "PID:P1" and d.state != RenderState.HIDDEN]
        assert not visible, f"expected no visible entity at frame {f} across cut"


def test_camera_stabilized_interpolation_during_pan():
    # Player stationary in stabilised space; camera moves +50px right between samples.
    # Bbox at frame 0 is at x=100, bbox at frame 5 is at x=150 (player moved with camera).
    # In raw pixel space the bbox appears to translate +50; in stabilised space it's stationary.
    # At frame 2 (40% of the way), cumulative offset is +20px; rendered bbox center
    # should land at ~120 (stab_center 100 + curr_off 20).
    frames = [
        _frame(0, [_player(playerId="P1", rawTrackId=1, identity_valid=True,
                           assignment_source="locked",
                           bbox=[80.0, 200.0, 120.0, 320.0])]),  # center x=100
        _frame(5, [_player(playerId="P1", rawTrackId=1, identity_valid=True,
                           assignment_source="locked",
                           bbox=[130.0, 200.0, 170.0, 320.0])]),  # center x=150
    ]
    motions = [
        {"frameIndex": 0, "dx": 0.0, "dy": 0.0, "motion_class": "stable"},
        {"frameIndex": 5, "dx": 50.0, "dy": 0.0, "motion_class": "fast_pan"},
    ]
    decisions = render_frames(_basic_tracking(frames), motions=motions, total_raw_frames=6,
                              frame_dims=(1920, 1080))
    f2 = [d for d in decisions[2] if d.key == "PID:P1" and d.state == RenderState.INTERPOLATED]
    assert f2, "expected interpolated PID:P1 at frame 2"
    bb = f2[0].bbox
    cx = (bb[0] + bb[2]) / 2.0
    # Stabilised: 100 across; raw at frame 2 = 100 + offset(frame2)=20 -> 120.
    assert cx == pytest.approx(120.0, abs=1.5), f"expected raw cx~120, got {cx}"


def test_duplicate_pid_suppressed():
    # Two entities both labelled P7 at the same frame; lower confidence loses.
    bbox_a = [100.0, 100.0, 140.0, 220.0]
    bbox_b = [400.0, 100.0, 440.0, 220.0]
    frames = [
        _frame(0, [
            _player(playerId="P7", rawTrackId=1, identity_valid=True,
                    assignment_source="locked", bbox=bbox_a, identity_confidence=0.95),
            _player(playerId="P7", rawTrackId=2, identity_valid=True,
                    assignment_source="locked", bbox=bbox_b, identity_confidence=0.40),
        ]),
    ]
    decisions = render_frames(_basic_tracking(frames), motions=[], total_raw_frames=1,
                              frame_dims=(1920, 1080))
    surviving = [d for d in decisions[0] if d.pid == "P7" and d.state != RenderState.HIDDEN]
    suppressed = [d for d in decisions[0] if d.hidden_reason == HideReason.DUPLICATE_PID_SUPPRESSED]
    assert len(surviving) == 1
    assert len(suppressed) == 1
    # The kept one must be the higher-confidence (tid=1).
    assert surviving[0].latest_tid == 1


def test_hold_expires_after_max_hold_frames():
    bbox = [100.0, 100.0, 140.0, 220.0]
    # Only one observation at frame 0; nothing after. Should hold up to MAX_HOLD_RAW_FRAMES, then hide.
    frames = [
        _frame(0, [_player(playerId="P1", rawTrackId=1, identity_valid=True,
                           assignment_source="locked", bbox=bbox)]),
    ]
    total = MAX_HOLD_RAW_FRAMES + 3
    decisions = render_frames(_basic_tracking(frames), motions=[], total_raw_frames=total,
                              frame_dims=(1920, 1080))
    # Within the hold window: HELD or VISIBLE
    for f in range(0, MAX_HOLD_RAW_FRAMES + 1):
        kept = [d for d in decisions[f] if d.key == "PID:P1" and d.state in
                (RenderState.VISIBLE, RenderState.HELD)]
        assert kept, f"expected entity within hold window at frame {f}"
    # Past the hold window: hidden with HOLD_EXPIRED
    expired_frame = MAX_HOLD_RAW_FRAMES + 1
    expired = [d for d in decisions[expired_frame]
               if d.key == "PID:P1" and d.hidden_reason == HideReason.HOLD_EXPIRED]
    assert expired, f"expected HOLD_EXPIRED at frame {expired_frame}"


# ---- Manifest invariants -------------------------------------------------------


def test_manifest_invariants(tmp_path):
    bbox = [100.0, 100.0, 140.0, 220.0]
    frames = [
        _frame(0, [_player(playerId="P1", rawTrackId=1, identity_valid=True,
                           assignment_source="locked", bbox=bbox)]),
        _frame(5, [_player(playerId="P1", rawTrackId=1, identity_valid=True,
                           assignment_source="locked", bbox=bbox)]),
    ]
    decisions = render_frames(_basic_tracking(frames), motions=[], total_raw_frames=6,
                              frame_dims=(1920, 1080))
    manifest = build_manifest(
        decisions=decisions,
        total_raw_frames=6,
        source_video="/dev/null",
        output_video=str(tmp_path / "out.mp4"),
        source_fps=30.0,
        identity_metrics={},
        motions=[],
        tracking_results=_basic_tracking(frames),
    )
    assert manifest["rendered_frames"] == manifest["total_raw_frames"]
    assert manifest["new_identities_created_during_render"] == 0
    assert manifest["max_duplicate_pid_per_frame"] == 0
    assert manifest["identity_sources_rendered"].get("hungarian", 0) == 0
    assert manifest["identity_sources_rendered"].get("provisional", 0) == 0
    assert manifest["tracked_first_frame"] == 0
    assert manifest["tracked_last_frame"] == 5
    assert manifest["tracking_coverage_raw_frames"] == 6
    assert manifest["render_untracked_tail_frames"] == 0


def test_manifest_records_color_thread_metrics(tmp_path):
    decisions = render_frames(_basic_tracking([]), motions=[], total_raw_frames=4,
                              frame_dims=(1920, 1080))
    manifest = build_manifest(
        decisions=decisions,
        total_raw_frames=4,
        source_video="/dev/null",
        output_video=str(tmp_path / "out.mp4"),
        source_fps=30.0,
        identity_metrics={},
        motions=[],
        tracking_results=_basic_tracking([]),
        color_thread_metrics={
            "color_threads_path": "temp/job/tracking/color_threads.json",
            "color_threads_present": True,
            "color_threads_count": 2,
            "color_thread_review_events": 1,
        },
    )
    assert manifest["new_identities_created_during_render"] == 0
    assert manifest["color_threads_present"] is True
    assert manifest["color_threads_count"] == 2
    assert manifest["color_thread_review_events"] == 1


def test_tracking_frame_range_reports_partial_coverage():
    frames = [
        _frame(10, [_player(playerId="P1", identity_valid=True, assignment_source="locked")]),
        _frame(15, [_player(playerId="P1", identity_valid=True, assignment_source="locked")]),
    ]
    assert tracking_frame_range(_basic_tracking(frames)) == (10, 15, 6)

    decisions = render_frames(_basic_tracking(frames), motions=[], total_raw_frames=30,
                              frame_dims=(1920, 1080))
    manifest = build_manifest(
        decisions=decisions,
        total_raw_frames=30,
        source_video="/dev/null",
        output_video="/tmp/out.mp4",
        source_fps=25.0,
        identity_metrics={},
        motions=[],
        tracking_results=_basic_tracking(frames),
        rendered_frames=30,
    )
    assert manifest["tracked_first_frame"] == 10
    assert manifest["tracked_last_frame"] == 15
    assert manifest["tracking_coverage_raw_frames"] == 6
    assert manifest["render_untracked_head_frames"] == 10
    assert manifest["render_untracked_tail_frames"] == 14
    assert manifest["tracking_coverage_ratio"] == pytest.approx(0.2)


def test_manifest_counts_fast_pan_raw_frames_not_samples(tmp_path):
    bbox = [100.0, 100.0, 140.0, 220.0]
    frames = [
        _frame(0, [_player(playerId="P1", rawTrackId=1, identity_valid=True,
                           assignment_source="locked", bbox=bbox)]),
        _frame(10, [_player(playerId="P1", rawTrackId=1, identity_valid=True,
                            assignment_source="locked", bbox=bbox)]),
    ]
    motions = [
        {"frameIndex": 0, "dx": 0.0, "dy": 0.0, "motion_class": "stable"},
        {"frameIndex": 5, "dx": 80.0, "dy": 0.0, "motion_class": "fast_pan"},
        {"frameIndex": 10, "dx": 0.0, "dy": 0.0, "motion_class": "stable"},
    ]
    decisions = render_frames(_basic_tracking(frames), motions=motions, total_raw_frames=12,
                              frame_dims=(1920, 1080))
    manifest = build_manifest(
        decisions=decisions,
        total_raw_frames=12,
        source_video="/dev/null",
        output_video=str(tmp_path / "out.mp4"),
        source_fps=30.0,
        identity_metrics={},
        motions=motions,
        tracking_results=_basic_tracking(frames),
    )
    assert manifest["camera_motion_samples"] == 3
    assert manifest["fast_pan_frames"] == 5
    assert manifest["cut_frames"] == 0


def test_manifest_records_missing_camera_motion_warning(tmp_path):
    decisions = render_frames(_basic_tracking([]), motions=[], total_raw_frames=4,
                              frame_dims=(1920, 1080))
    manifest = build_manifest(
        decisions=decisions,
        total_raw_frames=4,
        source_video="/dev/null",
        output_video=str(tmp_path / "out.mp4"),
        source_fps=30.0,
        identity_metrics={},
        motions=[],
        tracking_results=_basic_tracking([]),
        camera_motion_path="temp/example/tracking/camera_motion.json",
        camera_motion_present=False,
    )
    assert manifest["camera_motion_present"] is False
    assert manifest["camera_motion_samples"] == 0
    assert "MISSING_CAMERA_MOTION" in manifest["warnings"]


def test_manifest_warns_when_source_video_name_looks_annotated(tmp_path):
    decisions = render_frames(_basic_tracking([]), motions=[], total_raw_frames=4,
                              frame_dims=(1920, 1080))
    manifest = build_manifest(
        decisions=decisions,
        total_raw_frames=4,
        source_video="/tmp/villa_psg_30s_v3_annotated.mp4",
        output_video=str(tmp_path / "out.mp4"),
        source_fps=30.0,
        identity_metrics={},
        motions=[],
        tracking_results=_basic_tracking([]),
        camera_motion_present=False,
    )
    assert "SOURCE_VIDEO_NAME_LOOKS_ANNOTATED" in manifest["warnings"]


# ---- Color-thread overlay ------------------------------------------------------


def test_load_color_threads_missing_returns_empty(tmp_path):
    assert load_color_threads(str(tmp_path / "missing.json")) == {}


def test_color_thread_runtime_normalizes_points_and_colors():
    color_threads = {
        "threads": [
            {
                "thread_id": "CT01",
                "color": {"hex": "#010203"},
                "segments": [
                    {
                        "segment_id": "seg_00001",
                        "raw_track_id": 1,
                        "start_frame": 0,
                        "end_frame": 2,
                        "first_center": [-10, 20],
                        "last_center": [150, 90],
                    }
                ],
                "events": [],
            }
        ],
        "events": [],
    }
    tracking = _basic_tracking([
        _frame(1, [_player(rawTrackId=1, bbox=[80, 40, 140, 90])]),
    ])

    runtime = build_color_thread_runtime_index(color_threads, tracking, frame_dims=(100, 80))

    points = _color_thread_points_for_frame(tracking["frames"][0], 1, runtime, frame_dims=(100, 80))
    point = points[0]
    assert point["thread_id"] == "CT01"
    assert point["color"] == (3, 2, 1)
    assert point["center"] == (99, 65)
    assert runtime["metrics"]["color_threads_count"] == 1


def test_draw_color_threads_draws_trail_pixels():
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    runtime = {
        "predicted_by_frame": {},
        "warnings_by_frame": {},
    }
    history = {}

    _draw_color_threads(
        frame,
        0,
        [{"frame": 0, "thread_id": "CT01", "center": (10, 10), "color": (0, 255, 0)}],
        runtime,
        history,
    )
    metrics = _draw_color_threads(
        frame,
        1,
        [{"frame": 1, "thread_id": "CT01", "center": (40, 40), "color": (0, 255, 0)}],
        runtime,
        history,
    )

    assert metrics["color_thread_trail_segments_drawn"] == 1
    assert int(frame.sum()) > 0


def test_color_thread_raw_boxes_draw_unassigned_player():
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    points = [
        {
            "frame": 0,
            "thread_id": "CT01",
            "raw_track_id": 7,
            "bbox": [10, 10, 30, 50],
            "color": (0, 255, 0),
            "player": _player(rawTrackId=7, identity_valid=False, assignment_source="unassigned"),
        }
    ]

    drawn = _draw_color_thread_raw_boxes(frame, points, show_raw_id=True)

    assert drawn == 1
    assert int(frame.sum()) > 0


def test_strict_fails_on_dimension_mismatch():
    from services.fullfps_tracking_renderer import RendererDimensionError, validate_bbox_dimensions

    # Bbox extends past declared dims.
    frames = [
        _frame(0, [_player(playerId="P1", rawTrackId=1, identity_valid=True,
                           assignment_source="locked",
                           bbox=[100.0, 100.0, 2000.0, 1200.0])]),
    ]
    with pytest.raises(RendererDimensionError):
        validate_bbox_dimensions(frames, frame_dims=(1920, 1080), strict=True)
    # Non-strict: returns warning string instead of raising.
    warnings = validate_bbox_dimensions(frames, frame_dims=(1920, 1080), strict=False)
    assert warnings


# ---- Observation build ---------------------------------------------------------


def test_observation_build_drops_officials_and_invalid():
    frames = [
        _frame(0, [
            _player(playerId="P1", rawTrackId=1, identity_valid=True, assignment_source="locked"),
            _player(playerId=None, rawTrackId=2, identity_valid=False, assignment_source="unassigned"),
            _player(playerId="P9", rawTrackId=3, identity_valid=True, assignment_source="locked",
                    is_official=True),
            _player(playerId="P4", rawTrackId=4, identity_valid=True, assignment_source="hungarian"),
        ]),
    ]
    obs, counters, _ = build_observations(_basic_tracking(frames),
                                       debug_unknown=False, debug_officials=False)
    # Only PID:P1 should be present.
    assert set(obs.keys()) == {"PID:P1"}
    assert counters["identity_sources_rendered"]["locked"] == 1
    assert counters["identity_sources_rendered"].get("hungarian", 0) == 0
    # Hungarian and unknown contribute to unknown_suppressed_object_frames.
    assert counters["unknown_suppressed_object_frames"] >= 2
    # Official suppressed counter.
    assert counters["official_suppressed_object_frames"] == 1
