import csv

from services.color_thread_service import (
    apply_review_corrections,
    build_color_threads,
    load_review_actions,
    write_review_csv,
)


def _player(raw_tid, bbox, **overrides):
    base = {
        "rawTrackId": raw_tid,
        "trackId": raw_tid,
        "bbox": bbox,
        "confidence": 0.9,
        "identity_valid": True,
        "assignment_source": "locked",
        "playerId": overrides.pop("playerId", None),
        "displayId": overrides.pop("displayId", None),
        "team_id": overrides.pop("team_id", 1),
        "is_official": False,
    }
    base.update(overrides)
    return base


def _frame(frame_index, players):
    return {"frameIndex": frame_index, "players": players}


def _tracking(frames):
    return {"frames": frames}


def test_color_thread_builder_reconnects_raw_tracklets_and_flags_gap():
    frames = [
        _frame(0, [_player(1, [90, 100, 110, 150], playerId="P1")]),
        _frame(1, [_player(1, [95, 100, 115, 150], playerId="P1")]),
        _frame(20, [_player(2, [155, 100, 175, 150], playerId="P1")]),
        _frame(21, [_player(2, [160, 100, 180, 150], playerId="P1")]),
    ]

    color_threads = build_color_threads(
        _tracking(frames),
        max_reconnect_gap=30,
        max_reconnect_distance=260.0,
        min_reconnect_confidence=0.35,
    )

    assert color_threads["stats"]["threads"] == 1
    thread = color_threads["threads"][0]
    assert thread["raw_track_ids"] == [1, 2]
    assert thread["frame_range"] == [0, 21]
    assert thread["status"] == "needs_review"
    assert thread["events"][0]["type"] == "gap_reconnect"
    assert thread["events"][0]["status"] == "needs_review"


def test_color_thread_builder_splits_raw_track_on_impossible_jump():
    frames = [
        _frame(0, [_player(1, [90, 100, 110, 150], playerId="P1")]),
        _frame(1, [_player(1, [500, 100, 520, 150], playerId="P1")]),
    ]

    color_threads = build_color_threads(_tracking(frames), max_segment_jump=100.0)

    assert color_threads["stats"]["segments"] == 2
    assert color_threads["threads"][1]["segments"][0]["split_reason"].startswith("jump=")


def test_review_csv_writes_uncertain_events(tmp_path):
    frames = [
        _frame(0, [_player(1, [90, 100, 110, 150], playerId="P1")]),
        _frame(20, [_player(2, [155, 100, 175, 150], playerId="P1")]),
    ]
    color_threads = build_color_threads(
        _tracking(frames),
        max_reconnect_gap=30,
        max_reconnect_distance=260.0,
        min_reconnect_confidence=0.35,
    )
    out = tmp_path / "thread_review.csv"

    rows_written = write_review_csv(color_threads, out)

    assert rows_written == 1
    with out.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["thread_id"] == "CT01"
    assert rows[0]["action"] == ""
    assert rows[0]["segment_id"] == "seg_00002"


def test_load_review_actions_ignores_blank_actions(tmp_path):
    review = tmp_path / "review.csv"
    review.write_text(
        "event_id,action,thread_id,target_thread_id,segment_id,frame,reason,status,label,notes\n"
        "e1,,CT01,,seg_00001,0,reason,needs_review,,\n"
        "e2,assign_label,CT01,,seg_00001,0,reason,needs_review,P7,\n",
        encoding="utf-8",
    )

    actions = load_review_actions(review)

    assert len(actions) == 1
    assert actions[0]["action"] == "assign_label"


def test_apply_assign_label_updates_matching_thread_frames():
    frames = [
        _frame(0, [_player(1, [90, 100, 110, 150])]),
        _frame(1, [_player(1, [95, 100, 115, 150])]),
    ]
    tracking = _tracking(frames)
    color_threads = build_color_threads(tracking)

    corrected, _, summary = apply_review_corrections(
        tracking,
        color_threads,
        [{"action": "assign_label", "thread_id": "CT01", "label": "P7"}],
    )

    assert summary["actions_applied"] == 1
    for frame in corrected["frames"]:
        player = frame["players"][0]
        assert player["colorThreadId"] == "CT01"
        assert player["playerId"] == "P7"
        assert player["assignment_source"] == "color_thread_review"


def test_apply_assign_label_ignores_review_frame_by_default():
    frames = [
        _frame(0, [_player(1, [90, 100, 110, 150])]),
        _frame(5, [_player(1, [95, 100, 115, 150])]),
    ]
    tracking = _tracking(frames)
    color_threads = build_color_threads(tracking)

    corrected, _, summary = apply_review_corrections(
        tracking,
        color_threads,
        [{"action": "assign_label", "thread_id": "CT01", "frame": "5", "label": "P7"}],
    )

    assert summary["actions_applied"] == 1
    assert corrected["frames"][0]["players"][0]["playerId"] == "P7"
    assert corrected["frames"][1]["players"][0]["playerId"] == "P7"


def test_apply_merge_threads_reassigns_color_thread_annotation():
    frames = [
        _frame(0, [_player(1, [90, 100, 110, 150], playerId="P1")]),
        _frame(3, [_player(2, [900, 100, 920, 150], playerId="P2")]),
    ]
    tracking = _tracking(frames)
    color_threads = build_color_threads(tracking, max_reconnect_distance=120.0)
    assert color_threads["stats"]["threads"] == 2

    corrected, corrected_threads, summary = apply_review_corrections(
        tracking,
        color_threads,
        [{"action": "merge_threads", "thread_id": "CT02", "target_thread_id": "CT01"}],
    )

    assert summary["actions_applied"] == 1
    assert len(corrected_threads["threads"]) == 1
    assert corrected["frames"][1]["players"][0]["colorThreadId"] == "CT01"


def test_apply_merge_threads_rejects_overlapping_segments_without_force():
    frames = [
        _frame(0, [
            _player(1, [90, 100, 110, 150], playerId="P1"),
            _player(2, [900, 100, 920, 150], playerId="P2"),
        ]),
    ]
    tracking = _tracking(frames)
    color_threads = build_color_threads(tracking)
    assert color_threads["stats"]["threads"] == 2

    corrected, corrected_threads, summary = apply_review_corrections(
        tracking,
        color_threads,
        [{"action": "merge_threads", "thread_id": "CT02", "target_thread_id": "CT01"}],
    )

    assert summary["actions_applied"] == 0
    assert summary["actions_skipped"] == 1
    assert len(corrected_threads["threads"]) == 2
    assert corrected["frames"][0]["players"][1]["colorThreadId"] == "CT02"


def test_apply_split_thread_after_frame_creates_new_thread_for_later_segment():
    frames = [
        _frame(0, [_player(1, [90, 100, 110, 150], playerId="P1")]),
        _frame(1, [_player(1, [95, 100, 115, 150], playerId="P1")]),
        _frame(20, [_player(2, [155, 100, 175, 150], playerId="P1")]),
        _frame(21, [_player(2, [160, 100, 180, 150], playerId="P1")]),
    ]
    tracking = _tracking(frames)
    color_threads = build_color_threads(
        tracking,
        max_reconnect_gap=30,
        max_reconnect_distance=260.0,
        min_reconnect_confidence=0.35,
    )
    assert color_threads["stats"]["threads"] == 1

    corrected, corrected_threads, summary = apply_review_corrections(
        tracking,
        color_threads,
        [{"action": "split_thread_after_frame", "thread_id": "CT01", "frame": "5"}],
    )

    assert summary["actions_applied"] == 1
    assert len(corrected_threads["threads"]) == 2
    assert corrected["frames"][0]["players"][0]["colorThreadId"] == "CT01"
    assert corrected["frames"][2]["players"][0]["colorThreadId"] == "CT02"


def test_apply_split_thread_after_frame_can_split_inside_segment():
    frames = [
        _frame(0, [_player(1, [90, 100, 110, 150], playerId="P1")]),
        _frame(5, [_player(1, [120, 100, 140, 150], playerId="P1")]),
    ]
    tracking = _tracking(frames)
    color_threads = build_color_threads(tracking, max_segment_gap=8)
    assert color_threads["stats"]["segments"] == 1

    corrected, corrected_threads, summary = apply_review_corrections(
        tracking,
        color_threads,
        [{"action": "split_thread_after_frame", "thread_id": "CT01", "frame": "2"}],
    )

    assert summary["actions_applied"] == 1
    assert len(corrected_threads["threads"]) == 2
    assert corrected_threads["threads"][0]["segments"][0]["end_frame"] == 2
    assert corrected["frames"][0]["players"][0]["colorThreadId"] == "CT01"
    assert corrected["frames"][1]["players"][0]["colorThreadId"] == "CT02"


def test_apply_split_marks_review_event_resolved():
    frames = [
        _frame(0, [_player(1, [90, 100, 110, 150], playerId="P1")]),
        _frame(20, [_player(2, [155, 100, 175, 150], playerId="P1")]),
    ]
    tracking = _tracking(frames)
    color_threads = build_color_threads(
        tracking,
        max_reconnect_gap=30,
        max_reconnect_distance=260.0,
        min_reconnect_confidence=0.35,
    )
    event_id = color_threads["events"][0]["event_id"]

    _, corrected_threads, summary = apply_review_corrections(
        tracking,
        color_threads,
        [{"event_id": event_id, "action": "split_thread_after_frame", "thread_id": "CT01", "frame": "5"}],
    )

    assert summary["actions_applied"] == 1
    statuses = [event["status"] for thread in corrected_threads["threads"] for event in thread["events"]]
    assert "reviewed" in statuses
    assert corrected_threads["stats"]["review_events"] == 0


def test_apply_mark_unknown_clears_identity_after_frame():
    frames = [
        _frame(0, [_player(1, [90, 100, 110, 150], playerId="P1")]),
        _frame(5, [_player(1, [95, 100, 115, 150], playerId="P1")]),
    ]
    tracking = _tracking(frames)
    color_threads = build_color_threads(tracking)

    corrected, _, summary = apply_review_corrections(
        tracking,
        color_threads,
        [{"action": "mark_unknown", "thread_id": "CT01", "frame": "5"}],
    )

    assert summary["actions_applied"] == 1
    assert corrected["frames"][0]["players"][0]["playerId"] == "P1"
    assert corrected["frames"][1]["players"][0]["playerId"] is None
    assert corrected["frames"][1]["players"][0]["assignment_source"] == "color_thread_unknown"
