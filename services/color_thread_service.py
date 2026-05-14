"""Post-tracking color-thread identity helpers.

This module deliberately sits after detection/tracking. It treats tracker ids as
raw evidence, groups contiguous raw tracklets into longer visual threads, and
keeps uncertain joins explicit so a human can repair them with a small sidecar
file.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


RGBColor = Tuple[int, int, int]
Point = Tuple[float, float]


PALETTE: Sequence[RGBColor] = (
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190),
    (0, 128, 128),
    (230, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (0, 0, 128),
    (128, 128, 0),
    (255, 215, 180),
    (0, 0, 0),
)


REVIEW_HEADERS = [
    "event_id",
    "action",
    "thread_id",
    "target_thread_id",
    "segment_id",
    "frame",
    "reason",
    "status",
    "label",
    "notes",
]


@dataclass
class TrackPoint:
    frame: int
    raw_track_id: int
    bbox: List[float]
    center: Point
    stabilized_center: Point
    player_id: Optional[str] = None
    display_id: Optional[str] = None
    team_id: Optional[Any] = None
    confidence: Optional[float] = None
    identity_valid: Optional[bool] = None
    assignment_source: Optional[str] = None


@dataclass
class Segment:
    segment_id: str
    raw_track_id: int
    points: List[TrackPoint] = field(default_factory=list)
    split_reason: Optional[str] = None

    @property
    def start_frame(self) -> int:
        return self.points[0].frame

    @property
    def end_frame(self) -> int:
        return self.points[-1].frame

    @property
    def first_center(self) -> Point:
        return self.points[0].center

    @property
    def last_center(self) -> Point:
        return self.points[-1].center

    @property
    def first_stabilized_center(self) -> Point:
        return self.points[0].stabilized_center

    @property
    def last_stabilized_center(self) -> Point:
        return self.points[-1].stabilized_center

    def velocity(self) -> Point:
        if len(self.points) < 2:
            return (0.0, 0.0)
        first = self.points[max(0, len(self.points) - 6)]
        last = self.points[-1]
        gap = max(1, last.frame - first.frame)
        return (
            (last.stabilized_center[0] - first.stabilized_center[0]) / gap,
            (last.stabilized_center[1] - first.stabilized_center[1]) / gap,
        )

    def player_ids(self) -> List[str]:
        ids = [p.player_id for p in self.points if p.player_id]
        return sorted(set(ids))

    def team_mode(self) -> Optional[Any]:
        values = [p.team_id for p in self.points if p.team_id is not None]
        if not values:
            return None
        return Counter(values).most_common(1)[0][0]

    def mean_confidence(self) -> float:
        values = [float(p.confidence) for p in self.points if p.confidence is not None]
        if not values:
            return 0.72
        return max(0.0, min(1.0, sum(values) / len(values)))


@dataclass
class ThreadDraft:
    thread_id: str
    color: RGBColor
    segments: List[Segment] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def start_frame(self) -> int:
        return min(s.start_frame for s in self.segments)

    @property
    def end_frame(self) -> int:
        return max(s.end_frame for s in self.segments)

    def last_segment(self) -> Segment:
        return max(self.segments, key=lambda s: s.end_frame)


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def rgb_to_hex(color: RGBColor) -> str:
    return "#{:02x}{:02x}{:02x}".format(*color)


def color_for_index(index: int) -> RGBColor:
    if index < len(PALETTE):
        return PALETTE[index]
    # Deterministic high-contrast fallback using the golden angle.
    hue = (index * 137.508) % 360.0
    return _hsv_to_rgb(hue, 0.74, 0.92)


def _hsv_to_rgb(h: float, s: float, v: float) -> RGBColor:
    c = v * s
    x = c * (1.0 - abs((h / 60.0) % 2.0 - 1.0))
    m = v - c
    if h < 60:
        rp, gp, bp = c, x, 0
    elif h < 120:
        rp, gp, bp = x, c, 0
    elif h < 180:
        rp, gp, bp = 0, c, x
    elif h < 240:
        rp, gp, bp = 0, x, c
    elif h < 300:
        rp, gp, bp = x, 0, c
    else:
        rp, gp, bp = c, 0, x
    return (int((rp + m) * 255), int((gp + m) * 255), int((bp + m) * 255))


def _frame_index(frame: Dict[str, Any], fallback: int) -> int:
    return int(frame.get("frameIndex", frame.get("frame", fallback)))


def _players_from_frame(frame: Dict[str, Any]) -> List[Dict[str, Any]]:
    players = frame.get("players")
    if isinstance(players, list):
        return players
    detections = frame.get("detections")
    if isinstance(detections, list):
        return [d for d in detections if _is_player_like(d)]
    return []


def _is_player_like(player: Dict[str, Any]) -> bool:
    role = str(player.get("role", player.get("class_name", player.get("class", "")))).lower()
    if role in {"official", "referee", "ref", "ball"}:
        return False
    if player.get("is_official") or player.get("official"):
        return False
    return True


def _raw_track_id(player: Dict[str, Any]) -> Optional[int]:
    for key in ("rawTrackId", "raw_track_id", "trackId", "track_id", "tid", "id"):
        value = player.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _bbox(player: Dict[str, Any]) -> Optional[List[float]]:
    value = player.get("bbox")
    if value is None:
        value = player.get("box")
    if isinstance(value, dict):
        if {"x1", "y1", "x2", "y2"}.issubset(value):
            return [float(value["x1"]), float(value["y1"]), float(value["x2"]), float(value["y2"])]
        if {"x", "y", "w", "h"}.issubset(value):
            x, y, w, h = float(value["x"]), float(value["y"]), float(value["w"]), float(value["h"])
            return [x, y, x + w, y + h]
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
    return None


def _center(bbox: Sequence[float]) -> Point:
    return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)


def _player_id(player: Dict[str, Any]) -> Optional[str]:
    value = player.get("playerId", player.get("player_id", player.get("pid")))
    return str(value) if value not in (None, "", "None") else None


def _display_id(player: Dict[str, Any]) -> Optional[str]:
    value = player.get("displayId", player.get("display_id", player.get("label")))
    return str(value) if value not in (None, "", "None") else None


def _team_id(player: Dict[str, Any]) -> Optional[Any]:
    return player.get("teamId", player.get("team_id", player.get("team")))


def _confidence(player: Dict[str, Any]) -> Optional[float]:
    for key in ("identity_confidence", "identityConfidence", "confidence", "score"):
        if player.get(key) is not None:
            try:
                return float(player[key])
            except (TypeError, ValueError):
                return None
    return None


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_cumulative_offsets(
    camera_motion: Optional[Dict[str, Any]],
    max_frame: int,
) -> Dict[int, Point]:
    """Build approximate camera offsets indexed by frame.

    The tracking renderer has its own camera-motion path and depends on cv2.
    This pure-Python approximation is intentionally conservative; it only
    stabilizes endpoint comparisons enough to avoid penalizing steady pans.
    """

    if not camera_motion:
        return {i: (0.0, 0.0) for i in range(max_frame + 1)}
    samples = camera_motion.get("frames", camera_motion.get("motions", camera_motion))
    if not isinstance(samples, list):
        return {i: (0.0, 0.0) for i in range(max_frame + 1)}

    points: List[Tuple[int, float, float]] = []
    cum_x = 0.0
    cum_y = 0.0
    for idx, sample in enumerate(samples):
        if not isinstance(sample, dict):
            continue
        fi = int(sample.get("frameIndex", sample.get("frame", idx)))
        if fi < 0:
            continue
        if str(sample.get("motion_class", sample.get("motionClass", ""))).lower() == "cut":
            cum_x = 0.0
            cum_y = 0.0
        if idx > 0:
            cum_x += float(sample.get("dx", sample.get("translation_x", 0.0)) or 0.0)
            cum_y += float(sample.get("dy", sample.get("translation_y", 0.0)) or 0.0)
        points.append((fi, cum_x, cum_y))

    if not points:
        return {i: (0.0, 0.0) for i in range(max_frame + 1)}

    points.sort(key=lambda item: item[0])
    offsets: Dict[int, Point] = {}
    cursor = 0
    for frame in range(max_frame + 1):
        while cursor + 1 < len(points) and points[cursor + 1][0] <= frame:
            cursor += 1
        if cursor + 1 >= len(points):
            offsets[frame] = (points[cursor][1], points[cursor][2])
            continue
        left = points[cursor]
        right = points[cursor + 1]
        span = max(1, right[0] - left[0])
        alpha = max(0.0, min(1.0, (frame - left[0]) / span))
        offsets[frame] = (
            left[1] + (right[1] - left[1]) * alpha,
            left[2] + (right[2] - left[2]) * alpha,
        )
    return offsets


def extract_track_points(
    track_results: Dict[str, Any],
    camera_motion: Optional[Dict[str, Any]] = None,
) -> List[TrackPoint]:
    frames = track_results.get("frames", [])
    if not isinstance(frames, list):
        return []

    max_frame = max((_frame_index(frame, idx) for idx, frame in enumerate(frames)), default=0)
    offsets = build_cumulative_offsets(camera_motion, max_frame)

    points: List[TrackPoint] = []
    for idx, frame in enumerate(frames):
        frame_idx = _frame_index(frame, idx)
        offset = offsets.get(frame_idx, (0.0, 0.0))
        for player in _players_from_frame(frame):
            if not _is_player_like(player):
                continue
            raw_tid = _raw_track_id(player)
            box = _bbox(player)
            if raw_tid is None or box is None:
                continue
            center = _center(box)
            points.append(
                TrackPoint(
                    frame=frame_idx,
                    raw_track_id=raw_tid,
                    bbox=box,
                    center=center,
                    stabilized_center=(center[0] - offset[0], center[1] - offset[1]),
                    player_id=_player_id(player),
                    display_id=_display_id(player),
                    team_id=_team_id(player),
                    confidence=_confidence(player),
                    identity_valid=player.get("identity_valid", player.get("identityValid")),
                    assignment_source=player.get("assignment_source", player.get("assignmentSource")),
                )
            )
    points.sort(key=lambda p: (p.raw_track_id, p.frame))
    return points


def build_segments(
    points: Sequence[TrackPoint],
    *,
    max_segment_gap: int = 8,
    max_segment_jump: float = 180.0,
) -> List[Segment]:
    by_track: Dict[int, List[TrackPoint]] = defaultdict(list)
    for point in points:
        by_track[point.raw_track_id].append(point)

    segments: List[Segment] = []
    next_id = 1
    for raw_tid in sorted(by_track):
        active: Optional[Segment] = None
        previous: Optional[TrackPoint] = None
        for point in sorted(by_track[raw_tid], key=lambda p: p.frame):
            split_reason: Optional[str] = None
            if previous is not None:
                gap = point.frame - previous.frame
                jump = _distance(point.stabilized_center, previous.stabilized_center)
                if gap > max_segment_gap:
                    split_reason = f"gap={gap} > max_segment_gap={max_segment_gap}"
                elif jump > max_segment_jump:
                    split_reason = f"jump={jump:.1f}px > max_segment_jump={max_segment_jump:.1f}"
            if active is None or split_reason:
                active = Segment(
                    segment_id=f"seg_{next_id:05d}",
                    raw_track_id=raw_tid,
                    split_reason=split_reason,
                )
                next_id += 1
                segments.append(active)
            active.points.append(point)
            previous = point
    return sorted(segments, key=lambda s: (s.start_frame, s.raw_track_id, s.segment_id))


def _thread_id(index: int) -> str:
    return f"CT{index:02d}"


def _event_id(index: int) -> str:
    return f"cte_{index:05d}"


def _compatible_team(a: Optional[Any], b: Optional[Any]) -> bool:
    return a is None or b is None or a == b


def _segment_score(
    previous: Segment,
    candidate: Segment,
    *,
    max_reconnect_gap: int,
    max_reconnect_distance: float,
) -> Optional[Tuple[float, float, Dict[str, Any]]]:
    gap = candidate.start_frame - previous.end_frame
    if gap <= 0 or gap > max_reconnect_gap:
        return None

    prev_team = previous.team_mode()
    cand_team = candidate.team_mode()
    if not _compatible_team(prev_team, cand_team):
        return None

    vx, vy = previous.velocity()
    predicted = (
        previous.last_stabilized_center[0] + vx * gap,
        previous.last_stabilized_center[1] + vy * gap,
    )
    dist = _distance(predicted, candidate.first_stabilized_center)
    if dist > max_reconnect_distance:
        return None

    prev_ids = set(previous.player_ids())
    cand_ids = set(candidate.player_ids())
    same_pid_bonus = 0.14 if prev_ids and cand_ids and bool(prev_ids & cand_ids) else 0.0
    same_raw_bonus = 0.18 if previous.raw_track_id == candidate.raw_track_id else 0.0
    team_bonus = 0.08 if prev_team is not None and prev_team == cand_team else 0.0
    distance_quality = 1.0 - (dist / max(1.0, max_reconnect_distance))
    gap_penalty = gap / max(1.0, max_reconnect_gap * 2.5)
    confidence = max(0.0, min(1.0, distance_quality - gap_penalty + same_pid_bonus + same_raw_bonus + team_bonus))
    score = (dist / max_reconnect_distance) + (gap / max_reconnect_gap) * 0.4 - same_pid_bonus - same_raw_bonus
    detail = {
        "gap": gap,
        "distance": round(dist, 3),
        "predicted_center": [round(predicted[0], 2), round(predicted[1], 2)],
        "candidate_center": [round(candidate.first_stabilized_center[0], 2), round(candidate.first_stabilized_center[1], 2)],
        "previous_segment_id": previous.segment_id,
        "next_segment_id": candidate.segment_id,
    }
    return score, confidence, detail


def build_color_threads(
    track_results: Dict[str, Any],
    camera_motion: Optional[Dict[str, Any]] = None,
    *,
    max_segment_gap: int = 8,
    max_segment_jump: float = 180.0,
    max_reconnect_gap: int = 45,
    max_reconnect_distance: float = 260.0,
    min_reconnect_confidence: float = 0.42,
) -> Dict[str, Any]:
    points = extract_track_points(track_results, camera_motion)
    segments = build_segments(points, max_segment_gap=max_segment_gap, max_segment_jump=max_segment_jump)

    threads: List[ThreadDraft] = []
    event_counter = 1

    for segment in segments:
        best_thread: Optional[ThreadDraft] = None
        best_score = float("inf")
        best_confidence = 0.0
        best_detail: Dict[str, Any] = {}

        for thread in threads:
            last = thread.last_segment()
            candidate_score = _segment_score(
                last,
                segment,
                max_reconnect_gap=max_reconnect_gap,
                max_reconnect_distance=max_reconnect_distance,
            )
            if candidate_score is None:
                continue
            score, confidence, detail = candidate_score
            if confidence >= min_reconnect_confidence and score < best_score:
                best_thread = thread
                best_score = score
                best_confidence = confidence
                best_detail = detail

        if best_thread is None:
            thread_index = len(threads)
            best_thread = ThreadDraft(
                thread_id=_thread_id(thread_index + 1),
                color=color_for_index(thread_index),
                segments=[],
            )
            threads.append(best_thread)

        if best_thread.segments:
            status = "needs_review" if best_confidence < 0.70 or best_detail.get("gap", 0) > 14 else "inferred"
            event = {
                "event_id": _event_id(event_counter),
                "type": "gap_reconnect",
                "thread_id": best_thread.thread_id,
                "frame": int(segment.start_frame),
                "status": status,
                "confidence": round(best_confidence, 3),
                "reason": (
                    f"raw tracklet reconnected after {best_detail.get('gap')} frames "
                    f"at {best_detail.get('distance')}px"
                ),
                **best_detail,
            }
            event_counter += 1
            best_thread.events.append(event)
        best_thread.segments.append(segment)

        if len(segment.player_ids()) > 1:
            best_thread.events.append(
                {
                    "event_id": _event_id(event_counter),
                    "type": "possible_switch",
                    "thread_id": best_thread.thread_id,
                    "segment_id": segment.segment_id,
                    "frame": int(segment.start_frame),
                    "status": "needs_review",
                    "confidence": 0.35,
                    "reason": "raw tracklet contains multiple player labels",
                    "player_ids": segment.player_ids(),
                }
            )
            event_counter += 1

    thread_dicts = [_thread_to_dict(thread) for thread in threads]
    all_events = [event for thread in thread_dicts for event in thread.get("events", [])]
    frames = track_results.get("frames", [])
    camera_samples = 0
    if isinstance(camera_motion, dict):
        samples = camera_motion.get("frames", camera_motion.get("motions", []))
        camera_samples = len(samples) if isinstance(samples, list) else 0
    return {
        "schema_version": 1,
        "source": {
            "track_results_frame_count": len(frames) if isinstance(frames, list) else 0,
            "camera_motion_samples": camera_samples,
        },
        "params": {
            "max_segment_gap": max_segment_gap,
            "max_segment_jump": max_segment_jump,
            "max_reconnect_gap": max_reconnect_gap,
            "max_reconnect_distance": max_reconnect_distance,
            "min_reconnect_confidence": min_reconnect_confidence,
        },
        "threads": thread_dicts,
        "events": all_events,
        "stats": {
            "raw_points": len(points),
            "segments": len(segments),
            "threads": len(thread_dicts),
            "review_events": sum(1 for event in all_events if event.get("status") == "needs_review"),
        },
    }


def _thread_to_dict(thread: ThreadDraft) -> Dict[str, Any]:
    events = list(thread.events)
    review_events = [event for event in events if event.get("status") == "needs_review"]
    event_conf = [float(event.get("confidence", 0.5)) for event in events if event.get("confidence") is not None]
    seg_conf = [segment.mean_confidence() for segment in thread.segments]
    confidence_values = seg_conf + event_conf
    confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.72
    status = "needs_review" if review_events else "confirmed"
    raw_ids = sorted({segment.raw_track_id for segment in thread.segments})
    return {
        "thread_id": thread.thread_id,
        "color": {"rgb": list(thread.color), "hex": rgb_to_hex(thread.color)},
        "segments": [_segment_to_dict(segment) for segment in sorted(thread.segments, key=lambda s: s.start_frame)],
        "raw_track_ids": raw_ids,
        "frame_range": [int(thread.start_frame), int(thread.end_frame)],
        "status": status,
        "confidence": round(max(0.0, min(1.0, confidence)), 3),
        "events": events,
    }


def _segment_to_dict(segment: Segment) -> Dict[str, Any]:
    return {
        "segment_id": segment.segment_id,
        "raw_track_id": int(segment.raw_track_id),
        "start_frame": int(segment.start_frame),
        "end_frame": int(segment.end_frame),
        "frame_count": len(segment.points),
        "first_bbox": [round(v, 2) for v in segment.points[0].bbox],
        "last_bbox": [round(v, 2) for v in segment.points[-1].bbox],
        "first_center": [round(segment.first_center[0], 2), round(segment.first_center[1], 2)],
        "last_center": [round(segment.last_center[0], 2), round(segment.last_center[1], 2)],
        "player_ids": segment.player_ids(),
        "team_id_mode": segment.team_mode(),
        "status": "observed",
        "split_reason": segment.split_reason,
        "sampled_centers": _sample_centers(segment.points),
    }


def _sample_centers(points: Sequence[TrackPoint], limit: int = 24) -> List[Dict[str, Any]]:
    if len(points) <= limit:
        sampled = points
    else:
        step = (len(points) - 1) / max(1, limit - 1)
        sampled = [points[int(round(i * step))] for i in range(limit)]
    return [
        {
            "frame": int(point.frame),
            "center": [round(point.center[0], 2), round(point.center[1], 2)],
        }
        for point in sampled
    ]


def write_review_csv(color_threads: Dict[str, Any], path: str | Path) -> int:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for event in color_threads.get("events", []):
        if event.get("status") != "needs_review":
            continue
        rows.append(
            {
                "event_id": event.get("event_id", ""),
                "action": "",
                "thread_id": event.get("thread_id", ""),
                "target_thread_id": "",
                "segment_id": event.get("segment_id", event.get("next_segment_id", "")),
                "frame": event.get("frame", ""),
                "reason": event.get("reason", event.get("type", "")),
                "status": event.get("status", ""),
                "label": "",
                "notes": "",
            }
        )
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REVIEW_HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def build_and_save(
    *,
    track_results_path: str | Path,
    camera_motion_path: Optional[str | Path],
    color_threads_path: str | Path,
    review_csv_path: Optional[str | Path] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    track_results = load_json(track_results_path)
    camera_motion = load_json(camera_motion_path) if camera_motion_path and Path(camera_motion_path).exists() else None
    color_threads = build_color_threads(track_results, camera_motion, **kwargs)
    save_json(color_threads, color_threads_path)
    if review_csv_path:
        write_review_csv(color_threads, review_csv_path)
    return color_threads


def load_review_actions(path: str | Path) -> List[Dict[str, Any]]:
    review_path = Path(path)
    if review_path.suffix.lower() == ".json":
        data = load_json(review_path)
        if isinstance(data, list):
            rows = data
        else:
            rows = data.get("corrections", data.get("actions", []))
        return [dict(row) for row in rows if str(row.get("action", "")).strip()]

    with review_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader if str(row.get("action", "")).strip()]


def apply_review_corrections(
    track_results: Dict[str, Any],
    color_threads: Dict[str, Any],
    review_actions: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    corrected_threads = json.loads(json.dumps(color_threads))
    actions = [dict(row) for row in review_actions if str(row.get("action", "")).strip()]
    summary = {
        "actions_requested": len(actions),
        "actions_applied": 0,
        "actions_skipped": 0,
        "applied": [],
        "skipped": [],
    }

    direct_rules: List[Dict[str, Any]] = []
    for row in actions:
        action = str(row.get("action", "")).strip().lower()
        if action == "merge_threads":
            ok, reason = _merge_threads(corrected_threads, row)
        elif action == "split_thread_after_frame":
            ok, reason = _split_thread_after_frame(corrected_threads, row)
        elif action in {"mark_unknown", "assign_label"}:
            ok, reason = True, "queued_for_track_results"
            direct_rules.append(row)
        else:
            ok, reason = False, f"unsupported action: {action}"
        if ok:
            _mark_review_event_resolved(corrected_threads, row, action)
            summary["actions_applied"] += 1
            summary["applied"].append({"action": action, "reason": reason, **row})
        else:
            summary["actions_skipped"] += 1
            summary["skipped"].append({"action": action, "reason": reason, **row})

    _refresh_thread_metadata(corrected_threads)
    corrected_track_results = _annotate_track_results(track_results, corrected_threads, direct_rules)
    summary["players_annotated"] = corrected_track_results.get("_color_thread_apply_stats", {}).get("players_annotated", 0)
    corrected_track_results.pop("_color_thread_apply_stats", None)
    return corrected_track_results, corrected_threads, summary


def _thread_map(color_threads: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(thread.get("thread_id")): thread for thread in color_threads.get("threads", [])}


def _merge_threads(color_threads: Dict[str, Any], row: Dict[str, Any]) -> Tuple[bool, str]:
    source_id = str(row.get("thread_id", "")).strip()
    target_id = str(row.get("target_thread_id", "")).strip()
    if not source_id or not target_id:
        return False, "merge_threads requires thread_id and target_thread_id"
    if source_id == target_id:
        return False, "source and target thread are the same"
    threads = _thread_map(color_threads)
    source = threads.get(source_id)
    target = threads.get(target_id)
    if source is None or target is None:
        return False, "source or target thread not found"
    force = str(row.get("force", "")).strip().lower() in {"1", "true", "yes", "y"}
    if not force and _has_temporal_overlap(source.get("segments", []), target.get("segments", [])):
        return False, "merge would create overlapping player observations; set force=true to override"
    target.setdefault("segments", []).extend(source.get("segments", []))
    for event in source.get("events", []):
        moved = dict(event)
        moved["thread_id"] = target_id
        moved["source_thread_id"] = source_id
        target.setdefault("events", []).append(moved)
    target.setdefault("events", []).append(
        {
            "event_id": f"manual_merge_{source_id}_into_{target_id}",
            "type": "manual_merge",
            "thread_id": target_id,
            "source_thread_id": source_id,
            "status": "reviewed",
            "confidence": 1.0,
            "reason": str(row.get("notes") or "manual sidecar merge"),
        }
    )
    color_threads["threads"] = [thread for thread in color_threads.get("threads", []) if thread.get("thread_id") != source_id]
    return True, f"merged {source_id} into {target_id}"


def _has_temporal_overlap(a_segments: Sequence[Dict[str, Any]], b_segments: Sequence[Dict[str, Any]]) -> bool:
    for a in a_segments:
        a_start = int(a.get("start_frame", 0))
        a_end = int(a.get("end_frame", a_start))
        for b in b_segments:
            b_start = int(b.get("start_frame", 0))
            b_end = int(b.get("end_frame", b_start))
            if max(a_start, b_start) <= min(a_end, b_end):
                return True
    return False


def _split_thread_after_frame(color_threads: Dict[str, Any], row: Dict[str, Any]) -> Tuple[bool, str]:
    source_id = str(row.get("thread_id", "")).strip()
    try:
        frame = int(float(row.get("frame", "")))
    except (TypeError, ValueError):
        return False, "split_thread_after_frame requires frame"
    threads = _thread_map(color_threads)
    source = threads.get(source_id)
    if source is None:
        return False, "thread not found"

    moving: List[Dict[str, Any]] = []
    staying: List[Dict[str, Any]] = []
    for segment in source.get("segments", []):
        start = int(segment.get("start_frame", 0))
        end = int(segment.get("end_frame", start))
        if end <= frame:
            staying.append(segment)
        elif start > frame:
            moving.append(segment)
        else:
            before, after = _split_segment_dict(segment, frame)
            staying.append(before)
            moving.append(after)
    if not moving or not staying:
        return False, "split frame does not separate any segment"

    new_id = _next_available_thread_id(color_threads)
    new_color = color_for_index(len(color_threads.get("threads", [])))
    new_thread = {
        "thread_id": new_id,
        "color": {"rgb": list(new_color), "hex": rgb_to_hex(new_color)},
        "segments": moving,
        "raw_track_ids": [],
        "frame_range": [0, 0],
        "status": "needs_review",
        "confidence": source.get("confidence", 0.6),
        "events": [
            {
                "event_id": f"manual_split_{source_id}_{frame}",
                "type": "manual_split",
                "thread_id": new_id,
                "source_thread_id": source_id,
                "frame": frame,
                "status": "reviewed",
                "confidence": 1.0,
                "reason": str(row.get("notes") or "manual sidecar split"),
            }
        ],
    }
    source["segments"] = staying
    source.setdefault("events", []).append(
        {
            "event_id": f"manual_split_source_{source_id}_{frame}",
            "type": "manual_split_source",
            "thread_id": source_id,
            "new_thread_id": new_id,
            "frame": frame,
            "status": "reviewed",
            "confidence": 1.0,
            "reason": str(row.get("notes") or "manual sidecar split"),
        }
    )
    color_threads.setdefault("threads", []).append(new_thread)
    return True, f"split {source_id} after frame {frame} into {new_id}"


def _split_segment_dict(segment: Dict[str, Any], frame: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    before = dict(segment)
    after = dict(segment)
    segment_id = str(segment.get("segment_id", "segment"))
    start = int(segment.get("start_frame", frame))
    end = int(segment.get("end_frame", frame + 1))
    split_center = _segment_center_at_frame(segment, frame)
    next_center = _segment_center_at_frame(segment, frame + 1)

    before["end_frame"] = frame
    before["frame_count"] = max(1, frame - start + 1)
    if split_center is not None:
        before["last_center"] = split_center
    before["split_reason"] = str(before.get("split_reason") or f"manual_split_after_frame={frame}")

    after["segment_id"] = f"{segment_id}_split_{frame + 1}"
    after["start_frame"] = frame + 1
    after["frame_count"] = max(1, end - frame)
    if next_center is not None:
        after["first_center"] = next_center
    after["split_reason"] = f"manual_split_after_frame={frame}"
    return before, after


def _segment_center_at_frame(segment: Dict[str, Any], frame: int) -> Optional[List[float]]:
    samples = segment.get("sampled_centers") or []
    if samples:
        sorted_samples = sorted(samples, key=lambda item: int(item.get("frame", 0)))
        candidates = [item for item in sorted_samples if int(item.get("frame", 0)) <= frame]
        chosen = candidates[-1] if candidates else sorted_samples[0]
        center = chosen.get("center")
        if isinstance(center, list) and len(center) >= 2:
            return [float(center[0]), float(center[1])]
    key = "last_center" if frame >= int(segment.get("end_frame", 0)) else "first_center"
    center = segment.get(key)
    if isinstance(center, list) and len(center) >= 2:
        return [float(center[0]), float(center[1])]
    return None


def _next_available_thread_id(color_threads: Dict[str, Any]) -> str:
    existing = set(_thread_map(color_threads))
    idx = 1
    while _thread_id(idx) in existing:
        idx += 1
    return _thread_id(idx)


def _refresh_thread_metadata(color_threads: Dict[str, Any]) -> None:
    events: List[Dict[str, Any]] = []
    for thread in color_threads.get("threads", []):
        segments = sorted(thread.get("segments", []), key=lambda seg: (int(seg.get("start_frame", 0)), str(seg.get("segment_id", ""))))
        thread["segments"] = segments
        raw_ids = sorted({int(seg.get("raw_track_id")) for seg in segments if seg.get("raw_track_id") is not None})
        frame_ranges = [
            (int(seg.get("start_frame", 0)), int(seg.get("end_frame", 0)))
            for seg in segments
        ]
        thread["raw_track_ids"] = raw_ids
        if frame_ranges:
            thread["frame_range"] = [min(start for start, _ in frame_ranges), max(end for _, end in frame_ranges)]
        else:
            thread["frame_range"] = [0, 0]
        if any(event.get("status") == "needs_review" for event in thread.get("events", [])):
            thread["status"] = "needs_review"
        elif thread.get("status") not in {"unknown", "manual"}:
            thread["status"] = "confirmed"
        for event in thread.get("events", []):
            if not event.get("thread_id"):
                event["thread_id"] = thread.get("thread_id")
            events.append(event)
    color_threads["events"] = events
    color_threads.setdefault("stats", {})["threads"] = len(color_threads.get("threads", []))
    color_threads.setdefault("stats", {})["segments"] = sum(len(thread.get("segments", [])) for thread in color_threads.get("threads", []))
    color_threads.setdefault("stats", {})["review_events"] = sum(1 for event in events if event.get("status") == "needs_review")


def _mark_review_event_resolved(color_threads: Dict[str, Any], row: Dict[str, Any], action: str) -> None:
    event_id = str(row.get("event_id", "")).strip()
    if not event_id:
        return
    for thread in color_threads.get("threads", []):
        for event in thread.get("events", []):
            if str(event.get("event_id", "")).strip() != event_id:
                continue
            event["status"] = "reviewed"
            event["resolved_by"] = action
            if row.get("notes"):
                event["review_notes"] = str(row.get("notes"))


def _annotate_track_results(
    track_results: Dict[str, Any],
    color_threads: Dict[str, Any],
    direct_rules: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    corrected = json.loads(json.dumps(track_results))
    segment_index = _segment_index(color_threads)
    stats = {"players_annotated": 0, "players_marked_unknown": 0, "players_labeled": 0}

    for frame_pos, frame in enumerate(corrected.get("frames", [])):
        frame_idx = _frame_index(frame, frame_pos)
        for player in _players_from_frame(frame):
            raw_tid = _raw_track_id(player)
            if raw_tid is None:
                continue
            hit = _lookup_segment(segment_index, raw_tid, frame_idx)
            if hit is None:
                continue
            thread, segment = hit
            player["colorThreadId"] = thread["thread_id"]
            player["colorThreadColor"] = thread.get("color", {}).get("hex")
            player["colorThreadStatus"] = thread.get("status")
            stats["players_annotated"] += 1
            for rule in direct_rules:
                if not _rule_matches(rule, thread, segment, frame_idx):
                    continue
                action = str(rule.get("action", "")).strip().lower()
                if action == "mark_unknown":
                    player["playerId"] = None
                    player["displayId"] = None
                    player["identity_valid"] = False
                    player["assignment_source"] = "color_thread_unknown"
                    stats["players_marked_unknown"] += 1
                elif action == "assign_label":
                    label = str(rule.get("label", "")).strip()
                    if not label:
                        continue
                    player["playerId"] = label
                    player["displayId"] = label
                    player["identity_valid"] = True
                    player["assignment_source"] = "color_thread_review"
                    stats["players_labeled"] += 1
    corrected["_color_thread_apply_stats"] = stats
    return corrected


def _segment_index(color_threads: Dict[str, Any]) -> Dict[int, List[Tuple[int, int, Dict[str, Any], Dict[str, Any]]]]:
    index: Dict[int, List[Tuple[int, int, Dict[str, Any], Dict[str, Any]]]] = defaultdict(list)
    for thread in color_threads.get("threads", []):
        for segment in thread.get("segments", []):
            raw_tid = segment.get("raw_track_id")
            if raw_tid is None:
                continue
            index[int(raw_tid)].append(
                (
                    int(segment.get("start_frame", 0)),
                    int(segment.get("end_frame", 0)),
                    thread,
                    segment,
                )
            )
    for spans in index.values():
        spans.sort(key=lambda item: (item[0], item[1]))
    return index


def _lookup_segment(
    segment_index: Dict[int, List[Tuple[int, int, Dict[str, Any], Dict[str, Any]]]],
    raw_tid: int,
    frame_idx: int,
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    for start, end, thread, segment in segment_index.get(int(raw_tid), []):
        if start <= frame_idx <= end:
            return thread, segment
    return None


def _rule_matches(
    row: Dict[str, Any],
    thread: Dict[str, Any],
    segment: Dict[str, Any],
    frame_idx: int,
) -> bool:
    action = str(row.get("action", "")).strip().lower()
    thread_id = str(row.get("thread_id", "")).strip()
    segment_id = str(row.get("segment_id", "")).strip()
    frame = str(row.get("frame", "")).strip()
    if thread_id and thread_id != str(thread.get("thread_id")):
        return False
    if segment_id and segment_id != str(segment.get("segment_id")):
        return False
    frame_is_range_start = action != "assign_label" or str(row.get("scope", "")).strip().lower() in {
        "after_frame",
        "from_frame",
    }
    if frame and frame_is_range_start:
        try:
            frame_value = int(float(frame))
        except (TypeError, ValueError):
            return False
        if frame_idx < frame_value:
            return False
    return True


def apply_and_save(
    *,
    track_results_path: str | Path,
    color_threads_path: str | Path,
    review_file_path: str | Path,
    out_path: str | Path,
    threads_out_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    track_results = load_json(track_results_path)
    color_threads = load_json(color_threads_path)
    actions = load_review_actions(review_file_path)
    corrected_track_results, corrected_threads, summary = apply_review_corrections(track_results, color_threads, actions)
    corrected_track_results.setdefault("metadata", {})["color_thread_corrections"] = summary
    save_json(corrected_track_results, out_path)
    if threads_out_path:
        save_json(corrected_threads, threads_out_path)
    return summary
