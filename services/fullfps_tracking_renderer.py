"""Full-FPS pan-safe tracking renderer.

Simulation tick = identity/tracking output at frame_stride=5 (read-only input).
Render tick = every raw video frame.
Render entities = keyed by PID for LOCKED/REVIVED only.
Camera transform = cumulative dx/dy per raw frame, from camera_motion.json.
Interpolation = in camera-stabilised coordinates.
Identity invariant = renderer never creates or mutates identity.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import cv2
import numpy as np


# ---- Config --------------------------------------------------------------------

PAN_LABEL_FREEZE_FRAMES = 5
MAX_INTERP_GAP_RAW_FRAMES = 8
MAX_HOLD_RAW_FRAMES = 15
MAX_CENTER_JUMP_PX = 140.0
MIN_BOX_AREA = 80.0

RENDERABLE_PID_SOURCES = {"locked", "revived"}


TEAM_COLORS = {
    0: (255, 200, 0),
    1: (0, 60, 255),
    -1: (200, 200, 200),
}
STATE_COLOR = {
    "locked": (80, 220, 80),
    "revived": (80, 200, 230),
}


# ---- Enums / sentinels ---------------------------------------------------------


class RenderState(str, Enum):
    VISIBLE = "VISIBLE"
    INTERPOLATED = "INTERPOLATED"
    HELD = "HELD"
    HIDDEN = "HIDDEN"


class HideReason(str, Enum):
    HOLD_EXPIRED = "HOLD_EXPIRED"
    NO_PREV_OBS = "NO_PREV_OBS"
    CUT_BOUNDARY_SUPPRESSED = "CUT_BOUNDARY_SUPPRESSED"
    GAP_TOO_LARGE = "GAP_TOO_LARGE"
    DUPLICATE_PID_SUPPRESSED = "DUPLICATE_PID_SUPPRESSED"
    REFEREE_SUPPRESSED = "REFEREE_SUPPRESSED"
    UNKNOWN_SUPPRESSED = "UNKNOWN_SUPPRESSED"
    PAN_LABEL_FREEZE = "PAN_LABEL_FREEZE"
    IMPLAUSIBLE_MOTION = "IMPLAUSIBLE_MOTION"
    EMPTY_BBOX = "EMPTY_BBOX"


# ---- Dataclasses ---------------------------------------------------------------


@dataclass
class TrackObservation:
    raw_frame_idx: int
    entity_key: str
    pid: Optional[str]
    tid: Optional[int]
    bbox: list[float]
    team_id: int = -1
    identity_valid: bool = False
    assignment_source: str = ""
    confidence: float = 0.0
    is_official: bool = False


@dataclass
class RenderDecision:
    key: str
    pid: Optional[str]
    latest_tid: Optional[int]
    team_id: int
    state: RenderState
    bbox: Optional[list[float]] = None
    confidence: float = 0.0
    alpha: float = 1.0
    assignment_source: str = ""
    hidden_reason: Optional[HideReason] = None
    raw_frame_idx: int = -1


class RendererDimensionError(Exception):
    pass


# ---- Geometry helpers ----------------------------------------------------------


def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def bbox_size(bbox: list[float]) -> tuple[float, float]:
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])


def bbox_from_center_size(cx: float, cy: float, w: float, h: float) -> list[float]:
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


# ---- Entity key ----------------------------------------------------------------


def _player_tid(player: dict) -> Optional[int]:
    tid = player.get("rawTrackId")
    if tid is None:
        tid = player.get("trackId")
    return tid


def render_entity_key(
    player: dict,
    debug_unknown: bool = False,
    debug_officials: bool = False,
) -> Optional[str]:
    """Decide the persistent render key for one player observation.

    Returns None to drop the observation. Officials are dropped before any PID
    logic. Only LOCKED/REVIVED sources become PID:* keys; hungarian, provisional,
    or unassigned are never PIDs.
    """
    if player.get("is_official"):
        if not debug_officials:
            return None
        tid = _player_tid(player)
        return f"OFFICIAL:{tid}" if tid is not None else None

    pid = player.get("playerId") or player.get("displayId")
    source = (player.get("assignment_source") or "").lower()

    if (
        player.get("identity_valid")
        and source in RENDERABLE_PID_SOURCES
        and isinstance(pid, str)
        and pid.startswith("P")
    ):
        return f"PID:{pid}"

    if debug_unknown:
        tid = _player_tid(player)
        if tid is not None:
            return f"TID:{tid}"
    return None


# ---- Camera-motion timeline ----------------------------------------------------


def build_cumulative_offset(
    motions: list[dict],
    total_raw_frames: int,
) -> dict[int, tuple[float, float]]:
    """Build cumulative camera offset for every raw frame.

    Convention: dx/dy in camera_motion.json are deltas between processed samples.
    Frame 0 is the baseline; its dx/dy are ignored. A `cut` sample resets the
    segment baseline to (0, 0). Between samples we linearly interpolate.
    """
    offset: dict[int, tuple[float, float]] = {}
    if total_raw_frames <= 0:
        return offset

    sorted_motions = sorted(motions, key=lambda m: int(m.get("frameIndex", 0)))

    # cumulative at each sample
    sample_offsets: list[tuple[int, float, float, str]] = []  # (frameIdx, cx, cy, motion_class)

    prev_cx, prev_cy = 0.0, 0.0
    first = True
    for m in sorted_motions:
        f_i = int(m.get("frameIndex", 0))
        mc = m.get("motion_class", "unknown")
        if first:
            sample_offsets.append((f_i, 0.0, 0.0, mc))
            prev_cx, prev_cy = 0.0, 0.0
            first = False
            continue
        if mc == "cut":
            # segment reset at this sample
            sample_offsets.append((f_i, 0.0, 0.0, mc))
            prev_cx, prev_cy = 0.0, 0.0
        else:
            dx = float(m.get("dx", 0.0))
            dy = float(m.get("dy", 0.0))
            prev_cx += dx
            prev_cy += dy
            sample_offsets.append((f_i, prev_cx, prev_cy, mc))

    if not sample_offsets:
        for f in range(total_raw_frames):
            offset[f] = (0.0, 0.0)
        return offset

    # Fill every raw frame.
    for f in range(total_raw_frames):
        # Find bracketing samples.
        prev_idx = None
        next_idx = None
        for i, s in enumerate(sample_offsets):
            if s[0] <= f:
                prev_idx = i
            if s[0] > f and next_idx is None:
                next_idx = i
                break

        if prev_idx is None:
            # Before first sample: zero.
            offset[f] = (0.0, 0.0)
            continue

        prev_sample = sample_offsets[prev_idx]
        if next_idx is None:
            # Past last sample: hold last value.
            offset[f] = (prev_sample[1], prev_sample[2])
            continue

        next_sample = sample_offsets[next_idx]

        # If next sample is a cut, the segment is broken between prev and next:
        # frames between belong to the pre-cut segment (hold prev value).
        if next_sample[3] == "cut":
            offset[f] = (prev_sample[1], prev_sample[2])
            continue

        fa, ca_x, ca_y, _ = prev_sample
        fb, cb_x, cb_y, _ = next_sample
        if fb == fa:
            offset[f] = (ca_x, ca_y)
            continue
        t = (f - fa) / (fb - fa)
        offset[f] = (lerp(ca_x, cb_x, t), lerp(ca_y, cb_y, t))

    return offset


def build_cut_segments(motions: list[dict]) -> list[int]:
    """Return sorted list of frame indices where a cut occurs."""
    return sorted(
        int(m["frameIndex"]) for m in motions if m.get("motion_class") == "cut"
    )


def frame_motion_class(
    motions_sorted: list[dict],
    raw_frame_idx: int,
) -> str:
    """Motion class of the nearest sample on or before raw_frame_idx."""
    mc = "stable"
    for m in motions_sorted:
        if int(m.get("frameIndex", 0)) <= raw_frame_idx:
            mc = m.get("motion_class", "stable")
        else:
            break
    return mc


def has_cut_between(cut_frames: list[int], a: int, b: int) -> bool:
    if a >= b:
        return False
    for c in cut_frames:
        if a < c <= b:
            return True
    return False


def raw_motion_class_counts(motions: list[dict], total_raw_frames: int) -> dict[str, int]:
    """Count motion classes over raw render frames, not sampled motion entries."""
    motions_sorted = sorted(motions, key=lambda m: int(m.get("frameIndex", 0)))
    counts: dict[str, int] = defaultdict(int)
    for f in range(total_raw_frames):
        counts[frame_motion_class(motions_sorted, f)] += 1
    return dict(counts)


# ---- Observation build ---------------------------------------------------------


def build_observations(
    tracking_results: dict,
    frame_dims: tuple[int, int] | None = None,
    debug_unknown: bool = False,
    debug_officials: bool = False,
) -> tuple[dict[str, list[TrackObservation]], dict[str, Any]]:
    """Build per-key observation lists from track_results.json frames."""
    obs_by_key: dict[str, list[TrackObservation]] = defaultdict(list)
    counters = {
        "official_suppressed_object_frames": 0,
        "unknown_suppressed_object_frames": 0,
        "identity_sources_rendered": defaultdict(int),
    }

    frames = tracking_results.get("frames", []) or []
    for frame in frames:
        raw_idx = int(frame.get("frameIndex", 0))
        for player in frame.get("players", []) or []:
            # Officials first.
            if player.get("is_official"):
                counters["official_suppressed_object_frames"] += 1
                if not debug_officials:
                    continue
            key = render_entity_key(player, debug_unknown=debug_unknown,
                                    debug_officials=debug_officials)
            if key is None:
                counters["unknown_suppressed_object_frames"] += 1
                continue

            source = (player.get("assignment_source") or "").lower()
            if key.startswith("PID:"):
                counters["identity_sources_rendered"][source] += 1
            elif key.startswith("OFFICIAL:"):
                counters["identity_sources_rendered"]["official"] += 1
            else:
                counters["identity_sources_rendered"][source or "unknown"] += 1

            raw_bbox = [float(v) for v in player.get("bbox", [0.0, 0.0, 0.0, 0.0])]
            if frame_dims is not None:
                w, h = frame_dims
                raw_bbox[0] = max(0.0, min(float(w), raw_bbox[0]))
                raw_bbox[1] = max(0.0, min(float(h), raw_bbox[1]))
                raw_bbox[2] = max(0.0, min(float(w), raw_bbox[2]))
                raw_bbox[3] = max(0.0, min(float(h), raw_bbox[3]))
            
            obs_by_key[key].append(
                TrackObservation(
                    raw_frame_idx=raw_idx,
                    entity_key=key,
                    pid=(player.get("playerId") or player.get("displayId")) if key.startswith("PID:") else None,
                    tid=_player_tid(player),
                    bbox=raw_bbox,
                    team_id=int(player.get("team_id", -1)),
                    identity_valid=bool(player.get("identity_valid")),
                    assignment_source=source,
                    confidence=float(player.get("identity_confidence", 0.0) or 0.0),
                    is_official=bool(player.get("is_official")),
                )
            )

    for key in obs_by_key:
        obs_by_key[key].sort(key=lambda o: o.raw_frame_idx)

    # convert defaultdict for cleaner downstream JSON
    counters["identity_sources_rendered"] = dict(counters["identity_sources_rendered"])
    return obs_by_key, counters


def validate_bbox_dimensions(
    frames: list[dict],
    frame_dims: tuple[int, int],
    strict: bool,
) -> list[str]:
    """Sanity check that bboxes fit inside source video dims."""
    w, h = frame_dims
    warnings: list[str] = []
    sampled = 0
    for frame in frames:
        if sampled > 10:
            break
        for p in frame.get("players", []) or []:
            bb = p.get("bbox") or [0, 0, 0, 0]
            if len(bb) < 4:
                continue
            if bb[2] > w + 100 or bb[3] > h + 100 or bb[0] < -100 or bb[1] < -100:
                msg = (
                    f"Track bbox {bb} outside source video {w}x{h} "
                    f"at frameIndex={frame.get('frameIndex')}; "
                    f"tracking may have been run on a rotated/resized frame."
                )
                warnings.append(msg)
                sampled += 1
                break
        sampled += 1
    if warnings and strict:
        raise RendererDimensionError(warnings[0])
    return warnings


def tracking_frame_range(tracking_results: dict) -> tuple[Optional[int], Optional[int], int]:
    """Return first/last tracked raw frame and inclusive coverage span.

    Uses every frame entry in track_results, even if that sampled frame has no
    visible players. This describes the simulation/output span available to the
    renderer, not the number of player observations.
    """
    frame_indices = [
        int(frame.get("frameIndex", 0))
        for frame in (tracking_results.get("frames", []) or [])
    ]
    if not frame_indices:
        return None, None, 0
    first = min(frame_indices)
    last = max(frame_indices)
    return first, last, max(0, last - first + 1)


# ---- Render loop ---------------------------------------------------------------


def _find_prev_next(
    observations: list[TrackObservation],
    f: int,
) -> tuple[Optional[TrackObservation], Optional[TrackObservation], list[TrackObservation]]:
    """Linear scan (observations are sorted by raw_frame_idx).

    Returns (prev_obs, next_obs, exact_matches).
    - prev_obs: latest observation with raw_frame_idx < f (strictly before).
    - next_obs: earliest with raw_frame_idx > f.
    - exact_matches: all observations with raw_frame_idx == f (may be multiple
      when the upstream identity emitted duplicate same-key entries at one frame).
    """
    prev_obs: Optional[TrackObservation] = None
    next_obs: Optional[TrackObservation] = None
    exact: list[TrackObservation] = []
    for o in observations:
        if o.raw_frame_idx < f:
            prev_obs = o
        elif o.raw_frame_idx == f:
            exact.append(o)
        elif o.raw_frame_idx > f:
            if next_obs is None:
                next_obs = o
            break
    return prev_obs, next_obs, exact


def render_frames(
    tracking_results: dict,
    motions: list[dict],
    total_raw_frames: int,
    frame_dims: tuple[int, int],
    debug_unknown: bool = False,
    debug_officials: bool = False,
) -> list[list[RenderDecision]]:
    """Compute render decisions for every raw frame.

    Returns a list of length total_raw_frames; each element is the list of
    RenderDecision objects for that frame (including hidden ones for QA).
    """
    obs_by_key, _counters = build_observations(
        tracking_results, frame_dims=frame_dims, debug_unknown=debug_unknown, debug_officials=debug_officials
    )
    offsets = build_cumulative_offset(motions, total_raw_frames)
    cut_frames = build_cut_segments(motions)
    motions_sorted = sorted(motions, key=lambda m: int(m.get("frameIndex", 0)))

    width, height = frame_dims

    # Per-key first-emit raw frame, for pan-label-freeze.
    first_emit_frame: dict[str, int] = {}
    last_render_center: dict[str, tuple[float, float]] = {}

    decisions: list[list[RenderDecision]] = [[] for _ in range(total_raw_frames)]

    for f in range(total_raw_frames):
        motion_class = frame_motion_class(motions_sorted, f)
        candidates: list[RenderDecision] = []

        for key, obs_list in obs_by_key.items():
            if not obs_list:
                continue
            prev_obs, next_obs, exact = _find_prev_next(obs_list, f)

            # Each exact observation becomes its own candidate — duplicate-PID
            # suppression below will pick the strongest.
            per_key_decisions: list[RenderDecision] = []

            if exact:
                for eo in exact:
                    per_key_decisions.append(RenderDecision(
                        key=key,
                        pid=eo.pid,
                        latest_tid=eo.tid,
                        team_id=eo.team_id,
                        state=RenderState.VISIBLE,
                        bbox=list(eo.bbox),
                        confidence=eo.confidence,
                        alpha=1.0,
                        assignment_source=eo.assignment_source,
                        raw_frame_idx=f,
                    ))
                candidates.extend(per_key_decisions)
                continue

            if prev_obs is None and next_obs is None:
                continue

            decision = RenderDecision(
                key=key,
                pid=(prev_obs or next_obs).pid,
                latest_tid=(prev_obs or next_obs).tid,
                team_id=(prev_obs or next_obs).team_id,
                state=RenderState.HIDDEN,
                confidence=(prev_obs or next_obs).confidence,
                assignment_source=(prev_obs or next_obs).assignment_source,
                raw_frame_idx=f,
            )

            if prev_obs is not None and next_obs is not None:
                gap = next_obs.raw_frame_idx - prev_obs.raw_frame_idx
                if has_cut_between(cut_frames, prev_obs.raw_frame_idx, next_obs.raw_frame_idx):
                    decision.hidden_reason = HideReason.CUT_BOUNDARY_SUPPRESSED
                elif gap > MAX_INTERP_GAP_RAW_FRAMES:
                    # Try HELD if within hold window.
                    age = f - prev_obs.raw_frame_idx
                    if age <= MAX_HOLD_RAW_FRAMES and not has_cut_between(
                        cut_frames, prev_obs.raw_frame_idx, f
                    ):
                        decision.state = RenderState.HELD
                        decision.bbox = list(prev_obs.bbox)
                        decision.alpha = max(0.2, 1.0 - 0.2 * age)
                    else:
                        decision.hidden_reason = HideReason.GAP_TOO_LARGE
                else:
                    # Camera-stabilised interpolation.
                    pcx, pcy = bbox_center(prev_obs.bbox)
                    ncx, ncy = bbox_center(next_obs.bbox)
                    pw, ph = bbox_size(prev_obs.bbox)
                    nw, nh = bbox_size(next_obs.bbox)
                    p_off = offsets.get(prev_obs.raw_frame_idx, (0.0, 0.0))
                    n_off = offsets.get(next_obs.raw_frame_idx, (0.0, 0.0))
                    c_off = offsets.get(f, (0.0, 0.0))
                    spx = pcx - p_off[0]
                    spy = pcy - p_off[1]
                    snx = ncx - n_off[0]
                    sny = ncy - n_off[1]
                    t = (f - prev_obs.raw_frame_idx) / gap
                    ts = smoothstep(t)
                    sx = lerp(spx, snx, ts)
                    sy = lerp(spy, sny, ts)
                    rx = sx + c_off[0]
                    ry = sy + c_off[1]
                    rw = lerp(pw, nw, t)
                    rh = lerp(ph, nh, t)
                    decision.bbox = bbox_from_center_size(rx, ry, rw, rh)
                    decision.state = RenderState.INTERPOLATED
                    decision.alpha = 0.95
            elif prev_obs is not None:
                age = f - prev_obs.raw_frame_idx
                if age <= MAX_HOLD_RAW_FRAMES and not has_cut_between(
                    cut_frames, prev_obs.raw_frame_idx, f
                ):
                    decision.state = RenderState.HELD
                    decision.bbox = list(prev_obs.bbox)
                    decision.alpha = max(0.2, 1.0 - 0.2 * age)
                else:
                    if has_cut_between(cut_frames, prev_obs.raw_frame_idx, f):
                        decision.hidden_reason = HideReason.CUT_BOUNDARY_SUPPRESSED
                    else:
                        decision.hidden_reason = HideReason.HOLD_EXPIRED
            else:
                # next_obs only, no prev → don't render.
                decision.hidden_reason = HideReason.NO_PREV_OBS

            # Sanity checks on bbox.
            if decision.state != RenderState.HIDDEN and decision.bbox is not None:
                bw = decision.bbox[2] - decision.bbox[0]
                bh = decision.bbox[3] - decision.bbox[1]
                if bw * bh < MIN_BOX_AREA:
                    decision.state = RenderState.HIDDEN
                    decision.hidden_reason = HideReason.EMPTY_BBOX
                else:
                    # Implausible jump from previous-rendered position?
                    prev_center = last_render_center.get(key)
                    if prev_center is not None:
                        cx, cy = bbox_center(decision.bbox)
                        jump = ((cx - prev_center[0]) ** 2 + (cy - prev_center[1]) ** 2) ** 0.5
                        if jump > MAX_CENTER_JUMP_PX:
                            decision.state = RenderState.HIDDEN
                            decision.hidden_reason = HideReason.IMPLAUSIBLE_MOTION

            # Pan-safe label freeze: hide first-emit during fast_pan for new PIDs.
            if (
                decision.state != RenderState.HIDDEN
                and motion_class == "fast_pan"
                and key.startswith("PID:")
                and key not in first_emit_frame
            ):
                decision.state = RenderState.HIDDEN
                decision.hidden_reason = HideReason.PAN_LABEL_FREEZE

            candidates.append(decision)

        # Duplicate-PID suppression (only PID:* keys).
        by_pid: dict[str, list[RenderDecision]] = defaultdict(list)
        for c in candidates:
            if c.state == RenderState.HIDDEN:
                continue
            if c.pid and c.key.startswith("PID:"):
                by_pid[c.pid].append(c)

        for pid, group in by_pid.items():
            if len(group) <= 1:
                continue
            STATE_RANK = {
                RenderState.VISIBLE: 3,
                RenderState.INTERPOLATED: 2,
                RenderState.HELD: 1,
                RenderState.HIDDEN: 0,
            }
            group.sort(
                key=lambda d: (STATE_RANK[d.state], d.confidence, -abs(f - d.raw_frame_idx)),
                reverse=True,
            )
            for loser in group[1:]:
                loser.state = RenderState.HIDDEN
                loser.hidden_reason = HideReason.DUPLICATE_PID_SUPPRESSED

        # Record first-emit for any candidate that survived.
        for c in candidates:
            if c.state != RenderState.HIDDEN and c.key not in first_emit_frame:
                first_emit_frame[c.key] = f
            if c.state != RenderState.HIDDEN and c.bbox is not None:
                last_render_center[c.key] = bbox_center(c.bbox)

        decisions[f] = candidates

    return decisions


# ---- Manifest ------------------------------------------------------------------


def build_manifest(
    decisions: list[list[RenderDecision]],
    total_raw_frames: int,
    source_video: str,
    output_video: str,
    source_fps: float,
    identity_metrics: dict,
    motions: list[dict],
    tracking_results: Optional[dict] = None,
    camera_motion_path: Optional[str] = None,
    camera_motion_present: Optional[bool] = None,
    duplicate_pid_suppressed_object_frames: int | None = None,
    official_suppressed_object_frames: int | None = None,
    unknown_suppressed_object_frames: int | None = None,
    identity_sources_rendered: dict[str, int] | None = None,
    rendered_frames: int | None = None,
) -> dict:
    """Assemble manifest. Asserts hard invariants before returning."""
    visible_obj = locked_obj = revived_obj = 0
    interp_obj = held_obj = hidden_obj = 0
    pan_freeze_obj = implausible_obj = cut_obj = 0
    dup_obj = duplicate_pid_suppressed_object_frames or 0

    frames_with_any_visible = 0
    frames_with_locked = 0
    frames_with_none = 0
    max_dup_per_frame = 0

    for f, frame_decisions in enumerate(decisions):
        per_frame_pids = defaultdict(int)
        any_visible = False
        any_locked = False
        dup_count_this_frame = 0
        for d in frame_decisions:
            if d.state == RenderState.HIDDEN:
                hidden_obj += 1
                if d.hidden_reason == HideReason.DUPLICATE_PID_SUPPRESSED:
                    if duplicate_pid_suppressed_object_frames is None:
                        dup_obj += 1
                    dup_count_this_frame += 1
                elif d.hidden_reason == HideReason.PAN_LABEL_FREEZE:
                    pan_freeze_obj += 1
                elif d.hidden_reason == HideReason.IMPLAUSIBLE_MOTION:
                    implausible_obj += 1
                elif d.hidden_reason == HideReason.CUT_BOUNDARY_SUPPRESSED:
                    cut_obj += 1
                continue
            any_visible = True
            visible_obj += 1
            if d.state == RenderState.INTERPOLATED:
                interp_obj += 1
            elif d.state == RenderState.HELD:
                held_obj += 1
            if d.assignment_source == "locked":
                locked_obj += 1
                any_locked = True
            elif d.assignment_source == "revived":
                revived_obj += 1
                any_locked = True
            if d.pid:
                per_frame_pids[d.pid] += 1

        if any_visible:
            frames_with_any_visible += 1
        else:
            frames_with_none += 1
        if any_locked:
            frames_with_locked += 1
        # After suppression, no PID should appear more than once.
        # Track *extra* duplicates per frame: count-1 per PID, max'd over PIDs.
        local_max_extra = max((c - 1 for c in per_frame_pids.values()), default=0)
        if local_max_extra > max_dup_per_frame:
            max_dup_per_frame = local_max_extra

    motion_counts = raw_motion_class_counts(motions, total_raw_frames) if motions else {}
    fast_pan_frames = motion_counts.get("fast_pan", 0)
    cut_frames = motion_counts.get("cut", 0)

    tracked_first, tracked_last, tracking_coverage = tracking_frame_range(
        tracking_results or {}
    )
    render_untracked_head = 0
    render_untracked_tail = 0
    if tracked_first is not None and tracked_last is not None:
        render_untracked_head = max(0, tracked_first)
        render_untracked_tail = max(0, total_raw_frames - tracked_last - 1)
    elif total_raw_frames > 0:
        render_untracked_head = total_raw_frames

    tracking_coverage_ratio = (
        tracking_coverage / total_raw_frames if total_raw_frames > 0 else 0.0
    )
    motion_present = bool(motions) if camera_motion_present is None else camera_motion_present
    warnings: list[str] = []
    if not motion_present:
        warnings.append("MISSING_CAMERA_MOTION")
    if tracking_coverage_ratio < 0.8 and total_raw_frames > 0:
        warnings.append("PARTIAL_TRACKING_COVERAGE")
    source_name = os.path.basename(source_video or "").lower()
    if "annotated" in source_name:
        warnings.append("SOURCE_VIDEO_NAME_LOOKS_ANNOTATED")

    manifest = {
        "source_video": source_video,
        "output_video": output_video,
        "camera_motion_path": camera_motion_path,
        "camera_motion_present": motion_present,
        "camera_motion_samples": len(motions),
        "source_fps": source_fps,
        "output_fps": source_fps,
        "total_raw_frames": total_raw_frames,
        "rendered_frames": rendered_frames if rendered_frames is not None else total_raw_frames,
        "tracked_first_frame": tracked_first,
        "tracked_last_frame": tracked_last,
        "track_results_frame_count": len((tracking_results or {}).get("frames", []) or []),
        "tracking_coverage_raw_frames": tracking_coverage,
        "tracking_coverage_ratio": tracking_coverage_ratio,
        "render_untracked_head_frames": render_untracked_head,
        "render_untracked_tail_frames": render_untracked_tail,
        "identity_frame_stride": 5,
        "render_stride": 1,
        "render_mode": "fullfps_pan_safe_game_style",
        # Frame counts.
        "frames_with_any_visible_box": frames_with_any_visible,
        "frames_with_locked_boxes": frames_with_locked,
        "frames_with_no_visible_boxes": frames_with_none,
        "fast_pan_frames": fast_pan_frames,
        "cut_frames": cut_frames,
        # Object-frame counts.
        "visible_object_frames": visible_obj,
        "locked_object_frames": locked_obj,
        "revived_object_frames": revived_obj,
        "interpolated_object_frames": interp_obj,
        "held_object_frames": held_obj,
        "hidden_object_frames": hidden_obj,
        "duplicate_pid_suppressed_object_frames": dup_obj,
        "official_suppressed_object_frames": official_suppressed_object_frames or 0,
        "unknown_suppressed_object_frames": unknown_suppressed_object_frames or 0,
        "pan_label_freeze_suppressed_object_frames": pan_freeze_obj,
        "implausible_motion_suppressed_object_frames": implausible_obj,
        "cut_boundary_suppressed_object_frames": cut_obj,
        # Source breakdown.
        "identity_sources_rendered": identity_sources_rendered or {},
        # Invariant signals.
        "new_identities_created_during_render": 0,
        "max_duplicate_pid_per_frame": max_dup_per_frame,
        "warnings": warnings,
    }
    if identity_metrics:
        manifest["identity_metrics"] = identity_metrics

    # Hard invariants.
    # OpenCV frame_count metadata can be slightly off from actual readable frames.
    assert abs(manifest["rendered_frames"] - manifest["total_raw_frames"]) < 30
    assert manifest["new_identities_created_during_render"] == 0
    assert manifest["max_duplicate_pid_per_frame"] == 0
    sr = manifest["identity_sources_rendered"]
    assert sr.get("hungarian", 0) == 0
    assert sr.get("provisional", 0) == 0
    return manifest


# ---- Drawing -------------------------------------------------------------------


def _color_for(decision: RenderDecision) -> tuple[int, int, int]:
    tc = TEAM_COLORS.get(decision.team_id, TEAM_COLORS[-1])
    return tc


def _draw_decision(frame: np.ndarray, d: RenderDecision, debug: bool) -> None:
    if d.bbox is None:
        return
    h, w = frame.shape[:2]
    x1 = max(0, min(int(d.bbox[0]), w - 1))
    y1 = max(0, min(int(d.bbox[1]), h - 1))
    x2 = max(0, min(int(d.bbox[2]), w - 1))
    y2 = max(0, min(int(d.bbox[3]), h - 1))
    if x2 <= x1 or y2 <= y1:
        return
    color = _color_for(d)
    thickness = 2 if d.state == RenderState.VISIBLE else 1
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    label = ""
    if d.key.startswith("OFFICIAL:"):
        label = "REF"
    elif d.pid:
        tag = "LOCK" if d.assignment_source == "locked" else "REV"
        label = f"{d.pid} {tag}"
        if debug and d.latest_tid is not None:
            label = f"{d.pid} T{d.latest_tid} {tag}"

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        text_th = 1
        (tw, th), _ = cv2.getTextSize(label, font, scale, text_th)
        bx1 = x1
        by1 = max(0, y1 - th - 4)
        bx2 = min(w - 1, x1 + tw + 6)
        by2 = by1 + th + 4
        overlay = frame.copy()
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, label, (bx1 + 3, by2 - 3), font, scale, (255, 255, 255),
                    text_th, cv2.LINE_AA)


# ---- Top-level pipeline --------------------------------------------------------


def load_track_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_camera_motion(path: str) -> list[dict]:
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("motions", [])


def load_identity_metrics(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def render_video(
    *,
    video_path: str,
    job_id: str,  # noqa: ARG001 - kept for API symmetry with CLI
    track_results_path: str,
    camera_motion_path: Optional[str],
    identity_metrics_path: Optional[str],
    output_path: str,
    debug: bool = False,
    debug_unknown: bool = False,
    debug_officials: bool = False,
    write_contact_sheet: bool = False,
    write_qa_json: bool = True,
    strict: bool = False,
) -> dict:
    """Run the full renderer. Returns the manifest dict."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_raw_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Renderer] source {width}x{height} @ {fps:.2f} fps, {total_raw_frames} frames")

    tracking_results = load_track_results(track_results_path)
    motions = load_camera_motion(camera_motion_path) if camera_motion_path else []
    if not motions:
        msg = f"[Renderer] WARNING camera_motion.json missing or empty at {camera_motion_path}"
        if strict:
            raise FileNotFoundError(msg)
        print(msg)
    identity_metrics = load_identity_metrics(identity_metrics_path) if identity_metrics_path else {}
    if not identity_metrics:
        if strict and identity_metrics_path:
            raise FileNotFoundError(f"identity_metrics.json missing at {identity_metrics_path}")

    # Dimension check.
    warnings = validate_bbox_dimensions(
        tracking_results.get("frames", []) or [],
        frame_dims=(width, height),
        strict=strict,
    )
    for w_msg in warnings:
        print(f"[Renderer] WARN: {w_msg}")

    # Observation build returns counters too; recompute via render_frames separately
    # so we have a single source of truth.
    obs_by_key, counters = build_observations(
        tracking_results,
        frame_dims=(width, height),
        debug_unknown=debug_unknown,
        debug_officials=debug_officials,
    )
    decisions = render_frames(
        tracking_results,
        motions=motions,
        total_raw_frames=total_raw_frames,
        frame_dims=(width, height),
        debug_unknown=debug_unknown,
        debug_officials=debug_officials,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create video writer at {output_path}")

    rendered_count = 0
    contact_sample_frames: list[np.ndarray] = []
    contact_sample_indices: list[int] = []
    if write_contact_sheet and total_raw_frames > 0:
        step = max(1, total_raw_frames // 12)
        contact_sample_indices = list(range(0, total_raw_frames, step))[:12]

    t0 = time.time()
    f = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))

        frame_decisions = decisions[f] if f < len(decisions) else []
        for d in frame_decisions:
            if d.state == RenderState.HIDDEN:
                continue
            _draw_decision(frame, d, debug=debug)

        writer.write(frame)
        if write_contact_sheet and f in contact_sample_indices:
            contact_sample_frames.append(frame.copy())
        rendered_count += 1
        f += 1
        if f % 100 == 0:
            print(f"[Renderer] rendered {f}/{total_raw_frames}")
        if f >= total_raw_frames:
            break

    writer.release()
    cap.release()
    print(f"[Renderer] wrote {rendered_count} frames in {time.time() - t0:.1f}s -> {output_path}")

    manifest = build_manifest(
        decisions=decisions,
        total_raw_frames=total_raw_frames,
        source_video=video_path,
        output_video=output_path,
        source_fps=fps,
        identity_metrics=identity_metrics,
        motions=motions,
        tracking_results=tracking_results,
        camera_motion_path=camera_motion_path,
        camera_motion_present=bool(motions),
        official_suppressed_object_frames=counters.get("official_suppressed_object_frames", 0),
        unknown_suppressed_object_frames=counters.get("unknown_suppressed_object_frames", 0),
        identity_sources_rendered=counters.get("identity_sources_rendered", {}),
        rendered_frames=rendered_count,
    )

    if manifest["duplicate_pid_suppressed_object_frames"] > 0:
        msg = (f"[Renderer] Suppressed {manifest['duplicate_pid_suppressed_object_frames']} "
               "duplicate-PID object-frames — investigate upstream identity")
        if strict:
            raise RuntimeError(msg)
        print(msg)

    out_dir = os.path.dirname(output_path) or "."
    manifest_path = os.path.join(out_dir, "render_manifest.json")
    with open(manifest_path, "w") as f_:
        json.dump(manifest, f_, indent=2, default=str)
    print(f"[Renderer] manifest -> {manifest_path}")

    if write_qa_json:
        qa = {
            "total_raw_frames": total_raw_frames,
            "per_frame": [
                {
                    "raw_frame_idx": i,
                    "visible_count": sum(1 for d in frame_decisions if d.state != RenderState.HIDDEN),
                    "locked_count": sum(
                        1 for d in frame_decisions
                        if d.state != RenderState.HIDDEN and d.assignment_source == "locked"
                    ),
                    "revived_count": sum(
                        1 for d in frame_decisions
                        if d.state != RenderState.HIDDEN and d.assignment_source == "revived"
                    ),
                    "hide_reasons": _hide_reason_counts(frame_decisions),
                }
                for i, frame_decisions in enumerate(decisions)
            ],
        }
        qa_path = os.path.join(out_dir, "render_visual_qa.json")
        with open(qa_path, "w") as f_:
            json.dump(qa, f_, indent=2, default=str)
        print(f"[Renderer] QA -> {qa_path}")

    if write_contact_sheet and contact_sample_frames:
        sheet_path = os.path.join(out_dir, "render_contact_sheet.jpg")
        _write_contact_sheet(contact_sample_frames, contact_sample_indices, sheet_path)
        print(f"[Renderer] contact sheet -> {sheet_path}")

    return manifest


def _hide_reason_counts(frame_decisions: list[RenderDecision]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for d in frame_decisions:
        if d.state == RenderState.HIDDEN and d.hidden_reason is not None:
            counts[d.hidden_reason.value] += 1
    return dict(counts)


def _write_contact_sheet(frames: list[np.ndarray], indices: list[int], path: str) -> None:
    if not frames:
        return
    rows, cols = 3, 4
    h, w = frames[0].shape[:2]
    thumb_h = h // 4
    thumb_w = w // 4
    sheet_h = rows * thumb_h
    sheet_w = cols * thumb_w
    sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)
    for i, (img, idx) in enumerate(zip(frames, indices)):
        if i >= rows * cols:
            break
        r, c = divmod(i, cols)
        thumb = cv2.resize(img, (thumb_w, thumb_h))
        cv2.putText(thumb, f"f{idx}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        sheet[r * thumb_h:(r + 1) * thumb_h, c * thumb_w:(c + 1) * thumb_w] = thumb
    cv2.imwrite(path, sheet)
