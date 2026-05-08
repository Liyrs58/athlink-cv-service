"""Performance Zone overlay renderer — UEFA-style broadcast graphics.

Story-based clip:
  - foot rings only on ball carrier + nearest k players per team within radius_m
  - pass arrow on pass events
  - dashed carry path for the carrier
  - top-left caption (BUILD UP / TRANSITION / HIGH PRESS / CHANCE CREATED)
  - no debug text in product mode
  - optional manual frame_range; otherwise auto-pick the densest action window

Default usage:
    render_performance_zone(
        video_path="clip.mov",
        results_json="temp/job/tracking/track_results.json",
        pitch_map_json="temp/job/pitch/pitch_map.json",
        team_results_json="temp/job/tracking/team_results.json",
        events_json="temp/job/events/event_timeline.json",   # NEW
        output_path="clip_pz.mp4",
        # auto_clip=True picks an 8s window from events; pass frame_range to override
    )
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from services.tactical_overlay import (
    CAPTION_FOR_EVENT,  # noqa: F401  (re-exported for downstream)
    caption_for_window,
    draw_arrow,
    draw_banner,
    draw_blocked_lane,
    draw_caption,
    draw_curved_arrow,
    draw_dashed_path,
    draw_glow_line,
    draw_local_callout,
    draw_role_pill,
    draw_zone_hull,
)
from services.world_to_image import (
    invert_homography,
    project_world_point,
    project_world_radius,
)


# Team palette in BGR
TEAM_HOME = (255, 99, 29)    # blue
TEAM_AWAY = (37, 197, 34)    # green
NEUTRAL = (200, 200, 200)
BALL = (40, 220, 255)        # warm yellow

RING_RADIUS_M = 0.55
RING_THICKNESS = 3
RING_SEGMENTS = 16
RING_FILLED_RATIO = 0.65

DEFAULT_RADIUS_M = 12.0
DEFAULT_K_PER_TEAM = 4


# ── Loaders ───────────────────────────────────────────────────────────────

def _load_team_map(team_json_path: str) -> Dict[int, int]:
    p = Path(team_json_path)
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            data = json.load(f)
        out: Dict[int, int] = {}
        if isinstance(data, list):
            entries = data
        else:
            entries = data.get("tracks", []) or data.get("teams", []) or []
        for entry in entries:
            tid = entry.get("trackId") or entry.get("track_id")
            team = entry.get("teamId", entry.get("team_id"))
            if tid is not None and team is not None:
                out[int(tid)] = int(team)
        return out
    except Exception:
        return {}


def _load_pitch_map(pitch_json_path: str):
    p = Path(pitch_json_path)
    if not p.exists():
        return {}, {}, {}
    with open(p) as f:
        data = json.load(f)
    homographies_inv: Dict[int, np.ndarray] = {}
    for fi_str, H_list in (data.get("homographies") or {}).items():
        try:
            H = np.asarray(H_list, dtype=np.float64)
            inv = invert_homography(H)
            if inv is not None:
                homographies_inv[int(fi_str)] = inv
        except Exception:
            continue
    world_pos: Dict[Tuple[int, int], Tuple[float, float]] = {}
    ball_pos: Dict[int, Tuple[float, float]] = {}
    for entry in data.get("players", []):
        if entry.get("is_ball"):
            for pt in entry.get("trajectory2d", []) or []:
                ball_pos[int(pt["frameIndex"])] = (float(pt["x"]), float(pt["y"]))
            continue
        tid = entry.get("trackId")
        if tid is None:
            continue
        for pt in entry.get("trajectory2d", []) or []:
            world_pos[(int(tid), int(pt["frameIndex"]))] = (float(pt["x"]), float(pt["y"]))
    return homographies_inv, world_pos, ball_pos


def _load_events(events_json_path: Optional[str]) -> List[dict]:
    if not events_json_path:
        return []
    p = Path(events_json_path)
    if not p.exists():
        return []
    try:
        with open(p) as f:
            data = json.load(f)
    except Exception:
        return []
    out: List[dict] = []
    if isinstance(data, dict):
        for key in ("pass_events", "shot_events", "tackle_events", "turnover_events",
                    "carry_events", "events"):
            arr = data.get(key)
            if isinstance(arr, list):
                # tag the type if missing
                tag = key.replace("_events", "")
                for e in arr:
                    if "type" not in e:
                        e = {**e, "type": tag}
                    out.append(e)
    elif isinstance(data, list):
        out.extend(data)
    return out


def _load_roster(roster_json_path: Optional[str]) -> Dict[int, dict]:
    if not roster_json_path:
        return {}
    p = Path(roster_json_path)
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            data = json.load(f)
    except Exception:
        return {}
    out: Dict[int, dict] = {}
    for entry in data.get("players", []) or []:
        tid = entry.get("trackId")
        if tid is None:
            continue
        out[int(tid)] = entry
    return out


# ── Selection ─────────────────────────────────────────────────────────────

def _select_around_ball(
    frame_idx: int,
    world_pos: Dict[Tuple[int, int], Tuple[float, float]],
    ball_pos: Dict[int, Tuple[float, float]],
    team_map: Dict[int, int],
    *,
    radius_m: float = DEFAULT_RADIUS_M,
    k_per_team: int = DEFAULT_K_PER_TEAM,
) -> Tuple[Optional[int], List[int]]:
    """Returns (carrier_tid, [tids_to_render]) for this frame.
    Carrier = nearest player to the ball; selection = carrier + k nearest from
    each team within radius. When ball is missing entirely (no ball-model
    available), falls back to "centre of player cluster" so the renderer still
    produces a tactical view.
    """
    # Frame's player positions
    players_now = [
        (tid, (px, py))
        for (tid, fi), (px, py) in world_pos.items()
        if fi == frame_idx
    ]
    if not players_now:
        return None, []

    # Resolve ball position, with ±15 frame search, then cluster-centroid fallback
    ball = ball_pos.get(frame_idx)
    if ball is None:
        for delta in range(1, 16):
            if (b1 := ball_pos.get(frame_idx - delta)) is not None:
                ball = b1; break
            if (b2 := ball_pos.get(frame_idx + delta)) is not None:
                ball = b2; break
    if ball is None:
        # Cluster-centroid fallback: median of player positions = "where the action is"
        xs = sorted(p[0] for _, p in players_now)
        ys = sorted(p[1] for _, p in players_now)
        mid = len(xs) // 2
        ball = (xs[mid], ys[mid])
    bx, by = ball

    candidates = []
    for tid, (px, py) in players_now:
        d = ((px - bx) ** 2 + (py - by) ** 2) ** 0.5
        candidates.append((d, tid))
    candidates.sort()
    carrier_tid = candidates[0][1]
    selected = {carrier_tid}
    by_team: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
    for d, tid in candidates:
        if d > radius_m:
            continue
        team = team_map.get(int(tid), -1)
        by_team[team].append((d, tid))
    for team, lst in by_team.items():
        if team not in (0, 1, 2):
            continue
        for _, tid in lst[:k_per_team]:
            selected.add(tid)
    # Always have at least k_per_team * 2 visible players if available, so
    # when ball is unknown we still see 8 rings around the action centre.
    if len(selected) < min(8, len(candidates)):
        for d, tid in candidates[: 8]:
            if d <= radius_m * 1.3:
                selected.add(tid)
    return int(carrier_tid), list(selected)


# ── Drawing primitives ────────────────────────────────────────────────────

def _team_color(team_id: Optional[int]) -> Tuple[int, int, int]:
    if team_id == 0:
        return TEAM_HOME
    if team_id == 1:
        return TEAM_AWAY
    return NEUTRAL


def _draw_foot_ring(
    img: np.ndarray,
    centre_px: Tuple[int, int],
    radius_px: int,
    color: Tuple[int, int, int],
    *,
    style: str = "solid",
    thickness: int = RING_THICKNESS,
) -> None:
    cx, cy = centre_px
    rx = max(2, radius_px)
    ry = max(2, int(radius_px * 0.5))
    if style == "solid":
        cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, color, thickness, cv2.LINE_AA)
        return
    n = RING_SEGMENTS
    seg_deg = 360.0 / n
    keep_every_n = 2 if style == "dashed" else 3
    for k in range(n):
        if k % keep_every_n != 0:
            continue
        a0 = k * seg_deg
        a1 = a0 + seg_deg * RING_FILLED_RATIO
        cv2.ellipse(img, (cx, cy), (rx, ry), 0, a0, a1, color, thickness, cv2.LINE_AA)


def _draw_nameplate(
    img: np.ndarray,
    anchor_px: Tuple[int, int],
    color: Tuple[int, int, int],
    pid: str,
    *,
    name: Optional[str] = None,
    number: Optional[str] = None,
) -> None:
    cx, cy = anchor_px
    n_text = number or "".join(ch for ch in pid if ch.isdigit()) or "?"
    body_text = (name or pid).strip()

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.5
    th = 1
    (nw, hh), _ = cv2.getTextSize(n_text, font, fs, th)
    (bw, _), _ = cv2.getTextSize(body_text, font, fs, th)

    pad = 6
    h = hh + 2 * pad
    tab_w = nw + 2 * pad
    body_w = bw + 2 * pad
    plate_w = tab_w + body_w
    x1 = cx - plate_w // 2
    y2 = max(cy - 18, h)
    y1 = y2 - h
    x2 = x1 + plate_w

    cv2.rectangle(img, (x1, y1), (x2, y2), (245, 245, 245), -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1), (x1 + tab_w, y2), color, -1, cv2.LINE_AA)
    cv2.putText(img, n_text, (x1 + pad, y2 - pad), font, fs, (255, 255, 255), th + 1, cv2.LINE_AA)
    cv2.putText(img, body_text, (x1 + tab_w + pad, y2 - pad), font, fs, (40, 40, 40), th, cv2.LINE_AA)


def _draw_ball_trail(img: np.ndarray, trail: List[Tuple[int, int]], color: Tuple[int, int, int] = BALL) -> None:
    if not trail:
        return
    n = len(trail)
    for i, (x, y) in enumerate(trail):
        alpha = (i + 1) / n
        radius = max(2, int(3 + 4 * alpha))
        overlay = img.copy()
        cv2.circle(overlay, (x, y), radius, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)


# ── Stroke style ──────────────────────────────────────────────────────────

def _ring_style(p_entry: dict) -> Optional[str]:
    """Product mode: only locked or revived/hungarian. Uncertain players are
    not rendered at all (return None)."""
    if not p_entry.get("identity_valid", False):
        return None
    src = p_entry.get("assignment_source", "")
    if src == "locked":
        return "solid"
    return "dashed"


# ── Main render ───────────────────────────────────────────────────────────

def render_performance_zone(
    video_path: str,
    results_json: str,
    pitch_map_json: str,
    team_results_json: str,
    output_path: str,
    *,
    events_json: Optional[str] = None,
    roster_json: Optional[str] = None,
    frame_range: Optional[Tuple[int, int]] = None,
    auto_clip: bool = True,
    auto_clip_seconds: float = 8.0,
    radius_m: float = DEFAULT_RADIUS_M,
    k_per_team: int = DEFAULT_K_PER_TEAM,
    show_ball: bool = True,
    show_carry_path: bool = True,
    show_pass_arrows: bool = True,
    show_caption: bool = True,
    show_debug_chip: bool = False,
    sample_frames: Optional[Iterable[int]] = None,
    frame_dir: Optional[str] = None,
    verbose: bool = True,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    with open(results_json) as f:
        track_data = json.load(f)
    frames_data = {f["frameIndex"]: f for f in track_data.get("frames", [])}

    H_inv_by_frame, world_pos, ball_pos = _load_pitch_map(pitch_map_json)
    team_map = _load_team_map(team_results_json)
    events = _load_events(events_json)
    roster = _load_roster(roster_json)

    # Decide frame range
    lo, hi = 0, total - 1
    if frame_range is not None:
        lo, hi = int(frame_range[0]), int(frame_range[1])
    elif auto_clip:
        try:
            from services.tactics_clip_service import pick_clip_window
            base = Path(results_json).resolve().parents[2].name  # job_id
            win = pick_clip_window(base, win_s=auto_clip_seconds, fps=fps)
            if win:
                lo, hi = win
        except Exception as e:
            if verbose:
                print(f"[render_performance_zone] auto-clip failed: {e}")
    lo = max(0, lo)
    hi = min(total - 1, hi) if total else hi
    if verbose:
        print(f"[render_performance_zone] window=[{lo}, {hi}]  ({(hi - lo + 1) / fps:.1f}s)")

    # Caption for the window
    caption_text = caption_for_window(events, lo, hi) if (show_caption and events) else None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    sample_set = set(sample_frames or [])
    if frame_dir and sample_set:
        Path(frame_dir).mkdir(parents=True, exist_ok=True)

    # Index pass events by their nominal frame for arrow drawing
    pass_events_by_frame: Dict[int, List[dict]] = defaultdict(list)
    for e in events:
        if str(e.get("type")) == "pass":
            f = e.get("frame", e.get("frameIndex"))
            if f is None:
                continue
            try:
                pass_events_by_frame[int(f)].append(e)
            except Exception:
                pass

    # Pre-compute carrier trajectories so carry path is smooth
    def _carrier_path_world(carrier_tid: int, frame_idx: int, lookback: int = 12) -> List[Tuple[float, float]]:
        pts = []
        for k in range(lookback, -1, -1):
            wp = world_pos.get((carrier_tid, frame_idx - k))
            if wp is not None:
                pts.append(wp)
        return pts

    ball_trail: List[Tuple[int, int]] = []
    frame_idx = 0
    written = 0
    extracted = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx < lo or frame_idx > hi:
            frame_idx += 1
            continue
        rec = frames_data.get(frame_idx, {})

        H_inv = H_inv_by_frame.get(frame_idx)
        if H_inv is None and H_inv_by_frame:
            nearest = min(H_inv_by_frame.keys(), key=lambda k: abs(k - frame_idx))
            H_inv = H_inv_by_frame[nearest]

        # Selection
        carrier_tid, selected_tids = _select_around_ball(
            frame_idx, world_pos, ball_pos, team_map,
            radius_m=radius_m, k_per_team=k_per_team,
        )
        selected_set = set(selected_tids)

        # Build a quick lookup of player entry by raw tid
        by_tid: Dict[int, dict] = {}
        for p in rec.get("players", []):
            if p.get("is_official", False):
                continue
            tid = p.get("rawTrackId") or p.get("trackId")
            if tid is None:
                continue
            by_tid[int(tid)] = p

        # ── Carry path (drawn first, behind rings) ──
        if show_carry_path and carrier_tid is not None and H_inv is not None:
            world_pts = _carrier_path_world(int(carrier_tid), frame_idx, lookback=12)
            pixel_pts: List[Tuple[int, int]] = []
            for wp in world_pts:
                pp = project_world_point(wp, H_inv)
                if pp is not None:
                    pixel_pts.append(pp)
            carrier_team = team_map.get(int(carrier_tid), -1)
            draw_dashed_path(frame, pixel_pts, _team_color(carrier_team), thickness=2)

        # ── Pass arrows in a small window around current frame ──
        if show_pass_arrows and H_inv is not None:
            for f_off in range(-6, 7):
                for ev in pass_events_by_frame.get(frame_idx + f_off, []):
                    passer_tid = ev.get("carrier_id") or ev.get("passer_id") or ev.get("from")
                    receiver_tid = ev.get("receiver_id") or ev.get("to")
                    if passer_tid is None or receiver_tid is None:
                        continue
                    try:
                        p_w = world_pos.get((int(passer_tid), frame_idx + f_off))
                        r_w = world_pos.get((int(receiver_tid), frame_idx + f_off))
                    except Exception:
                        continue
                    if p_w is None or r_w is None:
                        continue
                    p_px = project_world_point(p_w, H_inv)
                    r_px = project_world_point(r_w, H_inv)
                    if p_px is None or r_px is None:
                        continue
                    team = team_map.get(int(passer_tid), -1)
                    draw_arrow(frame, p_px, r_px, _team_color(team), thickness=3)
                    break  # one arrow per frame is enough

        # ── Player rings + nameplates ──
        for tid in selected_set:
            p = by_tid.get(int(tid))
            if not p:
                continue
            style = _ring_style(p)
            if style is None:
                continue
            wp = world_pos.get((int(tid), frame_idx))
            centre_px = None
            radius_px = None
            if wp is not None and H_inv is not None:
                centre_px = project_world_point(wp, H_inv)
                radius_px = project_world_radius(RING_RADIUS_M, wp, H_inv)
            if centre_px is None or radius_px is None:
                bbox = p.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = map(int, bbox)
                centre_px = ((x1 + x2) // 2, y2)
                bw = max(1, x2 - x1)
                radius_px = max(8, int(bw * 0.35))

            team = team_map.get(int(tid))
            color = _team_color(team)
            _draw_foot_ring(frame, centre_px, radius_px, color, style=style)

            # Nameplate ONLY for the ball carrier in product mode
            if int(tid) == carrier_tid:
                rinfo = roster.get(int(tid)) or {}
                name = rinfo.get("name")
                number = str(rinfo.get("number")) if rinfo.get("number") is not None else None
                pid = p.get("playerId") or p.get("displayId") or f"P{tid}"
                _draw_nameplate(frame, centre_px, color, str(pid), name=name, number=number)

        # ── Ball trail ──
        if show_ball:
            wpb = ball_pos.get(frame_idx)
            if wpb is not None and H_inv is not None:
                bxy = project_world_point(wpb, H_inv)
                if bxy is not None:
                    ball_trail.append(bxy)
            if len(ball_trail) > 8:
                ball_trail = ball_trail[-8:]
            _draw_ball_trail(frame, ball_trail)

        # ── Caption ──
        if show_caption and caption_text:
            accent_team = team_map.get(int(carrier_tid)) if carrier_tid is not None else -1
            accent = _team_color(accent_team) if accent_team in (0, 1) else None
            draw_caption(frame, caption_text, accent=accent)

        # ── Debug chip (off by default) ──
        if show_debug_chip:
            n_locked = sum(1 for p in rec.get("players", []) if p.get("assignment_source") == "locked")
            n_revived = sum(1 for p in rec.get("players", []) if p.get("assignment_source") == "revived")
            n_uncertain = sum(
                1 for p in rec.get("players", [])
                if not p.get("identity_valid", False) and not p.get("is_official", False)
            )
            chip = f"F{frame_idx}  locked={n_locked} revived={n_revived} uncertain={n_uncertain}"
            cv2.putText(frame, chip, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        out.write(frame)
        written += 1
        if frame_dir and frame_idx in sample_set:
            cv2.imwrite(str(Path(frame_dir) / f"pz_frame_{frame_idx:05d}.jpg"), frame)
            extracted += 1
        frame_idx += 1

    cap.release()
    out.release()
    if verbose:
        print(f"[render_performance_zone] {written} frames -> {output_path}")
        if extracted:
            print(f"[render_performance_zone] {extracted} samples -> {frame_dir}")


# ── Phase 4: hardcoded story renderer ─────────────────────────────────────

ARROW_KIND_COLORS = {
    "pressure": (56, 56, 240),     # red — defenders closing carrier
    "pass":     None,              # resolved per arrow from carrier team
    "run":      None,              # resolved per arrow from cast team
    "support":  (40, 220, 255),    # ball-yellow
}

ROLE_RING_RADIUS_MULT = {
    "CARRIER":   1.55,
    "DEFENDER":  1.05,
    "RUNNER":    1.25,
    "OPTION":    1.05,
}


def _hex_to_bgr(hex_str: str) -> Tuple[int, int, int]:
    s = hex_str.lstrip("#")
    if len(s) != 6:
        return NEUTRAL
    r = int(s[0:2], 16); g = int(s[2:4], 16); b = int(s[4:6], 16)
    return (b, g, r)


def _foot_pixel(
    p_entry: dict,
    world_pos: Dict[Tuple[int, int], Tuple[float, float]],
    frame_idx: int,
    H_inv: Optional[np.ndarray],
) -> Optional[Tuple[int, int]]:
    """Best foot-point estimate for a player record on this frame."""
    tid = p_entry.get("rawTrackId") or p_entry.get("trackId")
    if tid is None:
        return None
    wp = world_pos.get((int(tid), frame_idx))
    if wp is not None and H_inv is not None:
        px = project_world_point(wp, H_inv)
        if px is not None:
            return px
    bbox = p_entry.get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = map(int, bbox)
    return ((x1 + x2) // 2, y2)


def _resolve_blocked_option(
    carrier_tid: int,
    anchor_frame: int,
    world_pos: Dict[Tuple[int, int], Tuple[float, float]],
    team_map: Dict[int, int],
    *,
    max_dist_m: float = 12.0,
) -> Optional[int]:
    """Find the carrier's nearest teammate at the anchor frame.
    Returns the trackId or None if no teammate is within max_dist_m."""
    cw = world_pos.get((int(carrier_tid), int(anchor_frame)))
    if cw is None:
        return None
    carrier_team = team_map.get(int(carrier_tid))
    best = None
    best_d = max_dist_m
    for (tid, fi), (x, y) in world_pos.items():
        if fi != anchor_frame or tid == carrier_tid:
            continue
        if team_map.get(int(tid)) != carrier_team:
            continue
        d = float(np.hypot(x - cw[0], y - cw[1]))
        if d <= best_d:
            best_d = d
            best = int(tid)
    return best


def _touchline_pixels(
    H_inv: np.ndarray,
    carrier_world: Optional[Tuple[float, float]],
    pitch_w: float = 105.0,
    pitch_h: float = 68.0,
    image_w: int = 1920,
    image_h: int = 1080,
    *,
    segment_len_m: float = 18.0,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Return (p1, p2) image-space endpoints of a SHORT touchline segment near
    the carrier. The segment is `segment_len_m` metres long along the world
    touchline, centred on the carrier's x-coord projected onto y=0 or y=68.

    Returns None if the carrier isn't reasonably close to a touchline (>14 m).
    """
    if carrier_world is None:
        return None
    cx_world, cy_world = carrier_world
    # Pick whichever touchline (y=0 or y=68) the carrier is closer to
    y_top = 0.0
    y_bot = pitch_h
    near_top = abs(cy_world - y_top)
    near_bot = abs(cy_world - y_bot)
    y_world = y_top if near_top <= near_bot else y_bot
    if min(near_top, near_bot) > 14.0:
        return None  # carrier is mid-pitch, touchline isn't part of the trap

    # Carrier projected onto the touchline (in world coords)
    half = segment_len_m / 2.0
    x_a = max(0.0, cx_world - half)
    x_b = min(pitch_w, cx_world + half)
    try:
        p1 = project_world_point((x_a, y_world), H_inv)
        p2 = project_world_point((x_b, y_world), H_inv)
    except Exception:
        return None
    if p1 is None or p2 is None:
        return None

    def _clamp(p):
        x = max(0, min(image_w - 1, int(p[0])))
        y = max(0, min(image_h - 1, int(p[1])))
        return (x, y)
    return _clamp(p1), _clamp(p2)


def _shrink_polygon(points_px: List[Tuple[int, int]], factor: float = 0.72) -> List[Tuple[int, int]]:
    """Shrink a polygon toward its centroid. `factor=0.72` keeps the trap zone
    tight to the cast instead of ballooning out to wide convex-hull edges."""
    if not points_px:
        return points_px
    arr = np.array(points_px, dtype=np.float64)
    cx, cy = arr.mean(axis=0)
    return [
        (int(round(cx + (x - cx) * factor)), int(round(cy + (y - cy) * factor)))
        for (x, y) in points_px
    ]


def render_story(
    video_path: str,
    results_json: str,
    pitch_map_json: str,
    team_results_json: str,
    story_json: str,
    output_path: str,
    *,
    verbose: bool = True,
) -> dict:
    """Anchor-frozen tactical clip render.

    Computes ring/zone/arrow/touchline geometry ONCE at the story's anchor
    frame, then draws the same overlay on every frame in frame_range.
    The broadcast plays underneath; the explanation stays still.

    Story JSON schema (Phase 4.3):
        {
          "title": "PRESSING TRAP",
          "subtitle": "TOUCHLINE 2V1",
          "frame_range": [140, 215],
          "anchor_frame": 165,            # optional, default = midpoint
          "fps": 25,
          "cast": [
            {"trackId": 587, "role": "BALL CARRIER"},
            {"trackId": 622, "role": "PRESSER"},
            {"trackId": 596, "role": "COVER"}
          ],
          "blocked_option_track": null,   # optional, null = auto-pick
          "show_blocked_lane": true,
          "show_touchline": true,
          "dim_alpha": 0.32,
          "zone_color": "#ff5544",
          "zone_alpha": 0.22,
          "callouts": [
            {"text": "PRESSING TRAP", "position": "top",     "frames":[140,215]},
            {"text": "TOUCHLINE 2V1", "position": "top_sub", "frames":[140,215]}
          ]
        }

    Returns a small validation manifest dict.
    """
    with open(story_json) as f:
        story = json.load(f)

    # ── Story config ──────────────────────────────────────────────────────
    frame_lo, frame_hi = story.get("frame_range", [0, 10**9])
    anchor_frame = int(story.get("anchor_frame") or (frame_lo + frame_hi) // 2)

    cast_list = story.get("cast", []) or []
    # Map roles into a canonical key and an optional display label override.
    # We accept any case; convert to upper for matching against ROLE_RING_RADIUS_MULT.
    raw_cast: List[Tuple[int, str]] = []
    for entry in cast_list:
        tid = entry.get("trackId")
        if tid is None:
            continue
        raw_cast.append((int(tid), str(entry.get("role", "PLAYER")).upper()))

    # Find carrier (any "CARRIER" role variant — "BALL CARRIER" maps to CARRIER)
    def _normalise_role(r: str) -> str:
        r = r.upper()
        if "CARRIER" in r:
            return "CARRIER"
        if "PRESSER" in r or r == "DEFENDER":
            return "DEFENDER"
        if r in ("COVER", "COVER DEFENDER"):
            return "DEFENDER"
        if "RUNNER" in r:
            return "RUNNER"
        if "BLOCKED" in r:
            return "OPTION"
        return r

    callouts = story.get("callouts", []) or []
    show_touchline = bool(story.get("show_touchline", True))
    show_blocked_lane = bool(story.get("show_blocked_lane", True))
    dim_alpha = float(story.get("dim_alpha", 0.32))
    zone_color = _hex_to_bgr(story.get("zone_color", "#ff5544"))
    zone_alpha = float(story.get("zone_alpha", 0.22))

    # ── Load track + pitch + team data ────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    with open(results_json) as f:
        track_data = json.load(f)
    frames_data = {int(fr["frameIndex"]): fr for fr in track_data.get("frames", [])}

    H_inv_by_frame, world_pos, _ball_pos = _load_pitch_map(pitch_map_json)
    team_map = _load_team_map(team_results_json)

    # ── Resolve anchor-frame H_inv ────────────────────────────────────────
    H_inv_anchor = H_inv_by_frame.get(anchor_frame)
    if H_inv_anchor is None and H_inv_by_frame:
        nearest = min(H_inv_by_frame.keys(), key=lambda k: abs(k - anchor_frame))
        H_inv_anchor = H_inv_by_frame[nearest]

    # ── Anchor-frame foot pixels for everyone in the JSON ────────────────
    anchor_rec = frames_data.get(anchor_frame, {})
    anchor_foot: Dict[int, Tuple[int, int]] = {}
    for p in anchor_rec.get("players", []):
        if p.get("is_official", False):
            continue
        tid = p.get("rawTrackId") or p.get("trackId")
        if tid is None:
            continue
        fp = _foot_pixel(p, world_pos, anchor_frame, H_inv_anchor)
        if fp is not None:
            anchor_foot[int(tid)] = fp

    # ── Carrier + cast resolution (with auto-resolve BLOCKED OPTION) ─────
    carrier_tid: Optional[int] = None
    cast_with_label: List[Tuple[int, str, str]] = []  # (tid, role_key, display_label)
    for tid, raw_role in raw_cast:
        norm = _normalise_role(raw_role)
        if norm == "CARRIER" and carrier_tid is None:
            carrier_tid = tid
        if tid in anchor_foot:
            cast_with_label.append((tid, norm, raw_role))

    if carrier_tid is None:
        raise RuntimeError("story has no CARRIER role")

    carrier_world = world_pos.get((int(carrier_tid), anchor_frame))

    # ── Tactical-accuracy filter ─────────────────────────────────────────
    # Drop cast members who are too far from the carrier at the anchor
    # frame to belong to the trap. The user reported the COVER label landing
    # on a defender far from the action; this filter removes that case.
    # Carrier is never dropped; OPTION is allowed up to 14 m (it's a passing
    # option, not a presser).
    presser_max_m = float(story.get("presser_max_m", 9.0))
    option_max_m = float(story.get("option_max_m", 14.0))
    if carrier_world is not None:
        kept: List[Tuple[int, str, str]] = []
        dropped_reasons: List[str] = []
        for (tid, role_key, display) in cast_with_label:
            if role_key == "CARRIER":
                kept.append((tid, role_key, display)); continue
            wp = world_pos.get((int(tid), anchor_frame))
            if wp is None:
                kept.append((tid, role_key, display)); continue
            d = float(np.hypot(wp[0] - carrier_world[0], wp[1] - carrier_world[1]))
            limit = option_max_m if role_key == "OPTION" else presser_max_m
            if d <= limit:
                kept.append((tid, role_key, display))
            else:
                dropped_reasons.append(f"{display}/tid={tid} {d:.1f}m>{limit}m")
        cast_with_label = kept
        if dropped_reasons and verbose:
            print(f"[render_story] dropped from cast: {dropped_reasons}")

    # Auto-resolve BLOCKED OPTION if not explicitly cast (and not already filtered out)
    blocked_tid_explicit = story.get("blocked_option_track")
    blocked_tid: Optional[int] = None
    if blocked_tid_explicit is not None:
        blocked_tid = int(blocked_tid_explicit)
    else:
        if not any(r == "OPTION" for _, r, _ in cast_with_label):
            blocked_tid = _resolve_blocked_option(
                carrier_tid, anchor_frame, world_pos, team_map, max_dist_m=12.0
            )
            if blocked_tid is not None and blocked_tid in anchor_foot:
                cast_with_label.append((blocked_tid, "OPTION", "BLOCKED OPTION"))

    # ── Pre-validate: minimum geometry must be present ────────────────────
    pressers = [t for (t, r, _) in cast_with_label if r == "DEFENDER"]
    if len(pressers) < 1:
        raise RuntimeError(
            "anchor frame does not satisfy minimum trap geometry "
            f"(no PRESSER/DEFENDER within {presser_max_m}m of carrier at frame {anchor_frame}). "
            "Try a different anchor_frame or relax presser_max_m in the story."
        )

    # ── Build OverlayPlan: cache every drawing primitive's pixel coords ──
    plan: Dict[str, object] = {}

    # Rings + role pills
    ring_specs: List[dict] = []
    for tid, role_key, display in cast_with_label:
        fp = anchor_foot.get(int(tid))
        if fp is None:
            continue
        team = team_map.get(int(tid))
        color = _team_color(team)
        base_radius = 24
        wp = world_pos.get((int(tid), anchor_frame))
        if wp is not None and H_inv_anchor is not None:
            rr = project_world_radius(RING_RADIUS_M, wp, H_inv_anchor)
            if rr is not None:
                base_radius = int(rr)
        mult = ROLE_RING_RADIUS_MULT.get(role_key, 1.0)
        radius = int(base_radius * mult)
        ring_specs.append({
            "tid": int(tid),
            "fp": fp,
            "radius": radius,
            "color": color,
            "role_key": role_key,
            "label": display,
        })
    plan["rings"] = ring_specs

    # Zone hull — carrier + pressers only (no OPTION). Shrink toward centroid
    # so the zone reads as the LOCAL trap, not the convex hull of half the pitch.
    zone_tids = [t for (t, r, _) in cast_with_label if r in ("CARRIER", "DEFENDER")]
    zone_pts_raw = [anchor_foot[t] for t in zone_tids if t in anchor_foot]
    if len(zone_pts_raw) >= 3:
        zone_shrink = float(story.get("zone_shrink", 0.72))
        plan["zone_pts"] = _shrink_polygon(zone_pts_raw, factor=zone_shrink)
    else:
        plan["zone_pts"] = []

    # Pressure arrows: each PRESSER -> CARRIER
    arrow_specs: List[dict] = []
    carrier_fp = anchor_foot.get(int(carrier_tid))
    for i, ptid in enumerate(pressers):
        pfp = anchor_foot.get(int(ptid))
        if pfp is None or carrier_fp is None:
            continue
        bend = 0.30 if i % 2 == 0 else -0.30
        arrow_specs.append({
            "from": pfp,
            "to": carrier_fp,
            "color": ARROW_KIND_COLORS["pressure"],
            "bend": bend,
        })
    plan["arrows"] = arrow_specs

    # Blocked passing lane: CARRIER -> BLOCKED OPTION (red dashed with X)
    blocked_lane = None
    if show_blocked_lane and blocked_tid is not None and blocked_tid in anchor_foot:
        blocked_lane = (carrier_fp, anchor_foot[int(blocked_tid)])
    plan["blocked_lane"] = blocked_lane

    # Touchline glow — short segment near the carrier, NOT a full sideline.
    touchline = None
    if show_touchline and H_inv_anchor is not None and carrier_world is not None:
        touchline = _touchline_pixels(
            H_inv_anchor, carrier_world,
            image_w=w, image_h=h,
            segment_len_m=float(story.get("touchline_segment_m", 18.0)),
        )
    plan["touchline"] = touchline

    # ── Per-frame loop: draw the cached plan, no recomputation ──────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    written = 0
    frame_idx = 0
    if verbose:
        print(
            f"[render_story] anchor_frame={anchor_frame}  window=[{frame_lo}, {frame_hi}]  "
            f"({(frame_hi - frame_lo + 1) / fps:.1f}s)  "
            f"cast={[(t, r) for (t, r, _) in cast_with_label]}  "
            f"blocked_option={blocked_tid}  touchline={'yes' if touchline else 'no'}"
        )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if total and frame_idx >= total:
            break
        if frame_idx < frame_lo:
            frame_idx += 1
            continue
        if frame_idx > frame_hi:
            break

        # Step 1: dim
        if dim_alpha > 0:
            black = np.zeros_like(frame)
            cv2.addWeighted(black, dim_alpha, frame, 1.0 - dim_alpha, 0, frame)

        # Step 2: zone hull
        if plan["zone_pts"]:
            draw_zone_hull(frame, plan["zone_pts"], zone_color, alpha=zone_alpha)

        # Step 3: touchline glow — short, thinner segment near the carrier
        if plan["touchline"]:
            tl_a, tl_b = plan["touchline"]
            draw_glow_line(frame, tl_a, tl_b, ARROW_KIND_COLORS["pressure"], thickness=6, glow_layers=2)

        # Step 4: pressure arrows (curved)
        for a in plan["arrows"]:
            draw_curved_arrow(
                frame, a["from"], a["to"], a["color"],
                thickness=4, bend=a["bend"], stop_short_px=32,
            )

        # Step 5: blocked passing lane
        if plan["blocked_lane"] is not None:
            ba, bb = plan["blocked_lane"]
            draw_blocked_lane(frame, ba, bb, color=ARROW_KIND_COLORS["pressure"], thickness=3)

        # Step 6: rings + role pills (BLOCKED OPTION pill goes BELOW its ring
        # so it doesn't pile on top of the BALL CARRIER pill / blocked-lane chip)
        for r in plan["rings"]:
            fp = r["fp"]
            color = r["color"]
            if r["role_key"] == "CARRIER":
                inner = max(3, r["radius"] // 3)
                cv2.ellipse(frame, fp, (inner, inner // 2), 0, 0, 360, color, -1, cv2.LINE_AA)
            _draw_foot_ring(frame, fp, r["radius"], color, style="solid", thickness=4)
            below = r["role_key"] == "OPTION"
            draw_role_pill(
                frame, fp, r["label"], color,
                above_offset_px=r["radius"] + 12, below=below,
            )

        # Step 7: title + subtitle banners (rendered every frame)
        for c in callouts:
            c_lo, c_hi = c.get("frames", [frame_lo, frame_hi])
            if not (c_lo <= frame_idx <= c_hi):
                continue
            text = str(c.get("text", "")).strip()
            if not text:
                continue
            position = str(c.get("position", "top"))
            draw_banner(frame, text, position=position)

        # Step 8: anchored callouts derived from the plan
        if plan["touchline"]:
            tl_a, tl_b = plan["touchline"]
            # Anchor at the midpoint of the short segment, slightly above-line
            tl_mx = (tl_a[0] + tl_b[0]) // 2
            tl_my = (tl_a[1] + tl_b[1]) // 2 - 18
            draw_local_callout(
                frame, (tl_mx, tl_my), "TOUCHLINE = EXTRA DEFENDER",
                side="left", accent=ARROW_KIND_COLORS["pressure"],
            )
        if plan["blocked_lane"] is not None and carrier_fp is not None:
            ba, bb = plan["blocked_lane"]
            mx = (ba[0] + bb[0]) // 2
            # Push the chip perpendicularly off the line so it doesn't sit on top of the X marker
            dy = bb[1] - ba[1]
            offset = 22 if dy >= 0 else -22
            my = (ba[1] + bb[1]) // 2 + offset
            draw_local_callout(
                frame, (mx, my), "PASSING LANE CLOSED",
                side="right", accent=ARROW_KIND_COLORS["pressure"],
            )

        out.write(frame)
        written += 1
        frame_idx += 1

    cap.release()
    out.release()

    # ── Validation manifest ───────────────────────────────────────────────
    manifest = {
        "anchor_frame": anchor_frame,
        "frames_written": written,
        "total_frames_window": frame_hi - frame_lo + 1,
        "rings": len(plan["rings"]),
        "arrows": len(plan["arrows"]),
        "zone_pts": len(plan["zone_pts"]) if plan["zone_pts"] else 0,
        "touchline": bool(plan["touchline"]),
        "blocked_lane": bool(plan["blocked_lane"]),
        "carrier_tid": carrier_tid,
        "blocked_tid": blocked_tid,
    }
    if verbose:
        print(f"[render_story] manifest={manifest}")
        if manifest["rings"] < 3:
            print(f"[render_story] WARN rings={manifest['rings']} < 3 — story too thin")
        if manifest["arrows"] < 1:
            print(f"[render_story] WARN no arrows in plan")
        if manifest["zone_pts"] < 3:
            print(f"[render_story] WARN no zone hull (need >=3 points at anchor)")
        if not manifest["touchline"]:
            print(f"[render_story] WARN touchline not resolved (homography missing?)")
    return manifest


if __name__ == "__main__":
    import sys
    job = sys.argv[1] if len(sys.argv) > 1 else "stride5_test"
    video = sys.argv[2] if len(sys.argv) > 2 else "/Users/rudra/Downloads/villa_psg_30s.mp4"
    out = sys.argv[3] if len(sys.argv) > 3 else f"temp/{job}/pz.mp4"
    render_performance_zone(
        video_path=video,
        results_json=f"temp/{job}/tracking/track_results.json",
        pitch_map_json=f"temp/{job}/pitch/pitch_map.json",
        team_results_json=f"temp/{job}/tracking/team_results.json",
        events_json=f"temp/{job}/events/event_timeline.json",
        output_path=out,
    )
