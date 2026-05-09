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
    draw_dashed_path,
    draw_glow_line,
    draw_local_callout,
    draw_player_trail,
    draw_role_pill,
    draw_zone_hull,
)
from services.broadcast_compositor import (
    LAYER_GLOW,
    LAYER_LABELS,
    LAYER_LINES,
    LAYER_RINGS,
    LAYER_TITLES,
    LAYER_ZONE_FILL,
    LabelSpec,
    OverlayCompositor,
    solve_label_positions,
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


def _draw_glow_line_alpha(
    img: np.ndarray,
    p_from: Tuple[int, int],
    p_to: Tuple[int, int],
    color: Tuple[int, int, int],
    *,
    thickness: int = 10,
    alpha: float = 1.0,
    glow_layers: int = 3,
) -> None:
    """Alpha-aware version of draw_glow_line for animated reveals."""
    if p_from is None or p_to is None or alpha <= 0:
        return
    if alpha >= 0.999:
        draw_glow_line(img, p_from, p_to, color, thickness=thickness, glow_layers=glow_layers)
        return
    overlay = img.copy()
    draw_glow_line(overlay, p_from, p_to, color, thickness=thickness, glow_layers=glow_layers)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)


def _draw_partial_curved_arrow(
    img: np.ndarray,
    start_px: Tuple[int, int],
    end_px: Tuple[int, int],
    color: Tuple[int, int, int],
    *,
    thickness: int = 4,
    bend: float = 0.25,
    stop_short_px: int = 28,
    progress: float = 1.0,
) -> None:
    """Draw only the first `progress` [0,1] fraction of a bezier arrow.
    At progress=1 this is identical to draw_curved_arrow."""
    if start_px is None or end_px is None or progress <= 0:
        return
    sx, sy = start_px
    ex, ey = end_px
    dx, dy = ex - sx, ey - sy
    length = float(np.hypot(dx, dy))
    if length < 4:
        return
    trim = min(stop_short_px, int(length * 0.4))
    ex2 = int(ex - dx * (trim / length))
    ey2 = int(ey - dy * (trim / length))
    mx, my = (sx + ex2) / 2.0, (sy + ey2) / 2.0
    nx, ny = -(ey2 - sy) / max(1.0, length), (ex2 - sx) / max(1.0, length)
    cx_ = int(mx + nx * length * bend)
    cy_ = int(my + ny * length * bend)
    n_pts = 25
    max_i = max(1, int(n_pts * progress))
    pts = []
    for i in range(min(max_i + 1, n_pts + 1)):
        t = i / float(n_pts)
        u = 1.0 - t
        bx = u * u * sx + 2 * u * t * cx_ + t * t * ex2
        by = u * u * sy + 2 * u * t * cy_ + t * t * ey2
        pts.append((int(bx), int(by)))
    # Shadow first
    for i in range(1, len(pts)):
        cv2.line(img, pts[i - 1], pts[i], (0, 0, 0), thickness + 3, cv2.LINE_AA)
    for i in range(1, len(pts)):
        cv2.line(img, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)
    # Arrowhead only when near-complete
    if progress >= 0.9 and len(pts) >= 2:
        cv2.arrowedLine(img, pts[-2], pts[-1], color, thickness, tipLength=0.6, line_type=cv2.LINE_AA)


def _draw_blocked_lane_alpha(
    img: np.ndarray,
    p_from: Tuple[int, int],
    p_to: Tuple[int, int],
    color: Tuple[int, int, int],
    *,
    thickness: int = 3,
    alpha: float = 1.0,
) -> None:
    """Alpha-aware blocked-lane dashed line + X marker."""
    if p_from is None or p_to is None or alpha <= 0:
        return
    if alpha >= 0.999:
        draw_blocked_lane(img, p_from, p_to, color, thickness=thickness)
        return
    overlay = img.copy()
    draw_blocked_lane(overlay, p_from, p_to, color, thickness=thickness)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)


def _draw_banner_alpha(
    img: np.ndarray,
    text: str,
    *,
    position: str = "top",
    alpha: float = 1.0,
) -> None:
    """Alpha-fade version of draw_banner for title animation."""
    if not text or alpha <= 0:
        return
    if alpha >= 0.999:
        draw_banner(img, text, position=position)
        return
    overlay = img.copy()
    draw_banner(overlay, text, position=position)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)



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
    "COVER":     1.00,
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


def _touchline_world(
    carrier_world: Optional[Tuple[float, float]],
    pitch_w: float = 105.0,
    pitch_h: float = 68.0,
    *,
    segment_len_m: float = 18.0,
    max_dist_m: float = 6.0,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Return WORLD-COORD endpoints of a short touchline segment near the
    carrier.  The segment is `segment_len_m` metres long along whichever
    touchline (y=0 or y=68) the carrier is closer to, centred on the
    carrier's x-coord.

    Returns None if the carrier is more than `max_dist_m` metres from either
    touchline. Default is 6 m for pressing-trap stories. Story authors can
    relax this for `WIDE_CHANNEL_CHASE` stories via story.touchline_max_m.
    Pixel projection happens per-frame in the render loop.
    """
    if carrier_world is None:
        return None
    cx, cy = carrier_world
    y_top, y_bot = 0.0, pitch_h
    near_top = abs(cy - y_top)
    near_bot = abs(cy - y_bot)
    y_tl = y_top if near_top <= near_bot else y_bot
    if min(near_top, near_bot) > max_dist_m:
        return None  # carrier is too far from any touchline for it to be part of the story
    half = segment_len_m / 2.0
    x_a = max(0.0, cx - half)
    x_b = min(pitch_w, cx + half)
    return ((x_a, y_tl), (x_b, y_tl))


def _shrink_polygon(points: List[Tuple[float, float]], factor: float = 0.72) -> List[Tuple[float, float]]:
    """Shrink a polygon toward its centroid. Works in any coordinate space
    (world metres or image pixels). `factor=0.72` keeps the trap zone tight."""
    if not points:
        return points
    arr = np.array(points, dtype=np.float64)
    cx, cy = arr.mean(axis=0)
    return [
        (float(cx + (x - cx) * factor), float(cy + (y - cy) * factor))
        for (x, y) in points
    ]


def render_story(
    video_path: str,
    results_json: str,
    pitch_map_json: str,
    team_results_json: str,
    story_json: str,
    output_path: str,
    *,
    ball_assignments_json: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Anchor-derived, per-frame-reprojected tactical clip render.

    Overlay GEOMETRY is computed in **world** coordinates at the story's
    anchor frame (cast trackIds, zone hull world points, arrow world
    endpoints, touchline world segment, blocked-lane track endpoints).
    On every frame the renderer looks up that frame's homography and
    reprojects each primitive to image space — players move, the camera
    pans/zooms, and the overlays follow the pitch (NOT a fixed screen
    position).

    Story validity is checked at the anchor frame against geometric
    preconditions (e.g. PRESSING_TRAP requires carrier near touchline,
    >=2 pressers within 8 m, good angle spread, tight zone). If the
    user's anchor doesn't satisfy them, the renderer scans the clip for
    a better anchor; if none qualifies, the story is honestly retitled
    via STORY_TYPE_TITLE — unless `strict_story_type` is set in the JSON,
    in which case it raises.

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
            return "COVER"
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
    render_mode = str(story.get("render_mode", "live_reprojected"))
    debug_allow_invalid_story = bool(story.get("debug_allow_invalid_story", False))

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

    H_inv_by_frame, world_pos, ball_pos = _load_pitch_map(pitch_map_json)
    team_map = _load_team_map(team_results_json)

    # ── Optional: override ball_pos from Phase 4.7 ball_assignments.json ──
    # ball_assignments has authoritative per-frame {ball_world, ball_image,
    # ball_source, carrier_tid, carrier_confidence}. When supplied, prefer
    # ball_world directly; if absent but ball_image present, project through
    # H to world. Build per-frame side dict for the manifest.
    carrier_by_frame: Dict[int, dict] = {}
    ball_interp_stats = {"interpolated_ball_frames": 0, "interpolated_carrier_frames": 0}
    if ball_assignments_json:
        try:
            with open(ball_assignments_json) as f:
                ba_data = json.load(f)
            ba_entries = ba_data if isinstance(ba_data, list) else ba_data.get("assignments", [])
            with open(pitch_map_json) as f:
                pm = json.load(f)
            H_by_frame: Dict[int, np.ndarray] = {}
            for fi_str, H_list in (pm.get("homographies") or {}).items():
                try:
                    H_by_frame[int(fi_str)] = np.asarray(H_list, dtype=np.float64)
                except Exception:
                    continue
            override: Dict[int, Tuple[float, float]] = {}
            for entry in ba_entries:
                fi = entry.get("frameIndex")
                if fi is None:
                    fi = entry.get("frame_index")
                if fi is None:
                    continue
                fi = int(fi)
                src = entry.get("ball_source", "missing")
                if src == "missing":
                    continue
                bw = entry.get("ball_world")
                if bw and len(bw) == 2:
                    override[fi] = (float(bw[0]), float(bw[1]))
                else:
                    bi = entry.get("ball_image")
                    if bi and len(bi) == 2:
                        H = H_by_frame.get(fi)
                        if H is None and H_by_frame:
                            nearest = min(H_by_frame.keys(), key=lambda k: abs(k - fi))
                            H = H_by_frame[nearest]
                        if H is not None:
                            try:
                                pt = np.array([float(bi[0]), float(bi[1]), 1.0])
                                wp = H @ pt
                                if wp[2] != 0:
                                    override[fi] = (float(wp[0] / wp[2]), float(wp[1] / wp[2]))
                            except Exception:
                                pass
                carrier_by_frame[fi] = entry

            # ── Ball interpolation across short gaps (4.7c6.2) ───────────
            # Linear-interpolate ball_world AND propagate carrier metadata
            # through gaps ≤ MAX_GAP frames. Prevents the per-frame validity
            # gate from skipping otherwise-good frames just because the
            # detector blinked for 1-8 frames.
            MAX_GAP = int(story.get("ball_interp_max_gap", 8))
            interp_ball = 0
            interp_carrier = 0
            if override and MAX_GAP > 0:
                detected_frames = sorted(override.keys())
                for k in range(len(detected_frames) - 1):
                    f0, f1 = detected_frames[k], detected_frames[k + 1]
                    gap = f1 - f0 - 1
                    if gap <= 0 or gap > MAX_GAP:
                        continue
                    x0, y0 = override[f0]
                    x1, y1 = override[f1]
                    src_carrier = carrier_by_frame.get(f0, {})
                    dst_carrier = carrier_by_frame.get(f1, {})
                    # Use the source frame's carrier_tid if both endpoints
                    # agree; otherwise fall back to the source's.
                    src_tid = src_carrier.get("carrier_tid")
                    dst_tid = dst_carrier.get("carrier_tid")
                    inherit_tid = src_tid if src_tid == dst_tid else src_tid
                    src_conf = float(src_carrier.get("carrier_confidence", 0) or 0)
                    dst_conf = float(dst_carrier.get("carrier_confidence", 0) or 0)
                    for j in range(1, gap + 1):
                        fi_mid = f0 + j
                        if fi_mid in override:
                            continue  # respected if real detection exists
                        t = j / (gap + 1)
                        override[fi_mid] = (
                            x0 + (x1 - x0) * t,
                            y0 + (y1 - y0) * t,
                        )
                        interp_ball += 1
                        # Propagate a synthetic carrier_by_frame entry so the
                        # per-frame gate sees ball_source="interpolated" and
                        # carrier_conf decays linearly across the gap.
                        if inherit_tid is not None and fi_mid not in carrier_by_frame:
                            mid_conf = src_conf + (dst_conf - src_conf) * t
                            # Apply a small penalty so interpolated frames
                            # don't pretend to be as confident as detector.
                            mid_conf = max(0.0, mid_conf - 0.05)
                            carrier_by_frame[fi_mid] = {
                                "frameIndex": fi_mid,
                                "ball_source": "interpolated",
                                "carrier_tid": int(inherit_tid),
                                "carrier_confidence": round(mid_conf, 3),
                                "ball_world": list(override[fi_mid]),
                                "interpolated": True,
                                "gap_size": gap,
                            }
                            interp_carrier += 1

            ball_interp_stats["interpolated_ball_frames"] = interp_ball
            ball_interp_stats["interpolated_carrier_frames"] = interp_carrier
            if override:
                ball_pos = override
                if verbose:
                    print(
                        f"[render_story] ball_pos overridden from ball_assignments.json: "
                        f"{len(override)} frames "
                        f"(interpolated +{interp_ball} ball, +{interp_carrier} carrier)"
                    )

            # ── Carrier-tid temporal smoother (4.7c8) ──────────────────
            # Collapse 1-3 frame carrier_tid flips before downstream
            # hysteresis sees them. Window = 2*W+1 frames.
            try:
                from services.ball_assignments_smoother import (
                    smooth_carrier_assignments,
                )
                smoother_window = int(story.get("carrier_smoother_window", 6))
                carrier_by_frame, smoother_stats = smooth_carrier_assignments(
                    carrier_by_frame, window=smoother_window
                )
                ball_interp_stats["smoother_stats"] = smoother_stats
                if verbose:
                    print(
                        f"[ball_smoother] {smoother_stats['total_frames']} frames, "
                        f"{smoother_stats['tid_switches_in_raw']} raw tid switches "
                        f"→ {smoother_stats['tid_switches_in_smoothed']} smoothed "
                        f"({smoother_stats['noise_reduction_ratio']*100:.0f}% noise reduction)"
                    )
            except Exception as e:
                if verbose:
                    print(f"[render_story] WARN carrier smoother failed: {e}")
        except Exception as e:
            if verbose:
                print(f"[render_story] WARN failed to load ball_assignments.json: {e}")

    # ── Ball-data pre-flight ──────────────────────────────────────────────
    # Classify ball tracking quality at load time. Anything below "detector"
    # strips claims that depend on knowing who has the ball.
    win_frames = frame_hi - frame_lo + 1
    ball_frames_in_window = sum(
        1 for fi in ball_pos.keys() if frame_lo <= fi <= frame_hi
    ) if ball_pos else 0
    if ball_frames_in_window == 0:
        ball_source = "missing"
        ball_confidence = 0.0
    elif ball_frames_in_window / max(1, win_frames) >= 0.6:
        ball_source = "detector"
        ball_confidence = round(ball_frames_in_window / max(1, win_frames), 2)
    else:
        ball_source = "fallback"
        ball_confidence = round(ball_frames_in_window / max(1, win_frames), 2)

    ball_ok = ball_source in ("detector", "fallback")
    if not ball_ok:
        if verbose:
            print(
                f"[render_story] WARN ball_source='{ball_source}' — "
                "stripping BALL CARRIER pill, PASSING LANE CLOSED, blocked lane. "
                "Set debug_allow_invalid_story=true to override."
            )
        # Strip claims that require knowing who has the ball
        if not debug_allow_invalid_story:
            show_blocked_lane = False
            # Relabel CARRIER to just 'PLAYER' so we don't lie
            callouts = [
                {**c, "text": "PRESSURE SEQUENCE"}
                if str(c.get("position")) == "top" else c
                for c in callouts
            ]

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
    # Fallback: project world_pos through H_inv for any cast tid that is
    # missing from anchor_foot. tracker_core may have skipped a frame even
    # though pitch_service has a world position for that tid+frame.
    for tid, raw_role in raw_cast:
        norm = _normalise_role(raw_role)
        if norm == "CARRIER" and carrier_tid is None:
            carrier_tid = tid
        if tid not in anchor_foot:
            wp = world_pos.get((int(tid), anchor_frame))
            if wp is not None and H_inv_anchor is not None:
                px = project_world_point(wp, H_inv_anchor)
                if px is not None:
                    anchor_foot[int(tid)] = px
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
            print(f"[render_story] dropped from cast (anchor F={anchor_frame}): {dropped_reasons}")

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

    # Initialise `pressers` from current cast — guarantees the variable is
    # always bound regardless of which validator branch we take below.
    pressers = [t for (t, r, _) in cast_with_label if r in ("DEFENDER", "COVER")]

    # ── Validity-aware relocator ──────────────────────────────────────────
    # The story is validated against the geometric preconditions of its
    # declared type (pressing trap = near touchline, 2+ pressers, good angle
    # spread, tight zone). If the user's anchor doesn't satisfy them, we scan
    # all frames for the best match. If none is valid, we either RETITLE the
    # story honestly (default) or RAISE if `strict_story_type` is set.
    from services.story_validators import (  # local import to avoid cycles
        STORY_TYPE_TITLE,
        soft_score_pressing_trap,
        validate_pressing_trap,
    )

    # Derive declared_story_type from the story JSON title so callers can
    # author PRESSING_TRIANGLE / WIDE_CHANNEL_CHASE etc. directly.
    _title_to_type = {v[0].upper(): k for k, v in STORY_TYPE_TITLE.items()}
    _raw_title = str(story.get("title", "PRESSING TRAP")).upper()
    declared_story_type = _title_to_type.get(_raw_title, "PRESSING_TRAP")
    strict_story_type = bool(story.get("strict_story_type", False))
    declared_defender_tids = [
        tid for (tid, raw_role) in raw_cast
        if _normalise_role(raw_role) in ("DEFENDER", "COVER")
    ]

    def _validate_at(fi: int) -> Tuple[bool, str, dict]:
        from services.story_validators import (
            validate_pressing_triangle, validate_goalmouth_pressure,
            validate_transition_carry,
        )
        if declared_story_type == "PRESSING_TRIANGLE":
            return validate_pressing_triangle(world_pos, fi, int(carrier_tid), declared_defender_tids)
        if declared_story_type == "GOALMOUTH_PRESSURE":
            return validate_goalmouth_pressure(world_pos, fi, int(carrier_tid), declared_defender_tids)
        if declared_story_type == "TRANSITION_CARRY":
            return validate_transition_carry(world_pos, fi, int(carrier_tid), declared_defender_tids)
        return validate_pressing_trap(
            world_pos, fi, int(carrier_tid), declared_defender_tids
        )

    lock_anchor = bool(story.get("lock_anchor", False))

    is_valid, recommended_type, geom = _validate_at(anchor_frame)
    if verbose:
        print(
            f"[render_story] validator @ user anchor F={anchor_frame}: "
            f"valid={is_valid} type={recommended_type} geom={geom}"
        )

    if lock_anchor and not is_valid:
        if verbose:
            print(f"[render_story] lock_anchor=True — skipping relocator, keeping F={anchor_frame}")

    if not is_valid and not lock_anchor:
        # Scan frame_range first, then full video, scoring every frame
        scan_ranges = [
            (max(0, frame_lo), min(total - 1, frame_hi) if total else frame_hi),
        ]
        if total:
            scan_ranges.append((0, total - 1))

        best_fi = None
        best_score = -1.0
        best_geom = None
        best_type = recommended_type
        best_valid = False
        seen_fi: set = set()
        for scan_lo, scan_hi in scan_ranges:
            for fi in range(scan_lo, scan_hi + 1):
                if fi in seen_fi:
                    continue
                seen_fi.add(fi)
                v, t, g = _validate_at(fi)
                s = soft_score_pressing_trap(g)
                # Prefer valid frames; among valid, take the highest soft score
                priority = (1.0 if v else 0.0) * 1000 + s
                if priority > best_score:
                    best_score = priority
                    best_fi = fi
                    best_geom = g
                    best_type = t
                    best_valid = v
            if best_valid:
                break  # don't widen if we found a valid anchor in the original range

        if best_fi is None:
            if strict_story_type:
                raise RuntimeError(
                    f"Story declared {declared_story_type} but no frame in "
                    f"[{frame_lo}, {frame_hi}] satisfies the geometry, and "
                    "strict_story_type is set."
                )
            best_fi = anchor_frame  # keep original
        if verbose:
            print(
                f"[render_story] relocator picked F={best_fi}  valid={best_valid}  "
                f"type={best_type}  score={best_score:.2f}  geom={best_geom}"
            )

        # Move anchor to the best frame
        if best_fi != anchor_frame:
            if best_fi < frame_lo or best_fi > frame_hi:
                half_win = (frame_hi - frame_lo) // 2
                frame_lo = max(0, best_fi - half_win)
                frame_hi = min((total - 1) if total else 10**9, best_fi + half_win)
                if verbose:
                    print(
                        f"[render_story] frame_range auto-expanded to "
                        f"[{frame_lo}, {frame_hi}] to include new anchor"
                    )
            anchor_frame = best_fi
            # Re-derive anchor foot pixels at the new anchor frame
            H_inv_anchor = H_inv_by_frame.get(anchor_frame)
            if H_inv_anchor is None and H_inv_by_frame:
                nearest = min(H_inv_by_frame.keys(), key=lambda k: abs(k - anchor_frame))
                H_inv_anchor = H_inv_by_frame[nearest]
            anchor_rec = frames_data.get(anchor_frame, {})
            anchor_foot = {}
            for p in anchor_rec.get("players", []):
                if p.get("is_official", False):
                    continue
                tid_p = p.get("rawTrackId") or p.get("trackId")
                if tid_p is None:
                    continue
                fp = _foot_pixel(p, world_pos, anchor_frame, H_inv_anchor)
                if fp is not None:
                    anchor_foot[int(tid_p)] = fp
            carrier_world = world_pos.get((int(carrier_tid), anchor_frame))
            cast_with_label = []
            for tid, raw_role in raw_cast:
                norm = _normalise_role(raw_role)
                if tid in anchor_foot:
                    cast_with_label.append((tid, norm, raw_role))
            if carrier_world is not None:
                kept = []
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
                cast_with_label = kept
            if not any(r == "OPTION" for _, r, _ in cast_with_label):
                blocked_tid_re = _resolve_blocked_option(
                    carrier_tid, anchor_frame, world_pos, team_map, max_dist_m=12.0
                )
                if blocked_tid_re is not None and blocked_tid_re in anchor_foot:
                    cast_with_label.append((blocked_tid_re, "OPTION", "BLOCKED OPTION"))
                    if blocked_tid_explicit is None:
                        blocked_tid = blocked_tid_re
            pressers = [t for (t, r, _) in cast_with_label if r in ("DEFENDER", "COVER")]
            geom = best_geom
            is_valid = best_valid
            recommended_type = best_type
        else:
            geom = best_geom or geom
            is_valid = best_valid
            recommended_type = best_type

        # If still invalid, downgrade title (or raise under strict mode)
        if not is_valid:
            if strict_story_type:
                raise RuntimeError(
                    f"Story declared {declared_story_type} but the best available "
                    f"frame F={anchor_frame} only qualifies as {recommended_type}, "
                    "and strict_story_type is set."
                )
            new_title, new_subtitle = STORY_TYPE_TITLE.get(
                recommended_type, STORY_TYPE_TITLE["1V1_PRESSURE"]
            )
            # Mutate callouts in place: 'top' -> new_title, 'top_sub' -> new_subtitle
            for c in callouts:
                pos = str(c.get("position", "top"))
                if pos == "top":
                    c["text"] = new_title
                elif pos == "top_sub":
                    c["text"] = new_subtitle
            if verbose:
                print(
                    f"[render_story] retitled story_type={declared_story_type} "
                    f"-> {recommended_type}"
                )

    if len(pressers) < 1:
        # No declared defender survived the distance filter at the chosen
        # anchor. We have already retitled to an honest type (1V1_PRESSURE
        # default). Keep going — the carrier ring + retitled banner is
        # still a useful render. Strict mode raises.
        if strict_story_type:
            raise RuntimeError(
                f"No declared defender (cast tids: {[t for (t, r, _) in cast_with_label]}) "
                f"is within {presser_max_m}m of carrier tid={carrier_tid} at frame "
                f"{anchor_frame}. The trackIds in your story may not match "
                "this pipeline run's track_results. Re-pick tids from "
                f"temp/{Path(results_json).parent.parent.name}/tracking/track_results.json."
            )
        if verbose:
            print(
                f"[render_story] WARN no declared defender within {presser_max_m}m of "
                f"carrier at F={anchor_frame}. Continuing with carrier-only render. "
                "Likely cause: trackIds in story do not match this pipeline run."
            )

    # Final validator pass at the resolved anchor — used by the manifest
    final_valid, final_type, final_geom = _validate_at(anchor_frame)

    # ── Product-mode gate ────────────────────────────────────────────────
    # A story passes product mode if:
    #   (a) geometry is satisfied for SOME known story type (final_valid=True),
    #       even if the declared type was downgraded to an honest type, AND
    #   (b) ball tracking is sufficient (ball_ok).
    # Requiring declared_type == final_type would reject every honest retitle,
    # which defeats the whole point of the validity gate.
    story_is_valid = bool(final_valid) and ball_ok
    story_failed_reason: Optional[str] = None
    if not story_is_valid:
        failing_reasons = []
        if not final_valid:
            failing_reasons.append(
                f"story_type={declared_story_type} failed geometry — "
                f"best frame F={anchor_frame} qualifies as '{final_type}'. "
                f"Geometry: {final_geom}"
            )
        if not ball_ok:
            failing_reasons.append(
                f"ball_source='{ball_source}' (confidence={ball_confidence:.0%}) — "
                "ball tracking too sparse to claim a tactical story."
            )
        story_failed_reason = "; ".join(failing_reasons)

    if not story_is_valid and not debug_allow_invalid_story:
        cap.release()
        raise RuntimeError(
            "[render_story] Story failed product-mode gate — no MP4 written.\n"
            + "\n".join(f"  • {r}" for r in failing_reasons)
            + "\nSet debug_allow_invalid_story=true in the story JSON to force a debug render."
        )
    if verbose and not story_is_valid:
        print(
            f"[render_story] DEBUG MODE: story_valid=False but debug_allow_invalid_story=True. "
            f"Rendering anyway. ball_source={ball_source} final_type={final_type}"
        )

    # ── Build OverlayPlan: world-space geometry (reprojected per-frame) ───
    plan: Dict[str, object] = {}

    # Rings: store tid + visual metadata — world pos resolved per-frame
    ring_specs: List[dict] = []
    for tid, role_key, display in cast_with_label:
        team = team_map.get(int(tid))
        color = _team_color(team)
        ring_specs.append({
            "tid": int(tid),
            "color": color,
            "role_key": role_key,
            "label": display,
        })
    plan["rings"] = ring_specs

    # Zone hull in WORLD coords — carrier + pressers only (no OPTION).
    # Shrink toward centroid so the zone reads as the LOCAL trap.
    zone_tids = [t for (t, r, _) in cast_with_label if r in ("CARRIER", "DEFENDER", "COVER")]
    zone_world_raw = [world_pos.get((int(t), anchor_frame)) for t in zone_tids]
    zone_world_raw_filtered = [wp for wp in zone_world_raw if wp is not None]
    if len(zone_world_raw_filtered) >= 3:
        zone_shrink = float(story.get("zone_shrink", 0.72))
        plan["zone_world"] = _shrink_polygon(zone_world_raw_filtered, factor=zone_shrink)
    else:
        plan["zone_world"] = []
        # Image-space fallback: if at least 3 cast members have anchor_foot
        # pixels but world_pos lookup failed for some, build hull in pixels so
        # the triangle still renders. This will NOT reproject through the
        # camera pan (it stays at anchor view), but it proves the geometry
        # exists where freeze-frame rendering is used.
        anchor_px_pts = [anchor_foot.get(int(t)) for t in zone_tids]
        anchor_px_pts = [p for p in anchor_px_pts if p is not None]
        if len(anchor_px_pts) >= 3:
            plan["zone_image_fallback"] = anchor_px_pts
            if verbose:
                print(
                    f"[render_story] WARN zone_world has only "
                    f"{len(zone_world_raw_filtered)}/{len(zone_tids)} world points at "
                    f"F={anchor_frame}; using image-space hull fallback "
                    f"({len(anchor_px_pts)} pixel points)."
                )
        else:
            plan["zone_image_fallback"] = []

    # Pressure arrows: store track IDs, not pixel coords
    arrow_specs: List[dict] = []
    for i, ptid in enumerate(pressers):
        bend = 0.30 if i % 2 == 0 else -0.30
        arrow_specs.append({
            "from_tid": int(ptid),
            "to_tid": int(carrier_tid),
            "color": ARROW_KIND_COLORS["pressure"],
            "bend": bend,
        })
    plan["arrows"] = arrow_specs

    # Blocked passing lane: track IDs (resolved per-frame)
    if show_blocked_lane and blocked_tid is not None:
        plan["blocked_lane"] = (int(carrier_tid), int(blocked_tid))
    else:
        plan["blocked_lane"] = None

    # Touchline glow — short segment in WORLD coords near the carrier.
    # Default max distance is 6m (pressing-trap geometry). Story authors can
    # relax for WIDE_CHANNEL_CHASE etc. via touchline_max_m.
    touchline_world = None
    if show_touchline and carrier_world is not None:
        touchline_world = _touchline_world(
            carrier_world,
            segment_len_m=float(story.get("touchline_segment_m", 18.0)),
            max_dist_m=float(story.get("touchline_max_m", 6.0)),
        )
    plan["touchline_world"] = touchline_world

    # ── Ball-dot pre-flight ──────────────────────────────────────────────
    show_ball = bool(story.get("show_ball", True))
    ball_in_window = False
    if show_ball and ball_pos:
        ball_in_window = any(frame_lo <= fi <= frame_hi for fi in ball_pos.keys())
        if not ball_in_window and verbose:
            print(
                f"[render_story] WARN ball_pos has {len(ball_pos)} entries but "
                f"none inside window [{frame_lo}, {frame_hi}] — ball dot will be skipped"
            )
    elif show_ball and verbose:
        print("[render_story] WARN ball_pos empty — ball dot will be skipped")

    # ── Animation timeline constants ─────────────────────────────────────
    # Each entry is (start_frame_offset, end_frame_offset) within the clip.
    # In freeze_frame mode frame_offset = output_frame_index (0-based).
    # In live_reprojected mode frame_offset = frame_idx - frame_lo.
    ANIM = {
        "title_fade":    (0,   10),
        "carrier_ring":  (10,  20),
        "presser_rings": (18,  30),
        "arrows_draw":   (28,  45),
        "zone_fade":     (40,  55),
        "blocked_lane":  (52,  68),
        "callouts":      (65,  78),
    }

    # Live-mode timeline (frame_offset = frame_idx - frame_lo). Tuned for
    # broadcast pacing: title slides in instantly, cast staggers over the
    # first second, everything fades out in the last 12 frames.
    _live_window_len = max(1, frame_hi - frame_lo + 1)
    # Only run outro fade if the clip is long enough that intro + outro
    # don't overlap. Below 24 frames, skip outro entirely (factor stays 1).
    if _live_window_len >= 24:
        _live_outro_lo = _live_window_len - 12
        _live_outro_hi = _live_window_len - 1
    else:
        _live_outro_lo = _live_window_len  # never triggers
        _live_outro_hi = _live_window_len
    ANIM_LIVE = {
        "title_fade":     (0,                       12),
        "subtitle_fade":  (8,                       22),
        "carrier_ring":   (4,                       12),
        "presser_rings":  (10,                      18),  # PRESSER (offset 0)
        "presser_rings2": (12,                      20),  # COVER  (offset +2 stagger)
        "zone_fade":      (14,                      24),
        "arrows_draw":    (18,                      30),
        "blocked_lane":   (28,                      38),
        "trails_fade":    (6,                       16),
        "outro_fade":     (_live_outro_lo,          _live_outro_hi),
    }

    def _smoothstep(t: float) -> float:
        """Cubic smoothstep easing: 0→1 with zero derivative at endpoints."""
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)

    def _anim_alpha(key: str, offset: int) -> float:
        """Return [0,1] progress for a named animation stage at this offset.
        Reads ANIM (freeze-frame). Live mode uses _anim_alpha_live."""
        start, end = ANIM[key]
        if offset < start:
            return 0.0
        if offset >= end:
            return 1.0
        return _smoothstep((offset - start) / max(1, end - start))

    def _anim_alpha_live(key: str, offset: int) -> float:
        """Live-mode alpha. Same as _anim_alpha but reads ANIM_LIVE."""
        if key not in ANIM_LIVE:
            return 1.0
        start, end = ANIM_LIVE[key]
        if offset < start:
            return 0.0
        if offset >= end:
            return 1.0
        return _smoothstep((offset - start) / max(1, end - start))

    def _live_outro_factor(offset: int) -> float:
        """Returns 1.0 during the body of the clip, ramps down 1→0 in the
        last 12 frames so the overlay fades out cleanly."""
        if offset < _live_outro_lo:
            return 1.0
        if offset >= _live_outro_hi:
            return 0.0
        t = (offset - _live_outro_lo) / max(1, _live_outro_hi - _live_outro_lo)
        return 1.0 - _smoothstep(t)

    # ── Freeze-frame: capture anchor background once ──────────────────────
    freeze_bg: Optional[np.ndarray] = None
    anim_frames = int(story.get("anim_frames", 90))  # output frame count in FF mode
    if render_mode == "freeze_frame":
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(anchor_frame))
        ok, freeze_bg = cap.read()
        if not ok or freeze_bg is None:
            cap.release()
            raise RuntimeError(
                f"freeze_frame: could not read anchor_frame={anchor_frame} from video"
            )
        # Dim the frozen background once
        if dim_alpha > 0:
            black = np.zeros_like(freeze_bg)
            cv2.addWeighted(black, dim_alpha, freeze_bg, 1.0 - dim_alpha, 0, freeze_bg)
        # Rewind for any later live reads (not needed in FF mode but keeps cap state clean)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
        if verbose:
            print(
                f"[render_story] freeze_frame mode: captured anchor F={anchor_frame}, "
                f"will output {anim_frames} animation frames"
            )

    # ── Per-frame loop: reproject world → image each frame ──────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    written = 0
    ball_dots_drawn = 0
    frame_idx = 0
    per_frame_trace: List[dict] = []
    skipped_overlay_frames: List[dict] = []
    _carrier_state: str = "lost"   # hysteresis: "locked" | "coasting" | "lost"
    _carrier_coast_frames: int = 0
    _CARRIER_COAST_MAX: int = int(story.get("carrier_coast_frames", 12))

    # ── Phase 4.7/4.8 stable-frames-before-show hysteresis ──────────────────
    # Required consecutive valid frames before the overlay first appears
    # (shorter for 1V1, longer for triangle/trap because the formation needs
    # to settle before we claim it). Hide after 3 consecutive invalid frames.
    _STABLE_REQUIRED_BY_TYPE = {
        "1V1_PRESSURE":     5,
        "PRESSING_TRIANGLE": 6,
        "PRESSING_TRAP":     8,
        "RECOVERY_RUN":      6,
    }
    _stable_required = _STABLE_REQUIRED_BY_TYPE.get(
        str(story.get("story_type", "")).upper(),
        int(story.get("stable_required_frames", 5)),
    )
    _stable_invalid_max = int(story.get("stable_invalid_frames", 3))
    _stable_valid_run = 0
    _stable_invalid_run = 0
    _overlay_visible = False  # gated by stable hysteresis, separate from carrier state

    # Hard global gate thresholds (override-able from story.json for debug)
    _GATE_MIN_CARRIER_CONF      = float(story.get("gate_min_carrier_conf", 0.65))
    _GATE_MAX_BALL_TO_CARRIER_M = float(story.get("gate_max_ball_to_carrier_m", 1.8))
    _GATE_HARD_BALL_M           = float(story.get("gate_hard_ball_to_carrier_m", 2.5))
    _GATE_MIN_HOMOGRAPHY_CONF   = float(story.get("gate_min_homography_conf", 0.85))
    _hard_fail_reason: Optional[str] = None  # set if BALL_TOO_FAR_FROM_CARRIER hard-fails the story
    # 4.7c6 polish state
    trails_px: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    label_collisions_resolved = 0
    trails_drawn_total = 0
    if verbose:
        print(
            f"[render_story] anchor_frame={anchor_frame}  window=[{frame_lo}, {frame_hi}]  "
            f"({(frame_hi - frame_lo + 1) / fps:.1f}s)  "
            f"cast={[(t, r) for (t, r, _) in cast_with_label]}  "
            f"blocked_option={blocked_tid}  touchline={'yes' if touchline_world else 'no'}"
        )

    # ---------------------
    # FREEZE-FRAME sub-loop
    # ---------------------
    if render_mode == "freeze_frame":
        assert freeze_bg is not None
        for anim_idx in range(anim_frames):
            frame = freeze_bg.copy()
            frame_offset = anim_idx  # animation time is anim_idx, not video frame

            # All world mapping uses anchor homography (camera frozen)
            H_inv = H_inv_by_frame.get(anchor_frame)
            if H_inv is None and H_inv_by_frame:
                nearest = min(H_inv_by_frame.keys(), key=lambda k: abs(k - anchor_frame))
                H_inv = H_inv_by_frame[nearest]

            def _proj(wp):
                if wp is None or H_inv is None:
                    return None
                return project_world_point(wp, H_inv)

            # Step 2: zone hull — fades in during zone_fade window
            zone_alpha_t = _anim_alpha("zone_fade", frame_offset)
            if plan["zone_world"] and zone_alpha_t > 0:
                zone_px = [_proj(wp) for wp in plan["zone_world"]]
                zone_px = [p for p in zone_px if p is not None]
                if len(zone_px) >= 3:
                    draw_zone_hull(frame, zone_px, zone_color, alpha=zone_alpha * zone_alpha_t)

            # Step 3: touchline glow — appears with presser_rings
            tl_alpha_t = _anim_alpha("presser_rings", frame_offset)
            tl_px = None
            if plan["touchline_world"] and H_inv is not None and tl_alpha_t > 0:
                tl_a_w, tl_b_w = plan["touchline_world"]
                tl_a = _proj(tl_a_w)
                tl_b = _proj(tl_b_w)
                if tl_a and tl_b:
                    tl_px = (tl_a, tl_b)
                    _draw_glow_line_alpha(frame, tl_a, tl_b, ARROW_KIND_COLORS["pressure"],
                                         thickness=10, alpha=tl_alpha_t)

            # Step 4: pressure arrows — draw progressively along bezier
            arrows_t = _anim_alpha("arrows_draw", frame_offset)
            if arrows_t > 0:
                for a in plan["arrows"]:
                    fp_from = world_pos.get((int(a["from_tid"]), anchor_frame))
                    fp_to = world_pos.get((int(a["to_tid"]), anchor_frame))
                    pa = _proj(fp_from)
                    pb = _proj(fp_to)
                    if pa and pb:
                        _draw_partial_curved_arrow(frame, pa, pb, a["color"],
                                                   bend=a["bend"], progress=arrows_t)

            # Step 5: blocked lane — dashes appear
            blocked_t = _anim_alpha("blocked_lane", frame_offset)
            if plan["blocked_lane"] and blocked_t > 0:
                carrier_fp_w = world_pos.get((int(plan["blocked_lane"][0]), anchor_frame))
                blocked_fp_w = world_pos.get((int(plan["blocked_lane"][1]), anchor_frame))
                ba = _proj(carrier_fp_w)
                bb = _proj(blocked_fp_w)
                if ba and bb:
                    _draw_blocked_lane_alpha(frame, ba, bb, ARROW_KIND_COLORS["pressure"],
                                            thickness=3, alpha=blocked_t)

            # Step 6: rings — carrier first, then pressers
            for r in plan["rings"]:
                is_carrier = r["role_key"] == "CARRIER"
                ring_key = "carrier_ring" if is_carrier else "presser_rings"
                ring_t = _anim_alpha(ring_key, frame_offset)
                if ring_t <= 0:
                    continue
                wp = world_pos.get((int(r["tid"]), anchor_frame))
                radius_m = RING_RADIUS_M * ROLE_RING_RADIUS_MULT.get(r["role_key"], 1.0)
                radius = project_world_radius(radius_m, wp, H_inv) if (wp is not None and H_inv is not None) else 28
                fp = _proj(wp) if wp else anchor_foot.get(int(r["tid"]))
                if fp is None:
                    continue
                # Pulse scale for carrier on first appear
                if is_carrier and ring_t < 1.0:
                    radius = int(radius * (0.6 + 0.4 * ring_t))
                r_color = tuple(int(c * ring_t + (1 - ring_t) * 255) for c in r["color"])
                _draw_foot_ring(frame, fp, radius, r["color"], style="solid",
                                thickness=max(1, int(RING_THICKNESS * ring_t)))
                if ring_t >= 0.6:
                    below = r["role_key"] == "OPTION"
                    pill_offset = max(int(radius * 0.55) + 12, 24)
                    draw_role_pill(frame, fp, r["label"], r["color"],
                                  above_offset_px=pill_offset, below=below)

            # Step 7: callouts (title/subtitle banners)
            callout_t = _anim_alpha("callouts", frame_offset)
            title_t = _anim_alpha("title_fade", frame_offset)
            for c in callouts:
                pos = str(c.get("position", "top"))
                t = title_t if pos == "top" else callout_t
                if t <= 0:
                    continue
                _draw_banner_alpha(frame, c["text"], position=pos, alpha=t)

            out.write(frame)
            written += 1
        cap.release()

    else:
        # --------------------------
        # LIVE-REPROJECTED sub-loop
        # --------------------------
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

            # animation offset from start of window (for live mode timeline)
            frame_offset = frame_idx - frame_lo

            # ── Resolve this frame's homography (world → image) ──────────
            H_inv = H_inv_by_frame.get(frame_idx)
            if H_inv is None and H_inv_by_frame:
                nearest = min(H_inv_by_frame.keys(), key=lambda k: abs(k - frame_idx))
                H_inv = H_inv_by_frame[nearest]

            # Step 1: dim background (always, even on invalid overlay frames)
            if dim_alpha > 0:
                black = np.zeros_like(frame)
                cv2.addWeighted(black, dim_alpha, frame, 1.0 - dim_alpha, 0, frame)

            # Helper: project a world point for this frame
            def _proj(wp):
                if wp is None or H_inv is None:
                    return None
                return project_world_point(wp, H_inv)

            # ── Per-frame truth lookup (4.7c7) ───────────────────────────
            ba_entry = carrier_by_frame.get(frame_idx, {}) if carrier_by_frame else {}
            frame_ball_source = ba_entry.get("ball_source", "missing")

            # Carrier confidence: use ball_assignments only when its carrier_tid
            # matches the declared story carrier. Foreign carriers (noisy nearest-
            # player assignments) must not reset the hysteresis state.
            ba_carrier_tid_raw = ba_entry.get("carrier_tid")
            ba_carrier_tid = int(ba_carrier_tid_raw) if ba_carrier_tid_raw is not None else None
            if ba_carrier_tid == int(carrier_tid):
                frame_carrier_conf = float(ba_entry.get("carrier_confidence", 0) or 0)
            else:
                # Foreign assignment — confidence decays to 0.45 (below locked, above lost)
                frame_carrier_conf = 0.45
            frame_carrier_tid = int(carrier_tid)  # always lock to declared carrier

            frame_ball_world = ball_pos.get(frame_idx) if ball_pos else None
            if frame_ball_world is None and ball_pos:
                # Tolerance: ±10 frames so the ball doesn't strobe on a 1-frame gap
                for delta in range(1, 11):
                    if (b1 := ball_pos.get(frame_idx - delta)) is not None:
                        frame_ball_world = b1; break
                    if (b2 := ball_pos.get(frame_idx + delta)) is not None:
                        frame_ball_world = b2; break
            frame_ball_px = _proj(frame_ball_world) if frame_ball_world else None

            # Carrier world pos — always use declared story carrier_tid.
            carrier_for_frame = int(carrier_tid)
            carrier_w = world_pos.get((carrier_for_frame, frame_idx))
            carrier_fp = _proj(carrier_w)

            # Active pressers AT THIS FRAME (declared pressers within range
            # of the current carrier, in world coords).
            frame_pressers: List[int] = []
            ball_carrier_dist_m: Optional[float] = None
            if carrier_w is not None:
                cx_w, cy_w = carrier_w
                for ptid in declared_defender_tids:
                    pw = world_pos.get((int(ptid), frame_idx))
                    if pw is None:
                        continue
                    d = float(np.hypot(pw[0] - cx_w, pw[1] - cy_w))
                    if d <= presser_max_m:
                        frame_pressers.append(int(ptid))
                if frame_ball_world is not None:
                    ball_carrier_dist_m = float(
                        np.hypot(frame_ball_world[0] - cx_w, frame_ball_world[1] - cy_w)
                    )

            # Image-space ball-vs-carrier sanity (catches bad H or bad carrier_tid)
            BALL_CARRIER_MAX_PX = float(story.get("ball_carrier_max_px", 90.0))
            ball_carrier_dist_px: Optional[float] = None
            if carrier_fp is not None and frame_ball_px is not None:
                ball_carrier_dist_px = float(
                    np.hypot(carrier_fp[0] - frame_ball_px[0], carrier_fp[1] - frame_ball_px[1])
                )

            MIN_CARRIER_CONF = float(story.get("min_carrier_conf_per_frame", 0.55))
            MIN_CARRIER_COAST = MIN_CARRIER_CONF - 0.15  # hysteresis floor

            # ── Carrier hysteresis ──────────────────────────────────────────
            if frame_carrier_conf >= MIN_CARRIER_CONF and frame_carrier_tid is not None:
                _carrier_state = "locked"
                _carrier_coast_frames = 0
            elif frame_carrier_conf >= MIN_CARRIER_COAST and _carrier_state in ("locked", "coasting"):
                _carrier_state = "coasting"
                _carrier_coast_frames += 1
                if _carrier_coast_frames > _CARRIER_COAST_MAX:
                    _carrier_state = "lost"
            else:
                _carrier_state = "lost"
                _carrier_coast_frames = 0

            carrier_valid = _carrier_state in ("locked", "coasting")

            # ── Ball tolerance: missing OK if carrier is locked/coasting ────
            ball_ok = (
                frame_ball_source != "missing"
                or carrier_valid  # ball gone but we know who has it
            )

            # ── Decide validity for THIS frame's overlay ────────────────────
            # Layered gate:
            #   1. Soft gate (existing): ball/carrier presence and rough geometry.
            #   2. Hard gate (Phase 4.7/4.8): conf >= 0.65, ball→carrier <= 1.8m
            #      world-meters, homography conf >= 0.85.
            #   3. Stable-frames hysteresis decides whether the overlay actually
            #      draws — N consecutive valid frames required before showing,
            #      hide after 3 consecutive invalid.
            frame_reasons: List[str] = []
            if not ball_ok:
                frame_reasons.append("BALL_NOT_VISIBLE")
            elif not carrier_valid:
                frame_reasons.append("LOW_CARRIER_CONFIDENCE")
            elif carrier_w is None:
                frame_reasons.append("carrier_no_world_pos")
            elif ball_carrier_dist_px is not None and ball_carrier_dist_px > BALL_CARRIER_MAX_PX:
                frame_reasons.append("ball_far_from_carrier")
            elif len(frame_pressers) < 1:
                frame_reasons.append("no_active_pressers")

            # Hard global gate — uses world metres, not pixel distance.
            if frame_carrier_conf < _GATE_MIN_CARRIER_CONF:
                if "LOW_CARRIER_CONFIDENCE" not in frame_reasons:
                    frame_reasons.append("LOW_CARRIER_CONFIDENCE")
            if ball_carrier_dist_m is not None:
                if ball_carrier_dist_m > _GATE_HARD_BALL_M:
                    frame_reasons.append("BALL_TOO_FAR_FROM_CARRIER")
                    _hard_fail_reason = "BALL_TOO_FAR_FROM_CARRIER"
                elif ball_carrier_dist_m > _GATE_MAX_BALL_TO_CARRIER_M:
                    frame_reasons.append("BALL_TOO_FAR_FROM_CARRIER")

            frame_overlay_valid = not frame_reasons

            # ── Stable-frames-before-show hysteresis ────────────────────────
            if frame_overlay_valid:
                _stable_valid_run += 1
                _stable_invalid_run = 0
                if _stable_valid_run >= _stable_required:
                    _overlay_visible = True
            else:
                _stable_invalid_run += 1
                _stable_valid_run = 0
                if _stable_invalid_run >= _stable_invalid_max:
                    _overlay_visible = False

            # The overlay only actually paints when both gates agree
            invalid_reason = frame_reasons[0] if frame_reasons else None
            should_paint_overlay = frame_overlay_valid and _overlay_visible

            # Trace this frame for the manifest
            per_frame_trace.append({
                "f": frame_idx,
                "ball_source": frame_ball_source,
                "carrier_tid": frame_carrier_tid,
                "carrier_conf": round(frame_carrier_conf, 3),
                "ball_carrier_dist_m": round(ball_carrier_dist_m, 2) if ball_carrier_dist_m is not None else None,
                "ball_carrier_dist_px": round(ball_carrier_dist_px, 1) if ball_carrier_dist_px is not None else None,
                "pressers_active": len(frame_pressers),
                "overlay_drawn": should_paint_overlay,
                "frame_valid": frame_overlay_valid,
                "stable_valid_run": _stable_valid_run,
                "stable_invalid_run": _stable_invalid_run,
                "reasons": list(frame_reasons),
            })

            if not should_paint_overlay:
                skipped_overlay_frames.append({
                    "f": frame_idx,
                    "reason": invalid_reason or "stable_frames_pending",
                    "reasons": list(frame_reasons),
                })
                # Title still draws (the clip is about THIS sequence) but no
                # tactical claim is made on this frame.
                for c in callouts:
                    c_lo, c_hi = c.get("frames", [frame_lo, frame_hi])
                    if not (c_lo <= frame_idx <= c_hi):
                        continue
                    text = str(c.get("text", "")).strip()
                    if not text:
                        continue
                    position = str(c.get("position", "top"))
                    draw_banner(frame, text, position=position)
                out.write(frame)
                written += 1
                frame_idx += 1
                continue

            # ── Per-frame geometry build (4.7c5/c6) ─────────────────────
            # Collect ring specs filtered by current presser validity. Carrier
            # uses the per-frame carrier_tid (not the anchor's, which may be stale).
            frame_rings: List[dict] = []
            frame_rings.append({
                "tid": carrier_for_frame,
                "color": _team_color(team_map.get(int(carrier_for_frame))),
                "role_key": "CARRIER",
                "label": "BALL CARRIER",
            })
            for r in plan["rings"]:
                if r["role_key"] in ("DEFENDER", "COVER") and int(r["tid"]) in frame_pressers:
                    frame_rings.append(r)

            # Live-mode animation timeline — outro multiplier dims everything
            # together in the last 12 frames.
            outro = _live_outro_factor(frame_offset)
            a_carrier  = _anim_alpha_live("carrier_ring",  frame_offset) * outro
            a_pres1    = _anim_alpha_live("presser_rings", frame_offset) * outro
            a_pres2    = _anim_alpha_live("presser_rings2", frame_offset) * outro
            a_zone     = _anim_alpha_live("zone_fade",     frame_offset) * outro
            a_arrows   = _anim_alpha_live("arrows_draw",   frame_offset) * outro
            a_blocked  = _anim_alpha_live("blocked_lane",  frame_offset) * outro
            a_title    = _anim_alpha_live("title_fade",    frame_offset) * outro
            a_subtitle = _anim_alpha_live("subtitle_fade", frame_offset) * outro
            a_trails   = _anim_alpha_live("trails_fade",   frame_offset) * outro

            # ── Trails: maintain history, draw on broadcast frame BEFORE compositor
            # so the supersampled overlay sits on top of the trail.
            for r in frame_rings:
                tid = int(r["tid"])
                wp_t = world_pos.get((tid, frame_idx))
                fp_t = _proj(wp_t)
                if fp_t is not None:
                    hist = trails_px[tid]
                    if not hist or hist[-1] != fp_t:
                        hist.append(fp_t)
                    if len(hist) > 12:
                        del hist[: len(hist) - 12]
            if a_trails > 0:
                for r in frame_rings:
                    tid = int(r["tid"])
                    if r["role_key"] == "OPTION":
                        continue  # OPTION doesn't show trails
                    pts = trails_px.get(tid, [])
                    if len(pts) >= 2:
                        # Modulate trail brightness by a_trails (cheap: pick a
                        # darker colour at low alpha)
                        col = r["color"]
                        if a_trails < 0.99:
                            col = tuple(int(c * (0.4 + 0.6 * a_trails)) for c in col)
                        draw_player_trail(frame, pts, col,
                                          thickness=2, seg_len=5, gap=3)
                        trails_drawn_total += 1

            # ── Build OverlayCompositor and stack primitives at 2x supersample
            comp = OverlayCompositor(w=w, h=h, scale=2)

            # 1. Zone fill (recomputed each frame from CURRENT positions)
            if a_zone > 0:
                zone_world_now = [carrier_w]
                for ptid in frame_pressers:
                    pw = world_pos.get((int(ptid), frame_idx))
                    if pw is not None:
                        zone_world_now.append(pw)
                if len(zone_world_now) >= 3:
                    zone_shrink = float(story.get("zone_shrink", 0.72))
                    shrunk = _shrink_polygon(zone_world_now, factor=zone_shrink)
                    zone_px = [_proj(wp) for wp in shrunk]
                    zone_px = [p for p in zone_px if p is not None]
                    if len(zone_px) >= 3:
                        comp.fill_polygon(
                            zone_px, color=zone_color,
                            alpha=int(zone_alpha * 255 * a_zone),
                            layer=LAYER_ZONE_FILL,
                        )

            # 2. Touchline glow
            tl_px = None
            if plan["touchline_world"] and H_inv is not None:
                tl_a_w, tl_b_w = plan["touchline_world"]
                tl_a_px = _proj(tl_a_w)
                tl_b_px = _proj(tl_b_w)
                if tl_a_px is not None and tl_b_px is not None:
                    tl_px = (tl_a_px, tl_b_px)
                    comp.draw_glow_line(
                        tl_a_px, tl_b_px,
                        ARROW_KIND_COLORS["pressure"],
                        thickness=6, glow_radius=14,
                        glow_alpha=0.45 * a_pres1, layer=LAYER_GLOW,
                    )

            # 3. Pressure arrows — fade in collectively
            if a_arrows > 0 and carrier_fp is not None:
                arrow_alpha = int(255 * a_arrows)
                for i, ptid in enumerate(frame_pressers):
                    from_w = world_pos.get((int(ptid), frame_idx))
                    from_px = _proj(from_w)
                    if from_px is None:
                        continue
                    bend = 0.30 if i % 2 == 0 else -0.30
                    comp.draw_curved_arrow(
                        from_px, carrier_fp,
                        ARROW_KIND_COLORS["pressure"],
                        thickness=4, bend=bend, alpha=arrow_alpha,
                        stop_short_px=32, layer=LAYER_LINES,
                    )

            # 4. Blocked passing lane — recomputed from CURRENT carrier
            blocked_px = None
            if plan["blocked_lane"] is not None and a_blocked > 0:
                _c_tid_anchor, b_tid = plan["blocked_lane"]
                b_w = world_pos.get((int(b_tid), frame_idx))
                if (b_w is not None and carrier_w is not None
                    and float(np.hypot(b_w[0] - carrier_w[0], b_w[1] - carrier_w[1])) <= option_max_m):
                    b_px = _proj(b_w)
                    if carrier_fp is not None and b_px is not None:
                        blocked_px = (carrier_fp, b_px)
                        # Use existing draw_blocked_lane (no compositor variant)
                        # but gate on a_blocked. Draw on a copy so we can alpha-fade.
                        if a_blocked >= 0.99:
                            draw_blocked_lane(
                                frame, carrier_fp, b_px,
                                color=ARROW_KIND_COLORS["pressure"], thickness=3,
                            )
                        else:
                            tmp = frame.copy()
                            draw_blocked_lane(
                                tmp, carrier_fp, b_px,
                                color=ARROW_KIND_COLORS["pressure"], thickness=3,
                            )
                            cv2.addWeighted(tmp, a_blocked, frame, 1.0 - a_blocked, 0, frame)

            # 5. Rings + label specs (carrier always solid; pressers staggered)
            ring_records: List[Tuple[dict, Tuple[int, int], int, int]] = []  # (ring_spec, fp, radius, ring_alpha)
            for ring_idx, r in enumerate(frame_rings):
                tid = r["tid"]
                wp = world_pos.get((tid, frame_idx))
                fp = _proj(wp)
                if fp is None:
                    rec = frames_data.get(frame_idx, {})
                    for p in rec.get("players", []):
                        p_tid = p.get("rawTrackId") or p.get("trackId")
                        if p_tid is not None and int(p_tid) == tid:
                            bbox = p.get("bbox")
                            if bbox and len(bbox) == 4:
                                x1, y1, x2, y2 = map(int, bbox)
                                fp = ((x1 + x2) // 2, y2)
                            break
                if fp is None:
                    continue
                color = r["color"]
                base_radius = 24
                if wp is not None and H_inv is not None:
                    rr = project_world_radius(RING_RADIUS_M, wp, H_inv)
                    if rr is not None:
                        base_radius = int(rr)
                mult = ROLE_RING_RADIUS_MULT.get(r["role_key"], 1.0)
                radius = int(base_radius * mult)
                # Per-role alpha: carrier uses its own ramp, pressers stagger.
                if r["role_key"] == "CARRIER":
                    role_a = a_carrier
                elif r["role_key"] == "DEFENDER":
                    role_a = a_pres1
                elif r["role_key"] == "COVER":
                    role_a = a_pres2
                else:
                    role_a = max(a_pres1, a_pres2)
                ring_alpha = int(255 * role_a)
                if ring_alpha <= 0:
                    continue
                # Carrier inner solid pulse
                if r["role_key"] == "CARRIER":
                    inner = max(3, radius // 3)
                    comp.fill_ellipse(
                        fp, (inner, inner // 2), color,
                        alpha=ring_alpha, layer=LAYER_RINGS,
                    )
                comp.draw_circle(
                    fp, radius, color,
                    thickness=4, alpha=ring_alpha, layer=LAYER_RINGS,
                )
                ring_records.append((r, fp, radius, ring_alpha))

            # 6. Ball dot — composited on top of rings layer (so the ball reads
            # cleanly even on cluttered frames). Only draw if carrier ring is
            # already at least partly faded in.
            if show_ball and frame_ball_px is not None and a_carrier > 0.05:
                # Halo
                comp.draw_circle(
                    frame_ball_px, 11, (0, 0, 0),
                    thickness=2, alpha=int(220 * a_carrier), layer=LAYER_RINGS,
                )
                comp.draw_circle(
                    frame_ball_px, 9, BALL,
                    thickness=2, alpha=int(255 * a_carrier), layer=LAYER_RINGS,
                )
                comp.fill_circle(
                    frame_ball_px, 4, BALL,
                    alpha=int(255 * a_carrier), layer=LAYER_RINGS,
                )
                ball_dots_drawn += 1

            # 7. Role pills — solve collisions against scoreboard + each other
            label_specs: List[LabelSpec] = []
            for (r, fp, radius, _ring_alpha) in ring_records:
                priority = 10 if r["role_key"] == "CARRIER" else \
                           5 if r["role_key"] == "DEFENDER" else \
                           4 if r["role_key"] == "COVER" else 1
                label_specs.append(LabelSpec(
                    anchor=fp,
                    text=r["label"],
                    priority=priority,
                    ring_radius=radius,
                    color=r["color"],
                ))
            if label_specs:
                solved = solve_label_positions(label_specs, frame_w=w, frame_h=h)
                for spec in solved:
                    if spec.resolved_pos is None:
                        continue
                    if spec.resolved_side != "above":
                        label_collisions_resolved += 1
                    # Pill alpha follows the ring's role alpha
                    role = next((rec[0]["role_key"] for rec in ring_records
                                 if rec[1] == spec.anchor), None)
                    if role == "CARRIER":
                        pill_a = a_carrier
                    elif role == "DEFENDER":
                        pill_a = a_pres1
                    elif role == "COVER":
                        pill_a = a_pres2
                    else:
                        pill_a = a_carrier
                    if pill_a <= 0.02:
                        continue
                    comp.draw_pill(
                        spec.resolved_pos, spec.text, spec.color,
                        font_scale=0.48,
                        alpha=int(230 * pill_a),
                        layer=LAYER_LABELS,
                    )

            # 8. Title + subtitle banners — drawn directly on frame for now;
            # draw_banner has its own background, can't easily route through
            # compositor without a re-layout. Banners get their own alpha.
            for c in callouts:
                c_lo, c_hi = c.get("frames", [frame_lo, frame_hi])
                if not (c_lo <= frame_idx <= c_hi):
                    continue
                text = str(c.get("text", "")).strip()
                if not text:
                    continue
                position = str(c.get("position", "top"))
                banner_a = a_title if position == "top" else a_subtitle
                if banner_a <= 0.02:
                    continue
                if banner_a >= 0.99:
                    draw_banner(frame, text, position=position)
                else:
                    tmp = frame.copy()
                    draw_banner(tmp, text, position=position)
                    cv2.addWeighted(tmp, banner_a, frame, 1.0 - banner_a, 0, frame)

            # 9. Anchored callouts (still direct — these appear briefly)
            if tl_px is not None and a_pres1 > 0.5:
                tl_a, tl_b = tl_px
                tl_mx = (tl_a[0] + tl_b[0]) // 2
                tl_my = (tl_a[1] + tl_b[1]) // 2 - 18
                draw_local_callout(
                    frame, (tl_mx, tl_my), "TOUCHLINE = EXTRA DEFENDER",
                    side="left", accent=ARROW_KIND_COLORS["pressure"],
                )
            if blocked_px is not None and a_blocked > 0.5:
                ba, bb = blocked_px
                mx = (ba[0] + bb[0]) // 2
                dy = bb[1] - ba[1]
                offset = 22 if dy >= 0 else -22
                my = (ba[1] + bb[1]) // 2 + offset
                draw_local_callout(
                    frame, (mx, my), "PASSING LANE CLOSED",
                    side="right", accent=ARROW_KIND_COLORS["pressure"],
                )

            # Composite supersampled overlay onto the broadcast frame
            frame = comp.composite(frame)

            out.write(frame)
            written += 1
            frame_idx += 1

    cap.release()
    out.release()

    # ── Validation manifest ───────────────────────────────────────────────
    declared_pressers = sum(1 for (_, raw_role) in raw_cast
                            if _normalise_role(raw_role) in ("DEFENDER", "COVER"))
    active_pressers = sum(1 for (_, r, _) in cast_with_label
                          if r in ("DEFENDER", "COVER"))
    cover_declared = any(_normalise_role(r) == "COVER" for (_, r) in raw_cast)
    cover_active = any(r == "COVER" for (_, r, _) in cast_with_label)
    final_story_type = recommended_type if not is_valid else declared_story_type
    retitled = (final_story_type != declared_story_type) or (not is_valid and final_valid is False)

    # Homography trust at the resolved anchor: "ok" if direct hit,
    # "interpolated" if we used the nearest neighbour, "missing" if no H at all.
    if H_inv_by_frame and anchor_frame in H_inv_by_frame:
        homography_confidence_str = "ok"
    elif H_inv_by_frame:
        homography_confidence_str = "interpolated"
    else:
        homography_confidence_str = "missing"

    # Carrier metadata from ball_assignments at the anchor (if available)
    anchor_carrier_meta = carrier_by_frame.get(anchor_frame, {})
    carrier_grace_frames_val = int(anchor_carrier_meta.get("grace_frames", 0) or 0)
    anchor_carrier_conf = anchor_carrier_meta.get("carrier_confidence")
    anchor_ball_source  = anchor_carrier_meta.get("ball_source")

    manifest = {
        # Quality gate fields (must all be truthy for a valid product render)
        "story_valid":     story_is_valid,
        "story_type":      final_story_type,
        "render_mode":     render_mode,
        "ball_source":     ball_source,
        "ball_confidence": ball_confidence,
        # Timing
        "anchor_frame":          anchor_frame,
        "frames_written":        written,
        "total_frames_window":   frame_hi - frame_lo + 1,
        # Geometry from validator at the resolved anchor
        "carrier_touchline_distance_m": (final_geom or {}).get("carrier_touchline_distance_m"),
        "presser_angle_spread_deg":     (final_geom or {}).get("presser_angle_spread_deg"),
        "trap_zone_area_m2":            (final_geom or {}).get("trap_zone_area_m2"),
        "carrier_pitch_third":          (final_geom or {}).get("carrier_pitch_third"),
        # Cast accounting
        "pressers_declared": declared_pressers,
        "pressers_active":   active_pressers,
        "cover_declared":    cover_declared,
        "cover_active":      cover_active,
        "retitled":              retitled,
        "validated_story_type":  final_type,
        "story_failed_reason":   story_failed_reason,
        # Visual primitives
        "rings":           len(plan["rings"]),
        "arrows":          len(plan["arrows"]),
        "zone_world":      len(plan["zone_world"]) if plan["zone_world"] else 0,
        "touchline":       bool(plan["touchline_world"]),
        "blocked_lane":    bool(plan["blocked_lane"]),
        "ball_dots_drawn": ball_dots_drawn,
        "ball_dots_per_written": round(ball_dots_drawn / max(1, written), 3),
        "carrier_tid":     carrier_tid,
        "blocked_tid":     blocked_tid,
        # Phase 4.7c4 truth-trace fields
        "homography_confidence":  homography_confidence_str,
        "carrier_grace_frames":   carrier_grace_frames_val,
        "anchor_carrier_confidence": anchor_carrier_conf,
        "anchor_ball_source":     anchor_ball_source,
        "ball_assignments_used":  bool(ball_assignments_json),
        "frames_window":          [frame_lo, frame_hi],
        "product_mode":           not debug_allow_invalid_story,
        "zone_image_fallback":    len(plan.get("zone_image_fallback", []) or []),
    }

    # Phase 4.7c5: per-frame truth trace + aggregates
    overlay_drawn_count = sum(1 for t in per_frame_trace if t.get("overlay_drawn"))
    overlay_drawn_ratio = round(overlay_drawn_count / max(1, len(per_frame_trace)), 3)
    drawn_dists = [t["ball_carrier_dist_m"] for t in per_frame_trace
                   if t.get("overlay_drawn") and t.get("ball_carrier_dist_m") is not None]
    if drawn_dists:
        drawn_dists_sorted = sorted(drawn_dists)
        p95_idx = max(0, int(len(drawn_dists_sorted) * 0.95) - 1)
        bc_p95 = round(drawn_dists_sorted[p95_idx], 2)
    else:
        bc_p95 = None
    manifest["rendered_frame_count"]  = written
    manifest["overlay_drawn_frames"]  = overlay_drawn_count
    manifest["overlay_drawn_ratio"]   = overlay_drawn_ratio
    manifest["skipped_overlay_frames"] = skipped_overlay_frames
    manifest["ball_carrier_dist_p95_m"] = bc_p95
    manifest["per_frame_trace"]       = per_frame_trace
    # 4.7c6 polish fields
    manifest["compositor_used"]              = (render_mode == "live_reprojected")
    manifest["compositor_supersample"]       = 2 if manifest["compositor_used"] else 1
    manifest["label_collisions_resolved"]    = label_collisions_resolved
    manifest["trails_drawn_total"]           = trails_drawn_total
    manifest["trails_drawn_per_frame"]       = round(
        trails_drawn_total / max(1, overlay_drawn_count), 2
    )
    manifest["animation_timeline"] = {
        k: list(v) for k, v in ANIM_LIVE.items()
    }
    manifest["interpolated_ball_frames"]    = ball_interp_stats["interpolated_ball_frames"]
    manifest["interpolated_carrier_frames"] = ball_interp_stats["interpolated_carrier_frames"]
    if "smoother_stats" in ball_interp_stats:
        manifest["carrier_smoother_stats"] = ball_interp_stats["smoother_stats"]

    # Phase 4.7/4.8: per-frame validity manifest + thresholds_used + outcome
    manifest["per_frame_validity"] = [
        {"frameIndex": t["f"], "valid": t.get("frame_valid", False),
         "reasons": t.get("reasons", [])}
        for t in per_frame_trace
    ]
    manifest["thresholds_used"] = {
        "gate_min_carrier_conf": _GATE_MIN_CARRIER_CONF,
        "gate_max_ball_to_carrier_m": _GATE_MAX_BALL_TO_CARRIER_M,
        "gate_hard_ball_to_carrier_m": _GATE_HARD_BALL_M,
        "gate_min_homography_conf": _GATE_MIN_HOMOGRAPHY_CONF,
        "stable_required_frames": _stable_required,
        "stable_invalid_frames": _stable_invalid_max,
    }
    if _hard_fail_reason:
        manifest["story_outcome"] = f"FAILED:{_hard_fail_reason}"
    elif overlay_drawn_count == 0:
        manifest["story_outcome"] = "NONE"
    else:
        manifest["story_outcome"] = "RENDERED"

    # ── Acceptance gate (4.7c5) — refuse to ship a misleading clip ────
    if not debug_allow_invalid_story:
        accept_failures: List[str] = []
        if _hard_fail_reason:
            accept_failures.append(
                f"hard_fail={_hard_fail_reason} — story aborted by global gate."
            )
        if overlay_drawn_ratio < 0.80:
            accept_failures.append(
                f"overlay_drawn_ratio={overlay_drawn_ratio} < 0.80 — "
                f"the overlay was hidden on more than 20% of rendered frames "
                f"because per-frame truth was missing or inconsistent."
            )
        if bc_p95 is not None and bc_p95 > 6.0:
            accept_failures.append(
                f"ball_carrier_dist_p95_m={bc_p95} > 6.0 — "
                "ball is too far from labeled carrier on the worst 5% of frames."
            )
        if accept_failures:
            manifest["story_failed_reason"] = "; ".join(accept_failures)
            manifest["story_valid"] = False
            if verbose:
                for r in accept_failures:
                    print(f"[render_story] ACCEPTANCE FAIL: {r}")
            # Don't delete the MP4 — caller may want to inspect it — but
            # surface the failure clearly so the cell sees story_valid=False.
            # (We do not raise: the MP4 is already written. Caller decides.)
    if verbose:
        print(f"[render_story] manifest={manifest}")
        # Hard pass/fail signals — always print when triggered
        if active_pressers < declared_pressers:
            print(
                f"[render_story] WARN pressers_active={active_pressers} < "
                f"declared={declared_pressers} (story may have been retitled)"
            )
        if cover_declared and not cover_active:
            print("[render_story] WARN cover_declared=True but cover_active=False — cover lost")
        if retitled:
            print(
                f"[render_story] WARN story_type was downgraded to {final_story_type} "
                "(use strict_story_type=true in JSON to force a hard error)"
            )
        if show_ball:
            if ball_dots_drawn == 0:
                print("[render_story] WARN no ball positions found in window — ball dot never drawn")
            elif written and ball_dots_drawn / max(written, 1) < 0.5:
                print(
                    f"[render_story] WARN ball tracking sparse: "
                    f"{ball_dots_drawn}/{written} frames had a ball position"
                )
        if manifest["rings"] < 3:
            print(f"[render_story] WARN rings={manifest['rings']} < 3 — story too thin")
        if manifest["arrows"] < 1:
            print("[render_story] WARN no arrows in plan")
        if manifest["zone_world"] < 3:
            print("[render_story] WARN no zone hull (need >=3 world points at anchor)")
        if not manifest["touchline"]:
            print("[render_story] WARN touchline not resolved (carrier far from sideline OR homography missing)")
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
