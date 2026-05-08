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

    declared_story_type = "PRESSING_TRAP"  # only PRESSING_TRAP is implemented
    strict_story_type = bool(story.get("strict_story_type", False))
    declared_defender_tids = [
        tid for (tid, raw_role) in raw_cast
        if _normalise_role(raw_role) in ("DEFENDER", "COVER")
    ]

    def _validate_at(fi: int) -> Tuple[bool, str, dict]:
        return validate_pressing_trap(
            world_pos, fi, int(carrier_tid), declared_defender_tids
        )

    is_valid, recommended_type, geom = _validate_at(anchor_frame)
    if verbose:
        print(
            f"[render_story] validator @ user anchor F={anchor_frame}: "
            f"valid={is_valid} type={recommended_type} geom={geom}"
        )

    if not is_valid:
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
    # In product mode (debug_allow_invalid_story not set), an invalid story
    # must NOT produce a rendered clip. We raise here — before opening the
    # VideoWriter — so no partial MP4 is written.
    story_is_valid = bool(final_valid) and ball_ok
    if not story_is_valid and not debug_allow_invalid_story:
        cap.release()
        failing_reasons = []
        if not final_valid:
            failing_reasons.append(
                f"story_type={declared_story_type} failed geometry validation "
                f"(best frame F={anchor_frame} qualifies as '{final_type}'). "
                f"Geometry: {final_geom}"
            )
        if not ball_ok:
            failing_reasons.append(
                f"ball_source='{ball_source}' (confidence={ball_confidence:.0%}) — "
                "ball tracking too sparse to claim a tactical story."
            )
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
    zone_world_raw = [wp for wp in zone_world_raw if wp is not None]
    if len(zone_world_raw) >= 3:
        zone_shrink = float(story.get("zone_shrink", 0.72))
        plan["zone_world"] = _shrink_polygon(zone_world_raw, factor=zone_shrink)
    else:
        plan["zone_world"] = []

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

    def _smoothstep(t: float) -> float:
        """Cubic smoothstep easing: 0→1 with zero derivative at endpoints."""
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)

    def _anim_alpha(key: str, offset: int) -> float:
        """Return [0,1] progress for a named animation stage at this offset."""
        start, end = ANIM[key]
        if offset < start:
            return 0.0
        if offset >= end:
            return 1.0
        return _smoothstep((offset - start) / max(1, end - start))

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

            # Step 1: dim
            if dim_alpha > 0:
                black = np.zeros_like(frame)
                cv2.addWeighted(black, dim_alpha, frame, 1.0 - dim_alpha, 0, frame)

            # Helper: project a world point for this frame
            def _proj(wp):
                if wp is None or H_inv is None:
                    return None
                return project_world_point(wp, H_inv)

        # Step 2: zone hull (reproject world coords → image)
        if plan["zone_world"]:
            zone_px = [_proj(wp) for wp in plan["zone_world"]]
            zone_px = [p for p in zone_px if p is not None]
            if len(zone_px) >= 3:
                draw_zone_hull(frame, zone_px, zone_color, alpha=zone_alpha)

        # Step 3: touchline glow (reproject world endpoints → image)
        tl_px: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
        if plan["touchline_world"] and H_inv is not None:
            tl_a_w, tl_b_w = plan["touchline_world"]
            tl_a_px = _proj(tl_a_w)
            tl_b_px = _proj(tl_b_w)
            if tl_a_px is not None and tl_b_px is not None:
                tl_px = (tl_a_px, tl_b_px)
                draw_glow_line(
                    frame, tl_a_px, tl_b_px,
                    ARROW_KIND_COLORS["pressure"], thickness=6, glow_layers=2,
                )

        # Step 4: pressure arrows (resolve world pos per-frame)
        for a in plan["arrows"]:
            from_w = world_pos.get((a["from_tid"], frame_idx))
            to_w = world_pos.get((a["to_tid"], frame_idx))
            from_px = _proj(from_w)
            to_px = _proj(to_w)
            if from_px is not None and to_px is not None:
                draw_curved_arrow(
                    frame, from_px, to_px, a["color"],
                    thickness=4, bend=a["bend"], stop_short_px=32,
                )

        # Step 5: blocked passing lane (resolve per-frame)
        blocked_px: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
        if plan["blocked_lane"] is not None:
            c_tid, b_tid = plan["blocked_lane"]
            c_w = world_pos.get((c_tid, frame_idx))
            b_w = world_pos.get((b_tid, frame_idx))
            c_px = _proj(c_w)
            b_px = _proj(b_w)
            if c_px is not None and b_px is not None:
                blocked_px = (c_px, b_px)
                draw_blocked_lane(
                    frame, c_px, b_px,
                    color=ARROW_KIND_COLORS["pressure"], thickness=3,
                )

        # Step 6: rings + role pills (resolve per-frame world pos → image)
        carrier_px: Optional[Tuple[int, int]] = None
        for r in plan["rings"]:
            tid = r["tid"]
            wp = world_pos.get((tid, frame_idx))
            fp = _proj(wp)
            if fp is None:
                # Fallback: derive foot pixel from bbox in tracking data
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
            # Compute ring radius in image space from world coords
            base_radius = 24
            if wp is not None and H_inv is not None:
                rr = project_world_radius(RING_RADIUS_M, wp, H_inv)
                if rr is not None:
                    base_radius = int(rr)
            mult = ROLE_RING_RADIUS_MULT.get(r["role_key"], 1.0)
            radius = int(base_radius * mult)
            if r["role_key"] == "CARRIER":
                inner = max(3, radius // 3)
                cv2.ellipse(frame, fp, (inner, inner // 2), 0, 0, 360, color, -1, cv2.LINE_AA)
                carrier_px = fp
            _draw_foot_ring(frame, fp, radius, color, style="solid", thickness=4)
            below = r["role_key"] == "OPTION"
            # Dynamic pill offset: ~half-radius at wide shots, capped low at zoom
            pill_offset = max(int(radius * 0.55) + 12, 24)
            draw_role_pill(
                frame, fp, r["label"], color,
                above_offset_px=pill_offset, below=below,
            )

        # Step 6.5: ball dot — yellow ring + filled inner dot at ball world pos
        if show_ball and ball_pos and H_inv is not None:
            ball_w = ball_pos.get(frame_idx)
            if ball_w is None:
                # Tolerance: ±10 frames
                for delta in range(1, 11):
                    if (b1 := ball_pos.get(frame_idx - delta)) is not None:
                        ball_w = b1; break
                    if (b2 := ball_pos.get(frame_idx + delta)) is not None:
                        ball_w = b2; break
            if ball_w is not None:
                ball_px = _proj(ball_w)
                if ball_px is not None:
                    # Outer halo + ring + filled inner dot
                    cv2.circle(frame, ball_px, 11, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.circle(frame, ball_px, 9, BALL, 2, cv2.LINE_AA)
                    cv2.circle(frame, ball_px, 4, BALL, -1, cv2.LINE_AA)
                    ball_dots_drawn += 1

        # Step 7: title + subtitle banners (screen-space, not pitch-anchored)
        for c in callouts:
            c_lo, c_hi = c.get("frames", [frame_lo, frame_hi])
            if not (c_lo <= frame_idx <= c_hi):
                continue
            text = str(c.get("text", "")).strip()
            if not text:
                continue
            position = str(c.get("position", "top"))
            draw_banner(frame, text, position=position)

        # Step 8: anchored callouts (derived from reprojected positions)
        if tl_px is not None:
            tl_a, tl_b = tl_px
            tl_mx = (tl_a[0] + tl_b[0]) // 2
            tl_my = (tl_a[1] + tl_b[1]) // 2 - 18
            draw_local_callout(
                frame, (tl_mx, tl_my), "TOUCHLINE = EXTRA DEFENDER",
                side="left", accent=ARROW_KIND_COLORS["pressure"],
            )
        if blocked_px is not None:
            ba, bb = blocked_px
            mx = (ba[0] + bb[0]) // 2
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
    declared_pressers = sum(1 for (_, raw_role) in raw_cast
                            if _normalise_role(raw_role) in ("DEFENDER", "COVER"))
    active_pressers = sum(1 for (_, r, _) in cast_with_label
                          if r in ("DEFENDER", "COVER"))
    cover_declared = any(_normalise_role(r) == "COVER" for (_, r) in raw_cast)
    cover_active = any(r == "COVER" for (_, r, _) in cast_with_label)
    final_story_type = recommended_type if not is_valid else declared_story_type
    retitled = (final_story_type != declared_story_type) or (not is_valid and final_valid is False)

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
        "retitled":          retitled,
        # Visual primitives
        "rings":           len(plan["rings"]),
        "arrows":          len(plan["arrows"]),
        "zone_world":      len(plan["zone_world"]) if plan["zone_world"] else 0,
        "touchline":       bool(plan["touchline_world"]),
        "blocked_lane":    bool(plan["blocked_lane"]),
        "ball_dots_drawn": ball_dots_drawn,
        "carrier_tid":     carrier_tid,
        "blocked_tid":     blocked_tid,
    }
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
