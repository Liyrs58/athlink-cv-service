"""Performance Zone overlay renderer — UEFA-style broadcast graphics.

Replaces the debug bbox renderer (render_video.py) with foot rings projected
through pitch homography, sparse nameplates for selected players, and an
optional ball trail. Officials are drawn as nothing (truly invisible to the
identity layer).

Usage:
    from render_performance_zone import render_performance_zone
    render_performance_zone(
        video_path="clip.mov",
        results_json="temp/job/tracking/track_results.json",
        pitch_map_json="temp/job/pitch/pitch_map.json",
        team_results_json="temp/job/tracking/team_results.json",
        output_path="clip_pz.mp4",
        selected_players=None,  # None => auto-pick top-3 by play time
        show_ball=True,
    )
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from services.world_to_image import (
    invert_homography,
    project_world_point,
    project_world_radius,
)


# Team palette in BGR — picked to read on green grass; matches mood board.
TEAM_HOME = (255, 99, 29)    # blue
TEAM_AWAY = (37, 197, 34)    # green
NEUTRAL = (200, 200, 200)
BALL = (40, 220, 255)        # warm yellow

RING_RADIUS_M = 0.55          # foot-ring radius in metres
RING_THICKNESS = 3
RING_SEGMENTS = 16            # ring is drawn as N alternating arc segments
RING_FILLED_RATIO = 0.65      # fraction of segments drawn for solid style


# ----------------------------------------------------------------------
# Data loaders
# ----------------------------------------------------------------------

def _load_team_map(team_json_path: str) -> Dict[int, int]:
    p = Path(team_json_path)
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            data = json.load(f)
        out: Dict[int, int] = {}
        for entry in data.get("tracks", []) or data.get("teams", []) or []:
            tid = entry.get("trackId") or entry.get("track_id")
            team = entry.get("teamId", entry.get("team_id"))
            if tid is not None and team is not None:
                out[int(tid)] = int(team)
        return out
    except Exception:
        return {}


def _load_pitch_map(pitch_json_path: str):
    """Returns (homographies_inv_by_frame, world_pos_by_pid_frame, ball_pos_by_frame)."""
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
    ball_pos: Dict[int, Tuple[float, float, float]] = {}
    for entry in data.get("players", []):
        if entry.get("is_ball"):
            for pt in entry.get("trajectory2d", []):
                ball_pos[int(pt["frameIndex"])] = (
                    float(pt["x"]),
                    float(pt["y"]),
                    float(pt.get("z", 0.0)),
                )
            continue
        tid = entry.get("trackId")
        if tid is None:
            continue
        for pt in entry.get("trajectory2d", []):
            world_pos[(int(tid), int(pt["frameIndex"]))] = (float(pt["x"]), float(pt["y"]))

    return homographies_inv, world_pos, ball_pos


def _team_color(team_id: Optional[int]) -> Tuple[int, int, int]:
    if team_id == 0:
        return TEAM_HOME
    if team_id == 1:
        return TEAM_AWAY
    return NEUTRAL


def _resolve_team(p_entry: dict, team_map: Dict[int, int]) -> Optional[int]:
    raw = p_entry.get("rawTrackId") or p_entry.get("trackId")
    if raw is None:
        return None
    return team_map.get(int(raw))


# ----------------------------------------------------------------------
# Drawing primitives
# ----------------------------------------------------------------------

def _draw_foot_ring(
    img: np.ndarray,
    centre_px: Tuple[int, int],
    radius_px: int,
    color: Tuple[int, int, int],
    *,
    style: str = "solid",
    thickness: int = RING_THICKNESS,
) -> None:
    """Draw a UEFA-style segmented ring centred at `centre_px`."""
    cx, cy = centre_px
    rx = max(2, radius_px)
    ry = max(2, int(radius_px * 0.5))   # squashed for ground perspective
    if style == "solid":
        cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, color, thickness, cv2.LINE_AA)
        return
    # segmented: alternate filled/empty arcs around 360deg
    n = RING_SEGMENTS
    seg_deg = 360.0 / n
    keep_every_n = 2 if style == "dashed" else 3   # dotted shows fewer
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
    name: Optional[str] = None,
) -> None:
    """Draw a small rounded plate above the anchor: [number tab][name]."""
    cx, cy = anchor_px
    number = "".join(ch for ch in pid if ch.isdigit()) or "?"
    label = (name or "").strip()

    n_text = number
    body_text = label or pid
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

    # Body (white) + number tab (team color)
    cv2.rectangle(img, (x1, y1), (x2, y2), (245, 245, 245), -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1), (x1 + tab_w, y2), color, -1, cv2.LINE_AA)
    cv2.putText(img, n_text, (x1 + pad, y2 - pad), font, fs, (255, 255, 255), th + 1, cv2.LINE_AA)
    cv2.putText(img, body_text, (x1 + tab_w + pad, y2 - pad), font, fs, (40, 40, 40), th, cv2.LINE_AA)


def _draw_ball_trail(
    img: np.ndarray,
    trail: List[Tuple[int, int]],
    color: Tuple[int, int, int] = BALL,
) -> None:
    if not trail:
        return
    n = len(trail)
    for i, (x, y) in enumerate(trail):
        alpha = (i + 1) / n
        radius = max(2, int(3 + 4 * alpha))
        overlay = img.copy()
        cv2.circle(overlay, (x, y), radius, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)


# ----------------------------------------------------------------------
# Selection helper
# ----------------------------------------------------------------------

def _auto_select_players(track_results: dict, n: int = 3) -> List[str]:
    """Top-N players by total assigned-frame count."""
    counts: Dict[str, int] = defaultdict(int)
    for f in track_results.get("frames", []):
        for p in f.get("players", []):
            pid = p.get("playerId") or p.get("displayId")
            if pid:
                counts[str(pid)] += 1
    return [pid for pid, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:n]]


# ----------------------------------------------------------------------
# Stroke style mapping
# ----------------------------------------------------------------------

def _ring_style(p_entry: dict) -> str:
    if not p_entry.get("identity_valid", False):
        return "dotted"
    src = p_entry.get("assignment_source", "")
    if src == "locked":
        return "solid"
    return "dashed"


# ----------------------------------------------------------------------
# Main render
# ----------------------------------------------------------------------

def render_performance_zone(
    video_path: str,
    results_json: str,
    pitch_map_json: str,
    team_results_json: str,
    output_path: str,
    *,
    selected_players: Optional[Iterable[str]] = None,
    selected_count: int = 3,
    show_ball: bool = True,
    sample_frames: Optional[Iterable[int]] = None,
    frame_dir: Optional[str] = None,
    verbose: bool = True,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {video_path}")

    with open(results_json) as f:
        track_data = json.load(f)
    frames_data = {
        f["frameIndex"]: f for f in track_data.get("frames", [])
    }

    H_inv_by_frame, world_pos, ball_pos = _load_pitch_map(pitch_map_json)
    team_map = _load_team_map(team_results_json)

    selected_set = set(selected_players or _auto_select_players(track_data, selected_count))

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    sample_set = set(sample_frames or [])
    if frame_dir and sample_set:
        Path(frame_dir).mkdir(parents=True, exist_ok=True)

    ball_trail: List[Tuple[int, int]] = []
    frame_idx = 0
    extracted = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rec = frames_data.get(frame_idx, {})
        H_inv = H_inv_by_frame.get(frame_idx)
        if H_inv is None and H_inv_by_frame:
            nearest = min(H_inv_by_frame.keys(), key=lambda k: abs(k - frame_idx))
            H_inv = H_inv_by_frame[nearest]

        # Players: foot rings
        for p in rec.get("players", []):
            if p.get("is_official", False):
                continue
            tid = p.get("rawTrackId") or p.get("trackId")
            wp = world_pos.get((int(tid), frame_idx)) if tid is not None else None
            centre_px = None
            radius_px = None
            if wp is not None and H_inv is not None:
                centre_px = project_world_point(wp, H_inv)
                radius_px = project_world_radius(RING_RADIUS_M, wp, H_inv)
            if centre_px is None or radius_px is None:
                # Fallback: bbox bottom-centre + heuristic radius
                bbox = p.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = map(int, bbox)
                centre_px = ((x1 + x2) // 2, y2)
                bw = max(1, x2 - x1)
                radius_px = max(8, int(bw * 0.35))

            team = _resolve_team(p, team_map)
            color = _team_color(team)
            style = _ring_style(p)
            _draw_foot_ring(frame, centre_px, radius_px, color, style=style)

            pid = p.get("playerId") or p.get("displayId")
            if pid and str(pid) in selected_set:
                _draw_nameplate(frame, centre_px, color, str(pid))

        # Ball trail
        if show_ball:
            wpb = ball_pos.get(frame_idx)
            if wpb is not None and H_inv is not None:
                bxy = project_world_point((wpb[0], wpb[1]), H_inv)
                if bxy is not None:
                    ball_trail.append(bxy)
            if len(ball_trail) > 8:
                ball_trail = ball_trail[-8:]
            _draw_ball_trail(frame, ball_trail)

        # Confidence chip (top-left)
        n_locked = sum(1 for p in rec.get("players", []) if p.get("assignment_source") == "locked")
        n_revived = sum(1 for p in rec.get("players", []) if p.get("assignment_source") == "revived")
        n_uncertain = sum(1 for p in rec.get("players", []) if not p.get("identity_valid", False) and not p.get("is_official", False))
        chip = f"F{frame_idx}  locked={n_locked} revived={n_revived} uncertain={n_uncertain}"
        cv2.putText(frame, chip, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(frame)
        if frame_dir and frame_idx in sample_set:
            cv2.imwrite(str(Path(frame_dir) / f"pz_frame_{frame_idx:05d}.jpg"), frame)
            extracted += 1
        frame_idx += 1

    cap.release()
    out.release()
    if verbose:
        print(f"[render_performance_zone] {frame_idx} frames -> {output_path}")
        if extracted:
            print(f"[render_performance_zone] {extracted} samples -> {frame_dir}")


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
        output_path=out,
    )
