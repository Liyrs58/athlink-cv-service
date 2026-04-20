"""Voronoi diagram analysis for space control.
"""

import math
from typing import List, Dict, Any, Tuple, Optional

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
GRID_RESOLUTION = 2.0  # 2m grid squares — fast enough, precise enough


def _closest_player(px: float, py: float,
                     positions: List[Tuple[float, float, int]]) -> int:
    """
    Returns team_id of player closest to pitch point (px, py).
    positions: list of (world_x, world_y, team_id)
    Returns -1 if no players.
    """
    best_dist = float("inf")
    best_team = -1
    for wx, wy, tid in positions:
        d = math.sqrt((px - wx)**2 + (py - wy)**2)
        if d < best_dist:
            best_dist = d
            best_team = tid
    return best_team


def compute_voronoi_control(
    tracks: List[Dict],
    frame_metadata: List[Dict],
    calibration: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Computes Voronoi-based pitch control for each team.
    Samples every 10th frame for performance.
    Returns time-averaged control percentages.
    """
    frame_indices = sorted(set(m.get("frameIndex", 0) for m in frame_metadata))
    sampled = frame_indices[::10] if len(frame_indices) > 10 else frame_indices

    team_control_frames = []

    for fi in sampled:
        # Get active player positions for this frame
        positions = []
        for t in tracks:
            if t.get("is_staff", False):
                continue
            team_id = t.get("teamId", -1)
            if team_id not in [0, 1]:
                continue
            traj = t.get("trajectory", [])
            entry = min(
                (e for e in traj if abs(e.get("frameIndex", 0) - fi) <= 4),
                key=lambda e: abs(e.get("frameIndex", 0) - fi),
                default=None,
            )
            if entry and "world_x" in entry:
                positions.append((
                    entry["world_x"],
                    entry["world_y"],
                    team_id,
                ))

        if len(positions) < 4:
            continue

        # Grid scan
        team_counts = {0: 0, 1: 0, -1: 0}
        total = 0
        x = 0.0
        while x <= PITCH_LENGTH:
            y = 0.0
            while y <= PITCH_WIDTH:
                owner = _closest_player(x, y, positions)
                team_counts[owner] = team_counts.get(owner, 0) + 1
                total += 1
                y += GRID_RESOLUTION
            x += GRID_RESOLUTION

        if total > 0:
            team_control_frames.append({
                "frame": fi,
                "team_0_pct": round(team_counts.get(0, 0) / total * 100, 1),
                "team_1_pct": round(team_counts.get(1, 0) / total * 100, 1),
            })

    if not team_control_frames:
        return {
            "status": "insufficient_data",
            "team_0_control_pct": None,
            "team_1_control_pct": None,
            "frames_analysed": 0,
        }

    avg_t0 = round(
        sum(f["team_0_pct"] for f in team_control_frames) / len(team_control_frames), 1
    )
    avg_t1 = round(
        sum(f["team_1_pct"] for f in team_control_frames) / len(team_control_frames), 1
    )

    # Find peak control moments
    peak_t0_frame = max(team_control_frames, key=lambda f: f["team_0_pct"])
    peak_t1_frame = max(team_control_frames, key=lambda f: f["team_1_pct"])

    return {
        "status": "ok",
        "team_0_control_pct": avg_t0,
        "team_1_control_pct": avg_t1,
        "frames_analysed": len(team_control_frames),
        "peak_control_team_0_frame": peak_t0_frame["frame"],
        "peak_control_team_1_frame": peak_t1_frame["frame"],
        "dominant_team": 0 if avg_t0 > avg_t1 else 1,
        "control_margin": round(abs(avg_t0 - avg_t1), 1),
        "frame_by_frame": team_control_frames,
    }
