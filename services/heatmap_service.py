"""Player distance heatmaps and spatial coverage per zone.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

PITCH_W = 105.0
PITCH_H = 68.0
GRID_W = 21
GRID_H = 14
SPRINT_THRESHOLD = 7.0
HIGH_INTENSITY_THRESHOLD = 5.0


def _base_job_id(job_id):
    # type: (str) -> str
    for suffix in ("_final_tactics", "_final_pitch", "_final", "_tactics", "_pitch"):
        if job_id.endswith(suffix):
            return job_id[:-len(suffix)]
    return job_id


def _load_json(path):
    # type: (Path) -> Optional[Dict]
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _rolling_median_3(arr):
    # type: (np.ndarray) -> np.ndarray
    """3-frame rolling median. Edges keep original values."""
    n = len(arr)
    if n < 3:
        return arr.copy()
    out = arr.copy()
    for i in range(1, n - 1):
        out[i] = np.median(arr[i - 1:i + 2])
    return out


def _normalise_heatmap(grid):
    # type: (np.ndarray) -> List[float]
    mx = grid.max()
    if mx > 0:
        grid = grid / mx
    return [round(float(v), 4) for v in grid.ravel()]


def compute_heatmaps(job_id):
    # type: (str) -> Dict[str, Any]
    """
    Compute per-player distance stats, sprint detection, and heatmaps.

    Reads:
        temp/{job_id}/pitch/pitch_map.json
        temp/{job_id}/tracking/track_results.json (for fps)
    """
    base = _base_job_id(job_id)
    candidates = [job_id]
    if base != job_id:
        candidates.append(base)

    # Load pitch_map.json
    pitch_data = None
    for jid in candidates:
        pitch_data = _load_json(Path("temp/{}/pitch/pitch_map.json".format(jid)))
        if pitch_data is not None:
            break
    if pitch_data is None:
        raise FileNotFoundError(
            "pitch_map.json not found for job '{}'".format(job_id)
        )

    # Load track_results.json for fps
    fps = 25.0
    for jid in candidates:
        track_data = _load_json(Path("temp/{}/tracking/track_results.json".format(jid)))
        if track_data is not None:
            metadata = track_data.get("metadata", {})
            fps = (
                track_data.get('fps') or
                track_data.get('metadata', {}).get('fps') or
                25.0
            )
            fps = float(fps) if fps else 25.0
            break

    # Collect per-player raw positions: trackId -> {team, frames, xs, ys}
    player_raw = {}  # type: Dict[int, Dict[str, Any]]
    for player in pitch_data.get("players", []):
        tid = player["trackId"]
        team = player.get("teamId", -1)
        traj = player.get("trajectory2d", [])
        if not traj:
            continue
        frames = []
        xs = []
        ys = []
        for pt in traj:
            frames.append(int(pt["frameIndex"]))
            xs.append(float(pt["x"]))
            ys.append(float(pt["y"]))
        player_raw[tid] = {
            "team": team,
            "frames": np.array(frames),
            "xs": np.array(xs),
            "ys": np.array(ys),
        }

    players_out = {}  # type: Dict[str, Dict[str, Any]]

    for tid, raw in player_raw.items():
        # Sort by frame
        order = np.argsort(raw["frames"])
        frames = raw["frames"][order]
        xs = raw["xs"][order]
        ys = raw["ys"][order]

        # Jitter filter: 3-frame rolling median
        xs = _rolling_median_3(xs)
        ys = _rolling_median_3(ys)

        n = len(frames)

        # Position heatmap
        pos_grid = np.zeros((GRID_H, GRID_W), dtype=np.float64)
        for i in range(n):
            cx = max(0.0, min(xs[i], PITCH_W))
            cy = max(0.0, min(ys[i], PITCH_H))
            col = int(cx / PITCH_W * GRID_W)
            row = int(cy / PITCH_H * GRID_H)
            col = min(col, GRID_W - 1)
            row = min(row, GRID_H - 1)
            pos_grid[row, col] += 1.0

        # Compute speeds and distances
        speeds = np.zeros(max(n - 1, 0), dtype=np.float64)
        distances = np.zeros(max(n - 1, 0), dtype=np.float64)
        seg_mid_x = np.zeros(max(n - 1, 0), dtype=np.float64)
        seg_mid_y = np.zeros(max(n - 1, 0), dtype=np.float64)
        valid = np.zeros(max(n - 1, 0), dtype=bool)

        MAX_HUMAN_SPEED_MS = 12.0  # m/s — fastest human sprint
        # At 25fps, max sprint ~12 m/s = 0.48m/frame. Allow 3m/frame
        # to handle stride + FPS variance without catching teleports.
        MAX_FRAME_DISPLACEMENT_M = 3.0

        for i in range(n - 1):
            frame_gap = int(frames[i + 1] - frames[i])
            dt = frame_gap / fps
            if dt <= 0 or dt > 2.0:
                continue
            dx = xs[i + 1] - xs[i]
            dy = ys[i + 1] - ys[i]
            d = np.sqrt(dx * dx + dy * dy)

            # Reject track-ID switches / teleports before computing speed
            displacement_per_frame = d / frame_gap
            if displacement_per_frame > MAX_FRAME_DISPLACEMENT_M:
                continue  # track break — skip segment entirely

            s = d / dt

            distances[i] = d
            speeds[i] = s
            seg_mid_x[i] = (xs[i] + xs[i + 1]) / 2.0
            seg_mid_y[i] = (ys[i] + ys[i + 1]) / 2.0
            valid[i] = True
        
        # Apply hard speed cap
        speeds = np.clip(speeds, 0, MAX_HUMAN_SPEED_MS)
        
        # Outlier rejection: replace spikes with local median
        for i in range(len(speeds)):
            if not valid[i]:
                continue
                
            # Get surrounding window (5 frames before and after)
            start_idx = max(0, i - 5)
            end_idx = min(len(speeds), i + 6)
            window_speeds = speeds[start_idx:end_idx]
            window_valid = valid[start_idx:end_idx]
            
            # Only consider valid speeds in window
            valid_window_speeds = window_speeds[window_valid]
            
            if len(valid_window_speeds) >= 3:
                local_median = np.median(valid_window_speeds)
                # Outlier threshold: 3x local median
                if speeds[i] > local_median * 3:
                    speeds[i] = local_median

        # Sprint and high-intensity heatmaps
        sprint_grid = np.zeros((GRID_H, GRID_W), dtype=np.float64)
        hi_grid = np.zeros((GRID_H, GRID_W), dtype=np.float64)

        total_distance = 0.0
        sprint_distance = 0.0
        hi_distance = 0.0
        top_speed = 0.0

        for i in range(len(speeds)):
            if not valid[i]:
                continue
            total_distance += distances[i]
            if speeds[i] > top_speed:
                top_speed = speeds[i]

            mx = max(0.0, min(seg_mid_x[i], PITCH_W))
            my = max(0.0, min(seg_mid_y[i], PITCH_H))
            col = min(int(mx / PITCH_W * GRID_W), GRID_W - 1)
            row = min(int(my / PITCH_H * GRID_H), GRID_H - 1)

            if speeds[i] > HIGH_INTENSITY_THRESHOLD:
                hi_distance += distances[i]
                hi_grid[row, col] += 1.0
            if speeds[i] > SPRINT_THRESHOLD:
                sprint_distance += distances[i]
                sprint_grid[row, col] += 1.0

        # Sprint count: distinct bursts above threshold
        # Burst broken when speed drops below threshold for 3+ consecutive frames
        sprint_count = 0
        in_sprint = False
        below_count = 0
        for i in range(len(speeds)):
            if not valid[i]:
                if in_sprint:
                    below_count += 1
                    if below_count >= 3:
                        in_sprint = False
                continue
            if speeds[i] > SPRINT_THRESHOLD:
                if not in_sprint:
                    sprint_count += 1
                    in_sprint = True
                below_count = 0
            else:
                if in_sprint:
                    below_count += 1
                    if below_count >= 3:
                        in_sprint = False

        # Average speed
        total_time = 0.0
        for i in range(n - 1):
            dt = (frames[i + 1] - frames[i]) / fps
            if dt <= 0 or dt > 2.0:
                continue
            total_time += dt
        avg_speed = total_distance / total_time if total_time > 0 else 0.0

        players_out[str(tid)] = {
            "team": raw["team"],
            "total_distance_m": round(total_distance, 2),
            "sprint_distance_m": round(sprint_distance, 2),
            "high_intensity_distance_m": round(hi_distance, 2),
            "top_speed_ms": round(min(top_speed, MAX_HUMAN_SPEED_MS), 2),
            "avg_speed_ms": round(avg_speed, 2),
            "sprint_count": sprint_count,
            "position_heatmap": _normalise_heatmap(pos_grid),
            "sprint_heatmap": _normalise_heatmap(sprint_grid),
            "high_intensity_heatmap": _normalise_heatmap(hi_grid),
            "heatmap_grid_w": GRID_W,
            "heatmap_grid_h": GRID_H,
        }

    # Team summary
    team_summary = {}  # type: Dict[str, Dict[str, Any]]
    for team_id in (0, 1):
        team_players = [p for p in players_out.values() if p["team"] == team_id]
        count = len(team_players)
        total_dist = sum(p["total_distance_m"] for p in team_players)
        sprint_dist = sum(p["sprint_distance_m"] for p in team_players)
        top_spd = max((p["top_speed_ms"] for p in team_players), default=0.0)
        avg_dist = total_dist / count if count > 0 else 0.0
        team_summary[str(team_id)] = {
            "total_distance_m": round(total_dist, 2),
            "sprint_distance_m": round(sprint_dist, 2),
            "top_speed_ms": round(top_spd, 2),
            "avg_distance_per_player_m": round(avg_dist, 2),
        }

    result = {
        "fps": fps,
        "players": players_out,
        "team_summary": team_summary,
    }

    logger.info(
        "Heatmap analysis: %d players, team0=%.0fm team1=%.0fm",
        len(players_out),
        team_summary["0"]["total_distance_m"],
        team_summary["1"]["total_distance_m"],
    )

    return result
