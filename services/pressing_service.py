import json
import logging
import math
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from services.pass_network_service import (
    _base_job_id,
    _load_json,
    _ball_to_pitch,
    compute_possession_frames,
)

logger = logging.getLogger(__name__)

PITCH_WIDTH = 105.0
PITCH_HEIGHT = 68.0
GRID_W = 21
GRID_H = 14
PRESS_RADIUS_M = 4.0
DEDUP_WINDOW = 10
DEF_THIRD_RIGHT = 70.0  # team 0 presses here (team 1's def third)
DEF_THIRD_LEFT = 35.0   # team 1 presses here (team 0's def third)


def _normalise_grid(grid):
    # type: (np.ndarray) -> List[float]
    mx = grid.max()
    if mx > 0:
        grid = grid / mx
    return [round(float(v), 4) for v in grid.ravel()]


def compute_pressing(job_id):
    # type: (str) -> Dict[str, Any]
    """
    Compute pressing intensity, PPDA, and recovery time.

    Reads:
        temp/{job_id}/tracking/track_results.json
        temp/{job_id}/pitch/pitch_map.json
        temp/{job_id}/tracking/team_results.json
    """
    base = _base_job_id(job_id)
    candidates = [job_id]
    if base != job_id:
        candidates.append(base)

    # Load track_results.json
    track_data = None
    for jid in candidates:
        track_data = _load_json(Path("temp/{}/tracking/track_results.json".format(jid)))
        if track_data is not None:
            break
    if track_data is None:
        raise FileNotFoundError(
            "track_results.json not found for job '{}'".format(job_id)
        )

    # Load team_results.json
    team_map = {}  # type: Dict[int, int]
    for jid in candidates:
        team_data = _load_json(Path("temp/{}/tracking/team_results.json".format(jid)))
        if team_data is not None:
            tracks_list = team_data.get("tracks", team_data) if isinstance(team_data, dict) else team_data
            for t in tracks_list:
                team_map[t["trackId"]] = t.get("teamId", -1)
            break

    # Load pitch_map.json
    pitch_data = None
    for jid in candidates:
        pitch_data = _load_json(Path("temp/{}/pitch/pitch_map.json".format(jid)))
        if pitch_data is not None:
            break

    # Build per-frame player world positions
    player_world = {}  # type: Dict[int, List[Dict[str, Any]]]
    if pitch_data is not None:
        for player in pitch_data.get("players", []):
            tid = player["trackId"]
            for pt in player.get("trajectory2d", []):
                fi = int(pt["frameIndex"])
                if fi not in player_world:
                    player_world[fi] = []
                player_world[fi].append({
                    "trackId": tid,
                    "x": float(pt["x"]),
                    "y": float(pt["y"]),
                })

    # Ball world positions
    ball_trajectory = track_data.get("ball_trajectory", [])
    ball_world = _ball_to_pitch(ball_trajectory, pitch_data)

    common_frames = sorted(set(ball_world.keys()) & set(player_world.keys()))

    # Compute possession using shared logic
    carrier_by_frame, contested_count = compute_possession_frames(
        common_frames, ball_world, player_world, team_map
    )

    # Defensive actions detection
    # {team: list of (frame, pressing_player_tid, carrier_x)}
    def_actions = {0: [], 1: []}  # type: Dict[int, List[Tuple[int, int, float]]]

    # Dedup: track last action frame per pressing player
    last_action_frame = {}  # type: Dict[int, int]

    for fi in common_frames:
        carrier_tid = carrier_by_frame.get(fi)
        if carrier_tid is None:
            continue

        carrier_team = team_map.get(carrier_tid, -1)
        if carrier_team not in (0, 1):
            continue

        # Find carrier world position
        carrier_x = 0.0
        carrier_y = 0.0
        for p in player_world.get(fi, []):
            if p["trackId"] == carrier_tid:
                carrier_x = p["x"]
                carrier_y = p["y"]
                break

        # Check non-possessing team players within PRESS_RADIUS of carrier
        pressing_team = 1 - carrier_team
        for p in player_world.get(fi, []):
            ptid = p["trackId"]
            if team_map.get(ptid, -1) != pressing_team:
                continue
            dx = p["x"] - carrier_x
            dy = p["y"] - carrier_y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < PRESS_RADIUS_M:
                # Dedup: one action per player per DEDUP_WINDOW frames
                prev_fi = last_action_frame.get(ptid, -DEDUP_WINDOW - 1)
                if fi - prev_fi >= DEDUP_WINDOW:
                    def_actions[pressing_team].append((fi, ptid, carrier_x))
                    last_action_frame[ptid] = fi

    # Count passes in defensive thirds (for PPDA denominator/numerator)
    # Import pass events logic: a pass = carrier change within same team
    # We need passes by location for PPDA
    passes_in_zone = {0: 0, 1: 0}  # type: Dict[int, int]
    # passes_in_zone[0] = passes by team 0 in x < 35 (team 0's def third)
    # passes_in_zone[1] = passes by team 1 in x > 70... wait, let me re-read spec

    # PPDA for team 0:
    #   numerator = passes completed by team 1 in x < 35
    #   denominator = defensive actions by team 0 in x > 70
    # PPDA for team 1:
    #   numerator = passes completed by team 0 in x > 70... wait
    # Actually re-reading: team 0 presses in x > 70 (team 1's def third)
    #   team_0_ppda numerator = passes by team 1 in x < 35
    #   team_0_ppda denominator = def actions by team 0 in x > 70

    # Count passes by each team in specific zones
    # Walk carrier transitions to find passes
    carrier_frames_sorted = sorted(carrier_by_frame.keys())
    team_passes_in_zone = {0: 0, 1: 0}  # type: Dict[int, int]
    # team_passes_in_zone[0] = passes by team 0 in x > 70 (for team 1's PPDA numerator)
    # team_passes_in_zone[1] = passes by team 1 in x < 35 (for team 0's PPDA numerator)

    prev_carrier = -1
    prev_fi = -1
    for fi in carrier_frames_sorted:
        curr_carrier = carrier_by_frame[fi]
        if prev_carrier >= 0 and curr_carrier != prev_carrier:
            prev_team = team_map.get(prev_carrier, -1)
            curr_team = team_map.get(curr_carrier, -1)
            if prev_team == curr_team and prev_team >= 0:
                # It's a pass — check ball position at pass frame
                bx = ball_world.get(fi, (0.0, 0.0))[0]
                if prev_team == 1 and bx < DEF_THIRD_LEFT:
                    team_passes_in_zone[1] += 1
                elif prev_team == 0 and bx > DEF_THIRD_RIGHT:
                    team_passes_in_zone[0] += 1
        prev_carrier = curr_carrier
        prev_fi = fi

    # Defensive actions in defensive thirds
    def_actions_in_third = {0: 0, 1: 0}  # type: Dict[int, int]
    for team in (0, 1):
        for fi, ptid, cx in def_actions[team]:
            if team == 0 and cx > DEF_THIRD_RIGHT:
                def_actions_in_third[0] += 1
            elif team == 1 and cx < DEF_THIRD_LEFT:
                def_actions_in_third[1] += 1

    # PPDA
    # team_0_ppda: passes by team 1 in x<35 / def actions by team 0 in x>70
    t0_ppda_num = team_passes_in_zone[1]
    t0_ppda_den = def_actions_in_third[0]
    t0_ppda = None  # type: Optional[float]
    if t0_ppda_den >= 5:
        t0_ppda = round(t0_ppda_num / t0_ppda_den, 2)

    # team_1_ppda: passes by team 0 in x>70 / def actions by team 1 in x<35
    t1_ppda_num = team_passes_in_zone[0]
    t1_ppda_den = def_actions_in_third[1]
    t1_ppda = None  # type: Optional[float]
    if t1_ppda_den >= 5:
        t1_ppda = round(t1_ppda_num / t1_ppda_den, 2)

    # Press height per team
    press_heights = {0: [], 1: []}  # type: Dict[int, List[float]]
    for team in (0, 1):
        for fi, ptid, cx in def_actions[team]:
            # Use pressing player's x, not the carrier's
            for p in player_world.get(fi, []):
                if p["trackId"] == ptid:
                    press_heights[team].append(p["x"])
                    break

    avg_press_height = {}  # type: Dict[int, float]
    for team in (0, 1):
        if press_heights[team]:
            avg_press_height[team] = round(float(np.mean(press_heights[team])), 2)
        else:
            avg_press_height[team] = 0.0

    # Pressing intensity grids
    press_grids = {}  # type: Dict[int, np.ndarray]
    for team in (0, 1):
        grid = np.zeros((GRID_H, GRID_W), dtype=np.float64)
        for fi, ptid, cx in def_actions[team]:
            for p in player_world.get(fi, []):
                if p["trackId"] == ptid:
                    px = max(0.0, min(p["x"], PITCH_WIDTH))
                    py = max(0.0, min(p["y"], PITCH_HEIGHT))
                    col = min(int(px / PITCH_WIDTH * GRID_W), GRID_W - 1)
                    row = min(int(py / PITCH_HEIGHT * GRID_H), GRID_H - 1)
                    grid[row, col] += 1.0
                    break
        press_grids[team] = grid

    # Recovery time: frames until pressing team gets within 4m of ball
    # after a possession change
    recovery_times = {0: [], 1: []}  # type: Dict[int, List[int]]
    prev_poss_team = -1
    transition_frame = -1

    for fi in carrier_frames_sorted:
        carrier = carrier_by_frame[fi]
        curr_team = team_map.get(carrier, -1)
        if curr_team not in (0, 1):
            continue

        if curr_team != prev_poss_team and prev_poss_team >= 0:
            # Possession changed to curr_team
            transition_frame = fi
            # Measure how many frames until curr_team player is within 4m of ball
            bpos = ball_world.get(fi, None)
            if bpos is not None:
                found = False
                for fj in carrier_frames_sorted:
                    if fj < fi:
                        continue
                    if fj - fi > 150:  # cap search at 150 frames
                        break
                    bp = ball_world.get(fj)
                    if bp is None:
                        continue
                    for p in player_world.get(fj, []):
                        if team_map.get(p["trackId"], -1) == curr_team:
                            d = math.sqrt((p["x"] - bp[0]) ** 2 + (p["y"] - bp[1]) ** 2)
                            if d < PRESS_RADIUS_M:
                                recovery_times[curr_team].append(fj - fi)
                                found = True
                                break
                    if found:
                        break

        prev_poss_team = curr_team

    avg_recovery = {}  # type: Dict[int, float]
    for team in (0, 1):
        if recovery_times[team]:
            avg_recovery[team] = round(float(np.mean(recovery_times[team])), 1)
        else:
            avg_recovery[team] = 0.0

    result = {
        "team_0": {
            "ppda": t0_ppda,
            "ppda_sample_size": t0_ppda_den,
            "defensive_actions": len(def_actions[0]),
            "defensive_actions_in_def_third": def_actions_in_third[0],
            "passes_conceded_in_def_third": t0_ppda_num,
            "pressing_intensity_map": _normalise_grid(press_grids[0]),
            "avg_press_height_m": avg_press_height[0],
            "avg_recovery_time_frames": avg_recovery[0],
        },
        "team_1": {
            "ppda": t1_ppda,
            "ppda_sample_size": t1_ppda_den,
            "defensive_actions": len(def_actions[1]),
            "defensive_actions_in_def_third": def_actions_in_third[1],
            "passes_conceded_in_def_third": t1_ppda_num,
            "pressing_intensity_map": _normalise_grid(press_grids[1]),
            "avg_press_height_m": avg_press_height[1],
            "avg_recovery_time_frames": avg_recovery[1],
        },
        "frame_count_analysed": len(common_frames),
        "contested_frames": contested_count,
    }

    logger.info(
        "Pressing: team0 %d actions (ppda=%s), team1 %d actions (ppda=%s)",
        len(def_actions[0]), t0_ppda,
        len(def_actions[1]), t1_ppda,
    )

    return result
