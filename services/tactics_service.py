"""Tactical analysis orchestrator: formations, heatmaps, events.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

PITCH_WIDTH  = 105.0
PITCH_HEIGHT = 68.0
HEATMAP_COLS = 10
HEATMAP_ROWS = 10

# Formation band thresholds along the pitch x-axis (metres).
# Players are grouped into defensive / midfield / attacking thirds.
DEF_MAX_X  = 40.0
MID_MAX_X  = 65.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_job_id(job_id: str) -> str:
    """Strip known suffixes to get the base job ID."""
    for suffix in ("_final_tactics", "_final_pitch", "_final", "_tactics", "_pitch"):
        if job_id.endswith(suffix):
            return job_id[: -len(suffix)]
    return job_id


def _load_pitch_map(job_id: str) -> Optional[Dict]:
    """
    Load pitch_map.json, trying the given job_id then the base job_id as fallback.
    """
    candidates = [job_id]
    base = _base_job_id(job_id)
    if base != job_id:
        candidates.append(base)

    for jid in candidates:
        path = Path(f"temp/{jid}/pitch/pitch_map.json")
        if path.exists():
            with open(path) as f:
                return json.load(f)

    return None


def _avg_position(trajectory_2d: List[Dict]) -> Optional[Dict[str, float]]:
    """Return the mean (x, y) over all trajectory points."""
    if not trajectory_2d:
        return None
    xs = [pt["x"] for pt in trajectory_2d]
    ys = [pt["y"] for pt in trajectory_2d]
    return {"x": float(np.mean(xs)), "y": float(np.mean(ys))}


def _detect_formation(avg_positions: List[Dict[str, Any]]) -> str:
    """
    Infer formation string (e.g. '4-3-3') from average player positions.

    We remove the player furthest from the opposing goal (goalkeeper),
    then bucket the remaining players into defensive / midfield / attacking
    bands based on their mean x-coordinate.
    """
    if not avg_positions:
        return "unknown"

    xs = [p["x"] for p in avg_positions]

    # Determine which end is 'our goal' by looking at where most players are.
    # If most players have low x → they're playing left-to-right (attacking right).
    # Goalkeeper is the one with the lowest x in that case.
    median_x = float(np.median(xs))
    if median_x < PITCH_WIDTH / 2:
        # Attacking right: GK has smallest x
        gk_idx = int(np.argmin(xs))
    else:
        # Attacking left: GK has largest x
        gk_idx = int(np.argmax(xs))

    outfield = [p for i, p in enumerate(avg_positions) if i != gk_idx]

    if not outfield:
        return "1"

    # Normalise so that our goal is at x=0
    out_xs = [p["x"] for p in outfield]
    if median_x >= PITCH_WIDTH / 2:
        out_xs = [PITCH_WIDTH - x for x in out_xs]

    def_band  = sum(1 for x in out_xs if x < DEF_MAX_X)
    mid_band  = sum(1 for x in out_xs if DEF_MAX_X <= x < MID_MAX_X)
    att_band  = sum(1 for x in out_xs if x >= MID_MAX_X)

    parts = []
    if def_band:  parts.append(str(def_band))
    if mid_band:  parts.append(str(mid_band))
    if att_band:  parts.append(str(att_band))

    return "-".join(parts) if parts else "unknown"


def _build_heatmap(trajectory_2d: List[Dict], all_player_trajectories: List[List[Dict]]) -> List[List[float]]:
    """
    Build a 10×10 heatmap for a set of player trajectories.
    Each cell is the fraction of all player-frame observations that fall
    in that cell, normalised so the max is 1.0.
    """
    grid = np.zeros((HEATMAP_ROWS, HEATMAP_COLS), dtype=np.float64)

    for traj in all_player_trajectories:
        for pt in traj:
            col = int(min(pt["x"] / PITCH_WIDTH  * HEATMAP_COLS, HEATMAP_COLS - 1))
            row = int(min(pt["y"] / PITCH_HEIGHT * HEATMAP_ROWS, HEATMAP_ROWS - 1))
            grid[row, col] += 1.0

    max_val = grid.max()
    if max_val > 0:
        grid /= max_val

    return grid.tolist()


# ---------------------------------------------------------------------------
# Team shape
# ---------------------------------------------------------------------------

MAX_PITCH_DIST = float(np.sqrt(PITCH_WIDTH ** 2 + PITCH_HEIGHT ** 2))  # ~125m


def compute_team_shape(avg_positions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute defensive/attacking shape metrics from average player positions.

    Returns a dict with defensiveWidth, defensiveDepth, teamLength, teamWidth,
    compactness, defensiveLine, attackingLine, and shape (formation string).
    """
    if not avg_positions:
        return {
            "defensiveWidth":  0.0,
            "defensiveDepth":  0.0,
            "teamLength":      0.0,
            "teamWidth":       0.0,
            "compactness":     0.0,
            "defensiveLine":   0.0,
            "attackingLine":   0.0,
            "shape":           "unknown",
        }

    pts = np.array([[p["x"], p["y"]] for p in avg_positions], dtype=np.float64)
    xs = pts[:, 0]
    ys = pts[:, 1]

    team_length = float(xs.max() - xs.min())
    team_width  = float(ys.max() - ys.min())

    # Determine attacking direction: if median x < half-pitch, team attacks right
    median_x = float(np.median(xs))
    attacking_right = median_x < PITCH_WIDTH / 2

    # Sort players by x: defenders are near own goal
    sorted_by_x = sorted(avg_positions, key=lambda p: p["x"])
    if attacking_right:
        defenders  = sorted_by_x[:4]   # smallest x = closest to own goal (left)
        attackers  = sorted_by_x[-3:]
    else:
        defenders  = sorted_by_x[-4:]  # largest x = closest to own goal (right)
        attackers  = sorted_by_x[:3]

    def_xs = [p["x"] for p in defenders]
    def_ys = [p["y"] for p in defenders]
    att_xs = [p["x"] for p in attackers]

    defensive_line  = float(np.mean(def_xs))
    attacking_line  = float(np.mean(att_xs))
    defensive_width = float(max(def_ys) - min(def_ys)) if len(def_ys) >= 2 else 0.0
    defensive_depth = float(max(def_xs) - min(def_xs)) if len(def_xs) >= 2 else 0.0

    # Compactness: 1 - (avg pairwise distance / max possible pitch distance)
    n = len(pts)
    if n >= 2:
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 2)
        dist_matrix = np.sqrt((diffs ** 2).sum(axis=2))        # (n, n)
        # Upper triangle only, excluding diagonal
        upper_idx = np.triu_indices(n, k=1)
        avg_pairwise = float(dist_matrix[upper_idx].mean())
        compactness = float(1.0 - avg_pairwise / MAX_PITCH_DIST)
        compactness = max(0.0, min(1.0, compactness))
    else:
        compactness = 1.0

    shape = _detect_formation(avg_positions)

    return {
        "defensiveWidth":  round(defensive_width, 2),
        "defensiveDepth":  round(defensive_depth, 2),
        "teamLength":      round(team_length, 2),
        "teamWidth":       round(team_width, 2),
        "compactness":     round(compactness, 4),
        "defensiveLine":   round(defensive_line, 2),
        "attackingLine":   round(attacking_line, 2),
        "shape":           shape,
    }


# ---------------------------------------------------------------------------
# Passing lanes & pressure map
# ---------------------------------------------------------------------------

def compute_passing_lanes(
    avg_positions_team: List[Dict[str, Any]],
    avg_positions_opponents: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    For each pair of teammates within 30m, check whether any opponent
    is within 2m of the straight line between them.

    Returns list of {"from": trackId, "to": trackId, "blocked": bool, "distance": float}.
    """
    opp_pts = np.array([[p["x"], p["y"]] for p in avg_positions_opponents], dtype=np.float64) \
        if avg_positions_opponents else np.empty((0, 2))

    lanes = []
    n = len(avg_positions_team)
    for i in range(n):
        for j in range(i + 1, n):
            a = avg_positions_team[i]
            b = avg_positions_team[j]
            ax, ay = a["x"], a["y"]
            bx, by = b["x"], b["y"]
            dist = float(np.sqrt((bx - ax) ** 2 + (by - ay) ** 2))
            if dist > 30.0:
                continue

            blocked = False
            if len(opp_pts) > 0:
                # Point-to-segment distance for each opponent
                ab = np.array([bx - ax, by - ay], dtype=np.float64)
                ab_len_sq = float(np.dot(ab, ab))
                for opp in opp_pts:
                    if ab_len_sq == 0:
                        seg_dist = float(np.linalg.norm(opp - np.array([ax, ay])))
                    else:
                        t = float(np.dot(opp - np.array([ax, ay]), ab) / ab_len_sq)
                        t = max(0.0, min(1.0, t))
                        proj = np.array([ax, ay]) + t * ab
                        seg_dist = float(np.linalg.norm(opp - proj))
                    if seg_dist < 2.0:
                        blocked = True
                        break

            lanes.append({
                "from": a["trackId"],
                "to": b["trackId"],
                "blocked": blocked,
                "distance": round(dist, 2),
            })

    return lanes


def compute_pressure_map(
    avg_positions_team: List[Dict[str, Any]],
    avg_positions_opponents: List[Dict[str, Any]],
) -> List[List[float]]:
    """
    10×10 grid over a 105×68 pitch.
    Each cell value = sum of 1/distance for all opponents within 10m of cell centre,
    normalised to 0–1.

    Returns [[float]] shape 10×10 (row-major, row=y, col=x).
    """
    grid = np.zeros((HEATMAP_ROWS, HEATMAP_COLS), dtype=np.float64)

    if not avg_positions_opponents:
        return grid.tolist()

    opp_pts = np.array([[p["x"], p["y"]] for p in avg_positions_opponents], dtype=np.float64)

    cell_w = PITCH_WIDTH  / HEATMAP_COLS
    cell_h = PITCH_HEIGHT / HEATMAP_ROWS

    for row in range(HEATMAP_ROWS):
        cy = (row + 0.5) * cell_h
        for col in range(HEATMAP_COLS):
            cx = (col + 0.5) * cell_w
            cell_center = np.array([cx, cy])
            dists = np.linalg.norm(opp_pts - cell_center, axis=1)
            nearby = dists[dists < 10.0]
            if len(nearby) > 0:
                grid[row, col] = float(np.sum(1.0 / np.maximum(nearby, 0.1)))

    max_val = grid.max()
    if max_val > 0:
        grid /= max_val

    return [[round(float(v), 4) for v in row] for row in grid.tolist()]


# ---------------------------------------------------------------------------
# Space occupation & dangerous zones  (Brick 20)
# ---------------------------------------------------------------------------

def compute_space_occupation(
    avg_positions_team0: List[Dict[str, Any]],
    avg_positions_team1: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute which team controls each cell on a 10x10 pitch grid.

    For each cell, the controlling team is whichever has a player closer
    to the cell centre.  The result includes a control grid (-1/0/1),
    per-team territory percentages, and a list of dangerous zones
    (cells within 20m of either goal with high opponent pressure).
    """
    control = np.full((HEATMAP_ROWS, HEATMAP_COLS), -1, dtype=np.int32)
    cell_w = PITCH_WIDTH / HEATMAP_COLS
    cell_h = PITCH_HEIGHT / HEATMAP_ROWS

    pts0 = np.array([[p["x"], p["y"]] for p in avg_positions_team0], dtype=np.float64) \
        if avg_positions_team0 else np.empty((0, 2))
    pts1 = np.array([[p["x"], p["y"]] for p in avg_positions_team1], dtype=np.float64) \
        if avg_positions_team1 else np.empty((0, 2))

    for row in range(HEATMAP_ROWS):
        cy = (row + 0.5) * cell_h
        for col in range(HEATMAP_COLS):
            cx = (col + 0.5) * cell_w
            center = np.array([cx, cy])
            min0 = float(np.min(np.linalg.norm(pts0 - center, axis=1))) if len(pts0) > 0 else 999.0
            min1 = float(np.min(np.linalg.norm(pts1 - center, axis=1))) if len(pts1) > 0 else 999.0
            if min0 < min1:
                control[row, col] = 0
            elif min1 < min0:
                control[row, col] = 1
            # tie → -1 (contested)

    total_cells = HEATMAP_ROWS * HEATMAP_COLS
    team0_pct = float(np.sum(control == 0)) / total_cells * 100.0
    team1_pct = float(np.sum(control == 1)) / total_cells * 100.0
    contested_pct = 100.0 - team0_pct - team1_pct

    # Dangerous zones: cells within 20m of either goal (x < 20 or x > 85)
    # that the opposing team controls
    dangerous_zones = []
    for row in range(HEATMAP_ROWS):
        cy = (row + 0.5) * cell_h
        for col in range(HEATMAP_COLS):
            cx = (col + 0.5) * cell_w
            owner = int(control[row, col])
            if owner < 0:
                continue
            # Near left goal (x < 20m) — dangerous if team1 controls
            if cx < 20.0 and owner == 1:
                dangerous_zones.append({
                    "row": row, "col": col, "x": round(cx, 1), "y": round(cy, 1),
                    "controlledBy": 1, "threatTo": 0,
                })
            # Near right goal (x > 85m) — dangerous if team0 controls
            elif cx > 85.0 and owner == 0:
                dangerous_zones.append({
                    "row": row, "col": col, "x": round(cx, 1), "y": round(cy, 1),
                    "controlledBy": 0, "threatTo": 1,
                })

    return {
        "controlGrid": control.tolist(),
        "team0Territory": round(team0_pct, 1),
        "team1Territory": round(team1_pct, 1),
        "contestedTerritory": round(contested_pct, 1),
        "dangerousZones": dangerous_zones,
    }


# ---------------------------------------------------------------------------
# Event detection  (Brick 21)
# ---------------------------------------------------------------------------

def detect_events(
    job_id: str,
    pitch_data: Dict[str, Any],
    ball_trajectory: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Detect football events from ball trajectory and player positions.

    Events detected:
      - pass:   ball moves > 8m between frames toward a different teammate
      - shot:   ball moves toward goal (x > 95 or x < 10) at speed > 5m/frame
      - tackle: two opponents converge within 3m of the ball
    """
    if not ball_trajectory or len(ball_trajectory) < 2:
        return []

    # Build per-frame player lookup from pitch_data
    frame_players = {}  # frameIndex -> list of {trackId, teamId, x, y}
    for player in pitch_data.get("players", []):
        tid = player["trackId"]
        team_id = player.get("teamId", -1)
        for pt in player.get("trajectory2d", []):
            fi = int(pt["frameIndex"])
            if fi not in frame_players:
                frame_players[fi] = []
            frame_players[fi].append({
                "trackId": tid, "teamId": team_id,
                "x": float(pt["x"]), "y": float(pt["y"]),
            })

    events = []
    ball_sorted = sorted(ball_trajectory, key=lambda b: b["frameIndex"])

    for idx in range(1, len(ball_sorted)):
        prev_b = ball_sorted[idx - 1]
        curr_b = ball_sorted[idx]

        bx0, by0 = prev_b["x"], prev_b["y"]
        bx1, by1 = curr_b["x"], curr_b["y"]
        ball_dist = float(np.sqrt((bx1 - bx0) ** 2 + (by1 - by0) ** 2))
        fi = int(curr_b["frameIndex"])
        ts = float(curr_b.get("timestampSeconds", 0.0))

        # --- Shot detection: ball heading toward goal at speed ---
        if ball_dist > 5.0:
            if bx1 > 95.0 or bx1 < 10.0:
                target_goal = "right" if bx1 > 95.0 else "left"
                events.append({
                    "type": "shot",
                    "frameIndex": fi,
                    "timestampSeconds": round(ts, 2),
                    "ballFrom": [round(bx0, 1), round(by0, 1)],
                    "ballTo": [round(bx1, 1), round(by1, 1)],
                    "speed": round(ball_dist, 2),
                    "targetGoal": target_goal,
                })
                continue

        # --- Pass detection: ball moves > 8m ---
        if ball_dist > 8.0:
            # Find nearest player to ball at start and end
            players_prev = frame_players.get(int(prev_b["frameIndex"]), [])
            players_curr = frame_players.get(fi, [])

            sender = _nearest_player(players_prev, bx0, by0)
            receiver = _nearest_player(players_curr, bx1, by1)

            if sender and receiver and sender["trackId"] != receiver["trackId"]:
                same_team = sender["teamId"] == receiver["teamId"]
                events.append({
                    "type": "pass",
                    "frameIndex": fi,
                    "timestampSeconds": round(ts, 2),
                    "fromTrackId": sender["trackId"],
                    "toTrackId": receiver["trackId"],
                    "fromTeamId": sender["teamId"],
                    "toTeamId": receiver["teamId"],
                    "distance": round(ball_dist, 2),
                    "successful": same_team,
                })
                continue

        # --- Tackle detection: opponents converge near ball ---
        players_here = frame_players.get(fi, [])
        near_ball = [p for p in players_here
                     if np.sqrt((p["x"] - bx1) ** 2 + (p["y"] - by1) ** 2) < 5.0]
        if len(near_ball) >= 2:
            team_ids = set(p["teamId"] for p in near_ball)
            if len(team_ids) >= 2:
                # Check if any two opponents are within 3m of each other
                for i_nb in range(len(near_ball)):
                    for j_nb in range(i_nb + 1, len(near_ball)):
                        p_a = near_ball[i_nb]
                        p_b = near_ball[j_nb]
                        if p_a["teamId"] == p_b["teamId"]:
                            continue
                        d = float(np.sqrt(
                            (p_a["x"] - p_b["x"]) ** 2 + (p_a["y"] - p_b["y"]) ** 2
                        ))
                        if d < 3.0:
                            events.append({
                                "type": "tackle",
                                "frameIndex": fi,
                                "timestampSeconds": round(ts, 2),
                                "player1": p_a["trackId"],
                                "player1Team": p_a["teamId"],
                                "player2": p_b["trackId"],
                                "player2Team": p_b["teamId"],
                                "distance": round(d, 2),
                            })
                            break
                    else:
                        continue
                    break

    return events


def _nearest_player(
    players: List[Dict[str, Any]], bx: float, by: float
) -> Optional[Dict[str, Any]]:
    """Return the player closest to (bx, by), or None."""
    if not players:
        return None
    best = None
    best_dist = float("inf")
    for p in players:
        d = (p["x"] - bx) ** 2 + (p["y"] - by) ** 2
        if d < best_dist:
            best_dist = d
            best = p
    return best


def _ball_to_pitch(
    ball_trajectory: List[Dict[str, Any]],
    pitch_data: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert ball pixel coordinates to pitch metres.

    If pitch_data has player trajectories, we use the frame dimensions
    and proportional mapping (same fallback as pitch_service).
    """
    if not ball_trajectory:
        return []

    # Try to get frame dimensions from pitch_data
    frame_w = pitch_data.get("frameWidth", 0) if pitch_data else 0
    frame_h = pitch_data.get("frameHeight", 0) if pitch_data else 0

    # Fallback: use a representative player's trajectory to estimate frame dims
    if (frame_w == 0 or frame_h == 0) and pitch_data:
        for player in pitch_data.get("players", []):
            traj = player.get("trajectory2d", [])
            if traj:
                # Use proportional mapping with assumed 4K landscape
                frame_w = frame_w or 3840
                frame_h = frame_h or 2160
                break

    if frame_w == 0:
        frame_w = 3840
    if frame_h == 0:
        frame_h = 2160

    result = []
    for b in ball_trajectory:
        # Ball coords are in pixel space (from tracking_service)
        px = b.get("x", 0.0)
        py = b.get("y", 0.0)
        # Proportional mapping to pitch coords
        pitch_x = px / frame_w * PITCH_WIDTH
        pitch_y = py / frame_h * PITCH_HEIGHT
        pitch_x = max(0.0, min(PITCH_WIDTH, pitch_x))
        pitch_y = max(0.0, min(PITCH_HEIGHT, pitch_y))
        result.append({
            "frameIndex": b["frameIndex"],
            "timestampSeconds": b.get("timestampSeconds", 0.0),
            "x": pitch_x,
            "y": pitch_y,
            "confidence": b.get("confidence", 0.0),
        })
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_tactics(
    job_id: str,
) -> Dict[str, Any]:
    """
    Compute formation and heatmap for each team from an existing pitch_map.json.

    Requires:
        temp/{job_id}/pitch/pitch_map.json   (produced by map_pitch())
    """
    pitch_data = _load_pitch_map(job_id)
    if pitch_data is None:
        raise ValueError(
            f"pitch_map.json not found for job '{job_id}'. "
            "Run /api/v1/pitch/map first."
        )

    players = pitch_data["players"]

    # Separate by team
    teams: Dict[int, List[Dict]] = {}
    for p in players:
        tid = p["teamId"]
        teams.setdefault(tid, []).append(p)

    team_results: Dict[int, Dict[str, Any]] = {}

    for team_id, team_players in teams.items():
        avg_positions = []
        for p in team_players:
            avg = _avg_position(p["trajectory2d"])
            if avg is not None:
                avg_positions.append({
                    "trackId": p["trackId"],
                    "x": round(avg["x"], 2),
                    "y": round(avg["y"], 2),
                })

        formation = _detect_formation(avg_positions)

        all_trajs = [p["trajectory2d"] for p in team_players]
        heatmap = _build_heatmap([], all_trajs)

        team_results[team_id] = {
            "formation": formation,
            "averagePositions": avg_positions,
            "heatmap": [[round(v, 4) for v in row] for row in heatmap],
        }

    # Compute passing lanes and pressure maps for teams 0 and 1
    avg0 = team_results.get(0, {}).get("averagePositions", [])
    avg1 = team_results.get(1, {}).get("averagePositions", [])

    for tid, own_avg, opp_avg in ((0, avg0, avg1), (1, avg1, avg0)):
        if tid in team_results:
            team_results[tid]["passingLanes"] = compute_passing_lanes(own_avg, opp_avg)
            team_results[tid]["pressureMap"]  = compute_pressure_map(own_avg, opp_avg)
            team_results[tid]["shape"]        = compute_team_shape(own_avg)

    # Brick 20: space occupation & dangerous zones
    space_occupation = compute_space_occupation(avg0, avg1)

    # Brick 21: event detection from ball trajectory + pitch positions
    base_jid = _base_job_id(job_id)
    track_path = Path(f"temp/{base_jid}/tracking/track_results.json")
    ball_trajectory = []
    if track_path.exists():
        with open(track_path) as f:
            track_data = json.load(f)
        ball_trajectory = track_data.get("ball_trajectory", [])

    # Transform ball pixel coords to pitch coords using pitch_data homography info
    ball_pitch = _ball_to_pitch(ball_trajectory, pitch_data)
    events = detect_events(job_id, pitch_data, ball_pitch)

    # Persist
    output_dir = Path(f"temp/{job_id}/tactics")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / "tactics_results.json")

    result: Dict[str, Any] = {
        "jobId": job_id,
        "outputPath": output_path,
    }

    # Include teams 0 and 1 explicitly (may be absent if no players assigned)
    for tid in (0, 1):
        key = f"team{tid}"
        result[key] = team_results.get(tid, {
            "formation": "unknown",
            "averagePositions": [],
            "heatmap": [[0.0] * HEATMAP_COLS for _ in range(HEATMAP_ROWS)],
            "passingLanes": [],
            "pressureMap": [[0.0] * HEATMAP_COLS for _ in range(HEATMAP_ROWS)],
            "shape": compute_team_shape([]),
        })

    result["spaceOccupation"] = space_occupation
    result["events"] = events

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(
        "Tactics analysis done. Team0=%s Team1=%s | "
        "territory=%.0f%%/%.0f%% | events=%d",
        result["team0"]["formation"],
        result["team1"]["formation"],
        space_occupation["team0Territory"],
        space_occupation["team1Territory"],
        len(events),
    )
    return result
