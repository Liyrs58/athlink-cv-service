import json
import logging
import math
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

PITCH_WIDTH = 105.0
PITCH_HEIGHT = 68.0

LEFT_GOAL = (0.0, 34.0)
RIGHT_GOAL = (105.0, 34.0)
GOAL_WIDTH = 7.32
HALF_GOAL = 3.66

SPEED_THRESHOLD = 8.0
PRE_SHOT_SPEED_MAX = 4.0
PRE_SHOT_WINDOW = 10
SHOT_DEDUP_FRAMES = 30
FINAL_THIRD_RIGHT = 68.0
FINAL_THIRD_LEFT = 37.0
MAX_INTERP_GAP = 5
SMOOTH_WINDOW = 5
CARRIER_SEARCH_DIST_M = 5.0
CARRIER_LOOKBACK = 10

INTERCEPT = -1.5
COEFF_DIST = -0.08
COEFF_ANGLE = 1.2


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


def _build_continuous_ball_track(ball_trajectory, pitch_data):
    # type: (List[Dict], Optional[Dict]) -> Tuple[Dict[int, Tuple[float, float]], set, int, int]
    """
    Build continuous ball track with interpolation for small gaps.

    Returns:
        positions: {frame: (world_x, world_y)}
        ball_lost_frames: set of frame indices marked as lost
        interpolated_count: number of interpolated frames
        lost_count: number of ball-lost frames
    """
    frame_w = 3840
    frame_h = 2160
    if pitch_data is not None:
        frame_w = pitch_data.get("frameWidth", 3840) or 3840
        frame_h = pitch_data.get("frameHeight", 2160) or 2160

    # Convert to world coords and sort
    raw = []  # type: List[Tuple[int, float, float]]
    for b in ball_trajectory:
        fi = int(b["frameIndex"])
        px = float(b.get("x", 0.0))
        py = float(b.get("y", 0.0))
        wx = px / frame_w * PITCH_WIDTH
        wy = py / frame_h * PITCH_HEIGHT
        raw.append((fi, wx, wy))
    raw.sort(key=lambda t: t[0])

    if not raw:
        return {}, set(), 0, 0

    positions = {}  # type: Dict[int, Tuple[float, float]]
    interpolated_frames = set()  # type: set
    ball_lost = set()  # type: set

    for fi, wx, wy in raw:
        positions[fi] = (wx, wy)

    # Interpolate small gaps, mark large gaps as ball_lost
    interpolated_count = 0
    lost_count = 0

    for i in range(len(raw) - 1):
        fi_a, xa, ya = raw[i]
        fi_b, xb, yb = raw[i + 1]
        gap = fi_b - fi_a

        if gap <= 1:
            continue

        missing = gap - 1  # frames between fi_a and fi_b exclusive

        if missing <= MAX_INTERP_GAP:
            # Linear interpolation
            for s in range(1, gap):
                frac = s / gap
                interp_fi = fi_a + s
                ix = xa + frac * (xb - xa)
                iy = ya + frac * (yb - ya)
                positions[interp_fi] = (ix, iy)
                interpolated_frames.add(interp_fi)
                interpolated_count += 1
        else:
            # Mark as ball lost
            for s in range(1, gap):
                lost_fi = fi_a + s
                ball_lost.add(lost_fi)
                lost_count += 1

    return positions, ball_lost, interpolated_count, lost_count


def _smooth_and_compute_speed(positions, ball_lost, fps):
    # type: (Dict[int, Tuple[float, float]], set, float) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]
    """
    Apply 5-frame rolling average then compute speed via central difference.

    Returns:
        smoothed_x: {frame: smoothed_world_x}
        smoothed_y: {frame: smoothed_world_y}
        speeds: {frame: speed_m_s}
    """
    frames = sorted(f for f in positions if f not in ball_lost)
    if len(frames) < 3:
        return {}, {}, {}

    # Build arrays
    n = len(frames)
    xs = np.array([positions[f][0] for f in frames], dtype=np.float64)
    ys = np.array([positions[f][1] for f in frames], dtype=np.float64)

    # 5-frame rolling average
    half_w = SMOOTH_WINDOW // 2
    sx = np.copy(xs)
    sy = np.copy(ys)
    for i in range(n):
        lo = max(0, i - half_w)
        hi = min(n, i + half_w + 1)
        sx[i] = np.mean(xs[lo:hi])
        sy[i] = np.mean(ys[lo:hi])

    smoothed_x = {}  # type: Dict[int, float]
    smoothed_y = {}  # type: Dict[int, float]
    for i, fi in enumerate(frames):
        smoothed_x[fi] = float(sx[i])
        smoothed_y[fi] = float(sy[i])

    # Central difference for speed
    speeds = {}  # type: Dict[int, float]
    MAX_BALL_SPEED_MS = 40.0  # m/s - reasonable max for ball speed
    
    for i in range(1, n - 1):
        fi_prev = frames[i - 1]
        fi_next = frames[i + 1]
        fi = frames[i]

        dt = (fi_next - fi_prev) / fps
        if dt <= 0:
            continue

        dx = sx[i + 1] - sx[i - 1]
        dy = sy[i + 1] - sy[i - 1]
        speed = math.sqrt(dx * dx + dy * dy) / dt
        
        # Apply speed cap
        speed = min(speed, MAX_BALL_SPEED_MS)
        speeds[fi] = speed

    return smoothed_x, smoothed_y, speeds


def _compute_xg_value(dist, angle):
    # type: (float, float) -> float
    z = INTERCEPT + COEFF_DIST * dist + COEFF_ANGLE * angle
    xg = 1.0 / (1.0 + math.exp(-z))
    return max(0.01, min(0.99, xg))


def compute_xg(job_id):
    # type: (str) -> Dict[str, Any]
    """
    Build xG model with ball track interpolation and smoothed velocity.

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

    # Load pitch_map.json
    pitch_data = None
    for jid in candidates:
        pitch_data = _load_json(Path("temp/{}/pitch/pitch_map.json".format(jid)))
        if pitch_data is not None:
            break

    # Load team_results.json
    team_map = {}  # type: Dict[int, int]
    for jid in candidates:
        team_data = _load_json(Path("temp/{}/tracking/team_results.json".format(jid)))
        if team_data is not None:
            tracks_list = team_data.get("tracks", team_data) if isinstance(team_data, dict) else team_data
            for t in tracks_list:
                team_map[t["trackId"]] = t.get("teamId", -1)
            break

    # Get FPS
    fps = (
        track_data.get('fps') or
        track_data.get('metadata', {}).get('fps') or
        25.0
    )
    fps = float(fps) if fps else 25.0

    # STEP 1: Build continuous ball track
    ball_trajectory = track_data.get("ball_trajectory", [])
    if not ball_trajectory:
        return {
            "shots": [],
            "xg_team_0": 0.0,
            "xg_team_1": 0.0,
            "shots_team_0": 0,
            "shots_team_1": 0,
            "ball_lost_frames": 0,
            "interpolated_frames": 0,
        }

    positions, ball_lost, interpolated_count, lost_count = _build_continuous_ball_track(
        ball_trajectory, pitch_data
    )

    # STEP 2: Smooth and compute speed
    smoothed_x, smoothed_y, speeds = _smooth_and_compute_speed(positions, ball_lost, fps)

    # Build per-frame player world positions for team attribution
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

    # Interpolated frame set for tagging
    interp_set = set()  # type: set
    # Rebuild from _build_continuous_ball_track — positions that aren't in original
    original_frames = set(int(b["frameIndex"]) for b in ball_trajectory)
    for fi in positions:
        if fi not in original_frames and fi not in ball_lost:
            interp_set.add(fi)

    # STEP 3: Shot detection
    speed_frames = sorted(speeds.keys())
    shot_candidates = []  # type: List[int]

    for fi in speed_frames:
        speed = speeds[fi]
        if speed <= SPEED_THRESHOLD:
            continue

        bx = smoothed_x.get(fi, 0.0)

        # Must be in final third
        if not (bx > FINAL_THIRD_RIGHT or bx < FINAL_THIRD_LEFT):
            continue

        # Pre-shot speed check: speed < 4.0 in the 10 frames before
        pre_slow = False
        for pf in speed_frames:
            if pf >= fi:
                break
            if fi - pf > PRE_SHOT_WINDOW:
                continue
            if speeds.get(pf, 999.0) < PRE_SHOT_SPEED_MAX:
                pre_slow = True
                break
        if not pre_slow:
            continue

        # Dedup: not within SHOT_DEDUP_FRAMES of another candidate
        if shot_candidates and fi - shot_candidates[-1] < SHOT_DEDUP_FRAMES:
            continue

        shot_candidates.append(fi)

    # STEP 4 & 5: xG calculation and team attribution
    shots = []  # type: List[Dict[str, Any]]

    for fi in shot_candidates:
        bx = smoothed_x.get(fi, 0.0)
        by = smoothed_y.get(fi, 0.0)

        # Determine target goal
        if bx > 52.5:
            goal = RIGHT_GOAL
        else:
            goal = LEFT_GOAL

        dist = math.sqrt((bx - goal[0]) ** 2 + (by - goal[1]) ** 2)

        # Angle subtended by goal
        if dist < 0.01:
            angle = math.pi / 2.0
        else:
            angle = math.atan(GOAL_WIDTH * dist / (dist * dist + HALF_GOAL * HALF_GOAL))

        xg = _compute_xg_value(dist, angle)

        # Team attribution: nearest player within 5m at shot frame
        team = -1
        best_tid = -1
        best_d = float("inf")
        for p in player_world.get(fi, []):
            dx = p["x"] - bx
            dy = p["y"] - by
            d = math.sqrt(dx * dx + dy * dy)
            if d < CARRIER_SEARCH_DIST_M and d < best_d:
                best_d = d
                best_tid = p["trackId"]

        if best_tid >= 0:
            team = team_map.get(best_tid, -1)

        # Lookback if no player found at shot frame
        if team < 0:
            for lookback in range(1, CARRIER_LOOKBACK + 1):
                lb_fi = fi - lookback
                for p in player_world.get(lb_fi, []):
                    lb_bx = smoothed_x.get(lb_fi, bx)
                    lb_by = smoothed_y.get(lb_fi, by)
                    dx = p["x"] - lb_bx
                    dy = p["y"] - lb_by
                    d = math.sqrt(dx * dx + dy * dy)
                    if d < CARRIER_SEARCH_DIST_M:
                        team = team_map.get(p["trackId"], -1)
                        break
                if team >= 0:
                    break

        shots.append({
            "frame": fi,
            "ball_x": round(bx, 2),
            "ball_y": round(by, 2),
            "dist_to_goal_m": round(dist, 2),
            "angle_rad": round(angle, 4),
            "xg": round(xg, 4),
            "team": team,
            "ball_speed_ms": round(speeds.get(fi, 0.0), 2),
            "interpolated_position": fi in interp_set,
        })

    # Aggregate
    xg_team_0 = 0.0
    xg_team_1 = 0.0
    shots_team_0 = 0
    shots_team_1 = 0
    for s in shots:
        if s["team"] == 0:
            xg_team_0 += s["xg"]
            shots_team_0 += 1
        elif s["team"] == 1:
            xg_team_1 += s["xg"]
            shots_team_1 += 1

    result = {
        "shots": shots,
        "xg_team_0": round(xg_team_0, 4),
        "xg_team_1": round(xg_team_1, 4),
        "shots_team_0": shots_team_0,
        "shots_team_1": shots_team_1,
        "ball_lost_frames": lost_count,
        "interpolated_frames": interpolated_count,
    }

    logger.info(
        "xG analysis: %d shots, team0=%.2f (%d), team1=%.2f (%d), "
        "%d interpolated, %d lost",
        len(shots), xg_team_0, shots_team_0, xg_team_1, shots_team_1,
        interpolated_count, lost_count,
    )

    return result
