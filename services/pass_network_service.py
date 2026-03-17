import json
import logging
import math
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

PITCH_WIDTH = 105.0
PITCH_HEIGHT = 68.0
CARRIER_RADIUS_M = 5.0
CARRIER_MIN_FRAMES = 2
PASS_MIN_DIST_M = 2.0
PASS_MAX_FRAMES = 60


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


def _check_ball_data_quality(job_id):
    # type: (str) -> dict
    """
    Returns a dict explaining ball data status:
    {
      "has_ball_world_coords": bool,
      "ball_frame_coverage": float,
      "calibration_valid": bool,
      "reason": str
    }
    """
    base = _base_job_id(job_id)
    candidates = [job_id]
    if base != job_id:
        candidates.append(base)

    pitch_data = None
    for jid in candidates:
        p = Path("temp/{}/pitch/pitch_map.json".format(jid))
        if p.exists():
            with open(p) as f:
                pitch_data = json.load(f)
            break

    if pitch_data is None:
        return {
            "has_ball_world_coords": False,
            "ball_frame_coverage": 0.0,
            "calibration_valid": False,
            "reason": "pitch_map_not_found",
        }

    # Check calibration flag written by pitch_service
    if pitch_data.get("calibration_failed", False) or not pitch_data.get("calibration_valid", True):
        return {
            "has_ball_world_coords": False,
            "ball_frame_coverage": 0.0,
            "calibration_valid": False,
            "reason": "homography_calibration_failed",
        }

    # Check for ball entry (trackId == -1)
    ball_entries = [p for p in pitch_data.get("players", []) if p.get("trackId") == -1]
    total_frames = pitch_data.get("framesProcessed", 1) or 1

    if not ball_entries:
        return {
            "has_ball_world_coords": False,
            "ball_frame_coverage": 0.0,
            "calibration_valid": True,
            "reason": "ball_not_in_pitch_map",
        }

    coverage = len(ball_entries[0].get("trajectory2d", [])) / total_frames
    reason = "ok" if coverage > 0.1 else "low_coverage"
    return {
        "has_ball_world_coords": True,
        "ball_frame_coverage": round(coverage, 3),
        "calibration_valid": True,
        "reason": reason,
    }


def _ball_to_pitch(ball_trajectory, pitch_data):
    # type: (List[Dict], Optional[Dict]) -> Dict[int, Tuple[float, float]]
    """Convert ball pixel coords to world coords, return {frame: (x, y)}."""
    frame_w = 3840
    frame_h = 2160
    if pitch_data is not None:
        frame_w = pitch_data.get("frameWidth", 3840) or 3840
        frame_h = pitch_data.get("frameHeight", 2160) or 2160

    result = {}  # type: Dict[int, Tuple[float, float]]
    for b in ball_trajectory:
        fi = int(b["frameIndex"])
        px = float(b.get("x", 0.0))
        py = float(b.get("y", 0.0))
        result[fi] = (px / frame_w * PITCH_WIDTH, py / frame_h * PITCH_HEIGHT)
    return result


def compute_possession_frames(
    common_frames,      # type: List[int]
    ball_world,         # type: Dict[int, Tuple[float, float]]
    player_world,       # type: Dict[int, List[Dict[str, Any]]]
    team_map,           # type: Dict[int, int]
):
    # type: (...) -> Tuple[Dict[int, int], int]
    """
    Determine ball carrier per frame with 3-frame consecutive rule.

    Returns:
        carrier_by_frame: {frame: trackId} (only confirmed carriers)
        contested_count: number of contested frames
    """
    # First pass: find closest player within CARRIER_RADIUS_M at each frame
    raw_closest = {}  # type: Dict[int, int]
    for fi in common_frames:
        bx, by = ball_world[fi]
        players = player_world.get(fi, [])
        best_tid = -1
        best_dist = float("inf")
        for p in players:
            dx = p["x"] - bx
            dy = p["y"] - by
            d = math.sqrt(dx * dx + dy * dy)
            if d < CARRIER_RADIUS_M and d < best_dist:
                best_dist = d
                best_tid = p["trackId"]
        if best_tid >= 0:
            raw_closest[fi] = best_tid

    # Second pass: require 3 consecutive frames with same player
    carrier_by_frame = {}  # type: Dict[int, int]
    contested = 0
    streak_tid = -1
    streak_count = 0

    for fi in common_frames:
        tid = raw_closest.get(fi, -1)
        if tid < 0:
            contested += 1
            streak_tid = -1
            streak_count = 0
            continue

        if tid == streak_tid:
            streak_count += 1
        else:
            streak_tid = tid
            streak_count = 1

        if streak_count >= CARRIER_MIN_FRAMES:
            carrier_by_frame[fi] = tid
            # Backfill the frames that built up to the streak
            idx = common_frames.index(fi)
            for back in range(1, CARRIER_MIN_FRAMES):
                if idx - back >= 0:
                    bf = common_frames[idx - back]
                    if raw_closest.get(bf, -1) == tid:
                        carrier_by_frame[bf] = tid
        elif fi not in carrier_by_frame:
            contested += 1

    return carrier_by_frame, contested


def compute_pass_network(job_id):
    # type: (str) -> Dict[str, Any]
    """
    Build a pass network with improved possession detection.

    Reads:
        temp/{job_id}/tracking/track_results.json
        temp/{job_id}/tracking/team_results.json
        temp/{job_id}/pitch/pitch_map.json
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

    # Build per-frame player world positions from pitch_map
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

    # Check ball data quality before proceeding
    ball_quality = _check_ball_data_quality(job_id)
    if not ball_quality["has_ball_world_coords"] or ball_quality["ball_frame_coverage"] < 0.05:
        logger.warning(
            "Pass network unavailable for job %s: %s", job_id, ball_quality["reason"]
        )
        return {
            "status": "unavailable",
            "reason": ball_quality["reason"],
            "total_passes": None,
            "nodes": [],
            "edges": [],
            "passes_by_team": {"0": None, "1": None},
            "contested_frames": None,
            "message": (
                "Pass detection requires ball world coordinates. "
                "Run pitch mapping with homography first."
            ),
            "ball_data_quality": ball_quality,
        }

    # Ball world positions — read from pitch_map ball entry (trackId==-1)
    ball_trajectory = track_data.get("ball_trajectory", [])
    ball_world = {}  # type: Dict[int, Tuple[float, float]]

    # Prefer world coords from pitch_map (already homography-transformed)
    if pitch_data is not None:
        for player in pitch_data.get("players", []):
            if player.get("trackId") == -1:
                for pt in player.get("trajectory2d", []):
                    fi = int(pt["frameIndex"])
                    ball_world[fi] = (float(pt["x"]), float(pt["y"]))
                break

    # Fall back to pixel conversion only if no ball in pitch_map
    if not ball_world:
        ball_world = _ball_to_pitch(ball_trajectory, pitch_data)

    if not ball_world:
        return {
            "status": "unavailable",
            "reason": "ball_trajectory_empty",
            "total_passes": None,
            "nodes": [],
            "edges": [],
            "passes_by_team": {"0": None, "1": None},
            "contested_frames": None,
            "message": "No ball trajectory data found.",
            "ball_data_quality": ball_quality,
        }

    # Frames where we have both ball and player data
    common_frames = sorted(set(ball_world.keys()) & set(player_world.keys()))

    # Compute possession
    carrier_by_frame, contested_count = compute_possession_frames(
        common_frames, ball_world, player_world, team_map
    )

    # Detect pass events
    pass_events = []  # type: List[Dict[str, Any]]
    carrier_frames = sorted(carrier_by_frame.keys())

    if len(carrier_frames) >= 2:
        # Walk through carrier transitions
        i = 0
        while i < len(carrier_frames):
            curr_fi = carrier_frames[i]
            curr_tid = carrier_by_frame[curr_fi]

            # Find the end of this carrier's run
            run_end = i
            while run_end + 1 < len(carrier_frames) and carrier_by_frame[carrier_frames[run_end + 1]] == curr_tid:
                run_end += 1

            last_carrier_fi = carrier_frames[run_end]

            # Look for next different carrier
            if run_end + 1 < len(carrier_frames):
                next_fi = carrier_frames[run_end + 1]
                next_tid = carrier_by_frame[next_fi]

                if next_tid != curr_tid:
                    # Check pass conditions
                    frame_gap = next_fi - last_carrier_fi
                    same_team = (team_map.get(curr_tid, -1) == team_map.get(next_tid, -1)
                                 and team_map.get(curr_tid, -1) >= 0)

                    # Ball distance between last A-frame and first B-frame
                    from_pos = ball_world.get(last_carrier_fi, (0.0, 0.0))
                    to_pos = ball_world.get(next_fi, (0.0, 0.0))
                    ball_dist = math.sqrt(
                        (to_pos[0] - from_pos[0]) ** 2 + (to_pos[1] - from_pos[1]) ** 2
                    )

                    if (same_team
                            and ball_dist >= PASS_MIN_DIST_M
                            and frame_gap <= PASS_MAX_FRAMES):
                        # Get player world positions at pass endpoints
                        from_player_pos = [0.0, 0.0]
                        to_player_pos = [0.0, 0.0]
                        for p in player_world.get(last_carrier_fi, []):
                            if p["trackId"] == curr_tid:
                                from_player_pos = [p["x"], p["y"]]
                                break
                        for p in player_world.get(next_fi, []):
                            if p["trackId"] == next_tid:
                                to_player_pos = [p["x"], p["y"]]
                                break

                        pass_dist = math.sqrt(
                            (to_player_pos[0] - from_player_pos[0]) ** 2
                            + (to_player_pos[1] - from_player_pos[1]) ** 2
                        )

                        pass_events.append({
                            "from_track_id": curr_tid,
                            "to_track_id": next_tid,
                            "from_pos": from_player_pos,
                            "to_pos": to_player_pos,
                            "frame": next_fi,
                            "distance_m": pass_dist,
                            "x_advance": to_player_pos[0] - from_player_pos[0],
                        })

            i = run_end + 1

    # Build pass matrix
    all_track_ids = sorted(set(
        [e["from_track_id"] for e in pass_events]
        + [e["to_track_id"] for e in pass_events]
    ))
    id_to_idx = {}  # type: Dict[int, int]
    for idx, tid in enumerate(all_track_ids):
        id_to_idx[tid] = idx

    n = len(all_track_ids)
    pass_matrix = np.zeros((n, n), dtype=np.int32)
    dist_matrix = np.zeros((n, n), dtype=np.float64)
    advance_matrix = np.zeros((n, n), dtype=np.float64)

    for e in pass_events:
        i = id_to_idx[e["from_track_id"]]
        j = id_to_idx[e["to_track_id"]]
        pass_matrix[i, j] += 1
        dist_matrix[i, j] += e["distance_m"]
        advance_matrix[i, j] += e["x_advance"]

    # Per-player stats
    passes_made = {}  # type: Dict[int, int]
    passes_received = {}  # type: Dict[int, int]
    avg_pass_dist = {}  # type: Dict[int, float]
    avg_advance = {}  # type: Dict[int, float]

    for tid in all_track_ids:
        idx = id_to_idx[tid]
        made = int(pass_matrix[idx, :].sum())
        passes_made[tid] = made
        passes_received[tid] = int(pass_matrix[:, idx].sum())
        total_dist = float(dist_matrix[idx, :].sum())
        total_adv = float(advance_matrix[idx, :].sum())
        avg_pass_dist[tid] = total_dist / made if made > 0 else 0.0
        avg_advance[tid] = total_adv / made if made > 0 else 0.0

    # Key passers: top 25%
    if all_track_ids:
        made_values = [passes_made[tid] for tid in all_track_ids]
        key_threshold = float(np.percentile(made_values, 75))
    else:
        key_threshold = 0.0

    # Progressive passers: avg x-advance > 10m toward opponent goal
    # Team 0 attacks right (positive x = forward)
    # Team 1 attacks left (negative x = forward)
    progressive_set = set()  # type: set
    for tid in all_track_ids:
        if passes_made[tid] == 0:
            continue
        team = team_map.get(tid, -1)
        adv = avg_advance[tid]
        if team == 0 and adv > 10.0:
            progressive_set.add(tid)
        elif team == 1 and adv < -10.0:
            progressive_set.add(tid)

    # Average world position per player
    avg_positions = {}  # type: Dict[int, Dict[str, float]]
    if pitch_data is not None:
        for player in pitch_data.get("players", []):
            tid = player["trackId"]
            traj = player.get("trajectory2d", [])
            if traj:
                xs = [float(pt["x"]) for pt in traj]
                ys = [float(pt["y"]) for pt in traj]
                avg_positions[tid] = {
                    "x": float(np.mean(xs)),
                    "y": float(np.mean(ys)),
                }

    # Build nodes
    nodes = []  # type: List[Dict[str, Any]]
    for tid in all_track_ids:
        avg = avg_positions.get(tid, {"x": 0.0, "y": 0.0})
        nodes.append({
            "track_id": tid,
            "team": team_map.get(tid, -1),
            "avg_x": round(avg["x"], 2),
            "avg_y": round(avg["y"], 2),
            "total_passes_made": passes_made[tid],
            "total_passes_received": passes_received[tid],
            "is_key_passer": passes_made[tid] >= key_threshold and key_threshold > 0,
            "is_progressive_passer": tid in progressive_set,
            "avg_pass_distance_m": round(avg_pass_dist[tid], 2),
        })

    # Build edges
    edge_counts = []  # type: List[int]
    edges_raw = []  # type: List[Dict[str, Any]]
    for i in range(n):
        for j in range(n):
            count = int(pass_matrix[i, j])
            if count > 0:
                avg_d = float(dist_matrix[i, j]) / count
                edges_raw.append({
                    "from": all_track_ids[i],
                    "to": all_track_ids[j],
                    "count": count,
                    "avg_distance_m": round(avg_d, 2),
                })
                edge_counts.append(count)

    if edge_counts:
        edge_threshold = float(np.percentile(edge_counts, 75))
    else:
        edge_threshold = 0.0

    edges = []  # type: List[Dict[str, Any]]
    for e in edges_raw:
        edges.append({
            "from": e["from"],
            "to": e["to"],
            "count": e["count"],
            "avg_distance_m": e["avg_distance_m"],
            "is_frequent": e["count"] >= edge_threshold and edge_threshold > 0,
        })

    # Passes by team
    passes_by_team = {"0": 0, "1": 0}  # type: Dict[str, int]
    for e in pass_events:
        team_id = team_map.get(e["from_track_id"], -1)
        key = str(team_id)
        if key in passes_by_team:
            passes_by_team[key] += 1

    result = {
        "nodes": nodes,
        "edges": edges,
        "total_passes": len(pass_events),
        "contested_frames": contested_count,
        "passes_by_team": passes_by_team,
        "ball_data_quality": ball_quality,
        "detection_method": "ball_tracking",
    }

    logger.info(
        "Pass network: %d passes, %d nodes, %d edges, %d contested frames",
        len(pass_events), len(nodes), len(edges), contested_count,
    )

    return result


def detect_passes_without_ball(job_id):
    # type: (str) -> Dict[str, Any]
    """
    Infer passes from player motion when ball world coords are unavailable.

    A pass is inferred when:
    1. Player A decelerates sharply (speed drops >60% over 5 frames)
    2. Player B on same team accelerates sharply (speed increases >60% over 5 frames)
    3. A and B are within 30m of each other
    4. B's acceleration direction points toward A's last position
    5. A and B transitions happen within 10 frames of each other

    All results are flagged confidence: "inferred_no_ball".
    """
    import math as _math

    base = _base_job_id(job_id)
    candidates = [job_id]
    if base != job_id:
        candidates.append(base)

    pitch_data = None
    for jid in candidates:
        pitch_data = _load_json(Path("temp/{}/pitch/pitch_map.json".format(jid)))
        if pitch_data is not None:
            break

    team_map = {}  # type: Dict[int, int]
    for jid in candidates:
        td = _load_json(Path("temp/{}/tracking/team_results.json".format(jid)))
        if td is not None:
            tracks_list = td.get("tracks", td) if isinstance(td, dict) else td
            for t in tracks_list:
                team_map[t["trackId"]] = t.get("teamId", -1)
            break

    track_data = None
    for jid in candidates:
        track_data = _load_json(Path("temp/{}/tracking/track_results.json".format(jid)))
        if track_data is not None:
            break

    if pitch_data is None or track_data is None:
        return {
            "status": "unavailable",
            "reason": "missing_input_files",
            "total_passes": None,
            "nodes": [],
            "edges": [],
            "passes_by_team": {"0": None, "1": None},
            "detection_method": "player_motion_inference",
        }

    fps = float(
        track_data.get("fps") or
        track_data.get("metadata", {}).get("fps") or
        25.0
    )
    if fps <= 0:
        fps = 25.0

    # Build per-player position series: {tid: [(frame, x, y), ...]}
    MAX_DISP = 3.0  # m/frame — same teleport guard as heatmap_service
    player_series = {}  # type: Dict[int, List[Tuple[int, float, float]]]
    for player in pitch_data.get("players", []):
        tid = player["trackId"]
        if tid == -1:
            continue
        traj = sorted(player.get("trajectory2d", []), key=lambda p: p["frameIndex"])
        player_series[tid] = [(int(pt["frameIndex"]), float(pt["x"]), float(pt["y"]))
                              for pt in traj]

    # Compute per-player speed at each frame
    player_speeds = {}  # type: Dict[int, Dict[int, float]]
    for tid, series in player_series.items():
        spd = {}
        for i in range(1, len(series)):
            fi_prev, xp, yp = series[i - 1]
            fi_curr, xc, yc = series[i]
            gap = fi_curr - fi_prev
            if gap <= 0 or gap > 10:
                continue
            dist = _math.sqrt((xc - xp) ** 2 + (yc - yp) ** 2)
            if dist / gap > MAX_DISP:
                continue
            spd[fi_curr] = (dist / gap) * fps
        player_speeds[tid] = spd

    # Detect deceleration events (speed drops >60% in 5 frames) per player
    WINDOW = 5
    THRESHOLD = 0.60

    def _get_avg_speed(speeds_dict, fi, window):
        # type: (Dict[int, float], int, int) -> float
        vals = [speeds_dict[f] for f in range(fi - window, fi) if f in speeds_dict]
        return sum(vals) / len(vals) if vals else 0.0

    decel_events = []  # type: List[Tuple[int, int, float, float]]  # (frame, tid, x, y)
    accel_events = []  # type: List[Tuple[int, int, float, float]]

    all_frames = sorted({fi for series in player_series.values() for fi, _, _ in series})

    pos_at = {}  # type: Dict[Tuple[int, int], Tuple[float, float]]  # (tid, frame) -> (x, y)
    for tid, series in player_series.items():
        for fi, x, y in series:
            pos_at[(tid, fi)] = (x, y)

    for tid, spd in player_speeds.items():
        frames = sorted(spd.keys())
        for i, fi in enumerate(frames):
            before_avg = _get_avg_speed(spd, fi, WINDOW)
            after_avg_vals = [spd[f] for f in frames if fi <= f < fi + WINDOW and f in spd]
            after_avg = sum(after_avg_vals) / len(after_avg_vals) if after_avg_vals else 0.0

            if before_avg > 1.0 and after_avg < before_avg * (1 - THRESHOLD):
                xy = pos_at.get((tid, fi), (0.0, 0.0))
                decel_events.append((fi, tid, xy[0], xy[1]))

            if after_avg > 1.0 and before_avg < after_avg * (1 - THRESHOLD):
                xy = pos_at.get((tid, fi), (0.0, 0.0))
                accel_events.append((fi, tid, xy[0], xy[1]))

    # Match decel (passer A) with accel (receiver B)
    inferred_passes = []  # type: List[Dict[str, Any]]
    FRAME_TOLERANCE = 10
    MAX_PASS_DIST_M = 30.0

    for d_fi, d_tid, d_x, d_y in decel_events:
        d_team = team_map.get(d_tid, -1)
        if d_team < 0:
            continue
        for a_fi, a_tid, a_x, a_y in accel_events:
            if a_tid == d_tid:
                continue
            if team_map.get(a_tid, -1) != d_team:
                continue
            if not (0 <= a_fi - d_fi <= FRAME_TOLERANCE):
                continue
            dist = _math.sqrt((a_x - d_x) ** 2 + (a_y - d_y) ** 2)
            if dist > MAX_PASS_DIST_M:
                continue
            # Direction check: B's acceleration direction roughly toward A
            # (simplified: just accept within distance)
            inferred_passes.append({
                "from_track_id": d_tid,
                "to_track_id": a_tid,
                "frame": a_fi,
                "distance_m": round(dist, 2),
                "from_pos": [round(d_x, 2), round(d_y, 2)],
                "to_pos": [round(a_x, 2), round(a_y, 2)],
                "detection_method": "player_motion_inference",
                "confidence": "low",
                "note": (
                    "Ball tracking unavailable. Pass inferred from "
                    "player acceleration patterns."
                ),
            })

    # Deduplicate (same passer within 10 frames)
    seen = {}  # type: Dict[int, int]
    deduped = []
    for p in inferred_passes:
        last = seen.get(p["from_track_id"], -FRAME_TOLERANCE - 1)
        if p["frame"] - last >= FRAME_TOLERANCE:
            deduped.append(p)
            seen[p["from_track_id"]] = p["frame"]

    passes_by_team = {"0": 0, "1": 0}
    for p in deduped:
        k = str(team_map.get(p["from_track_id"], -1))
        if k in passes_by_team:
            passes_by_team[k] += 1

    logger.info(
        "Player-motion pass inference for job %s: %d inferred passes",
        job_id, len(deduped)
    )

    return {
        "status": "inferred",
        "total_passes": len(deduped),
        "passes": deduped,
        "passes_by_team": passes_by_team,
        "nodes": [],
        "edges": [],
        "detection_method": "player_motion_inference",
        "confidence": "low",
        "note": (
            "Ball tracking unavailable. Passes inferred from "
            "player acceleration patterns."
        ),
    }
