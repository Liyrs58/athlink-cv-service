import json
import logging
import math
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Constants
PASS_MIN_DIST_M = 2.0
PASS_MAX_FRAMES = 60
SHOT_SPEED_MS = 8.0
SHOT_FINAL_THIRD_X = 68.0
DRIBBLE_RADIUS_M = 2.0
DRIBBLE_MIN_FRAMES = 8
DRIBBLE_OPPONENT_DIST = 3.5
TURNOVER_GAP_FRAMES = 45

# Pitch dimensions
PITCH_WIDTH = 105.0
PITCH_HEIGHT = 68.0

# Import helper functions from existing services
from services.pass_network_service import _base_job_id, _load_json, _check_ball_data_quality
try:
    from services.xg_service import compute_xg
    XG_AVAILABLE = True
except ImportError:
    XG_AVAILABLE = False
    logger.warning("xg_service not available, xG values will be null")

# Local copy of interpolation logic if xg_service not available
def _build_continuous_ball_track(ball_trajectory, pitch_data):
    """Build ball track from world coords or convert pixel coords."""
    ball_coords_are_pixels = False
    
    # Step A: Try to get ball world coords from pitch_map.json
    if pitch_data and "players" in pitch_data:
        ball_world_coords = {}
        for player in pitch_data["players"]:
            # Look for ball entries (track_id == -1 or class_id == 32)
            if player.get("trackId", -99) == -1 or player.get("class_id", 0) == 32:
                for pt in player.get("trajectory2d", []):
                    fi = int(pt["frameIndex"])
                    # Check if world_x/world_y exist
                    if "world_x" in pt and "world_y" in pt:
                        ball_world_coords[fi] = (pt["world_x"], pt["world_y"])
                    elif "x" in pt and "y" in pt:
                        # Assume x/y are world coords in this context
                        ball_world_coords[fi] = (pt["x"], pt["y"])
        
        if ball_world_coords:
            # Use world coords from pitch_map
            raw = [(fi, wx, wy) for fi, (wx, wy) in ball_world_coords.items()]
            raw.sort(key=lambda t: t[0])
        else:
            # Step B: Fall back to pixel conversion
            ball_coords_are_pixels = True
            frame_w = 3840
            frame_h = 2160
            if pitch_data is not None:
                frame_w = pitch_data.get("frameWidth", 3840) or 3840
                frame_h = pitch_data.get("frameHeight", 2160) or 2160

            # Convert pixel coords to world coords
            raw = []
            for b in ball_trajectory:
                fi = int(b["frameIndex"])
                px = float(b.get("x", 0.0))
                py = float(b.get("y", 0.0))
                wx = px / frame_w * PITCH_WIDTH
                wy = py / frame_h * PITCH_HEIGHT
                raw.append((fi, wx, wy))
            raw.sort(key=lambda t: t[0])
    else:
        # No pitch data, use pixel conversion
        ball_coords_are_pixels = True
        frame_w = 3840
        frame_h = 2160
        if pitch_data is not None:
            frame_w = pitch_data.get("frameWidth", 3840) or 3840
            frame_h = pitch_data.get("frameHeight", 2160) or 2160

        # Convert pixel coords to world coords
        raw = []
        for b in ball_trajectory:
            fi = int(b["frameIndex"])
            px = float(b.get("x", 0.0))
            py = float(b.get("y", 0.0))
            wx = px / frame_w * PITCH_WIDTH
            wy = py / frame_h * PITCH_HEIGHT
            raw.append((fi, wx, wy))
        raw.sort(key=lambda t: t[0])

    if not raw:
        return {}, set(), 0, 0, ball_coords_are_pixels

    positions = {}
    interpolated_frames = set()
    ball_lost = set()

    for fi, wx, wy in raw:
        positions[fi] = (wx, wy)

    # Interpolate small gaps (1-5 frames)
    interpolated_count = 0
    lost_count = 0
    MAX_INTERP_GAP = 5

    for i in range(len(raw) - 1):
        fi_a, xa, ya = raw[i]
        fi_b, xb, yb = raw[i + 1]
        gap = fi_b - fi_a

        if gap <= 1:
            continue

        missing = gap - 1

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

    return positions, ball_lost, interpolated_count, lost_count, ball_coords_are_pixels

def _smooth_and_compute_speed(positions, ball_lost, fps):
    """Local copy of smoothing and speed calculation from xg_service."""
    frames = sorted(f for f in positions if f not in ball_lost)
    if len(frames) < 3:
        return {}, {}, {}

    # Build arrays
    n = len(frames)
    xs = np.array([positions[f][0] for f in frames], dtype=np.float64)
    ys = np.array([positions[f][1] for f in frames], dtype=np.float64)

    # 5-frame rolling average
    SMOOTH_WINDOW = 5
    half_w = SMOOTH_WINDOW // 2
    sx = np.copy(xs)
    sy = np.copy(ys)
    for i in range(n):
        lo = max(0, i - half_w)
        hi = min(n, i + half_w + 1)
        sx[i] = np.mean(xs[lo:hi])
        sy[i] = np.mean(ys[lo:hi])

    smoothed_x = {}
    smoothed_y = {}
    for i, fi in enumerate(frames):
        smoothed_x[fi] = float(sx[i])
        smoothed_y[fi] = float(sy[i])

    # Central difference for speed
    speeds = {}
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
        speeds[fi] = speed

    return smoothed_x, smoothed_y, speeds

def load_event_inputs(job_id: str) -> dict:
    """Load required input files for event detection."""
    base = _base_job_id(job_id)
    candidates = [job_id]
    if base != job_id:
        candidates.append(base)

    # Load track_results.json
    track_data = None
    for jid in candidates:
        track_data = _load_json(Path(f"temp/{jid}/tracking/track_results.json"))
        if track_data is not None:
            break
    if track_data is None:
        raise FileNotFoundError(f"track_results.json not found for job '{job_id}'")

    # Load team_results.json
    team_data = None
    for jid in candidates:
        team_data = _load_json(Path(f"temp/{jid}/tracking/team_results.json"))
        if team_data is not None:
            break
    if team_data is None:
        raise FileNotFoundError(f"team_results.json not found for job '{job_id}'")

    # Load pitch_map.json
    pitch_data = None
    for jid in candidates:
        pitch_data = _load_json(Path(f"temp/{jid}/pitch/pitch_map.json"))
        if pitch_data is not None:
            break
    if pitch_data is None:
        raise FileNotFoundError(f"pitch_map.json not found for job '{job_id}'")

    return {
        "track_data": track_data,
        "team_data": team_data,
        "pitch_data": pitch_data
    }

def build_ball_track(pitch_map_data: dict) -> dict:
    """Build continuous ball track with interpolation and speed calculation."""
    # Extract ball trajectory from track_data (passed separately)
    # This function expects pitch_map_data to contain the ball trajectory
    # In practice, we'll get ball_trajectory from the track_data in detect_events
    
    # Return empty dict for now - will be populated in detect_events
    return {}

def build_possession_sequences(ball_track: dict, player_data: dict, team_data: dict, fps: float) -> list:
    """Build possession sequences from ball and player data."""
    sequences = []
    
    # Build team mapping
    team_map = {}
    tracks_list = team_data.get("tracks", team_data) if isinstance(team_data, dict) else team_data
    for t in tracks_list:
        team_map[t["trackId"]] = t.get("teamId", -1)
    
    # Build player world positions
    player_world = {}
    if player_data is not None:
        for player in player_data.get("players", []):
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
    
    # Get common frames
    ball_frames = set(ball_track.keys())
    player_frames = set(player_world.keys())
    common_frames = sorted(ball_frames & player_frames)
    
    if not common_frames:
        return sequences
    
    # Find ball carriers (within 3.0m for 3 consecutive frames)
    CARRIER_RADIUS_M = 3.0
    CARRIER_MIN_FRAMES = 3
    
    carrier_by_frame = {}
    streak_tid = -1
    streak_count = 0
    
    for fi in common_frames:
        bx, by = ball_track[fi]["x"], ball_track[fi]["y"]
        players = player_world.get(fi, [])
        
        # Find closest player within radius
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
            if best_tid == streak_tid:
                streak_count += 1
            else:
                streak_tid = best_tid
                streak_count = 1
            
            if streak_count >= CARRIER_MIN_FRAMES:
                carrier_by_frame[fi] = best_tid
        else:
            streak_tid = -1
            streak_count = 0
    
    # Build sequences from carrier frames
    carrier_frames = sorted(carrier_by_frame.keys())
    if not carrier_frames:
        return sequences
    
    i = 0
    while i < len(carrier_frames):
        curr_fi = carrier_frames[i]
        curr_tid = carrier_by_frame[curr_fi]
        
        # Find end of this carrier's run
        run_end = i
        while run_end + 1 < len(carrier_frames) and carrier_by_frame[carrier_frames[run_end + 1]] == curr_tid:
            run_end += 1
        
        start_fi = carrier_frames[i]
        end_fi = carrier_frames[run_end]
        
        # Get positions
        start_pos = ball_track[start_fi]
        end_pos = ball_track[end_fi]
        
        sequences.append({
            "track_id": curr_tid,
            "team": team_map.get(curr_tid, -1),
            "start_frame": start_fi,
            "end_frame": end_fi,
            "start_x": start_pos["x"],
            "start_y": start_pos["y"],
            "end_x": end_pos["x"],
            "end_y": end_pos["y"],
            "duration_frames": end_fi - start_fi + 1
        })
        
        i = run_end + 1
    
    return sequences

def detect_events(job_id: str) -> dict:
    """Main event detection function."""
    # Load inputs
    inputs = load_event_inputs(job_id)
    track_data = inputs["track_data"]
    team_data = inputs["team_data"]
    pitch_data = inputs["pitch_data"]
    
    # Get FPS
    fps = (
        track_data.get('fps') or
        track_data.get('metadata', {}).get('fps') or
        25.0
    )
    fps = float(fps) if fps else 25.0
    
    # Build ball track
    ball_trajectory = track_data.get("ball_trajectory", [])
    if not ball_trajectory:
        logger.warning("No ball trajectory found")
        return {"events": [], "summary": {}, "ball_track_frames": 0, "interpolated_frames": 0, "possession_sequences": 0}
    
    # Check ball data quality before running distance-based detection
    ball_quality = _check_ball_data_quality(job_id)
    if not ball_quality["has_ball_world_coords"] or ball_quality["ball_frame_coverage"] < 0.05:
        logger.warning(
            "Event detection unavailable for job %s: %s", job_id, ball_quality["reason"]
        )
        empty_summary = {
            "team_0": {"pass_count": 0, "shot_count": 0, "dribble_count": 0,
                       "dribble_success_rate": None, "turnover_count": 0,
                       "clearance_count": 0, "forward_carry_distance_m": 0.0},
            "team_1": {"pass_count": 0, "shot_count": 0, "dribble_count": 0,
                       "dribble_success_rate": None, "turnover_count": 0,
                       "clearance_count": 0, "forward_carry_distance_m": 0.0},
            "total_events": 0,
            "duration_seconds": 0.0,
        }
        result = {
            "status": "unavailable",
            "reason": ball_quality["reason"],
            "message": (
                "Event detection requires ball world coordinates. "
                "Run pitch mapping with homography first."
            ),
            "events": [],
            "summary": empty_summary,
            "ball_track_frames": 0,
            "interpolated_frames": 0,
            "possession_sequences": 0,
            "ball_data_quality": ball_quality,
        }
        output_dir = Path(f"temp/{job_id}/events")
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "event_timeline.json", "w") as f:
            json.dump(result, f, indent=2)
        return result

    positions, ball_lost, interpolated_count, lost_count, ball_coords_are_pixels = _build_continuous_ball_track(ball_trajectory, pitch_data)

    # Secondary check: if build_ball_track still ended up in pixel mode, refuse
    if ball_coords_are_pixels:
        return {
            "status": "unavailable",
            "reason": "ball_coords_are_pixels",
            "message": "Ball coordinates are pixel-space, not world-space. Homography required.",
            "events": [],
            "summary": {},
            "ball_track_frames": len(positions),
            "interpolated_frames": interpolated_count,
            "possession_sequences": 0,
        }
    smoothed_x, smoothed_y, speeds = _smooth_and_compute_speed(positions, ball_lost, fps)
    
    # Build ball track dict
    ball_track = {}
    for fi in positions:
        if fi not in ball_lost:
            ball_track[fi] = {
                "x": smoothed_x.get(fi, positions[fi][0]),
                "y": smoothed_y.get(fi, positions[fi][1]),
                "speed_ms": speeds.get(fi, 0.0),
                "interpolated": fi in set(f for f in positions if f not in ball_lost and f not in set(int(b["frameIndex"]) for b in ball_trajectory))
            }
    
    # Build possession sequences
    possession_sequences = build_possession_sequences(ball_track, pitch_data, team_data, fps)
    
    # Initialize events list
    events = []
    
    # PASS DETECTION
    for i in range(len(possession_sequences) - 1):
        seq_a = possession_sequences[i]
        seq_b = possession_sequences[i + 1]
        
        # Check if same team
        if seq_a["team"] != seq_b["team"] or seq_a["team"] < 0:
            continue
        
        gap_frames = seq_b["start_frame"] - seq_a["end_frame"]
        ball_dist = math.sqrt((seq_b["start_x"] - seq_a["end_x"])**2 + (seq_b["start_y"] - seq_a["end_y"])**2)
        
        if gap_frames <= PASS_MAX_FRAMES and ball_dist >= PASS_MIN_DIST_M:
            # Determine pass type
            sub_type = "simple"
            x_advance = seq_b["start_x"] - seq_a["end_x"]
            
            # Progressive pass
            if seq_a["team"] == 0 and x_advance > 10.0:  # Team 0 attacks right
                sub_type = "progressive"
            elif seq_a["team"] == 1 and x_advance < -10.0:  # Team 1 attacks left
                sub_type = "progressive"
            
            # Back pass
            if seq_a["team"] == 0 and x_advance < -5.0:
                sub_type = "back_pass"
            elif seq_a["team"] == 1 and x_advance > 5.0:
                sub_type = "back_pass"
            
            # Cross
            if ball_dist > 25.0 and abs(seq_b["start_y"] - seq_a["end_y"]) > 15.0:
                sub_type = "cross"
            
            events.append({
                "type": "pass",
                "frame": seq_b["start_frame"],
                "time_seconds": seq_b["start_frame"] / fps,
                "team": seq_a["team"],
                "player_track_id": seq_a["track_id"],
                "x": seq_a["end_x"],
                "y": seq_a["end_y"],
                "to_player_track_id": seq_b["track_id"],
                "xg": None,
                "distance_m": ball_dist,
                "sub_type": sub_type,
                "success": True
            })
    
    # SHOT DETECTION
    shot_candidates = []
    for fi, speed in speeds.items():
        if speed > SHOT_SPEED_MS:
            bx = smoothed_x.get(fi, 0.0)
            # Must be in final third
            if bx > SHOT_FINAL_THIRD_X or bx < (PITCH_WIDTH - SHOT_FINAL_THIRD_X):
                # Pre-shot speed check
                pre_slow = False
                for pf in range(max(0, fi - 10), fi):
                    if speeds.get(pf, 999.0) < 4.0:
                        pre_slow = True
                        break
                
                if pre_slow:
                    shot_candidates.append(fi)
    
    # Get xG values if available
    xg_results = {}
    if XG_AVAILABLE:
        try:
            xg_results = compute_xg(job_id)
        except Exception as e:
            logger.warning(f"Failed to compute xG: {e}")
    
    # Create shot events
    for fi in shot_candidates:
        bx = smoothed_x.get(fi, 0.0)
        by = smoothed_y.get(fi, 0.0)
        
        # Find team (nearest player)
        team = -1
        player_tid = -1
        team_map = {}
        tracks_list = team_data.get("tracks", team_data) if isinstance(team_data, dict) else team_data
        for t in tracks_list:
            team_map[t["trackId"]] = t.get("teamId", -1)
        
        if pitch_data:
            for p in pitch_data.get("players", []):
                for pt in p.get("trajectory2d", []):
                    if int(pt["frameIndex"]) == fi:
                        dx = pt.get("x", 0.0) - bx
                        dy = pt.get("y", 0.0) - by
                        d = math.sqrt(dx * dx + dy * dy)
                        if d < 5.0:  # Within 5m
                            team = team_map.get(p["trackId"], -1)
                            player_tid = p["trackId"]
                            break
                if team >= 0:
                    break
        
        # Get xG value
        xg = None
        if xg_results and "shots" in xg_results:
            for shot in xg_results["shots"]:
                if shot["frame"] == fi:
                    xg = shot.get("xg")
                    break
        
        events.append({
            "type": "shot",
            "frame": fi,
            "time_seconds": fi / fps,
            "team": team,
            "player_track_id": player_tid,
            "x": bx,
            "y": by,
            "to_player_track_id": None,
            "xg": xg,
            "distance_m": None,
            "sub_type": None,
            "success": None
        })
    
    # DRIBBLE DETECTION
    for seq in possession_sequences:
        if seq["duration_frames"] < DRIBBLE_MIN_FRAMES:
            continue
        
        # Check if player moved > 3m
        player_movement = math.sqrt((seq["end_x"] - seq["start_x"])**2 + (seq["end_y"] - seq["start_y"])**2)
        if player_movement < 3.0:
            continue
        
        # Check for opponent proximity during sequence
        has_opponent = False
        opponent_frame = None
        team_map = {}
        tracks_list = team_data.get("tracks", team_data) if isinstance(team_data, dict) else team_data
        for t in tracks_list:
            team_map[t["trackId"]] = t.get("teamId", -1)
        
        if pitch_data:
            for fi in range(seq["start_frame"], seq["end_frame"] + 1):
                player_x, player_y = None, None
                
                # Get player position at this frame
                for p in pitch_data.get("players", []):
                    if p["trackId"] == seq["track_id"]:
                        for pt in p.get("trajectory2d", []):
                            if int(pt["frameIndex"]) == fi:
                                player_x, player_y = pt.get("x", 0.0), pt.get("y", 0.0)
                                break
                        if player_x is not None:
                            break
                
                if player_x is not None:
                    # Check for opponents
                    for p in pitch_data.get("players", []):
                        if p["trackId"] != seq["track_id"] and team_map.get(p["trackId"], -1) != seq["team"]:
                            for pt in p.get("trajectory2d", []):
                                if int(pt["frameIndex"]) == fi:
                                    dx = pt.get("x", 0.0) - player_x
                                    dy = pt.get("y", 0.0) - player_y
                                    d = math.sqrt(dx * dx + dy * dy)
                                    if d < DRIBBLE_OPPONENT_DIST:
                                        has_opponent = True
                                        opponent_frame = fi
                                        break
                            if has_opponent:
                                break
                if has_opponent:
                    break
        
        if has_opponent:
            # Determine success/failure
            success = True
            if opponent_frame is not None:
                # Check if possession changes within 10 frames
                for fi in range(opponent_frame, min(opponent_frame + 10, seq["end_frame"] + 1)):
                    found_opponent_takeover = False
                    if pitch_data:
                        for p in pitch_data.get("players", []):
                            if p["trackId"] != seq["track_id"] and team_map.get(p["trackId"], -1) != seq["team"]:
                                for pt in p.get("trajectory2d", []):
                                    if int(pt["frameIndex"]) == fi:
                                        bx, by = ball_track.get(fi, {"x": 0, "y": 0})["x"], ball_track.get(fi, {"x": 0, "y": 0})["y"]
                                        dx = pt.get("x", 0.0) - bx
                                        dy = pt.get("y", 0.0) - by
                                        d = math.sqrt(dx * dx + dy * dy)
                                        if d < 3.0:  # Opponent gets ball
                                            found_opponent_takeover = True
                                            break
                                if found_opponent_takeover:
                                    break
                    if found_opponent_takeover:
                        success = False
                        break
            
            events.append({
                "type": "dribble",
                "frame": seq["start_frame"],
                "time_seconds": seq["start_frame"] / fps,
                "team": seq["team"],
                "player_track_id": seq["track_id"],
                "x": seq["start_x"],
                "y": seq["start_y"],
                "to_player_track_id": None,
                "xg": None,
                "distance_m": player_movement,
                "sub_type": None,
                "success": success
            })
    
    # TURNOVER DETECTION
    for i in range(len(possession_sequences) - 1):
        seq_a = possession_sequences[i]
        seq_b = possession_sequences[i + 1]
        
        # Check if team changes
        if seq_a["team"] == seq_b["team"] or seq_a["team"] < 0 or seq_b["team"] < 0:
            continue
        
        gap_frames = seq_b["start_frame"] - seq_a["end_frame"]
        
        if gap_frames <= TURNOVER_GAP_FRAMES:
            # Check if this was a pass (already recorded)
            is_pass = False
            for event in events:
                if event["type"] == "pass" and event["frame"] == seq_b["start_frame"]:
                    is_pass = True
                    break
            
            if not is_pass:
                # Determine turnover type
                sub_type = "turnover"
                ball_speed = ball_track.get(seq_a["end_frame"], {}).get("speed_ms", 0.0)
                
                # Clearance: high speed and in own half
                if ball_speed > 6.0:
                    if (seq_a["team"] == 0 and seq_a["end_x"] < PITCH_WIDTH / 2) or \
                       (seq_a["team"] == 1 and seq_a["end_x"] > PITCH_WIDTH / 2):
                        sub_type = "clearance"
                    else:
                        sub_type = "ball_recovery"
                else:
                    sub_type = "ball_recovery"
                
                events.append({
                    "type": "turnover",
                    "frame": seq_b["start_frame"],
                    "time_seconds": seq_b["start_frame"] / fps,
                    "team": seq_b["team"],  # Team that gained possession
                    "player_track_id": seq_b["track_id"],
                    "x": seq_b["start_x"],
                    "y": seq_b["start_y"],
                    "to_player_track_id": None,
                    "xg": None,
                    "distance_m": None,
                    "sub_type": sub_type,
                    "success": True
                })
    
    # CARRY DETECTION
    for seq in possession_sequences:
        if seq["duration_frames"] < 5:  # Need at least 5 frames
            continue
        
        player_movement = math.sqrt((seq["end_x"] - seq["start_x"])**2 + (seq["end_y"] - seq["start_y"])**2)
        
        if player_movement > 5.0:
            # Determine direction
            x_advance = seq["end_x"] - seq["start_x"]
            sub_type = "forward_carry"
            
            if seq["team"] == 0:  # Team 0 attacks right
                if x_advance < -2.0:
                    sub_type = "backward_carry"
                elif abs(x_advance) <= 2.0:
                    sub_type = "lateral_carry"
            else:  # Team 1 attacks left
                if x_advance > 2.0:
                    sub_type = "backward_carry"
                elif abs(x_advance) <= 2.0:
                    sub_type = "lateral_carry"
            
            events.append({
                "type": "carry",
                "frame": seq["start_frame"],
                "time_seconds": seq["start_frame"] / fps,
                "team": seq["team"],
                "player_track_id": seq["track_id"],
                "x": seq["start_x"],
                "y": seq["start_y"],
                "to_player_track_id": None,
                "xg": None,
                "distance_m": player_movement,
                "sub_type": sub_type,
                "success": True
            })
    
    # Sort events by frame
    events.sort(key=lambda e: e["frame"])
    
    # Build summary stats
    summary = {
        "team_0": {"pass_count": 0, "shot_count": 0, "dribble_count": 0, "dribble_success_rate": 0.0, 
                  "turnover_count": 0, "clearance_count": 0, "forward_carry_distance_m": 0.0},
        "team_1": {"pass_count": 0, "shot_count": 0, "dribble_count": 0, "dribble_success_rate": 0.0,
                  "turnover_count": 0, "clearance_count": 0, "forward_carry_distance_m": 0.0},
        "total_events": len(events),
        "duration_seconds": max([e["time_seconds"] for e in events]) if events else 0.0
    }
    
    # Calculate team stats
    team_0_dribbles = {"total": 0, "success": 0}
    team_1_dribbles = {"total": 0, "success": 0}
    
    for event in events:
        team_key = f"team_{event['team']}"
        if team_key in summary:
            if event["type"] == "pass":
                summary[team_key]["pass_count"] += 1
            elif event["type"] == "shot":
                summary[team_key]["shot_count"] += 1
            elif event["type"] == "dribble":
                if event["team"] == 0:
                    team_0_dribbles["total"] += 1
                    if event["success"]:
                        team_0_dribbles["success"] += 1
                else:
                    team_1_dribbles["total"] += 1
                    if event["success"]:
                        team_1_dribbles["success"] += 1
                summary[team_key]["dribble_count"] += 1
            elif event["type"] == "turnover":
                summary[team_key]["turnover_count"] += 1
                if event["sub_type"] == "clearance":
                    summary[team_key]["clearance_count"] += 1
            elif event["type"] == "carry" and event["sub_type"] == "forward_carry":
                summary[team_key]["forward_carry_distance_m"] += event["distance_m"] or 0.0
    
    # Calculate dribble success rates
    if team_0_dribbles["total"] > 0:
        summary["team_0"]["dribble_success_rate"] = team_0_dribbles["success"] / team_0_dribbles["total"]
    if team_1_dribbles["total"] > 0:
        summary["team_1"]["dribble_success_rate"] = team_1_dribbles["success"] / team_1_dribbles["total"]
    
    # Save results
    output_dir = Path(f"temp/{job_id}/events")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result = {
        "events": events,
        "summary": summary,
        "ball_track_frames": len(ball_track),
        "interpolated_frames": interpolated_count,
        "possession_sequences": len(possession_sequences)
    }
    
    with open(output_dir / "event_timeline.json", "w") as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Event detection complete: {len(events)} events, {len(possession_sequences)} possession sequences")
    
    return result
