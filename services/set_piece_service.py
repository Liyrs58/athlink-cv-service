"""Set piece detection: corners, throw-ins, free kicks.
"""

import json
import logging
import math
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Constants
CORNER_ZONE_RADIUS_M = 5.0
FREE_KICK_CLUSTER_DIST_M = 3.0
WALL_MIN_PLAYERS = 2
STATIC_BALL_FRAMES = 20
STATIC_BALL_SPEED_MS = 0.3
THROW_IN_SIDELINE_DIST_M = 2.0
PITCH_WIDTH = 105.0
PITCH_HEIGHT = 68.0

# Corner zones
CORNER_ZONES = [
    (0.0, 0.0),      # Bottom left
    (0.0, PITCH_HEIGHT),  # Top left
    (PITCH_WIDTH, 0.0),   # Bottom right
    (PITCH_WIDTH, PITCH_HEIGHT)  # Top right
]

# Goal areas
GOAL_AREA_LEFT = (18.0, 25.0, 45.0)  # x_start, y_start, y_end
GOAL_AREA_RIGHT = (87.0, 25.0, 105.0)  # x_start, y_start, x_end

# Penalty spot areas
PENALTY_AREA_LEFT = (9.0, 13.0, 30.0, 38.0)  # x_start, x_end, y_start, y_end
PENALTY_AREA_RIGHT = (92.0, 96.0, 30.0, 38.0)  # x_start, x_end, y_start, y_end

# Centre circle
CENTRE_CIRCLE = {
    "center": (52.5, 34.0),
    "radius": 9.15
}

# Import helper functions
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

def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def _is_near_corner(ball_x: float, ball_y: float) -> bool:
    """Check if ball is near any corner flag."""
    for corner_x, corner_y in CORNER_ZONES:
        if _distance((ball_x, ball_y), (corner_x, corner_y)) <= CORNER_ZONE_RADIUS_M:
            return True
    return False

def _is_in_goal_area(ball_x: float, ball_y: float) -> bool:
    """Check if ball is in either goal area."""
    # Left goal area
    if (GOAL_AREA_LEFT[0] <= ball_x <= GOAL_AREA_LEFT[1] and 
        GOAL_AREA_LEFT[2] <= ball_y <= GOAL_AREA_LEFT[3]):
        return True
    # Right goal area
    if (GOAL_AREA_RIGHT[0] <= ball_x <= GOAL_AREA_RIGHT[2] and 
        GOAL_AREA_RIGHT[1] <= ball_y <= GOAL_AREA_RIGHT[1]):
        return True
    return False

def _is_in_penalty_area(ball_x: float, ball_y: float) -> bool:
    """Check if ball is in penalty spot area."""
    # Left penalty area
    if (PENALTY_AREA_LEFT[0] <= ball_x <= PENALTY_AREA_LEFT[1] and 
        PENALTY_AREA_LEFT[2] <= ball_y <= PENALTY_AREA_LEFT[3]):
        return True
    # Right penalty area
    if (PENALTY_AREA_RIGHT[0] <= ball_x <= PENALTY_AREA_RIGHT[1] and 
        PENALTY_AREA_RIGHT[2] <= ball_y <= PENALTY_AREA_RIGHT[3]):
        return True
    return False

def _is_near_sideline(ball_y: float) -> bool:
    """Check if ball is near either sideline."""
    return ball_y < THROW_IN_SIDELINE_DIST_M or ball_y > (PITCH_HEIGHT - THROW_IN_SIDELINE_DIST_M)

def _is_near_centre_circle(ball_x: float, ball_y: float) -> bool:
    """Check if ball is near centre circle."""
    return _distance((ball_x, ball_y), CENTRE_CIRCLE["center"]) <= CENTRE_CIRCLE["radius"]

def _detect_defensive_wall(
    ball_x: float, ball_y: float, 
    defending_team: int, player_positions: List[Dict], team_map: Dict
) -> int:
    """Count defending players forming a wall near the ball."""
    wall_players = 0
    
    for player in player_positions:
        player_team = team_map.get(player["trackId"], -1)
        if player_team != defending_team:
            continue
        
        dist = _distance((player["x"], player["y"]), (ball_x, ball_y))
        if dist <= FREE_KICK_CLUSTER_DIST_M:
            wall_players += 1
    
    return wall_players

def _detect_corner_delivery(
    set_piece_frame: int, ball_track: Dict, fps: float, attacking_team: int
) -> Dict[str, Any]:
    """Analyze corner delivery type and outcome."""
    details = {
        "delivery_type": "unknown",
        "attacking_runners": 0,
        "shot_followed": False
    }
    
    # Look at next 30 frames after ball moves
    for fi in range(set_piece_frame + 1, min(set_piece_frame + 90, max(ball_track.keys()) + 1)):
        if fi not in ball_track:
            continue
        
        ball_speed = ball_track[fi].get("speed_ms", 0.0)
        if ball_speed > 2.0:  # Ball started moving
            # Analyze first 10 frames of movement
            movement_frames = []
            for mf in range(fi, min(fi + 10, max(ball_track.keys()) + 1)):
                if mf in ball_track:
                    movement_frames.append(mf)
            
            if len(movement_frames) >= 5:
                # Calculate delivery characteristics
                start_pos = (ball_track[movement_frames[0]]["x"], ball_track[movement_frames[0]]["y"])
                end_pos = (ball_track[movement_frames[-1]]["x"], ball_track[movement_frames[-1]]["y"])
                
                distance = _distance(start_pos, end_pos)
                avg_speed = ball_track[movement_frames[0]]["speed_ms"]
                
                # Classify delivery type
                if distance < 15.0:
                    details["delivery_type"] = "short_corner"
                elif avg_speed > 12.0:
                    details["delivery_type"] = "driven"
                else:
                    # Would need more complex analysis for in/outswinger detection
                    details["delivery_type"] = "standard"
            
            # Check for shot within 30 frames
            for sf in range(fi, min(fi + 30, max(ball_track.keys()) + 1)):
                if sf in ball_track and ball_track[sf].get("speed_ms", 0.0) > 8.0:
                    details["shot_followed"] = True
                    break
            
            break
    
    return details

def _detect_free_kick_details(
    set_piece_frame: int, ball_x: float, ball_y: float,
    ball_track: Dict, fps: float, attacking_team: int,
    defending_team: int, player_positions: List[Dict], team_map: Dict
) -> Dict[str, Any]:
    """Analyze free kick details."""
    # Distance from goal
    dist_to_left_goal = _distance((ball_x, ball_y), (0.0, 34.0))
    dist_to_right_goal = _distance((ball_x, ball_y), (PITCH_WIDTH, 34.0))
    dist_to_goal = min(dist_to_left_goal, dist_to_right_goal)
    
    # Wall size
    wall_size = _detect_defensive_wall(ball_x, ball_y, defending_team, player_positions, team_map)
    
    # Direct/indirect detection
    details = {
        "distance_to_goal_m": dist_to_goal,
        "wall_size": wall_size,
        "direct_free_kick": False,
        "outcome": "unknown"
    }
    
    # Look at next 20 frames to determine if direct or indirect
    for fi in range(set_piece_frame + 1, min(set_piece_frame + 20, max(ball_track.keys()) + 1)):
        if fi in ball_track and ball_track[fi].get("speed_ms", 0.0) > 8.0:
            details["direct_free_kick"] = True
            details["outcome"] = "shot"
            break
        elif fi in ball_track and ball_track[fi].get("speed_ms", 0.0) > 2.0:
            details["direct_free_kick"] = False
            details["outcome"] = "pass"
            break
    
    return details

def _detect_throw_in_details(
    set_piece_frame: int, ball_x: float, ball_y: float, ball_track: Dict, fps: float
) -> Dict[str, Any]:
    """Analyze throw-in details."""
    # Determine zone
    if ball_x < 35.0:
        zone = "defensive_third"
    elif ball_x > 70.0:
        zone = "attacking_third"
    else:
        zone = "middle_third"
    
    # Quick or slow throw-in
    details = {
        "zone": zone,
        "quick_throw": False
    }
    
    # Look at next 10 frames
    for fi in range(set_piece_frame + 1, min(set_piece_frame + 10, max(ball_track.keys()) + 1)):
        if fi in ball_track and ball_track[fi].get("speed_ms", 0.0) > 2.0:
            time_to_move = (fi - set_piece_frame) / fps
            details["quick_throw"] = time_to_move < 2.0
            break
    
    return details

def detect_set_pieces(job_id: str) -> dict:
    """Detect set pieces from tracking data."""
    # Load required files
    base = _base_job_id(job_id)
    candidates = [job_id]
    if base != job_id:
        candidates.append(base)
    
    # Load pitch_map.json
    pitch_data = None
    for jid in candidates:
        pitch_data = _load_json(Path(f"temp/{jid}/pitch/pitch_map.json"))
        if pitch_data is not None:
            break
    if pitch_data is None:
        raise FileNotFoundError(f"pitch_map.json not found for job '{job_id}'")
    
    # Load team_results.json
    team_data = None
    for jid in candidates:
        team_data = _load_json(Path(f"temp/{jid}/tracking/team_results.json"))
        if team_data is not None:
            break
    if team_data is None:
        raise FileNotFoundError(f"team_results.json not found for job '{job_id}'")
    
    # Load track_results.json for ball trajectory and FPS
    track_data = None
    for jid in candidates:
        track_data = _load_json(Path(f"temp/{jid}/tracking/track_results.json"))
        if track_data is not None:
            break
    if track_data is None:
        raise FileNotFoundError(f"track_results.json not found for job '{job_id}'")
    
    # Get FPS
    fps = 25.0
    if track_data is not None:
        metadata = track_data.get("metadata", {})
        fps = float(metadata.get("fps", 25))
        if fps <= 0:
            fps = 25.0
    
    # Build team mapping
    team_map = {}
    tracks_list = team_data.get("tracks", team_data) if isinstance(team_data, dict) else team_data
    for t in tracks_list:
        team_map[t["trackId"]] = t.get("teamId", -1)
    
    # Build ball track
    ball_trajectory = track_data.get("ball_trajectory", [])
    if not ball_trajectory:
        logger.warning("No ball trajectory found")
        return {"set_pieces": [], "summary": {}, "total_set_pieces": 0}
    
    # Convert ball trajectory to world coordinates
    frame_w = pitch_data.get("frameWidth", 3840) or 3840
    frame_h = pitch_data.get("frameHeight", 2160) or 2160
    
    ball_track = {}
    for b in ball_trajectory:
        fi = int(b["frameIndex"])
        px = float(b.get("x", 0.0))
        py = float(b.get("y", 0.0))
        wx = px / frame_w * PITCH_WIDTH
        wy = py / frame_h * PITCH_HEIGHT
        
        # Calculate speed (simplified)
        speed_ms = 0.0
        if fi - 1 in ball_track:
            prev_x, prev_y = ball_track[fi - 1]["x"], ball_track[fi - 1]["y"]
            dist = _distance((wx, wy), (prev_x, prev_y))
            speed_ms = dist * fps
        
        ball_track[fi] = {
            "x": wx,
            "y": wy,
            "speed_ms": speed_ms
        }
    
    # Build player positions per frame
    player_world = {}
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
    
    # Detect stationary ball periods
    stationary_periods = []
    current_stationary_start = None
    
    for fi in sorted(ball_track.keys()):
        speed = ball_track[fi]["speed_ms"]
        
        if speed < STATIC_BALL_SPEED_MS:
            if current_stationary_start is None:
                current_stationary_start = fi
        else:
            if current_stationary_start is not None:
                duration = fi - current_stationary_start
                if duration >= STATIC_BALL_FRAMES:
                    stationary_periods.append((current_stationary_start, fi - 1))
                current_stationary_start = None
    
    # Handle trailing stationary period
    if current_stationary_start is not None:
        last_frame = max(ball_track.keys())
        duration = last_frame - current_stationary_start + 1
        if duration >= STATIC_BALL_FRAMES:
            stationary_periods.append((current_stationary_start, last_frame))
    
    # Classify set pieces
    set_pieces = []
    
    for start_frame, end_frame in stationary_periods:
        # Use the middle frame of stationary period for classification
        mid_frame = (start_frame + end_frame) // 2
        
        if mid_frame not in ball_track:
            continue
        
        ball_x = ball_track[mid_frame]["x"]
        ball_y = ball_track[mid_frame]["y"]
        
        # Determine set piece type
        set_piece_type = "unknown_static"
        zone = "unknown"
        
        if _is_near_corner(ball_x, ball_y):
            set_piece_type = "corner"
            zone = "corner"
        elif _is_in_penalty_area(ball_x, ball_y):
            set_piece_type = "penalty"
            zone = "penalty_area"
        elif _is_near_centre_circle(ball_x, ball_y):
            set_piece_type = "kick_off"
            zone = "centre_circle"
        elif _is_in_goal_area(ball_x, ball_y):
            # Check if preceded by high speed (goal kick)
            high_speed_before = False
            for fi in range(max(0, start_frame - 5), start_frame):
                if fi in ball_track and ball_track[fi]["speed_ms"] > 10.0:
                    high_speed_before = True
                    break
            
            if high_speed_before:
                set_piece_type = "goal_kick"
                zone = "goal_area"
        elif _is_near_sideline(ball_y):
            set_piece_type = "throw_in"
            if ball_x < 35.0:
                zone = "defensive_third"
            elif ball_x > 70.0:
                zone = "attacking_third"
            else:
                zone = "middle_third"
        else:
            # Check for defensive wall (free kick)
            # Determine which team is attacking (has possession)
            attacking_team = -1
            defending_team = -1
            
            # Simple heuristic: look at player positions around ball
            if mid_frame in player_world:
                team_counts = {0: 0, 1: 0}
                for player in player_world[mid_frame]:
                    team = team_map.get(player["trackId"], -1)
                    if team in [0, 1]:
                        dist = _distance((player["x"], player["y"]), (ball_x, ball_y))
                        if dist < 10.0:  # Within 10m of ball
                            team_counts[team] += 1
                
                if team_counts[0] > team_counts[1]:
                    attacking_team = 0
                    defending_team = 1
                elif team_counts[1] > team_counts[0]:
                    attacking_team = 1
                    defending_team = 0
            
            if defending_team >= 0:
                wall_size = _detect_defensive_wall(ball_x, ball_y, defending_team, 
                                                player_world.get(mid_frame, []), team_map)
                if wall_size >= WALL_MIN_PLAYERS:
                    set_piece_type = "free_kick"
                    zone = "open_play"
        
        # Skip unknown static periods
        if set_piece_type == "unknown_static":
            continue
        
        # Determine which team has the set piece
        possessing_team = -1
        if mid_frame in player_world:
            # Find closest player to ball
            min_dist = float("inf")
            for player in player_world[mid_frame]:
                dist = _distance((player["x"], player["y"]), (ball_x, ball_y))
                if dist < min_dist:
                    min_dist = dist
                    possessing_team = team_map.get(player["trackId"], -1)
        
        if possessing_team not in [0, 1]:
            continue
        
        # Analyze type-specific details
        details = {}
        outcome = None
        
        if set_piece_type == "corner":
            details = _detect_corner_delivery(end_frame, ball_track, fps, possessing_team)
        elif set_piece_type == "free_kick":
            details = _detect_free_kick_details(
                end_frame, ball_x, ball_y, ball_track, fps,
                possessing_team, defending_team, player_world.get(mid_frame, []), team_map
            )
        elif set_piece_type == "throw_in":
            details = _detect_throw_in_details(end_frame, ball_x, ball_y, ball_track, fps)
        
        # Create set piece record
        set_piece = {
            "type": set_piece_type,
            "frame": start_frame,
            "time_s": start_frame / fps,
            "team": possessing_team,
            "ball_x": ball_x,
            "ball_y": ball_y,
            "zone": zone,
            "details": details,
            "outcome": outcome
        }
        
        set_pieces.append(set_piece)
    
    # Build summary statistics
    summary = {
        "team_0": {
            "corners": 0,
            "free_kicks": 0,
            "throw_ins": 0,
            "penalties": 0,
            "corner_shots_created": 0,
            "free_kick_shots_created": 0
        },
        "team_1": {
            "corners": 0,
            "free_kicks": 0,
            "throw_ins": 0,
            "penalties": 0,
            "corner_shots_created": 0,
            "free_kick_shots_created": 0
        }
    }
    
    for sp in set_pieces:
        team_key = f"team_{sp['team']}"
        if team_key in summary:
            if sp["type"] == "corner":
                summary[team_key]["corners"] += 1
                if sp["details"].get("shot_followed", False):
                    summary[team_key]["corner_shots_created"] += 1
            elif sp["type"] == "free_kick":
                summary[team_key]["free_kicks"] += 1
                if sp["details"].get("outcome") == "shot":
                    summary[team_key]["free_kick_shots_created"] += 1
            elif sp["type"] == "throw_in":
                summary[team_key]["throw_ins"] += 1
            elif sp["type"] == "penalty":
                summary[team_key]["penalties"] += 1
    
    # Save results
    result = {
        "set_pieces": set_pieces,
        "summary": summary,
        "total_set_pieces": len(set_pieces)
    }
    
    output_dir = Path(f"temp/{job_id}/set_pieces")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "set_pieces.json", "w") as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Set piece analysis complete: {len(set_pieces)} set pieces detected")
    
    return result
