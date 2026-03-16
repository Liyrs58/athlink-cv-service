import json
import logging
import math
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Constants
COUNTER_PRESS_WINDOW_S = 5.0
COUNTER_PRESS_RADIUS_M = 5.0
MIN_PRESSERS = 2
INTENSITY_THRESHOLD = 3
SUCCESS_REGAIN_S = 5.0
PITCH_WIDTH = 105.0
PITCH_HEIGHT = 68.0
GRID_W = 21
GRID_H = 14

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

def _ensure_events_exist(job_id: str) -> dict:
    """Ensure events exist by running detect_events if needed."""
    events_path = Path(f"temp/{job_id}/events/event_timeline.json")
    if not events_path.exists():
        # Import and run detect_events
        from services.event_service import detect_events
        logger.info(f"Running event detection for job {job_id}")
        return detect_events(job_id)
    else:
        return _load_json(events_path)

def _normalise_grid(grid):
    """Normalise grid values to 0-1 range."""
    mx = grid.max()
    if mx > 0:
        grid = grid / mx
    return [round(float(v), 4) for v in grid.ravel()]

def _detect_pressing_in_window(
    start_frame: int,
    end_frame: int,
    counter_press_team: int,
    player_world: dict,
    team_map: dict,
    ball_world: dict,
    carrier_by_frame: dict
) -> List[Tuple[int, int, float, float]]:
    """
    Detect pressing actions in a window.
    Returns list of (frame, pressing_player_id, ball_x, ball_y)
    """
    pressing_actions = []
    last_action_frame = {}  # Dedup: one action per player per 5 frames
    
    for fi in range(start_frame, min(end_frame + 1, max(carrier_by_frame.keys()) + 1)):
        if fi not in carrier_by_frame:
            continue
            
        carrier_tid = carrier_by_frame[fi]
        carrier_team = team_map.get(carrier_tid, -1)
        
        # Skip if carrier is from counter-pressing team (they have possession)
        if carrier_team == counter_press_team:
            continue
            
        # Get ball position
        ball_x, ball_y = 0.0, 0.0
        if fi in ball_world:
            ball_x, ball_y = ball_world[fi]
        elif fi in player_world:
            # Use carrier position as proxy
            for p in player_world[fi]:
                if p["trackId"] == carrier_tid:
                    ball_x, ball_y = p["x"], p["y"]
                    break
        
        # Check for pressing players
        if fi in player_world:
            for p in player_world[fi]:
                ptid = p["trackId"]
                if team_map.get(ptid, -1) != counter_press_team:
                    continue
                    
                dx = p["x"] - ball_x
                dy = p["y"] - ball_y
                dist = math.sqrt(dx * dx + dy * dy)
                
                if dist < COUNTER_PRESS_RADIUS_M:
                    # Dedup: one action per player per 5 frames
                    prev_fi = last_action_frame.get(ptid, -10)
                    if fi - prev_fi >= 5:
                        pressing_actions.append((fi, ptid, ball_x, ball_y))
                        last_action_frame[ptid] = fi
    
    return pressing_actions

def _classify_intensity(pressing_actions: List[Tuple[int, int, float, float]]) -> str:
    """Classify counter-press intensity based on simultaneous pressers."""
    if not pressing_actions:
        return "no_press"
    
    # Count maximum simultaneous pressers
    max_simultaneous = 0
    frames_with_actions = {}
    
    for fi, ptid, bx, by in pressing_actions:
        if fi not in frames_with_actions:
            frames_with_actions[fi] = []
        frames_with_actions[fi].append(ptid)
    
    for frame_pressers in frames_with_actions.values():
        max_simultaneous = max(max_simultaneous, len(frame_pressers))
    
    if max_simultaneous >= INTENSITY_THRESHOLD:
        return "high_intensity"
    elif max_simultaneous >= MIN_PRESSERS:
        return "medium_intensity"
    elif max_simultaneous >= 1:
        return "low_intensity"
    else:
        return "no_press"

def _determine_outcome(
    turnover_frame: int,
    counter_press_team: int,
    events: List[dict],
    fps: float
) -> str:
    """Determine the outcome of a counter-press attempt."""
    window_end_frame = turnover_frame + int(SUCCESS_REGAIN_S * fps)
    
    # Look for events in the window
    for event in events:
        if event["frame"] <= turnover_frame:
            continue
        if event["frame"] > window_end_frame:
            break
            
        # Check for regain by counter-pressing team
        if event["type"] == "turnover" and event["team"] == counter_press_team:
            return "success_regain"
        
        # Check for shot conceded
        if event["type"] == "shot" and event["team"] != counter_press_team:
            return "fail_shot_conceded"
        
        # Check for foul (if available)
        if event["type"] == "foul" and event["team"] != counter_press_team:
            return "success_foul_won"
    
    # If no specific outcome found, possession was lost
    return "fail_possession_lost"

def _calculate_team_retreat(
    start_frame: int,
    end_frame: int,
    team: int,
    player_world: dict,
    team_map: dict
) -> bool:
    """Check if team retreated (moved toward own goal > 5m)."""
    positions_start = []
    positions_end = []
    
    for fi in range(start_frame, min(start_frame + 10, end_frame + 1)):
        if fi in player_world:
            for p in player_world[fi]:
                if team_map.get(p["trackId"], -1) == team:
                    positions_start.append((p["x"], p["y"]))
    
    for fi in range(max(start_frame, end_frame - 10), end_frame + 1):
        if fi in player_world:
            for p in player_world[fi]:
                if team_map.get(p["trackId"], -1) == team:
                    positions_end.append((p["x"], p["y"]))
    
    if not positions_start or not positions_end:
        return False
    
    # Calculate average positions
    avg_x_start = np.mean([pos[0] for pos in positions_start])
    avg_x_end = np.mean([pos[0] for pos in positions_end])
    
    # Check retreat based on team direction
    if team == 0:  # Team 0 attacks right, retreat = move left
        return avg_x_end < avg_x_start - 5.0
    else:  # Team 1 attacks left, retreat = move right
        return avg_x_end > avg_x_start + 5.0

def compute_counter_press(job_id: str) -> dict:
    """Compute counter-pressing analysis."""
    # Ensure events exist
    events_data = _ensure_events_exist(job_id)
    events = events_data.get("events", [])
    
    # Load other required files
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
    
    # Load track_results.json for FPS and ball trajectory
    track_data = None
    for jid in candidates:
        track_data = _load_json(Path(f"temp/{jid}/tracking/track_results.json"))
        if track_data is not None:
            break
    
    # Get FPS
    fps = 25.0
    if track_data is not None:
        fps = (
            track_data.get('fps') or
            track_data.get('metadata', {}).get('fps') or
            25.0
        )
        fps = float(fps) if fps else 25.0
    
    # Build team mapping
    team_map = {}
    tracks_list = team_data.get("tracks", team_data) if isinstance(team_data, dict) else team_data
    for t in tracks_list:
        team_map[t["trackId"]] = t.get("teamId", -1)
    
    # Build player world positions
    player_world = {}
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
    
    # Build ball world positions
    ball_world = {}
    if track_data is not None:
        ball_trajectory = track_data.get("ball_trajectory", [])
        frame_w = pitch_data.get("frameWidth", 3840) or 3840
        frame_h = pitch_data.get("frameHeight", 2160) or 2160
        
        for b in ball_trajectory:
            fi = int(b["frameIndex"])
            px = float(b.get("x", 0.0))
            py = float(b.get("y", 0.0))
            ball_world[fi] = (px / frame_w * PITCH_WIDTH, py / frame_h * PITCH_HEIGHT)
    
    # Build possession frames (reuse from pressing_service logic)
    from services.pass_network_service import compute_possession_frames
    common_frames = sorted(set(ball_world.keys()) & set(player_world.keys()))
    carrier_by_frame, _ = compute_possession_frames(common_frames, ball_world, player_world, team_map)
    
    # Extract turnover events
    turnover_events = [e for e in events if e["type"] == "turnover"]
    
    # Initialize results
    results = {
        "team_0": {
            "total_attempts": 0,
            "high_intensity_pct": 0.0,
            "success_rate": 0.0,
            "avg_pressers": 0.0,
            "avg_time_to_first_press_s": 0.0,
            "intensity_map": [0.0] * (GRID_W * GRID_H),
            "windows": []
        },
        "team_1": {
            "total_attempts": 0,
            "high_intensity_pct": 0.0,
            "success_rate": 0.0,
            "avg_pressers": 0.0,
            "avg_time_to_first_press_s": 0.0,
            "intensity_map": [0.0] * (GRID_W * GRID_H),
            "windows": []
        },
        "total_turnovers_analysed": len(turnover_events)
    }
    
    # Process each turnover
    for turnover in turnover_events:
        turnover_frame = turnover["frame"]
        gaining_team = turnover["team"]  # Team that gained possession
        losing_team = 1 - gaining_team  # Team that lost possession and will counter-press
        
        # The counter-pressing team is the one that lost possession
        counter_press_team = losing_team
        
        # Define window
        window_end_frame = turnover_frame + int(COUNTER_PRESS_WINDOW_S * fps)
        
        # Detect pressing actions in window
        pressing_actions = _detect_pressing_in_window(
            turnover_frame, window_end_frame, counter_press_team,
            player_world, team_map, ball_world, carrier_by_frame
        )
        
        # Classify intensity
        intensity = _classify_intensity(pressing_actions)
        
        # Check for retreat (override intensity if retreated)
        if intensity == "no_press":
            if _calculate_team_retreat(turnover_frame, window_end_frame, counter_press_team, player_world, team_map):
                intensity = "no_press"  # Already no_press, but this indicates retreat
        
        # Determine outcome
        outcome = _determine_outcome(turnover_frame, counter_press_team, events, fps)
        
        # Calculate metrics
        peak_pressers = 0
        peak_positions = []
        first_press_time = None
        
        if pressing_actions:
            # Find peak frame with most pressers
            frames_with_pressers = {}
            for fi, ptid, bx, by in pressing_actions:
                if fi not in frames_with_pressers:
                    frames_with_pressers[fi] = []
                frames_with_pressers[fi].append({"track_id": ptid, "x": bx, "y": by})
            
            peak_frame = max(frames_with_pressers.keys(), key=lambda f: len(frames_with_pressers[f]))
            peak_pressers = len(frames_with_pressers[peak_frame])
            peak_positions = frames_with_pressers[peak_frame]
            
            # Time to first press
            first_press_frame = min(pressing_actions, key=lambda x: x[0])[0]
            first_press_time = (first_press_frame - turnover_frame) / fps
        
        # Build window data
        window_data = {
            "trigger_frame": turnover_frame,
            "trigger_time_s": turnover_frame / fps,
            "intensity": intensity,
            "outcome": outcome,
            "peak_pressers": peak_pressers,
            "duration_frames": min(window_end_frame, max(carrier_by_frame.keys())) - turnover_frame,
            "peak_positions": peak_positions
        }
        
        # Update team results
        team_key = f"team_{counter_press_team}"
        team_results = results[team_key]
        
        team_results["total_attempts"] += 1
        team_results["windows"].append(window_data)
        
        # Update intensity map for high intensity windows
        if intensity == "high_intensity" and peak_positions:
            for pos in peak_positions:
                # Convert to grid coordinates
                grid_x = int(pos["x"] / PITCH_WIDTH * GRID_W)
                grid_y = int(pos["y"] / PITCH_HEIGHT * GRID_H)
                grid_x = max(0, min(GRID_W - 1, grid_x))
                grid_y = max(0, min(GRID_H - 1, grid_y))
                grid_idx = grid_y * GRID_W + grid_x
                team_results["intensity_map"][grid_idx] += 1
        
        # Accumulate for averages
        if first_press_time is not None:
            team_results["avg_time_to_first_press_s"] += first_press_time
        team_results["avg_pressers"] += peak_pressers
    
    # Calculate final statistics for each team
    for team_id in [0, 1]:
        team_key = f"team_{team_id}"
        team_results = results[team_key]
        
        if team_results["total_attempts"] > 0:
            # High intensity percentage
            high_intensity_count = sum(1 for w in team_results["windows"] if w["intensity"] == "high_intensity")
            team_results["high_intensity_pct"] = (high_intensity_count / team_results["total_attempts"]) * 100.0
            
            # Success rate
            success_count = sum(1 for w in team_results["windows"] if w["outcome"] == "success_regain")
            team_results["success_rate"] = (success_count / team_results["total_attempts"]) * 100.0
            
            # Average pressers
            team_results["avg_pressers"] /= team_results["total_attempts"]
            
            # Average time to first press
            windows_with_press = [w for w in team_results["windows"] if w["peak_pressers"] > 0]
            if windows_with_press:
                total_time = sum(w["trigger_time_s"] - min(windows_with_press, key=lambda x: x["trigger_frame"])["trigger_time_s"] 
                              for w in windows_with_press)
                team_results["avg_time_to_first_press_s"] = total_time / len(windows_with_press)
            else:
                team_results["avg_time_to_first_press_s"] = 0.0
        
        # Normalise intensity map
        team_results["intensity_map"] = _normalise_grid(np.array(team_results["intensity_map"]))
    
    # Save results
    output_dir = Path(f"temp/{job_id}/counter_press")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "counter_press.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Counter-press analysis complete: {results['total_turnovers_analysed']} turnovers analysed")
    
    return results
