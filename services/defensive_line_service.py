import json
import logging
import math
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Constants
SMOOTHING_WINDOW = 15
OWN_HALF_X_TEAM0 = 52.5
OWN_HALF_X_TEAM1 = 52.5
MIN_DEFENDERS = 3
PITCH_WIDTH = 105.0
PITCH_HEIGHT = 68.0

# Import helper functions from formation_service
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

def _detect_goalkeeper(avg_positions, team_id):
    """Reuse GK detection from formation_service."""
    if team_id == 0:
        # Team 0 attacks right, GK has lowest x
        gk_idx = min(range(len(avg_positions)), key=lambda i: avg_positions[i]["x"])
    else:
        # Team 1 attacks left, GK has highest x
        gk_idx = max(range(len(avg_positions)), key=lambda i: avg_positions[i]["x"])
    return gk_idx

def _get_line_positions(outfield_positions, team_id):
    """Get defensive, midfield, and attacking line positions using same logic as formation_service."""
    if len(outfield_positions) < MIN_DEFENDERS:
        return None, None, None
    
    xs = np.array([p["x"] for p in outfield_positions])
    
    # Split into lines using x-percentile thresholds (same as formation_service)
    p33 = float(np.percentile(xs, 33))
    p66 = float(np.percentile(xs, 66))
    
    defenders = []
    midfielders = []
    attackers = []
    
    for p in outfield_positions:
        px = p["x"]
        if team_id == 0:
            if px <= p33:
                defenders.append(p)
            elif px <= p66:
                midfielders.append(p)
            else:
                attackers.append(p)
        else:
            # Team 1: lower x = more attacking
            if px >= p66:
                defenders.append(p)
            elif px >= p33:
                midfielders.append(p)
            else:
                attackers.append(p)
    
    # Calculate line positions
    def_line_x = None
    mid_line_x = None
    att_line_x = None
    
    if defenders:
        def_line_x = np.mean([p["x"] for p in defenders])
    if midfielders:
        mid_line_x = np.mean([p["x"] for p in midfielders])
    if attackers:
        att_line_x = np.mean([p["x"] for p in attackers])
    
    return def_line_x, mid_line_x, att_line_x

def _smooth_time_series(values, window):
    """Apply rolling average smoothing using numpy."""
    if len(values) < window:
        return values.copy()
    
    values_array = np.array(values, dtype=np.float64)
    smoothed = np.convolve(values_array, np.ones(window)/window, mode='valid')
    
    # Pad the beginning with original values to maintain same length
    result = values.copy()
    result[window-1:] = smoothed
    
    return result

def _detect_out_of_shape_events(timeline_data):
    """Detect windows where team is out of shape."""
    events = []
    current_event = None
    
    for frame_data in timeline_data:
        frame = frame_data["frame"]
        team_length = frame_data["team_length_m"]
        team_width = frame_data["team_width_m"]
        def_line_depth = frame_data["def_line_depth_m"]
        
        # Check out-of-shape conditions
        reasons = []
        if team_length > 40.0:
            reasons.append("stretched_too_long")
        if team_width > 55.0:
            reasons.append("too_wide")
        if def_line_depth < 18.0:
            reasons.append("defensive_line_in_own_box")
        
        is_out_of_shape = len(reasons) > 0
        
        if is_out_of_shape and current_event is None:
            # Start new event
            current_event = {
                "frame_start": frame,
                "reason": ", ".join(reasons)
            }
        elif not is_out_of_shape and current_event is not None:
            # End current event
            current_event["frame_end"] = frame - 1
            events.append(current_event)
            current_event = None
    
    # Close any open event
    if current_event is not None:
        current_event["frame_end"] = timeline_data[-1]["frame"]
        events.append(current_event)
    
    return events

def compute_defensive_lines(job_id: str) -> dict:
    """Compute defensive line height and team compactness analysis."""
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
    
    # Load track_results.json for FPS
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
    
    # Extract all frames from pitch_map
    all_frames = set()
    for player in pitch_data.get("players", []):
        for pt in player.get("trajectory2d", []):
            all_frames.add(int(pt["frameIndex"]))
    
    frame_list = sorted(all_frames)
    if not frame_list:
        logger.warning("No frame data found")
        return {"team_0": {}, "team_1": {}, "frame_count": 0}
    
    # Initialize data storage for each team
    team_data_storage = {
        0: {
            "def_line_depth_raw": [],
            "team_width_raw": [],
            "team_length_raw": [],
            "shape_area_raw": [],
            "def_mid_spacing_raw": [],
            "mid_att_spacing_raw": [],
            "timeline": []
        },
        1: {
            "def_line_depth_raw": [],
            "team_width_raw": [],
            "team_length_raw": [],
            "shape_area_raw": [],
            "def_mid_spacing_raw": [],
            "mid_att_spacing_raw": [],
            "timeline": []
        }
    }
    
    # Process each frame
    for frame in frame_list:
        # Get player positions for this frame
        team_positions = {0: [], 1: []}
        for player in pitch_data.get("players", []):
            for pt in player.get("trajectory2d", []):
                if int(pt["frameIndex"]) == frame:
                    team_id = team_map.get(player["trackId"], -1)
                    if team_id in [0, 1]:
                        team_positions[team_id].append({
                            "trackId": player["trackId"],
                            "x": float(pt["x"]),
                            "y": float(pt["y"])
                        })
                    break
        
        # Process each team
        for team_id in [0, 1]:
            positions = team_positions[team_id]
            if len(positions) < MIN_DEFENDERS + 1:  # Need at least GK + defenders
                continue
            
            # GK detection (reuse from formation_service)
            gk_idx = _detect_goalkeeper(positions, team_id)
            outfield = [p for i, p in enumerate(positions) if i != gk_idx]
            
            if len(outfield) < MIN_DEFENDERS:
                continue
            
            # Get line positions
            def_line_x, mid_line_x, att_line_x = _get_line_positions(outfield, team_id)
            
            if def_line_x is None:
                continue
            
            # Calculate defensive line depth (metres from own goal)
            if team_id == 0:
                def_line_depth = def_line_x  # Team 0 defends left
            else:
                def_line_depth = PITCH_WIDTH - def_line_x  # Team 1 defends right
            
            # Calculate team width
            team_width = max(p["y"] for p in outfield) - min(p["y"] for p in outfield)
            
            # Calculate team length
            team_length = max(p["x"] for p in outfield) - min(p["x"] for p in outfield)
            
            # Calculate shape area
            shape_area = team_width * team_length
            
            # Calculate inter-line spacing
            def_mid_spacing = 0.0
            mid_att_spacing = 0.0
            if mid_line_x is not None:
                def_mid_spacing = abs(mid_line_x - def_line_x)
            if att_line_x is not None and mid_line_x is not None:
                mid_att_spacing = abs(att_line_x - mid_line_x)
            
            # Store raw data
            storage = team_data_storage[team_id]
            storage["def_line_depth_raw"].append(def_line_depth)
            storage["team_width_raw"].append(team_width)
            storage["team_length_raw"].append(team_length)
            storage["shape_area_raw"].append(shape_area)
            storage["def_mid_spacing_raw"].append(def_mid_spacing)
            storage["mid_att_spacing_raw"].append(mid_att_spacing)
            storage["timeline"].append({
                "frame": frame,
                "time_s": frame / fps,
                "def_line_depth_m": def_line_depth,
                "team_width_m": team_width,
                "team_length_m": team_length,
                "shape_area_m2": shape_area,
                "def_mid_spacing_m": def_mid_spacing,
                "mid_att_spacing_m": mid_att_spacing
            })
    
    # Process and finalize results for each team
    result = {"team_0": {}, "team_1": {}, "frame_count": len(frame_list)}
    
    for team_id in [0, 1]:
        storage = team_data_storage[team_id]
        
        if not storage["timeline"]:
            result[f"team_{team_id}"] = {
                "avg_defensive_line_depth_m": None,
                "avg_team_width_m": None,
                "avg_team_length_m": None,
                "avg_shape_area_m2": None,
                "high_line_pct": None,
                "low_block_pct": None,
                "out_of_shape_events": [],
                "timeline": []
            }
            continue
        
        # Apply smoothing
        def_line_depth_smooth = _smooth_time_series(storage["def_line_depth_raw"], SMOOTHING_WINDOW)
        team_width_smooth = _smooth_time_series(storage["team_width_raw"], SMOOTHING_WINDOW)
        team_length_smooth = _smooth_time_series(storage["team_length_raw"], SMOOTHING_WINDOW)
        shape_area_smooth = _smooth_time_series(storage["shape_area_raw"], SMOOTHING_WINDOW)
        
        # Calculate averages from smoothed values
        avg_def_line_depth = np.mean(def_line_depth_smooth) if def_line_depth_smooth else None
        avg_team_width = np.mean(team_width_smooth) if team_width_smooth else None
        avg_team_length = np.mean(team_length_smooth) if team_length_smooth else None
        avg_shape_area = np.mean(shape_area_smooth) if shape_area_smooth else None
        
        # Calculate high line and low block percentages
        high_line_frames = sum(1 for depth in def_line_depth_smooth if depth > 40.0)
        low_block_frames = sum(1 for depth in def_line_depth_smooth if depth < 25.0)
        total_frames = len(def_line_depth_smooth)
        
        high_line_pct = (high_line_frames / total_frames * 100.0) if total_frames > 0 else None
        low_block_pct = (low_block_frames / total_frames * 100.0) if total_frames > 0 else None
        
        # Detect out-of-shape events
        out_of_shape_events = _detect_out_of_shape_events(storage["timeline"])
        
        # Downsample timeline to every 5 frames for output
        downsampled_timeline = storage["timeline"][::5]
        
        result[f"team_{team_id}"] = {
            "avg_defensive_line_depth_m": float(avg_def_line_depth) if avg_def_line_depth is not None else None,
            "avg_team_width_m": float(avg_team_width) if avg_team_width is not None else None,
            "avg_team_length_m": float(avg_team_length) if avg_team_length is not None else None,
            "avg_shape_area_m2": float(avg_shape_area) if avg_shape_area is not None else None,
            "high_line_pct": float(high_line_pct) if high_line_pct is not None else None,
            "low_block_pct": float(low_block_pct) if low_block_pct is not None else None,
            "out_of_shape_events": out_of_shape_events,
            "timeline": downsampled_timeline
        }
    
    # Save results
    output_dir = Path(f"temp/{job_id}/defensive_line")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "defensive_line.json", "w") as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Defensive line analysis complete: {result['frame_count']} frames processed")
    
    return result
