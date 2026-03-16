"""
services/analytics_overlay_service.py

Analytics overlay service - burns analytics overlays directly into video.
xG values on shots, pass lines, formation shapes, sprint trails.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Constants
OVERLAY_ALPHA = 0.65
PASS_LINE_THICKNESS = 2
SPRINT_TRAIL_FRAMES = 15
XG_FONT_SCALE = 0.7
XG_CIRCLE_RADIUS = 30
FORMATION_DOT_RADIUS = 8
TEAM_0_COLOR = (86, 190, 104)  # BGR green
TEAM_1_COLOR = (60, 130, 220)  # BGR orange-ish
BALL_COLOR = (255, 255, 255)   # BGR white
FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_overlay_data(job_id: str) -> Dict[str, Any]:
    """Load all available analytics for overlays."""
    base = Path("temp") / job_id
    data = {}
    
    # Load tracking data
    track_path = base / "tracking" / "track_results.json"
    if track_path.exists():
        with open(track_path) as f:
            data["track_data"] = json.load(f)
    else:
        data["track_data"] = None
    
    # Load team assignments
    team_path = base / "tracking" / "team_results.json"
    if team_path.exists():
        with open(team_path) as f:
            data["team_data"] = json.load(f)
    else:
        data["team_data"] = None
    
    # Load pitch map for homography
    pitch_path = base / "pitch" / "pitch_map.json"
    if pitch_path.exists():
        with open(pitch_path) as f:
            data["pitch_data"] = json.load(f)
    else:
        data["pitch_data"] = None
    
    # Load events
    events_path = base / "events" / "event_timeline.json"
    if events_path.exists():
        with open(events_path) as f:
            data["events"] = json.load(f)
    else:
        data["events"] = None
    
    # Load analytics report
    analytics_path = base / "analytics" / "analytics_report.json"
    if analytics_path.exists():
        with open(analytics_path) as f:
            data["analytics"] = json.load(f)
    else:
        data["analytics"] = None
    
    # Load pass network
    pass_path = base / "pass_network" / "pass_network.json"
    if pass_path.exists():
        with open(pass_path) as f:
            data["pass_network"] = json.load(f)
    else:
        data["pass_network"] = None
    
    # Load xG data
    xg_path = base / "xg" / "xg_results.json"
    if xg_path.exists():
        with open(xg_path) as f:
            data["xg"] = json.load(f)
    else:
        data["xg"] = None
    
    # Load heatmap data for sprint detection
    heatmap_path = base / "heatmap" / "heatmap_results.json"
    if heatmap_path.exists():
        with open(heatmap_path) as f:
            data["heatmap"] = json.load(f)
    else:
        data["heatmap"] = None
    
    # Load formation data
    formation_path = base / "formation" / "formation_results.json"
    if formation_path.exists():
        with open(formation_path) as f:
            data["formation"] = json.load(f)
    else:
        data["formation"] = None
    
    # Load highlights
    highlights_path = base / "highlights" / "highlights.json"
    if highlights_path.exists():
        with open(highlights_path) as f:
            data["highlights"] = json.load(f)
    else:
        data["highlights"] = None
    
    return data


def world_to_pixel(world_x: float, world_y: float, H_inv: np.ndarray, 
                   frame_w: int, frame_h: int) -> Tuple[int, int]:
    """Convert world coordinates to pixel coordinates using inverse homography."""
    if H_inv is None:
        # Fallback: simple proportional mapping
        px = int(world_x / 105.0 * frame_w)
        py = int(world_y / 68.0 * frame_h)
        return px, py
    
    # Use homography
    world_pt = np.array([[world_x, world_y, 1.0]], dtype=np.float64).reshape(-1, 1, 3)
    pixel_pt = cv2.perspectiveTransform(world_pt, H_inv)
    px, py = int(pixel_pt[0][0][0]), int(pixel_pt[0][0][1])
    return px, py


def get_frame_player_positions(frame_idx: int, track_data: Dict, 
                              team_data: Dict, pitch_data: Dict) -> Dict[int, Tuple[int, int]]:
    """Get pixel positions of all players in a specific frame."""
    positions = {}
    
    if not track_data or not team_data or not pitch_data:
        return positions
    
    # Build team map
    team_map = {}
    for entry in team_data.get("tracks", team_data) if isinstance(team_data, dict) else team_data:
        team_map[entry["trackId"]] = entry.get("teamId", -1)
    
    # Get homography inverse if available
    H_inv = None
    if pitch_data.get("homographyFound"):
        # For now, we'll use proportional mapping as fallback
        # In a real implementation, you'd store the inverse homography
        pass
    
    frame_w = pitch_data.get("frameWidth", 3840)
    frame_h = pitch_data.get("frameHeight", 2160)
    
    # Get player positions from pitch_map
    for player in pitch_data.get("players", []):
        track_id = player["trackId"]
        team_id = team_map.get(track_id, -1)
        
        for pt in player.get("trajectory2d", []):
            if pt["frameIndex"] == frame_idx:
                world_x, world_y = pt["x"], pt["y"]
                px, py = world_to_pixel(world_x, world_y, H_inv, frame_w, frame_h)
                positions[track_id] = (px, py, team_id)
                break
    
    return positions


def draw_pass_network_overlay(frame: np.ndarray, frame_idx: int, 
                              track_data: Dict, pass_network: Dict,
                              team_data: Dict) -> np.ndarray:
    """Draw pass network lines between players."""
    if not pass_network or not track_data or not team_data:
        return frame
    
    # Get current player positions
    pitch_data = load_overlay_data(list(track_data.values())[0].get("jobId", ""))["pitch_data"]
    positions = get_frame_player_positions(frame_idx, track_data, team_data, pitch_data)
    
    # Build team map
    team_map = {}
    for entry in team_data.get("tracks", team_data) if isinstance(team_data, dict) else team_data:
        team_map[entry["trackId"]] = entry.get("teamId", -1)
    
    # Draw pass edges
    for edge in pass_network.get("edges", []):
        if edge.get("count", 0) < 3:  # Skip weak connections
            continue
        
        source_id = None
        target_id = None
        
        # Find track IDs from node IDs
        for node in pass_network.get("nodes", []):
            if node["id"] == edge["source"]:
                source_id = node["trackId"]
            elif node["id"] == edge["target"]:
                target_id = node["trackId"]
        
        if source_id is None or target_id is None:
            continue
        
        # Get positions
        source_pos = positions.get(source_id)
        target_pos = positions.get(target_id)
        
        if source_pos and target_pos:
            sx, sy, steam = source_pos
            tx, ty, tteam = target_pos
            
            # Use source team color
            color = TEAM_0_COLOR if steam == 0 else TEAM_1_COLOR
            
            # Line thickness proportional to pass count
            thickness = min(PASS_LINE_THICKNESS + edge.get("count", 1) // 3, 8)
            
            # Draw semi-transparent line
            overlay = frame.copy()
            cv2.line(overlay, (sx, sy), (tx, ty), color, thickness)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    
    # Draw player dots
    for node in pass_network.get("nodes", []):
        track_id = node["trackId"]
        pos = positions.get(track_id)
        
        if pos:
            px, py, team = pos
            # Dot size proportional to total passes
            pass_count = node.get("passCount", 0)
            radius = max(4, min(12, 4 + pass_count // 5))
            
            color = TEAM_0_COLOR if team == 0 else TEAM_1_COLOR
            cv2.circle(frame, (px, py), radius, color, -1)
    
    return frame


def draw_xg_overlay(frame: np.ndarray, frame_idx: int, 
                    xg_data: Dict, fps: float) -> np.ndarray:
    """Draw xG shot indicators."""
    if not xg_data:
        return frame
    
    for shot in xg_data.get("shots", []):
        shot_frame = shot["frame"]
        xg_value = shot["xg"]
        team = shot.get("team", 0)
        
        # Show overlay for 2 seconds after shot
        frames_to_show = int(fps * 2)
        if frame_idx < shot_frame or frame_idx > shot_frame + frames_to_show:
            continue
        
        # Calculate fade
        frames_since_shot = frame_idx - shot_frame
        fade_frames = 10
        
        if frames_since_shot < fade_frames:
            # Fade in
            alpha = frames_since_shot / fade_frames
        elif frames_since_shot > frames_to_show - fade_frames:
            # Fade out
            alpha = (frames_to_show - frames_since_shot) / fade_frames
        else:
            # Full opacity
            alpha = 1.0
        
        # Get shot position (convert from world coords)
        # For now, we'll use a placeholder position
        # In real implementation, convert from shot["ball_x"], shot["ball_y"]
        shot_x, shot_y = frame.shape[1] // 2, frame.shape[0] // 2
        
        # Choose color based on xG value
        if xg_value > 0.3:
            color = (0, 100, 255)  # Coral/red in BGR
        elif xg_value > 0.15:
            color = (0, 165, 255)  # Amber in BGR
        else:
            color = (128, 128, 128)  # Grey in BGR
        
        # Draw circle
        overlay = frame.copy()
        cv2.circle(overlay, (shot_x, shot_y), XG_CIRCLE_RADIUS, color, 3)
        
        # Add xG text
        text = f"xG {xg_value:.2f}"
        text_size = cv2.getTextSize(text, FONT, XG_FONT_SCALE, 2)[0]
        text_x = shot_x - text_size[0] // 2
        text_y = shot_y + 5
        
        cv2.putText(overlay, text, (text_x, text_y), FONT, 
                   XG_FONT_SCALE, (255, 255, 255), 2)
        
        # Add SHOT label
        shot_text = "SHOT"
        shot_size = cv2.getTextSize(shot_text, FONT, XG_FONT_SCALE * 0.8, 1)[0]
        shot_y = shot_y - XG_CIRCLE_RADIUS - 10
        cv2.putText(overlay, shot_text, 
                   (shot_x - shot_size[0] // 2, shot_y), FONT,
                   XG_FONT_SCALE * 0.8, (255, 255, 255), 1)
        
        # Blend with alpha
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    return frame


def draw_sprint_trails(frame: np.ndarray, frame_idx: int,
                       track_data: Dict, heatmap_data: Dict,
                       team_data: Dict) -> np.ndarray:
    """Draw sprint trails for fast-moving players."""
    if not track_data or not team_data:
        return frame
    
    # Get current player speeds from heatmap data
    sprinting_players = set()
    
    if heatmap_data:
        for pid_str, player_data in heatmap_data.get("players", {}).items():
            pid = int(pid_str)
            # Check if player is sprinting (speed > 7.0 m/s)
            if player_data.get("top_speed_ms", 0) > 7.0:
                sprinting_players.add(pid)
    
    # Draw trails for sprinting players
    pitch_data = load_overlay_data(list(track_data.values())[0].get("jobId", ""))["pitch_data"]
    positions = get_frame_player_positions(frame_idx, track_data, team_data, pitch_data)
    
    for track_id in sprinting_players:
        team = None
        for entry in team_data.get("tracks", team_data) if isinstance(team_data, dict) else team_data:
            if entry["trackId"] == track_id:
                team = entry.get("teamId", -1)
                break
        
        if team is None:
            continue
        
        color = TEAM_0_COLOR if team == 0 else TEAM_1_COLOR
        
        # Draw trail dots for last SPRINT_TRAIL_FRAMES frames
        for i in range(SPRINT_TRAIL_FRAMES):
            trail_frame = frame_idx - i
            if trail_frame < 0:
                break
            
            trail_positions = get_frame_player_positions(trail_frame, track_data, team_data, pitch_data)
            pos = trail_positions.get(track_id)
            
            if pos:
                px, py, _ = pos
                # Fade trail
                alpha = 1.0 - (i / SPRINT_TRAIL_FRAMES)
                radius = max(1, 3 - i // 5)
                
                overlay = frame.copy()
                cv2.circle(overlay, (px, py), radius, color, -1)
                cv2.addWeighted(overlay, alpha * 0.7, frame, 1 - alpha * 0.7, 0, frame)
    
    return frame


def draw_formation_overlay(frame: np.ndarray, frame_idx: int,
                           formation_data: Dict, track_data: Dict,
                           team_data: Dict) -> np.ndarray:
    """Draw formation shapes connecting team players."""
    if not track_data or not team_data:
        return frame
    
    # Get current player positions
    pitch_data = load_overlay_data(list(track_data.values())[0].get("jobId", ""))["pitch_data"]
    positions = get_frame_player_positions(frame_idx, track_data, team_data, pitch_data)
    
    # Group positions by team
    team_positions = {0: [], 1: []}
    for track_id, (px, py, team) in positions.items():
        if team in team_positions:
            team_positions[team].append((px, py))
    
    # Draw formation shapes
    for team, pts in team_positions.items():
        if len(pts) < 3:  # Need at least 3 points for convex hull
            continue
        
        pts_array = np.array(pts, dtype=np.int32)
        
        # Calculate convex hull
        hull = cv2.convexHull(pts_array)
        
        # Draw filled polygon with low alpha
        color = TEAM_0_COLOR if team == 0 else TEAM_1_COLOR
        overlay = frame.copy()
        cv2.fillPoly(overlay, [hull], color)
        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
        
        # Draw player dots
        for px, py in pts:
            cv2.circle(frame, (px, py), FORMATION_DOT_RADIUS, color, -1)
    
    return frame


def draw_event_labels(frame: np.ndarray, frame_idx: int,
                      event_data: Dict, fps: float) -> np.ndarray:
    """Draw event labels for recent events."""
    if not event_data:
        return frame
    
    current_events = []
    
    # Find events within 1.5 seconds of current frame
    for event in event_data.get("events", []):
        event_frame = event.get("frame", 0)
        time_diff = abs(frame_idx - event_frame) / fps
        
        if time_diff <= 1.5:
            current_events.append(event)
    
    # Limit to 3 simultaneous labels
    current_events = current_events[:3]
    
    for event in current_events:
        event_frame = event.get("frame", 0)
        event_type = event.get("type", "")
        team = event.get("team", 0)
        
        # Calculate fade based on time distance
        time_diff = abs(frame_idx - event_frame) / fps
        alpha = max(0.3, 1.0 - (time_diff / 1.5))
        
        # Get event position (placeholder for now)
        ex, ey = frame.shape[1] // 2, frame.shape[0] // 2
        
        # Draw based on event type
        if event_type.upper() == "PASS":
            # Draw arrow
            color = TEAM_0_COLOR if team == 0 else TEAM_1_COLOR
            cv2.arrowedLine(frame, (ex - 30, ey), (ex + 30, ey), color, 2, tipLength=0.3)
            
        elif event_type.upper() == "SHOT":
            # Draw shot label with xG if available
            xg = event.get("details", {}).get("xg", 0.0)
            text = f"SHOT xG:{xg:.2f}"
            cv2.putText(frame, text, (ex - 40, ey - 20), FONT,
                       0.6, (255, 255, 255), 2)
            
        elif event_type.upper() == "DRIBBLE":
            # Draw dribble label with highlight ring
            cv2.putText(frame, "DRIBBLE", (ex - 30, ey), FONT,
                       0.6, (255, 255, 255), 2)
            cv2.circle(frame, (ex, ey), 25, (255, 255, 255), 2)
            
        elif event_type.upper() == "TURNOVER":
            # Draw turnover label in red
            cv2.putText(frame, "TURNOVER", (ex - 35, ey), FONT,
                       0.6, (0, 0, 255), 2)
    
    return frame


def merge_frame_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping frame ranges."""
    if not ranges:
        return []
    
    # Sort by start frame
    ranges.sort(key=lambda x: x[0])
    
    merged = []
    current_start, current_end = ranges[0]
    
    for start, end in ranges[1:]:
        if start <= current_end:  # Overlapping
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    
    merged.append((current_start, current_end))
    return merged


def render_analytics_highlight(
    job_id: str,
    output_path: str,
    overlays: List[str],
    highlight_frames: Optional[List[int]] = None,
    padding_seconds: float = 2.0
) -> str:
    """Render analytics highlight video with specified overlays."""
    
    # Load all overlay data
    data = load_overlay_data(job_id)
    
    if not data.get("track_data"):
        raise ValueError(f"No tracking data found for job '{job_id}'")
    
    track_data = data["track_data"]
    video_path = track_data.get("videoPath", "")
    
    if not video_path or not os.path.exists(video_path):
        raise ValueError(f"Source video not found: {video_path}")
    
    # Get video properties
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get highlight frames if not provided
    if not highlight_frames:
        highlight_frames = []
        if data.get("highlights"):
            highlight_frames = [h["frame"] for h in data["highlights"].get("highlights", [])]
        elif data.get("events"):
            highlight_frames = [e["frame"] for e in data["events"].get("events", []) 
                              if e.get("type") in ["shot", "dribble", "pass"]]
    
    # Build frame ranges to include
    padding_frames = int(padding_seconds * fps)
    ranges = []
    
    for frame in highlight_frames:
        start_frame = max(0, frame - padding_frames)
        end_frame = min(total_frames - 1, frame + padding_frames)
        ranges.append((start_frame, end_frame))
    
    # Merge overlapping ranges
    merged_ranges = merge_frame_ranges(ranges)
    
    # Setup output video writer
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
    
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {output_path}")
    
    # Process each frame in merged ranges
    for start_frame, end_frame in merged_ranges:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_idx in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply overlays in order (bottom to top layer)
            if "formation" in overlays:
                frame = draw_formation_overlay(frame, frame_idx, 
                                              data.get("formation"), track_data, 
                                              data.get("team_data"))
            
            if "pass_network" in overlays:
                frame = draw_pass_network_overlay(frame, frame_idx,
                                                 track_data, data.get("pass_network"),
                                                 data.get("team_data"))
            
            if "sprint_trails" in overlays:
                frame = draw_sprint_trails(frame, frame_idx,
                                           track_data, data.get("heatmap"),
                                           data.get("team_data"))
            
            if "xg" in overlays:
                frame = draw_xg_overlay(frame, frame_idx,
                                        data.get("xg"), fps)
            
            if "event_labels" in overlays:
                frame = draw_event_labels(frame, frame_idx,
                                          data.get("events"), fps)
            
            out.write(frame)
    
    # Cleanup
    cap.release()
    out.release()
    
    return output_path


def render_full_match_analytics(
    job_id: str,
    overlays: List[str]
) -> str:
    """Render full match analytics overlay video."""
    
    # Load all overlay data
    data = load_overlay_data(job_id)
    
    if not data.get("track_data"):
        raise ValueError(f"No tracking data found for job '{job_id}'")
    
    track_data = data["track_data"]
    video_path = track_data.get("videoPath", "")
    
    if not video_path or not os.path.exists(video_path):
        raise ValueError(f"Source video not found: {video_path}")
    
    # Get video properties
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup output video writer
    output_path = f"temp/{job_id}/render/analytics_full.mp4"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
    
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {output_path}")
    
    # Process every frame
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply overlays in order (bottom to top layer)
        if "formation" in overlays:
            frame = draw_formation_overlay(frame, frame_idx, 
                                          data.get("formation"), track_data, 
                                          data.get("team_data"))
        
        if "pass_network" in overlays:
            frame = draw_pass_network_overlay(frame, frame_idx,
                                             track_data, data.get("pass_network"),
                                             data.get("team_data"))
        
        if "sprint_trails" in overlays:
            frame = draw_sprint_trails(frame, frame_idx,
                                       track_data, data.get("heatmap"),
                                       data.get("team_data"))
        
        if "xg" in overlays:
            frame = draw_xg_overlay(frame, frame_idx,
                                    data.get("xg"), fps)
        
        if "event_labels" in overlays:
            frame = draw_event_labels(frame, frame_idx,
                                      data.get("events"), fps)
        
        out.write(frame)
    
    # Cleanup
    cap.release()
    out.release()
    
    return output_path
