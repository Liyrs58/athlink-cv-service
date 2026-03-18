from typing import Dict, List, Any
import math

SITUATIONS = ["OPEN_PLAY", "SET_PIECE", "TRANSITION", "HIGH_PRESS", "DEAD_BALL"]

# Track movement history for dead ball detection
_movement_history = {}

def detect_situation(tracks, ball, frame_idx, pitch_w=105.0, pitch_h=68.0):
    """
    Detect match situation from tracks and ball.

    DEAD_BALL triggers when:
    - Very few players detected (< 6), OR
    - Average player movement drops to near-zero (bench/cutaway)

    OPEN_PLAY/SET_PIECE based on ball and pitch activity.
    """

    # Calculate average movement velocity across all visible players
    total_movement = 0.0
    movement_samples = 0

    all_bboxes = [t for t in tracks if t.get("bbox")]

    if len(all_bboxes) < 4:
        # Very few players visible — dead ball / bench cutaway / stadium shot
        return {"situation": "DEAD_BALL", "confidence": 0.95, "details": {"reason": "low_detection", "tracks": len(all_bboxes)}}

    # Calculate average bbox movement from previous frame if available
    for track in all_bboxes:
        track_id = track.get("trackId", None)
        bbox = track.get("bbox", [])

        if track_id is None or len(bbox) < 4:
            continue

        # Get center of current bbox
        cx_curr = (bbox[0] + bbox[2]) / 2.0
        cy_curr = (bbox[1] + bbox[3]) / 2.0

        # Check if we have previous position
        if track_id in _movement_history:
            cx_prev, cy_prev = _movement_history[track_id]
            dx = cx_curr - cx_prev
            dy = cy_curr - cy_prev
            movement = math.sqrt(dx*dx + dy*dy)
            total_movement += movement
            movement_samples += 1

        # Update movement history
        _movement_history[track_id] = (cx_curr, cy_curr)

    # Clean up stale track history (tracks not seen for 5+ frames)
    seen_ids = {t.get("trackId") for t in all_bboxes if t.get("trackId")}
    stale_ids = set(_movement_history.keys()) - seen_ids
    for stale_id in stale_ids:
        _movement_history.pop(stale_id, None)

    # Determine average movement per player
    avg_movement = total_movement / max(movement_samples, 1)

    # DEAD_BALL: requires BOTH very low movement AND very few players visible
    # Logic:
    # - If 5+ players detected with 4+ movement samples → OPEN_PLAY (enough players playing)
    # - If fewer than 4 players AND movement < 3 pixels → DEAD_BALL (bench/cutaway)
    # - Otherwise → OPEN_PLAY (default for normal play)
    if len(all_bboxes) >= 5 and movement_samples >= 4:
        # Enough moving players — continue to OPEN_PLAY at line 95
        pass
    elif len(all_bboxes) < 4 and avg_movement < 3.0 and movement_samples >= 4:
        # Very few players AND almost no movement — this is dead ball (bench/cutaway)
        return {
            "situation": "DEAD_BALL",
            "confidence": 0.85,
            "details": {"reason": "low_movement", "avg_movement": round(avg_movement, 1), "tracks": len(all_bboxes)}
        }

    # Ball-based classification (if available)
    if ball:
        ball_vx = abs(ball.get("vx", 1.0))
        ball_vy = abs(ball.get("vy", 1.0))
        ball_speed = math.sqrt(ball_vx**2 + ball_vy**2)

        if ball_speed < 0.5:
            return {"situation": "SET_PIECE", "confidence": 0.8, "details": {"reason": "stationary_ball"}}
        elif ball_speed > 3.0:
            return {"situation": "TRANSITION", "confidence": 0.75, "details": {"reason": "fast_ball"}}

    # Pressing detection: players in upper half of frame (aggressive positioning)
    heights = [b[3] for b in [t.get("bbox", []) for t in all_bboxes] if len(b) >= 4]
    if heights:
        max_y = max(heights)
        frame_h = max_y / 0.88
        pressing = [b for b in [t.get("bbox", []) for t in all_bboxes] if len(b) >= 4 and b[3] < frame_h * 0.35]
        if len(pressing) >= 3:
            return {"situation": "HIGH_PRESS", "confidence": 0.7, "details": {"pressing": len(pressing)}}

    return {"situation": "OPEN_PLAY", "confidence": 0.7, "details": {"active_tracks": len(all_bboxes), "avg_movement": round(avg_movement, 1)}}

def get_situation_history(history, window=10):
    if not history:
        return "OPEN_PLAY"
    recent = history[-window:]
    counts = {}
    for h in recent:
        s = h.get("situation", "OPEN_PLAY")
        counts[s] = counts.get(s, 0) + 1
    return max(counts, key=counts.get)

def extract_situation_events(frame_results, fps=25.0, frame_stride=2):
    """
    Converts per-frame situation labels into discrete events.
    
    frame_results: list of dicts with 'frameIndex' and 'situation'
    Returns: list of event dicts
    """
    if not frame_results:
        return []

    events = []
    current_situation = None
    current_start = None
    current_start_frame = None

    for entry in frame_results:
        frame_idx = entry.get('frameIndex', 0)
        situation = entry.get('situation', 'OPEN_PLAY')
        timestamp = frame_idx / fps

        if situation != current_situation:
            if current_situation is not None:
                duration = (frame_idx - current_start_frame) / fps
                # Detect DEAD_BALL faster (0.2s min), but still filter flicker for other situations
                min_duration = 0.2 if current_situation == "DEAD_BALL" else 0.5
                if duration >= min_duration:
                    events.append({
                        'situation': current_situation,
                        'start_frame': current_start_frame,
                        'end_frame': frame_idx,
                        'start_time': round(current_start, 2),
                        'end_time': round(timestamp, 2),
                        'duration_seconds': round(duration, 2),
                    })
            current_situation = situation
            current_start = timestamp
            current_start_frame = frame_idx

    # Close final event
    if current_situation is not None and frame_results:
        last = frame_results[-1]
        last_frame = last.get('frameIndex', 0)
        duration = (last_frame - current_start_frame) / fps
        min_duration = 0.2 if current_situation == "DEAD_BALL" else 0.5
        if duration >= min_duration:
            events.append({
                'situation': current_situation,
                'start_frame': current_start_frame,
                'end_frame': last_frame,
                'start_time': round(current_start, 2),
                'end_time': round(last_frame / fps, 2),
                'duration_seconds': round(duration, 2),
            })

    return events
