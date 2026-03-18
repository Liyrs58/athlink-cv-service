from typing import Dict, List, Any

SITUATIONS = ["OPEN_PLAY", "SET_PIECE", "TRANSITION", "HIGH_PRESS", "DEAD_BALL"]

def detect_situation(tracks, ball, frame_idx, pitch_w=105.0, pitch_h=68.0):
    if len(tracks) < 6:
        return {"situation": "DEAD_BALL", "confidence": 0.9, "details": {"reason": "low_detection"}}
    if ball and abs(ball.get("vx", 1.0)) < 0.5 and abs(ball.get("vy", 1.0)) < 0.5:
        return {"situation": "SET_PIECE", "confidence": 0.8, "details": {}}
    if ball and abs(ball.get("vx", 0)) > 3.0:
        return {"situation": "TRANSITION", "confidence": 0.75, "details": {}}
    all_bboxes = [t.get("bbox", []) for t in tracks if t.get("bbox")]
    if all_bboxes:
        heights = [b[3] for b in all_bboxes if len(b) >= 4]
        if heights:
            max_y = max(heights)
            frame_h = max_y / 0.88
            pressing = [b for b in all_bboxes if len(b) >= 4 and b[3] < frame_h * 0.35]
            if len(pressing) >= 3:
                return {"situation": "HIGH_PRESS", "confidence": 0.7, "details": {"pressing": len(pressing)}}
    return {"situation": "OPEN_PLAY", "confidence": 0.7, "details": {"active_tracks": len(tracks)}}

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
                if duration >= 0.5:
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
        if duration >= 0.5:
            events.append({
                'situation': current_situation,
                'start_frame': current_start_frame,
                'end_frame': last_frame,
                'start_time': round(current_start, 2),
                'end_time': round(last_frame / fps, 2),
                'duration_seconds': round(duration, 2),
            })

    return events
