"""Pick a 5-10 second clip window from a job's outputs for the Performance
Zone renderer. Two strategies:
  1. event-anchored: find the densest cluster of pass/carry/shot events and
     widen ±N frames.
  2. ball-motion fallback: when there are no events (or pitch_map ball trail
     missing), use the highest-mean-pixel-motion window.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple


def _load(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _event_density(events: List[dict], frames_processed: int, fps: float, win_s: float = 8.0) -> Optional[Tuple[int, int]]:
    if not events or not frames_processed:
        return None
    win = int(round(win_s * fps))
    if win < 10:
        win = min(frames_processed, 100)
    # collect frame indices
    fis: List[int] = []
    for e in events:
        f = e.get("frame", e.get("frameIndex"))
        if f is None:
            continue
        try:
            fis.append(int(f))
        except Exception:
            continue
    if not fis:
        return None
    fis.sort()
    # sliding window: find the start that maximises count(fis ∈ [start, start+win])
    best_lo = fis[0]
    best_count = 1
    j = 0
    for i in range(len(fis)):
        while j < len(fis) and fis[j] - fis[i] <= win:
            j += 1
        c = j - i
        if c > best_count:
            best_count = c
            best_lo = fis[i]
    lo = max(0, best_lo - win // 4)
    hi = min(frames_processed - 1, lo + win)
    return lo, hi


def _ball_motion_window(pitch_data: dict, frames_processed: int, fps: float, win_s: float = 8.0) -> Optional[Tuple[int, int]]:
    """Pick the window with the largest mean ball pixel motion."""
    if not pitch_data:
        return None
    win = int(round(win_s * fps))
    ball_traj = None
    for entry in pitch_data.get("players", []):
        if entry.get("is_ball"):
            ball_traj = entry.get("trajectory2d") or []
            break
    if not ball_traj or len(ball_traj) < 4:
        return None
    by_frame = {}
    for pt in ball_traj:
        try:
            by_frame[int(pt["frameIndex"])] = (float(pt["x"]), float(pt["y"]))
        except Exception:
            continue
    fis = sorted(by_frame.keys())
    if len(fis) < 4:
        return None
    speeds = []
    for i in range(1, len(fis)):
        f0, f1 = fis[i - 1], fis[i]
        x0, y0 = by_frame[f0]
        x1, y1 = by_frame[f1]
        dt = max(1, f1 - f0)
        speeds.append((f0, ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / dt))
    if not speeds:
        return None
    # rolling sum within `win` frames
    best = (speeds[0][0], 0.0)
    j = 0
    cumsum = 0.0
    window = []
    for i, (f, sp) in enumerate(speeds):
        window.append((f, sp))
        cumsum += sp
        while window and (f - window[0][0]) > win:
            cumsum -= window.pop(0)[1]
        if cumsum > best[1]:
            best = (window[0][0], cumsum)
    lo = max(0, int(best[0]))
    hi = min(frames_processed - 1, lo + win)
    return lo, hi


def pick_clip_window(
    job_id: str,
    *,
    win_s: float = 8.0,
    fps: float = 25.0,
) -> Optional[Tuple[int, int]]:
    """Best 5-10s window for `job_id`. Returns (frame_lo, frame_hi) or None."""
    base = Path(f"temp/{job_id}")
    track = _load(base / "tracking" / "track_results.json")
    pitch = _load(base / "pitch" / "pitch_map.json")
    events = _load(base / "events" / "event_timeline.json")

    frames_processed = 0
    if track:
        frames_processed = int(
            track.get("framesProcessed")
            or track.get("total_frames")
            or len(track.get("frames", []))
            or 0
        )
    if not frames_processed:
        return None

    # 1. Event-anchored
    if events:
        all_events: List[dict] = []
        for key in ("pass_events", "shot_events", "tackle_events", "turnover_events", "carry_events", "events"):
            arr = events.get(key) if isinstance(events, dict) else None
            if isinstance(arr, list):
                all_events.extend(arr)
        win = _event_density(all_events, frames_processed, fps, win_s)
        if win:
            return win

    # 2. Ball motion fallback
    win = _ball_motion_window(pitch or {}, frames_processed, fps, win_s)
    if win:
        return win

    # 3. Last resort: middle of clip
    mid = frames_processed // 2
    half = int(round(win_s * fps / 2))
    return max(0, mid - half), min(frames_processed - 1, mid + half)
