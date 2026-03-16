"""
services/highlight_service.py

Brick 16 — Automatic highlight detection from pitch-mapped trajectories.
Detects long runs, acceleration bursts, and penetrating runs into the final third.
"""

import json
import logging
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def detect_highlights(
    job_id: str,
    min_run_distance: float = 15.0,
    min_acceleration: float = 8.0,
) -> dict:
    """Scan all tracks for highlight-worthy events."""
    base = Path("temp") / job_id

    # Load pitch map (required for 2D distances)
    pitch_path = base / "pitch" / "pitch_map.json"
    if not pitch_path.exists():
        raise ValueError(f"pitch_map.json not found for job '{job_id}'. Run pitch/map first.")

    with open(pitch_path) as f:
        pitch_data = json.load(f)

    # Load team assignments
    team_path = base / "tracking" / "team_results.json"
    track_to_team = {}
    if team_path.exists():
        with open(team_path) as f:
            raw_team = json.load(f)
        team_list = raw_team["tracks"] if isinstance(raw_team, dict) else raw_team
        for entry in team_list:
            tid = entry.get("trackId")
            team = entry.get("teamId")
            if tid is not None and team is not None:
                track_to_team[int(tid)] = int(team)

    # Get video duration for clamping
    track_path = base / "tracking" / "track_results.json"
    video_duration = 0.0
    if track_path.exists():
        with open(track_path) as f:
            td = json.load(f)
        video_path = td.get("videoPath", "")
        if video_path and os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            fps_val = float(cap.get(cv2.CAP_PROP_FPS))
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if fps_val > 0:
                video_duration = n_frames / fps_val

    fps = _get_fps_from_pitch(pitch_data, base)

    highlights: List[dict] = []

    for player in pitch_data.get("players", []):
        tid = int(player["trackId"])
        team_id = track_to_team.get(tid, -1)
        traj = player.get("trajectory2d", [])

        if len(traj) < 10:
            continue

        # Build numpy arrays: frame_indices, timestamps, x, y
        fi_arr = np.array([int(pt["frameIndex"]) for pt in traj], dtype=np.float64)
        x_arr = np.array([float(pt["x"]) for pt in traj], dtype=np.float64)
        y_arr = np.array([float(pt["y"]) for pt in traj], dtype=np.float64)
        ts_arr = fi_arr / fps

        # ── Long run detection ──────────────────────────────────────────
        _detect_long_runs(
            highlights, tid, team_id,
            fi_arr, ts_arr, x_arr, y_arr,
            fps, min_run_distance, video_duration,
        )

        # ── Acceleration burst ──────────────────────────────────────────
        _detect_acceleration_bursts(
            highlights, tid, team_id,
            fi_arr, ts_arr, x_arr, y_arr,
            fps, min_acceleration, video_duration,
        )

        # ── Penetrating run into final third ────────────────────────────
        _detect_penetrating_runs(
            highlights, tid, team_id,
            fi_arr, ts_arr, x_arr, y_arr,
            fps, video_duration,
        )

    # Sort by score descending
    highlights.sort(key=lambda h: h["score"], reverse=True)

    # Pad times for context and clamp
    for h in highlights:
        h["startSecond"] = max(0.0, h["startSecond"] - 1.5)
        h["endSecond"] = min(video_duration, h["endSecond"] + 2.0) if video_duration > 0 else h["endSecond"] + 2.0

    logger.info("Job %s: detected %d highlights", job_id, len(highlights))

    return {
        "jobId": job_id,
        "highlights": highlights,
    }


# ── Detection functions ──────────────────────────────────────────────────────

def _detect_long_runs(
    highlights, tid, team_id,
    fi_arr, ts_arr, x_arr, y_arr,
    fps, min_distance, video_duration,
):
    """Sliding 3-second window distance check."""
    n = len(ts_arr)
    # Distance between consecutive points
    dx = np.diff(x_arr)
    dy = np.diff(y_arr)
    seg_dist = np.sqrt(dx ** 2 + dy ** 2)

    window_seconds = 3.0
    # Find windows of 3-second spans
    j = 0
    for i in range(n - 1):
        # Advance j until the time span exceeds window
        while j < n - 1 and (ts_arr[j] - ts_arr[i]) < window_seconds:
            j += 1

        if j <= i:
            continue

        total_dist = float(np.sum(seg_dist[i:j]))
        if total_dist >= min_distance:
            score = min(1.0, total_dist / 30.0)
            highlights.append({
                "trackId": tid,
                "teamId": team_id,
                "type": "long_run",
                "startSecond": round(float(ts_arr[i]), 2),
                "endSecond": round(float(ts_arr[min(j, n - 1)]), 2),
                "score": round(score, 3),
                "description": f"Long run of {total_dist:.0f}m",
            })
            # Skip ahead to avoid duplicate overlapping windows
            break


def _detect_acceleration_bursts(
    highlights, tid, team_id,
    fi_arr, ts_arr, x_arr, y_arr,
    fps, min_acceleration, video_duration,
):
    """Detect speed spikes using 5-frame rolling average."""
    n = len(ts_arr)
    if n < 6:
        return

    # Per-frame speed
    dx = np.diff(x_arr)
    dy = np.diff(y_arr)
    dt = np.diff(ts_arr)
    dt[dt == 0] = 1e-6  # avoid division by zero
    speed = np.sqrt(dx ** 2 + dy ** 2) / dt

    # 5-frame rolling average
    kernel = np.ones(5) / 5.0
    if len(speed) >= 5:
        smooth_speed = np.convolve(speed, kernel, mode="valid")
    else:
        smooth_speed = speed

    peak_idx = int(np.argmax(smooth_speed))
    peak_speed = float(smooth_speed[peak_idx])

    if peak_speed >= min_acceleration:
        # Map back to original indices (convolve valid mode shifts by kernel//2)
        offset = 2 if len(speed) >= 5 else 0
        orig_idx = peak_idx + offset
        orig_idx = min(orig_idx, n - 2)

        # Window: 1 second around peak
        start_idx = max(0, orig_idx - int(fps / 2))
        end_idx = min(n - 1, orig_idx + int(fps / 2))

        score = min(1.0, peak_speed / 15.0)
        highlights.append({
            "trackId": tid,
            "teamId": team_id,
            "type": "acceleration_burst",
            "startSecond": round(float(ts_arr[start_idx]), 2),
            "endSecond": round(float(ts_arr[end_idx]), 2),
            "score": round(score, 3),
            "description": f"Acceleration burst to {peak_speed:.1f} m/s",
        })


def _detect_penetrating_runs(
    highlights, tid, team_id,
    fi_arr, ts_arr, x_arr, y_arr,
    fps, video_duration,
):
    """Detect runs from x < 70 to x > 85 within 4 seconds."""
    n = len(ts_arr)
    window_seconds = 4.0

    j = 0
    found = False
    for i in range(n):
        if x_arr[i] >= 70:
            continue  # Must start behind the 70m line

        # Scan forward within 4-second window
        for j in range(i + 1, n):
            if (ts_arr[j] - ts_arr[i]) > window_seconds:
                break
            if x_arr[j] > 85:
                final_x = float(x_arr[j])
                score = 0.7 + min(0.3, (final_x - 85.0) / 20.0)
                highlights.append({
                    "trackId": tid,
                    "teamId": team_id,
                    "type": "penetrating_run",
                    "startSecond": round(float(ts_arr[i]), 2),
                    "endSecond": round(float(ts_arr[j]), 2),
                    "score": round(score, 3),
                    "description": "Run into final third",
                })
                found = True
                break
        if found:
            break


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_fps_from_pitch(pitch_data: dict, base: Path) -> float:
    """Get FPS from the source video via track_results.json."""
    track_path = base / "tracking" / "track_results.json"
    if track_path.exists():
        with open(track_path) as f:
            td = json.load(f)
        video_path = td.get("videoPath", "")
        if video_path and os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            if fps > 0:
                return fps
    return 30.0
