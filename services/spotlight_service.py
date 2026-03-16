"""
services/spotlight_service.py

Brick 14 — Player selection (select_player)
Brick 15 — Spotlight rendering (render_spotlight)
Brick 17 — Clip export (export_clip)
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ── Brick 14: Player selection ──────────────────────────────────────────────

def select_player(
    job_id: str,
    track_id: int,
    start_second: float,
    end_second: float,
) -> dict:
    """Return trajectory frames for a single player within a time window."""
    base = Path("temp") / job_id

    track_path = base / "tracking" / "track_results.json"
    if not track_path.exists():
        raise ValueError(f"track_results.json not found for job '{job_id}'")

    with open(track_path) as f:
        track_data = json.load(f)

    fps = _get_fps(track_data.get("videoPath", ""))

    # Find the target track
    target_track = None
    for t in track_data.get("tracks", []):
        if int(t["trackId"]) == track_id:
            target_track = t
            break
    if target_track is None:
        raise ValueError(f"trackId {track_id} not found in job '{job_id}'")

    # Load pitch data if available
    pitch_lookup = _load_pitch_lookup(base)

    # Filter trajectory points by time window
    frames: List[dict] = []
    for pt in target_track.get("trajectory", []):
        fi = int(pt["frameIndex"])
        ts = float(pt.get("timestampSeconds", fi / fps))
        if ts < start_second or ts > end_second:
            continue

        pitch_x, pitch_y = pitch_lookup.get((track_id, fi), (None, None))

        frames.append({
            "frameIndex": fi,
            "timestampSeconds": round(ts, 3),
            "bbox": [float(v) for v in pt["bbox"]],
            "pitchX": pitch_x,
            "pitchY": pitch_y,
        })

    duration = end_second - start_second
    if frames:
        actual_start = frames[0]["timestampSeconds"]
        actual_end = frames[-1]["timestampSeconds"]
        duration = actual_end - actual_start

    return {
        "jobId": job_id,
        "trackId": track_id,
        "frames": frames,
        "totalFrames": len(frames),
        "durationSeconds": round(duration, 3),
    }


# ── Brick 15: Spotlight rendering ───────────────────────────────────────────

def render_spotlight(
    job_id: str,
    track_id: int,
    start_second: float,
    end_second: float,
    effect_style: str = "glow",
) -> dict:
    """Render a video clip spotlighting one player with the chosen effect."""
    base = Path("temp") / job_id

    track_path = base / "tracking" / "track_results.json"
    if not track_path.exists():
        raise ValueError(f"track_results.json not found for job '{job_id}'")

    with open(track_path) as f:
        track_data = json.load(f)

    video_path = track_data.get("videoPath", "")
    if not video_path or not os.path.exists(video_path):
        raise ValueError(f"Source video not found: {video_path!r}")

    fps = _get_fps(video_path)

    # Build interpolated bbox lookup for the target track
    keyframes = _build_track_keyframes(track_data, track_id, fps)
    if not keyframes:
        raise ValueError(f"trackId {track_id} has no trajectory points")

    bbox_lookup = _interpolate_keyframes(keyframes)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    is_portrait = raw_h > raw_w

    if is_portrait:
        out_w, out_h = raw_h, raw_w
    else:
        out_w, out_h = raw_w, raw_h

    start_frame = int(start_second * fps)
    end_frame = int(end_second * fps)

    out_dir = base / "spotlight"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / f"{track_id}_{effect_style}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames_rendered = 0
    for fi in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break

        if is_portrait:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        bbox = bbox_lookup.get(fi)

        if bbox is not None:
            frame = _apply_spotlight_effect(frame, bbox, effect_style, fi)
        else:
            # Player not visible — just dim the whole frame
            frame = (frame.astype(np.float32) * 0.4).astype(np.uint8)

        writer.write(frame)
        frames_rendered += 1

    cap.release()
    writer.release()

    return {
        "jobId": job_id,
        "trackId": track_id,
        "outputPath": out_path,
        "framesRendered": frames_rendered,
    }


# ── Brick 17: Clip export ───────────────────────────────────────────────────

def export_clip(
    job_id: str,
    track_id: int,
    start_second: float,
    end_second: float,
    effect_style: str = "glow",
    include_slowmo: bool = False,
    slowmo_section: float = 0.5,
) -> dict:
    """Render a clip with spotlight effect, optional slow-mo, and timestamp."""
    base = Path("temp") / job_id

    track_path = base / "tracking" / "track_results.json"
    if not track_path.exists():
        raise ValueError(f"track_results.json not found for job '{job_id}'")

    with open(track_path) as f:
        track_data = json.load(f)

    video_path = track_data.get("videoPath", "")
    if not video_path or not os.path.exists(video_path):
        raise ValueError(f"Source video not found: {video_path!r}")

    fps = _get_fps(video_path)

    keyframes = _build_track_keyframes(track_data, track_id, fps)
    if not keyframes:
        raise ValueError(f"trackId {track_id} has no trajectory points")

    bbox_lookup = _interpolate_keyframes(keyframes)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    is_portrait = raw_h > raw_w
    out_w, out_h = (raw_h, raw_w) if is_portrait else (raw_w, raw_h)

    start_frame = int(start_second * fps)
    end_frame = int(end_second * fps)
    total_clip_frames = end_frame - start_frame + 1

    # Slow-mo range: middle portion of the clip
    if include_slowmo and total_clip_frames > 4:
        margin = (1.0 - slowmo_section) / 2.0
        slowmo_start = start_frame + int(total_clip_frames * margin)
        slowmo_end = start_frame + int(total_clip_frames * (1.0 - margin))
    else:
        slowmo_start = -1
        slowmo_end = -1

    out_dir = base / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / f"{track_id}_{int(start_second)}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames_written = 0

    for fi in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break

        if is_portrait:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        bbox = bbox_lookup.get(fi)

        if bbox is not None:
            frame = _apply_spotlight_effect(frame, bbox, effect_style, fi)
        else:
            frame = (frame.astype(np.float32) * 0.4).astype(np.uint8)

        # Burn timestamp in bottom-left corner
        ts = fi / fps
        ts_text = f"{int(ts // 60):02d}:{ts % 60:05.2f}"
        cv2.putText(frame, ts_text, (10, out_h - 12),
                    FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        writer.write(frame)
        frames_written += 1

        # Slow-mo: write same frame a second time
        if slowmo_start <= fi <= slowmo_end:
            writer.write(frame)
            frames_written += 1

    cap.release()
    writer.release()

    duration = frames_written / fps if fps > 0 else 0.0

    return {
        "jobId": job_id,
        "trackId": track_id,
        "outputPath": out_path,
        "durationSeconds": round(duration, 3),
    }


# ── Private helpers ──────────────────────────────────────────────────────────

def _get_fps(video_path: str) -> float:
    if not video_path or not os.path.exists(video_path):
        return 30.0
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return fps if fps > 0 else 30.0


def _load_pitch_lookup(base: Path) -> dict:
    """Build (trackId, frameIndex) → (pitchX, pitchY) mapping."""
    pitch_path = base / "pitch" / "pitch_map.json"
    lookup = {}
    if pitch_path.exists():
        with open(pitch_path) as f:
            pitch_data = json.load(f)
        for player in pitch_data.get("players", []):
            tid = int(player["trackId"])
            for pt in player.get("trajectory2d", []):
                key = (tid, int(pt["frameIndex"]))
                lookup[key] = (float(pt["x"]), float(pt["y"]))
    return lookup


def _build_track_keyframes(
    track_data: dict, track_id: int, fps: float,
) -> List[dict]:
    """Extract sorted keyframe list for one track."""
    for t in track_data.get("tracks", []):
        if int(t["trackId"]) != track_id:
            continue
        kfs = []
        for pt in t.get("trajectory", []):
            fi = int(pt["frameIndex"])
            kfs.append({
                "fi": fi,
                "bbox": [float(v) for v in pt["bbox"]],
            })
        # EMA smooth (alpha=0.35)
        if len(kfs) >= 2:
            smoothed = list(kfs[0]["bbox"])
            for kf in kfs[1:]:
                smoothed = [
                    0.35 * kf["bbox"][i] + 0.65 * smoothed[i]
                    for i in range(4)
                ]
                kf["bbox"] = smoothed[:]
        return kfs
    return []


def _interpolate_keyframes(keyframes: List[dict]) -> dict:
    """Build frameIndex → [x1, y1, x2, y2] lookup with linear interpolation."""
    if not keyframes:
        return {}

    lookup = {}
    fi_first = keyframes[0]["fi"]
    fi_last = keyframes[-1]["fi"]
    n = len(keyframes)

    for fi in range(fi_first, fi_last + 1):
        # Binary search for last keyframe with fi <= current
        lo, hi = 0, n - 1
        prev_idx = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if keyframes[mid]["fi"] <= fi:
                prev_idx = mid
                lo = mid + 1
            else:
                hi = mid - 1

        prev_kf = keyframes[prev_idx]
        if prev_idx + 1 < n:
            next_kf = keyframes[prev_idx + 1]
            span = next_kf["fi"] - prev_kf["fi"]
            t = (fi - prev_kf["fi"]) / span if span > 0 else 0.0
            bbox = [
                prev_kf["bbox"][i] + t * (next_kf["bbox"][i] - prev_kf["bbox"][i])
                for i in range(4)
            ]
        else:
            bbox = prev_kf["bbox"]

        lookup[fi] = bbox

    return lookup


def _apply_spotlight_effect(
    frame: np.ndarray,
    bbox: list,
    style: str,
    frame_idx: int,
) -> np.ndarray:
    """Dim the frame, restore player region, draw the spotlight effect."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    if x2 <= x1 or y2 <= y1:
        return frame

    # Save the player region at full brightness
    player_roi = frame[y1:y2, x1:x2].copy()

    # Dim the entire frame
    dimmed = (frame.astype(np.float32) * 0.4).astype(np.uint8)

    # Restore player region
    dimmed[y1:y2, x1:x2] = player_roi

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    if style == "circle":
        radius = max((x2 - x1), (y2 - y1)) // 2 + 8
        cv2.circle(dimmed, (cx, cy), radius, (255, 255, 255), 3)

    elif style == "glow":
        radius = max((x2 - x1), (y2 - y1)) // 2 + 15
        overlay = dimmed.copy()
        # Outer glow — large yellow circle, blended softly
        cv2.circle(overlay, (cx, cy), radius, (0, 230, 255), -1)
        cv2.addWeighted(overlay, 0.15, dimmed, 0.85, 0, dimmed)
        # Inner glow — smaller, brighter
        inner_r = radius * 2 // 3
        overlay2 = dimmed.copy()
        cv2.circle(overlay2, (cx, cy), inner_r, (0, 255, 255), -1)
        cv2.addWeighted(overlay2, 0.20, dimmed, 0.80, 0, dimmed)
        # Bright white center ring
        cv2.circle(dimmed, (cx, cy), radius, (0, 200, 255), 2)

    elif style == "arrow":
        # Animated arrow that bounces every 10 frames
        bounce = int(8 * abs((frame_idx % 20) - 10) / 10.0)
        arrow_tip_y = y1 - 10 - bounce
        arrow_base_y = arrow_tip_y - 25

        # Arrow stem
        cv2.line(dimmed, (cx, arrow_base_y), (cx, arrow_tip_y),
                 (0, 255, 255), 3)

        # Arrow head (filled triangle)
        head_w = 12
        pts = np.array([
            [cx, arrow_tip_y + 5],
            [cx - head_w, arrow_tip_y - 10],
            [cx + head_w, arrow_tip_y - 10],
        ], dtype=np.int32)
        cv2.fillPoly(dimmed, [pts], (0, 255, 255))

    return dimmed
