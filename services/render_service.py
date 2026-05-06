"""
services/render_service.py

Brick 13 — burns player bounding boxes + team colours into video frames
and writes temp/{jobId}/render/output.mp4
"""

import json
import os
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from services.export_service import build_export


TEAM_BGR = {
    0: (255, 100,   0),   # Blue team (BGR: cyan)
    1: (  0,  50, 255),   # Orange team (BGR: orange)
    2: (  0, 255,   0),   # goalkeeper — bright green
    -2: (128, 128, 128),  # FIX 2: Official — grey
}
DEFAULT_BGR = (200, 200, 200)
OFFICIAL_BGR = (128, 128, 128)  # FIX 2: Grey for referees/linesmen

# FIX 4: Tracking state visualization colors
CONFIRMED_COLOR = (0, 255, 0)   # Green — confirmed detection
PREDICTED_COLOR = (0, 255, 255) # Cyan — Kalman prediction
STALE_COLOR = (128, 128, 128)   # Gray — stale track
BALL_DETECTED_COLOR = (255, 255, 255)  # White — YOLO detection
BALL_PREDICTED_COLOR = (128, 128, 128) # Gray — predicted

MINIMAP_W = 200
MINIMAP_H = 130
MINIMAP_MARGIN = 10
FONT = cv2.FONT_HERSHEY_SIMPLEX


def run_render(job_id: str, include_minimap: bool = False) -> dict:
    export = build_export(job_id)

    video_meta = export["videoMeta"]
    fps        = video_meta["fps"] if video_meta["fps"] > 0 else 30.0
    out_w      = video_meta["width"]
    out_h      = video_meta["height"]

    track_path = Path("temp") / job_id / "tracking" / "track_results.json"
    with open(track_path) as f:
        track_data = json.load(f)
    video_path = track_data.get("videoPath", "")

    if not video_path or not os.path.exists(video_path):
        raise ValueError(f"Source video not found: {video_path!r}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    is_portrait = raw_h > raw_w

    out_dir = Path("temp") / job_id / "render"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / "output.mp4")

    # Try avc1 (H.264) first; fall back to mp4v if unavailable
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

    # ── Build per-track keyframe list ────────────────────────────────────────
    # Collect every (frameIndex, bbox, teamId, pitchX, pitchY) observation per track.
    _track_keyframes: dict = {}  # trackId -> sorted list of (fi, bbox, meta)
    _sorted_frames = sorted(export["frames"], key=lambda e: e["frameIndex"])
    for frame_entry in _sorted_frames:
        for p in frame_entry["players"]:
            tid = p["trackId"]
            if tid not in _track_keyframes:
                _track_keyframes[tid] = []
            _track_keyframes[tid].append({
                "fi": frame_entry["frameIndex"],
                "bbox": list(p["bbox"]),
                "teamId": p.get("teamId", -1),
                "pitchX": p.get("pitchX"),
                "pitchY": p.get("pitchY"),
            })

    # ── Smooth each track's raw keyframes with EMA (alpha=0.35) ──────────────
    _ema_alpha = 0.35
    for tid, kfs in _track_keyframes.items():
        smoothed = list(kfs[0]["bbox"])
        kfs[0]["bbox"] = smoothed[:]
        for kf in kfs[1:]:
            smoothed = [
                _ema_alpha * kf["bbox"][i] + (1 - _ema_alpha) * smoothed[i]
                for i in range(4)
            ]
            kf["bbox"] = smoothed[:]

    # ── Interpolate / hold-last between keyframes ─────────────────────────────
    # For each video frame, every active track gets a bbox via linear interpolation
    # between its two surrounding keyframes (or hold-last after final keyframe).
    _total_frames = export["videoMeta"]["frameCount"]
    frame_lookup: dict = {}  # frameIndex -> {"players": [...], "ball": None}

    for fi in range(_total_frames):
        frame_lookup[fi] = {"players": [], "ball": None}

    for tid, kfs in _track_keyframes.items():
        if not kfs:
            continue
        fi_first = kfs[0]["fi"]
        fi_last  = kfs[-1]["fi"]
        n_kfs    = len(kfs)

        for fi in range(fi_first, fi_last + 1):
            # Binary-search for the last keyframe with fi <= current frame
            lo, hi = 0, n_kfs - 1
            prev_idx = 0
            while lo <= hi:
                mid = (lo + hi) // 2
                if kfs[mid]["fi"] <= fi:
                    prev_idx = mid
                    lo = mid + 1
                else:
                    hi = mid - 1

            prev_kf = kfs[prev_idx]
            # If there's a next keyframe, interpolate; otherwise hold last
            if prev_idx + 1 < n_kfs:
                next_kf = kfs[prev_idx + 1]
                span = next_kf["fi"] - prev_kf["fi"]
                t = (fi - prev_kf["fi"]) / span if span > 0 else 0.0
                bbox = [
                    prev_kf["bbox"][i] + t * (next_kf["bbox"][i] - prev_kf["bbox"][i])
                    for i in range(4)
                ]
            else:
                bbox = prev_kf["bbox"]

            frame_lookup[fi]["players"].append({
                "trackId": tid,
                "teamId":  prev_kf["teamId"],
                "bbox":    bbox,
                "pitchX":  prev_kf.get("pitchX"),
                "pitchY":  prev_kf.get("pitchY"),
            })

    # ── Fill ball from export ─────────────────────────────────────────────────
    for frame_entry in export["frames"]:
        fi = frame_entry["frameIndex"]
        if fi in frame_lookup and frame_entry.get("ball"):
            frame_lookup[fi]["ball"] = frame_entry["ball"]

    frames_rendered = 0
    frame_idx = 0
    ball_trail = deque(maxlen=5)  # stores recent ball bbox centers as (cx, cy)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if is_portrait:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        frame_data = frame_lookup.get(frame_idx, {})
        players = frame_data.get("players", [])
        ball = frame_data.get("ball")

        # FIX 4: Get frame metadata for validity display
        frame_meta = frame_data.get("frame_metadata", {})

        if players:
            frame = _draw_players(frame, players, include_minimap, out_w, out_h)
        if ball and ball.get("bbox"):
            bbox = ball["bbox"]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            ball_trail.append((cx, cy))
            # FIX 4: Pass ball source and staleness info
            ball_source = ball.get("source", "unknown")
            frames_since_detection = ball.get("frames_since_detection", 0)
            frame = _draw_ball(frame, bbox, list(ball_trail), ball_source, frames_since_detection)

        # FIX 4: Draw frame status if metadata available
        if frame_meta:
            frame = _draw_frame_status(frame, frame_meta, out_w, out_h)

        writer.write(frame)
        frames_rendered += 1
        frame_idx += 1

    cap.release()
    writer.release()

    # Upload to Supabase if configured
    render_url = None
    try:
        from services.storage_service import upload_file_from_path, BUCKET_RENDERS
        render_url = upload_file_from_path(
            BUCKET_RENDERS,
            "{}/output.mp4".format(job_id),
            out_path
        )
    except Exception:
        pass

    return {
        "jobId":          job_id,
        "outputPath":     out_path,
        "framesRendered": frames_rendered,
        "fps":            fps,
        "render_url":     render_url,
    }


def _draw_players(frame, players, include_minimap: bool, frame_w: int, frame_h: int):
    """
    Draw players with FIX 4 state visualization:
    - Confirmed detection (solid circle): YOLO matched this frame
    - Kalman prediction (dashed outline, 50% opacity): no YOLO match
    - Stale track (grey, 30% opacity): not seen for >10 frames (hidden after 20)
    """
    overlay = frame.copy()

    for p in players:
        team_id = p.get("teamId", -1)
        is_official = p.get("is_official", False)  # FIX 2: skip fill for officials
        x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
        x1 = max(0, min(x1, frame_w - 1))
        x2 = max(0, min(x2, frame_w - 1))
        y1 = max(0, min(y1, frame_h - 1))
        y2 = max(0, min(y2, frame_h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        if is_official:
            # FIX 2: Don't fill officials, just outline them
            continue
        color = TEAM_BGR.get(team_id, DEFAULT_BGR)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    frame = cv2.addWeighted(overlay, 0.40, frame, 0.60, 0)

    for p in players:
        track_id = p["trackId"]
        team_id  = p.get("teamId", -1)
        is_official = p.get("is_official", False)  # FIX 2: handle officials
        x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
        x1 = max(0, min(x1, frame_w - 1))
        x2 = max(0, min(x2, frame_w - 1))
        y1 = max(0, min(y1, frame_h - 1))
        y2 = max(0, min(y2, frame_h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        # FIX 2: Draw officials as thin grey box without label
        if is_official:
            cv2.rectangle(frame, (x1, y1), (x2, y2), OFFICIAL_BGR, 2)
            label = "REF"
            (lw, lh), baseline = cv2.getTextSize(label, FONT, 0.4, 1)
            label_x = x1
            label_y = max(y1 - 4, lh + 2)
            cv2.rectangle(frame,
                          (label_x, label_y - lh - baseline),
                          (label_x + lw + 2, label_y + baseline),
                          OFFICIAL_BGR, -1)
            cv2.putText(frame, label, (label_x + 1, label_y),
                        FONT, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            continue

        color = TEAM_BGR.get(team_id, DEFAULT_BGR)

        # FIX 4: Draw bbox based on tracking state
        is_stale = p.get("is_stale", False)
        is_predicted = p.get("is_predicted", False)

        if is_stale:
            # Stale track: grey dashed, 30% opacity
            cv2.rectangle(frame, (x1, y1), (x2, y2), STALE_COLOR, 1, cv2.LINE_4)
            frame = cv2.addWeighted(frame, 0.7, frame, 0.3, 0)
        elif is_predicted:
            # Kalman prediction: dashed outline, 50% opacity
            cv2.rectangle(frame, (x1, y1), (x2, y2), PREDICTED_COLOR, 1, cv2.LINE_4)
            overlay_pred = frame.copy()
            cv2.putText(overlay_pred, "?", (x1 + 3, y1 + 15),
                       FONT, 0.6, PREDICTED_COLOR, 1, cv2.LINE_AA)
            frame = cv2.addWeighted(overlay_pred, 0.5, frame, 0.5, 0)
        else:
            # Confirmed detection: solid rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        display_id = p.get("displayId")
        if not display_id:
            if p.get("identity_valid", False):
                display_id = p.get("playerId") or f"P{track_id}"
            elif "rawTrackId" in p:
                display_id = f"U T{p['rawTrackId']}"
            else:
                display_id = f"T{track_id}"
        label = str(display_id)
        font_scale = 0.45
        thickness  = 1
        (lw, lh), baseline = cv2.getTextSize(label, FONT, font_scale, thickness)
        label_x = x1
        label_y = max(y1 - 4, lh + 2)
        cv2.rectangle(frame,
                      (label_x, label_y - lh - baseline),
                      (label_x + lw + 2, label_y + baseline),
                      color, -1)
        cv2.putText(frame, label, (label_x + 1, label_y),
                    FONT, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    if include_minimap:
        frame = _draw_minimap(frame, players, frame_w, frame_h)

    # FIX 4: Draw legend
    frame = _draw_legend(frame, frame_h, frame_w)

    return frame


def _draw_ball(frame, bbox, trail=None, ball_source=None, frames_since_detection=0):
    """
    FIX 4: Draw ball with source-based visualization:
    - YOLO detection: white filled circle, radius 6px
    - Hough candidate: yellow outline circle, radius 6px
    - Kalman prediction: grey dashed circle, radius 4px
    - If prediction > 10 frames old: don't show ball at all
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # FIX 4: Hide ball if prediction is too stale (>10 frames without detection)
    if ball_source == "kalman_prediction" and frames_since_detection > 10:
        return frame

    # Draw fading trail — oldest first, radii 6..2, decreasing opacity
    if trail and len(trail) > 1:
        trail_radii = [6, 5, 4, 3, 2]
        trail_pts = trail[:-1]
        for i, (tx, ty) in enumerate(trail_pts):
            idx = len(trail_pts) - 1 - i
            radius = trail_radii[min(idx, len(trail_radii) - 1)]
            alpha = 0.3 + 0.1 * (len(trail_pts) - 1 - idx)
            alpha = min(alpha, 0.7)
            overlay = frame.copy()
            cv2.circle(overlay, (int(tx), int(ty)), radius, (0, 255, 255), -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # FIX 4: Draw ball based on source
    if ball_source == "yolo":
        # YOLO: white filled circle, radius 6px
        cv2.circle(frame, (cx, cy), 6, BALL_DETECTED_COLOR, -1)
        cv2.circle(frame, (cx, cy), 6, (0, 0, 0), 1)
    elif ball_source == "hough_candidate":
        # Hough: yellow outline circle, radius 6px
        cv2.circle(frame, (cx, cy), 6, (0, 255, 255), 2)
    elif ball_source == "kalman_prediction":
        # Kalman: grey dashed circle, radius 4px
        cv2.circle(frame, (cx, cy), 4, BALL_PREDICTED_COLOR, 1, cv2.LINE_4)
    else:
        # Fallback: white filled circle
        cv2.circle(frame, (cx, cy), 6, BALL_DETECTED_COLOR, -1)
        cv2.circle(frame, (cx, cy), 6, (0, 0, 0), 1)

    return frame


def _draw_legend(frame, frame_h: int, frame_w: int):
    """FIX 4: Draw legend box in bottom-left corner with FIX 2 official marker."""
    legend_y_start = frame_h - 150
    legend_x_start = 10
    legend_items = [
        ("Confirmed", CONFIRMED_COLOR),
        ("Predicted", PREDICTED_COLOR),
        ("Official", OFFICIAL_BGR),  # FIX 2: Add official marker
        ("Ball (YOLO)", BALL_DETECTED_COLOR),
        ("Ball (Pred)", BALL_PREDICTED_COLOR),
    ]

    for i, (label, color) in enumerate(legend_items):
        y = legend_y_start + i * 28
        # Draw colored dot
        cv2.circle(frame, (legend_x_start + 8, y + 8), 4, color, -1)
        # Draw label
        cv2.putText(frame, label, (legend_x_start + 20, y + 12),
                   FONT, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def _draw_frame_status(frame, frame_data, frame_w: int, frame_h: int):
    """FIX 4: Draw frame validity status in top-right corner."""
    is_valid = frame_data.get("analysis_valid", True)
    scene_cut = frame_data.get("scene_cut", False)

    status_x = frame_w - 150
    status_y = 20

    # Draw validity indicator
    if is_valid:
        cv2.circle(frame, (status_x, status_y), 6, (0, 255, 0), -1)
    else:
        cv2.circle(frame, (status_x, status_y), 6, (0, 0, 255), -1)
        cv2.putText(frame, "NON-PITCH", (status_x + 15, status_y + 5),
                   FONT, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    # Draw scene cut indicator
    if scene_cut:
        cv2.putText(frame, "SCENE CUT", (status_x - 50, status_y + 30),
                   FONT, 0.5, (0, 165, 255), 2, cv2.LINE_AA)

    return frame


def _draw_minimap(frame, players, frame_w: int, frame_h: int):
    mm = np.zeros((MINIMAP_H, MINIMAP_W, 3), dtype=np.uint8)
    mm[:] = (34, 100, 34)
    cv2.rectangle(mm, (2, 2), (MINIMAP_W - 3, MINIMAP_H - 3), (255, 255, 255), 1)
    cx = MINIMAP_W // 2
    cv2.line(mm, (cx, 2), (cx, MINIMAP_H - 3), (255, 255, 255), 1)
    cv2.circle(mm, (cx, MINIMAP_H // 2), 15, (255, 255, 255), 1)

    for p in players:
        px = p.get("pitchX")
        py = p.get("pitchY")
        if px is None or py is None:
            continue
        team_id = p.get("teamId", -1)
        color   = TEAM_BGR.get(team_id, DEFAULT_BGR)
        dot_x = int(px / 105.0 * (MINIMAP_W - 6)) + 3
        dot_y = int(py / 68.0  * (MINIMAP_H - 6)) + 3
        dot_x = max(3, min(dot_x, MINIMAP_W - 4))
        dot_y = max(3, min(dot_y, MINIMAP_H - 4))
        cv2.circle(mm, (dot_x, dot_y), 4, color, -1)
        cv2.circle(mm, (dot_x, dot_y), 4, (255, 255, 255), 1)

    x_off = max(0, frame_w - MINIMAP_W - MINIMAP_MARGIN)
    y_off = max(0, frame_h - MINIMAP_H - MINIMAP_MARGIN)
    roi = frame[y_off:y_off + MINIMAP_H, x_off:x_off + MINIMAP_W]
    if roi.shape[:2] == (MINIMAP_H, MINIMAP_W):
        blended = cv2.addWeighted(mm, 0.85, roi, 0.15, 0)
        frame[y_off:y_off + MINIMAP_H, x_off:x_off + MINIMAP_W] = blended

    return frame
