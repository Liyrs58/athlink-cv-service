#!/usr/bin/env python3
"""
Smooth full-FPS annotated render from stride-5 identity decisions.

- Identity runs at stride=5 (high confidence, sparse)
- Rendering interpolates at stride=1 (all frames, visual only)
- No new identities created during render
- No identity state changes during render
- Smooth motion, fade/hold rules, proper labeling
"""

import os
import json
import sys
from pathlib import Path
from collections import defaultdict
import time

import cv2
import numpy as np


def smoothstep(t):
    """Smooth interpolation: ease in/out."""
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    """Linear interpolation."""
    return a + (b - a) * t


def bbox_to_center_size(bbox):
    """Convert [x1, y1, x2, y2] to (cx, cy, w, h)."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return cx, cy, w, h


def center_size_to_bbox(cx, cy, w, h):
    """Convert (cx, cy, w, h) to [x1, y1, x2, y2]."""
    return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]


def get_render_key(pid, track_id, identity_state):
    """Decide visual key: prefer P-ID for locked/revived, else T-ID."""
    if pid and identity_state in ("locked", "revived"):
        return pid  # "P1", "P2", etc.
    return f"T{track_id}"


def render_smooth_tracking(
    video_path: str,
    tracking_results,
    identity_metrics: dict,
    job_id: str,
    output_dir: str = None,
    identity_frame_stride: int = 5,
    render_stride: int = 1,
    max_hold_raw_frames: int = 4,
    max_interp_gap_raw_frames: int = 10,
    ema_alpha: float = 0.25,
    show_unknown: bool = True,
    show_provisional: bool = True,
    show_trails: bool = True,
    trail_length: int = 10,
    show_hud: bool = True,
):
    """
    Render smooth full-FPS video from stride-5 tracking.

    tracking_results: list of frame dicts from run_tracking()
    identity_metrics: dict with locks_created, lock_retention_rate, etc.
    """
    if output_dir is None:
        output_dir = f"temp/{job_id}"
    os.makedirs(output_dir, exist_ok=True)

    # Open source video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"✗ Cannot open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Source: {width}x{height} @ {fps:.1f} fps, {total_frames} frames")

    # Output video
    output_video = f"{output_dir}/annotated_tracking_fullfps_smooth.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"✗ Cannot create video writer: {output_video}")
        sys.exit(1)

    # === Build render timeline ===
    # Map: key (P-ID or T-ID) -> list of observations
    render_observations = defaultdict(list)
    frame_is_sampled = {}

    for frame_idx, frame_data in enumerate(tracking_results):
        raw_frame_idx = frame_idx * identity_frame_stride
        frame_is_sampled[raw_frame_idx] = True

        if "players" not in frame_data:
            continue

        for player in frame_data["players"]:
            tid = player.get("rawTrackId")
            pid = player.get("playerId")  # "P1", "P2", etc., or None
            source = player.get("assignment_source", "unassigned")  # "locked", "revived", etc.
            bbox = player.get("bbox", [0, 0, 0, 0])
            team_id = player.get("team_id", -1)
            confidence = player.get("identity_confidence", 0.0)

            if tid is None:
                continue

            # Map assignment_source to state name
            if source == "locked":
                state = "locked"
            elif source == "revived":
                state = "revived"
            elif source == "provisional":
                state = "provisional"
            else:
                state = "unknown"

            # Only render if state is visible
            if state == "unknown" and not show_unknown:
                continue
            if state == "provisional" and not show_provisional:
                continue

            # Decide visual key
            key = get_render_key(pid, tid, state)

            render_observations[key].append({
                "frame": raw_frame_idx,
                "bbox": bbox,
                "track_id": tid,
                "pid": pid,
                "state": state,
                "team_id": team_id,
                "confidence": confidence,
            })

    # Sort observations by frame
    for key in render_observations:
        render_observations[key].sort(key=lambda x: x["frame"])

    print(f"Built render timeline: {len(render_observations)} keys, {sum(len(v) for v in render_observations.values())} observations")

    # === Render loop ===
    last_render_bbox = {}  # key -> last rendered bbox for EMA smoothing
    frame_idx = 0
    interpolated_count = 0
    held_count = 0
    hidden_count = 0
    new_identities_created = 0
    warned = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame is None or frame.size == 0:
                frame_idx += 1
                out.write(np.zeros((height, width, 3), dtype=np.uint8))
                hidden_count += 1
                continue

            # Find observations for this frame
            current_boxes = {}  # key -> {bbox, state, team_id, pid, track_id, confidence}

            for key, observations in render_observations.items():
                # Find prev and next observations
                prev_obs = None
                next_obs = None

                for obs in observations:
                    if obs["frame"] <= frame_idx:
                        prev_obs = obs
                    if obs["frame"] > frame_idx:
                        next_obs = obs
                        break

                if prev_obs is None and next_obs is None:
                    continue

                # Decide bbox
                bbox = None
                render_mode = "unknown"

                if prev_obs is not None and next_obs is not None:
                    # Both exist — interpolate
                    gap = next_obs["frame"] - prev_obs["frame"]
                    if gap <= max_interp_gap_raw_frames:
                        alpha = (frame_idx - prev_obs["frame"]) / gap
                        alpha_smooth = smoothstep(alpha)

                        # Interpolate center/size for smoother motion
                        cx1, cy1, w1, h1 = bbox_to_center_size(prev_obs["bbox"])
                        cx2, cy2, w2, h2 = bbox_to_center_size(next_obs["bbox"])

                        cx = lerp(cx1, cx2, alpha_smooth)
                        cy = lerp(cy1, cy2, alpha_smooth)
                        w = lerp(w1, w2, alpha_smooth)
                        h = lerp(h1, h2, alpha_smooth)

                        bbox = center_size_to_bbox(cx, cy, w, h)
                        render_mode = "interpolated"
                        interpolated_count += 1

                        # Use prev state for interpolation
                        obs = prev_obs

                elif prev_obs is not None:
                    # Only prev — hold if within max_hold
                    frames_since = frame_idx - prev_obs["frame"]
                    if frames_since <= max_hold_raw_frames:
                        bbox = prev_obs["bbox"]
                        render_mode = "held"
                        held_count += 1
                        obs = prev_obs
                    else:
                        hidden_count += 1
                        continue

                elif next_obs is not None:
                    # Only next — hold forward if within max_hold
                    frames_until = next_obs["frame"] - frame_idx
                    if frames_until <= max_hold_raw_frames:
                        bbox = next_obs["bbox"]
                        render_mode = "held_forward"
                        held_count += 1
                        obs = next_obs
                    else:
                        hidden_count += 1
                        continue

                if bbox is None:
                    continue

                # Apply EMA smoothing
                if key in last_render_bbox:
                    # Check for reasonable continuity
                    prev_cx, prev_cy, _, _ = bbox_to_center_size(last_render_bbox[key])
                    curr_cx, curr_cy, _, _ = bbox_to_center_size(bbox)
                    jump = np.sqrt((curr_cx - prev_cx)**2 + (curr_cy - prev_cy)**2)

                    if jump < 120:  # Reasonable movement threshold
                        # Apply EMA
                        px1, py1, pw, ph = bbox_to_center_size(last_render_bbox[key])
                        cx, cy, w, h = bbox_to_center_size(bbox)

                        smooth_cx = px1 + (cx - px1) * ema_alpha
                        smooth_cy = py1 + (cy - py1) * ema_alpha
                        smooth_w = pw + (w - pw) * ema_alpha
                        smooth_h = ph + (h - ph) * ema_alpha

                        bbox = center_size_to_bbox(smooth_cx, smooth_cy, smooth_w, smooth_h)

                last_render_bbox[key] = bbox

                # Compute opacity (fade in/out)
                opacity = 1.0
                if render_mode == "held":
                    frames_held = frame_idx - obs["frame"]
                    if frames_held == 1:
                        opacity = 0.80
                    elif frames_held == 2:
                        opacity = 0.65
                    elif frames_held == 3:
                        opacity = 0.45
                    elif frames_held == 4:
                        opacity = 0.25
                elif render_mode == "interpolated":
                    opacity = 0.95

                current_boxes[key] = {
                    "bbox": bbox,
                    "state": obs["state"],
                    "team_id": obs["team_id"],
                    "pid": obs["pid"],
                    "track_id": obs["track_id"],
                    "confidence": obs["confidence"],
                    "opacity": opacity,
                    "is_sampled": frame_idx in frame_is_sampled,
                }

            # === Draw boxes ===
            for key, data in current_boxes.items():
                bbox = data["bbox"]
                state = data["state"]
                team_id = data["team_id"]
                pid = data["pid"]
                track_id = data["track_id"]
                confidence = data["confidence"]
                opacity = data["opacity"]

                x1, y1, x2, y2 = [int(v) for v in bbox]
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))

                # Choose color by state and team
                if state == "locked":
                    if team_id == 0:
                        color = (255, 200, 0)  # Cyan
                    elif team_id == 1:
                        color = (0, 0, 255)  # Red
                    else:
                        color = (255, 255, 0)  # Yellow
                    thickness = 3
                elif state == "revived":
                    if team_id == 0:
                        color = (200, 150, 0)
                    elif team_id == 1:
                        color = (0, 0, 200)
                    else:
                        color = (200, 200, 0)
                    thickness = 2
                elif state == "provisional":
                    color = (128, 128, 255)  # Light red
                    thickness = 1
                else:  # unknown
                    color = (128, 128, 128)  # Gray
                    thickness = 1

                # Apply opacity
                color_rgb = np.array(color) * opacity
                color = tuple(int(c) for c in color_rgb)

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # Label
                if state == "locked":
                    label = f"{pid} T{track_id} LOCK"
                elif state == "revived":
                    label = f"{pid} T{track_id} REV"
                elif state == "provisional":
                    label = f"T{track_id} PROV"
                else:
                    label = f"T{track_id}"

                # Draw label with background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness_text = 1
                text_size = cv2.getTextSize(label, font, font_scale, thickness_text)[0]

                label_x = max(5, min(x1, width - text_size[0] - 10))
                label_y = max(text_size[1] + 5, y1 - 5)

                # Background for text
                bg_color = (50, 50, 50)
                cv2.rectangle(frame, (label_x - 2, label_y - text_size[1] - 2),
                             (label_x + text_size[0] + 2, label_y + 2), bg_color, -1)
                cv2.putText(frame, label, (label_x, label_y), font, font_scale, (255, 255, 255), thickness_text)

                # Motion trail for locked/revived only
                if show_trails and state in ("locked", "revived"):
                    # Simple: draw last few center points (would need history buffer for full trail)
                    cx, cy, _, _ = bbox_to_center_size(bbox)
                    cv2.circle(frame, (int(cx), int(cy)), 2, color, -1)

            # === Draw HUD ===
            if show_hud:
                hud_lines = [
                    f"Frame {frame_idx}/{total_frames}",
                    f"Stride: {identity_frame_stride} | Render: {render_stride}",
                    f"Locks: {identity_metrics.get('locks_created', 0)} | Live: {identity_metrics.get('locks_live_at_end', 0)} | Coverage: {identity_metrics.get('valid_id_coverage', 0.0):.2f}",
                ]
                for i, line in enumerate(hud_lines):
                    cv2.putText(frame, line, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Write frame
            out.write(frame)
            frame_idx += 1

            if frame_idx % 100 == 0:
                print(f"  Rendered {frame_idx}/{total_frames}")

    except KeyboardInterrupt:
        print("Render interrupted")

    out.release()
    cap.release()

    print(f"✓ Rendered {frame_idx} frames to {output_video}")
    print(f"  Interpolated: {interpolated_count}, Held: {held_count}, Hidden: {hidden_count}")

    # === Write manifest ===
    manifest = {
        "source_video": str(video_path),
        "output_video": output_video,
        "source_fps": fps,
        "output_fps": fps,
        "total_raw_frames": total_frames,
        "rendered_frames": frame_idx,
        "identity_frame_stride": identity_frame_stride,
        "render_stride": render_stride,
        "render_mode": "fullfps_visual_interpolation_from_stride5_identity",
        "interpolated_frames_count": interpolated_count,
        "held_frames_count": held_count,
        "hidden_frames_count": hidden_count,
        "max_hold_raw_frames": max_hold_raw_frames,
        "max_interp_gap_raw_frames": max_interp_gap_raw_frames,
        "new_identities_created_during_render": new_identities_created,
        "warnings": warned,
    }
    manifest.update(identity_metrics)

    manifest_path = f"{output_dir}/annotated_tracking_fullfps_smooth_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"✓ Manifest: {manifest_path}")
    print(json.dumps(manifest, indent=2))

    return output_video, manifest_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Source video path")
    parser.add_argument("--job-id", required=True, help="Job ID")
    parser.add_argument("--tracks", required=True, help="Path to tracking results JSON")
    parser.add_argument("--identity-metrics", help="Path to identity_metrics.json")
    parser.add_argument("--out", help="Output video path (default: temp/{job_id}/...)")
    parser.add_argument("--identity-stride", type=int, default=5)
    parser.add_argument("--render-stride", type=int, default=1)

    args = parser.parse_args()

    # Load tracking results
    with open(args.tracks) as f:
        data = json.load(f)
        tracking_results = data.get("frames", [])

    # Load identity metrics
    identity_metrics = {}
    if args.identity_metrics:
        with open(args.identity_metrics) as f:
            identity_metrics = json.load(f)

    # Render
    render_smooth_tracking(
        args.video,
        tracking_results,
        identity_metrics,
        args.job_id,
        output_dir=os.path.dirname(args.out) if args.out else None,
        identity_frame_stride=args.identity_stride,
        render_stride=args.render_stride,
    )
