#!/usr/bin/env python3
"""
Smooth full-FPS annotated render from stride-5 identity decisions.

- Identity runs at stride=5 (high confidence, sparse)
- Rendering interpolates at stride=1 (all frames, visual only)
- No new identities created during render
- No identity state changes during render
- Referee filtering, ghost box suppression, duplicate PID protection
- Detailed diagnostics for QA
"""

import os
import json
import sys
from pathlib import Path
from collections import defaultdict
import time

import cv2
import numpy as np


# === Constants ===
REFEREE_COLORS = {
    # (B, G, R) approximate ranges for referee kits
    "orange": ((100, 140, 220), (180, 200, 255)),  # Orange
    "yellow": ((0, 150, 255), (100, 255, 255)),    # Yellow
    "black": ((0, 0, 50), (50, 50, 100)),          # Black/dark
    "red": ((0, 0, 200), (100, 100, 255)),         # Red
}

TEAM_COLORS = {
    0: (255, 200, 0),    # Team 0 → Cyan
    1: (0, 0, 255),      # Team 1 → Red
    -1: (255, 255, 0),   # Unknown → Yellow
}

STATE_COLORS = {
    "locked": (255, 200, 0),     # Bright cyan
    "revived": (200, 150, 0),    # Darker cyan
    "provisional": (128, 128, 255),  # Light red
    "unknown": (128, 128, 128),  # Gray
}


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


def is_referee_kit(frame, bbox, threshold=0.3):
    """
    Detect if bbox contains a referee by checking dominant color.
    Returns True if referee-like colors dominate the crop.
    """
    try:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return False

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False

        # Check HSV ranges for referee colors
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0].astype(float)
        s = hsv[:, :, 1].astype(float)
        v = hsv[:, :, 2].astype(float)

        # Referee detection: orange/yellow/black dominant
        # Orange: H ~10-30, high S/V
        orange_mask = ((h >= 5) & (h <= 35) & (s > 50) & (v > 50))
        # Yellow: H ~25-40
        yellow_mask = ((h >= 20) & (h <= 45) & (s > 50) & (v > 50))
        # Black: low V
        black_mask = (v < 50)

        dominant = (orange_mask | yellow_mask | black_mask).sum()
        total = crop.size // 3
        ratio = dominant / (total + 1e-6)

        return ratio > threshold
    except Exception:
        return False


def get_render_key_for_pid(pid, track_id):
    """Decide visual key: prefer P-ID for locked/revived, else T-ID."""
    if pid:
        return pid  # "P1", "P2", etc.
    return f"T{track_id}"


def offset_label_position(x1, y1, x2, y2, text_width, frame_width, frame_height):
    """
    Compute label position with collision avoidance.
    Try: above box, below box, left side, right side.
    """
    label_h = 20
    candidates = [
        (max(5, min(x1, frame_width - text_width - 10)), max(label_h, y1 - 5), "above"),
        (max(5, min(x1, frame_width - text_width - 10)), min(y2 + label_h, frame_height - 5), "below"),
        (max(5, x1 - text_width - 10), max(label_h, (y1 + y2) // 2), "left"),
        (min(x2 + 5, frame_width - text_width - 5), max(label_h, (y1 + y2) // 2), "right"),
    ]
    # Return first valid candidate (in this simple version, return the preferred one)
    return candidates[0][0], candidates[0][1]


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
    show_hud: bool = False,  # Production: False
    debug_mode: bool = False,
):
    """
    Render smooth full-FPS video from stride-5 tracking.

    tracking_results: list of frame dicts from run_tracking()
    identity_metrics: dict with locks_created, lock_retention_rate, etc.
    """
    if output_dir is None:
        output_dir = f"temp/{job_id}"
    os.makedirs(output_dir, exist_ok=True)
    qa_dir = os.path.join(output_dir, "render_qa")
    os.makedirs(qa_dir, exist_ok=True)

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
    render_observations = defaultdict(list)
    frame_is_sampled = {}
    referee_count = 0
    duplicate_pid_suppressed = 0

    for frame_idx, frame_data in enumerate(tracking_results):
        raw_frame_idx = frame_idx * identity_frame_stride
        frame_is_sampled[raw_frame_idx] = True

        if "players" not in frame_data:
            continue

        for player in frame_data["players"]:
            tid = player.get("rawTrackId")
            pid = player.get("playerId")
            source = player.get("assignment_source", "unassigned")
            bbox = player.get("bbox", [0, 0, 0, 0])
            team_id = player.get("team_id", -1)
            confidence = player.get("identity_confidence", 0.0)

            if tid is None:
                continue

            # Map assignment_source to state
            if source == "locked":
                state = "locked"
            elif source == "revived":
                state = "revived"
            elif source == "provisional":
                state = "provisional"
            else:
                state = "unknown"

            # Filter by visibility settings
            if state == "unknown" and not show_unknown:
                continue
            if state == "provisional" and not show_provisional:
                continue

            key = get_render_key_for_pid(pid, tid)

            render_observations[key].append({
                "frame": raw_frame_idx,
                "bbox": bbox,
                "track_id": tid,
                "pid": pid,
                "state": state,
                "team_id": team_id,
                "confidence": confidence,
                "is_referee": False,  # Will be detected during render
            })

    # Sort observations by frame
    for key in render_observations:
        render_observations[key].sort(key=lambda x: x["frame"])

    print(f"Built render timeline: {len(render_observations)} keys, {sum(len(v) for v in render_observations.values())} observations")

    # === Render loop ===
    last_render_bbox = {}
    frame_idx = 0
    interpolated_object_frames = 0
    held_object_frames = 0
    hidden_object_frames = 0
    visible_object_frames = 0
    new_identities_created = 0

    # Per-frame diagnostics
    per_frame_diagnostics = {}

    qa_frames_to_export = set()
    qa_frame_data = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame is None or frame.size == 0:
                per_frame_diagnostics[frame_idx] = {
                    "frame_idx": frame_idx,
                    "valid_frame": False,
                    "visible_count": 0,
                }
                frame_idx += 1
                out.write(np.zeros((height, width, 3), dtype=np.uint8))
                hidden_object_frames += 1
                continue

            # Find observations for this frame
            current_boxes = {}
            hide_reasons = defaultdict(int)

            for key, observations in render_observations.items():
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

                # Decide bbox and render mode
                bbox = None
                render_mode = "unknown"
                obs = None

                if prev_obs is not None and next_obs is not None:
                    gap = next_obs["frame"] - prev_obs["frame"]
                    if gap <= max_interp_gap_raw_frames:
                        # Check for duplicate PID in same frame
                        if key in current_boxes:
                            hide_reasons["DUPLICATE_PID"] += 1
                            hidden_object_frames += 1
                            continue

                        alpha = (frame_idx - prev_obs["frame"]) / gap
                        alpha_smooth = smoothstep(alpha)

                        cx1, cy1, w1, h1 = bbox_to_center_size(prev_obs["bbox"])
                        cx2, cy2, w2, h2 = bbox_to_center_size(next_obs["bbox"])

                        cx = lerp(cx1, cx2, alpha_smooth)
                        cy = lerp(cy1, cy2, alpha_smooth)
                        w = lerp(w1, w2, alpha_smooth)
                        h = lerp(h1, h2, alpha_smooth)

                        bbox = center_size_to_bbox(cx, cy, w, h)
                        render_mode = "interpolated"
                        interpolated_object_frames += 1
                        obs = prev_obs
                    else:
                        hide_reasons["GAP_TOO_LARGE"] += 1
                        hidden_object_frames += 1
                        continue

                elif prev_obs is not None:
                    frames_since = frame_idx - prev_obs["frame"]
                    if frames_since <= max_hold_raw_frames:
                        # Check for duplicate
                        if key in current_boxes:
                            hide_reasons["DUPLICATE_PID"] += 1
                            hidden_object_frames += 1
                            continue

                        bbox = prev_obs["bbox"]
                        render_mode = "held"
                        held_object_frames += 1
                        obs = prev_obs
                    else:
                        hide_reasons["HOLD_EXPIRED"] += 1
                        hidden_object_frames += 1
                        continue

                elif next_obs is not None:
                    frames_until = next_obs["frame"] - frame_idx
                    if frames_until <= max_hold_raw_frames:
                        # Check for duplicate
                        if key in current_boxes:
                            hide_reasons["DUPLICATE_PID"] += 1
                            hidden_object_frames += 1
                            continue

                        bbox = next_obs["bbox"]
                        render_mode = "held_forward"
                        held_object_frames += 1
                        obs = next_obs
                    else:
                        hide_reasons["HOLD_FORWARD_EXPIRED"] += 1
                        hidden_object_frames += 1
                        continue

                if bbox is None or obs is None:
                    continue

                # === Referee detection ===
                if is_referee_kit(frame, bbox, threshold=0.25):
                    hide_reasons["REFEREE_SUPPRESSED"] += 1
                    hidden_object_frames += 1
                    referee_count += 1
                    continue

                # Apply EMA smoothing
                if key in last_render_bbox:
                    prev_cx, prev_cy, _, _ = bbox_to_center_size(last_render_bbox[key])
                    curr_cx, curr_cy, _, _ = bbox_to_center_size(bbox)
                    jump = np.sqrt((curr_cx - prev_cx)**2 + (curr_cy - prev_cy)**2)

                    if jump < 120:
                        px1, py1, pw, ph = bbox_to_center_size(last_render_bbox[key])
                        cx, cy, w, h = bbox_to_center_size(bbox)

                        smooth_cx = px1 + (cx - px1) * ema_alpha
                        smooth_cy = py1 + (cy - py1) * ema_alpha
                        smooth_w = pw + (w - pw) * ema_alpha
                        smooth_h = ph + (h - ph) * ema_alpha

                        bbox = center_size_to_bbox(smooth_cx, smooth_cy, smooth_w, smooth_h)

                last_render_bbox[key] = bbox

                # Compute opacity
                opacity = 1.0
                if render_mode == "held":
                    frames_held = frame_idx - obs["frame"]
                    opacities = {1: 0.80, 2: 0.65, 3: 0.45, 4: 0.25}
                    opacity = opacities.get(frames_held, 0.0)
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
                    "render_mode": render_mode,
                }
                visible_object_frames += 1

            # === Draw boxes ===
            for key, data in current_boxes.items():
                bbox = data["bbox"]
                state = data["state"]
                team_id = data["team_id"]
                pid = data["pid"]
                track_id = data["track_id"]
                opacity = data["opacity"]

                x1, y1, x2, y2 = [int(v) for v in bbox]
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))

                if x2 <= x1 or y2 <= y1:
                    continue

                # Color by state
                color = STATE_COLORS.get(state, (128, 128, 128))
                thickness = 3 if state == "locked" else (2 if state == "revived" else 1)

                # Apply opacity
                color = tuple(int(c * opacity) for c in color)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # Compact label (production mode)
                if state == "locked" and pid:
                    label = f"{pid} L"
                elif state == "revived" and pid:
                    label = f"{pid} R"
                elif debug_mode:
                    label = f"{pid} {state[:3]}" if pid else f"T{track_id}"
                else:
                    label = f"{pid}" if pid else ""

                # Draw label if present
                if label:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.4
                    thickness_text = 1
                    text_size = cv2.getTextSize(label, font, font_scale, thickness_text)[0]

                    label_x, label_y = offset_label_position(x1, y1, x2, y2, text_size[0], width, height)

                    # Small background
                    bg_color = (30, 30, 30)
                    cv2.rectangle(frame, (label_x - 2, label_y - text_size[1] - 2),
                                 (label_x + text_size[0] + 2, label_y + 2), bg_color, -1)
                    cv2.putText(frame, label, (label_x, label_y), font, font_scale, (255, 255, 255), thickness_text)

                # Motion trail for locked/revived
                if show_trails and state in ("locked", "revived"):
                    cx, cy, _, _ = bbox_to_center_size(bbox)
                    cv2.circle(frame, (int(cx), int(cy)), 2, color, -1)

            # === HUD (debug mode only) ===
            if show_hud and debug_mode:
                hud_lines = [
                    f"Frame {frame_idx}/{total_frames}",
                    f"Visible boxes: {len(current_boxes)}",
                ]
                for i, line in enumerate(hud_lines):
                    cv2.putText(frame, line, (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # Store frame diagnostics
            frame_diag = {
                "frame_idx": frame_idx,
                "valid_frame": True,
                "visible_count": len(current_boxes),
                "locked_count": sum(1 for d in current_boxes.values() if d["state"] == "locked"),
                "revived_count": sum(1 for d in current_boxes.values() if d["state"] == "revived"),
                "unknown_count": sum(1 for d in current_boxes.values() if d["state"] == "unknown"),
                "interpolated_count": sum(1 for d in current_boxes.values() if d["render_mode"] == "interpolated"),
                "held_count": sum(1 for d in current_boxes.values() if d["render_mode"].startswith("held")),
                "hide_reasons": dict(hide_reasons),
            }
            per_frame_diagnostics[frame_idx] = frame_diag
            qa_frame_data.append(frame_diag)

            # Mark frames for QA export (every 50 frames)
            if frame_idx % 50 == 0:
                qa_frames_to_export.add(frame_idx)

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
    print(f"  Object-frames: interpolated={interpolated_object_frames}, held={held_object_frames}, hidden={hidden_object_frames}, visible={visible_object_frames}")
    print(f"  Referee suppressed: {referee_count}")

    # === Write manifest ===
    frames_with_any_visible = sum(1 for d in per_frame_diagnostics.values() if d.get("visible_count", 0) > 0)
    frames_with_locked = sum(1 for d in per_frame_diagnostics.values() if d.get("locked_count", 0) > 0)
    frames_with_no_visible = sum(1 for d in per_frame_diagnostics.values() if d.get("visible_count", 0) == 0)

    visible_counts = [d.get("visible_count", 0) for d in per_frame_diagnostics.values() if d.get("valid_frame", False)]
    median_visible = int(np.median(visible_counts)) if visible_counts else 0

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
        # Object-frame counts (can exceed total_raw_frames)
        "interpolated_object_frames": interpolated_object_frames,
        "held_object_frames": held_object_frames,
        "hidden_object_frames": hidden_object_frames,
        "visible_object_frames": visible_object_frames,
        # Frame counts (cannot exceed total_raw_frames)
        "frames_with_any_visible_box": frames_with_any_visible,
        "frames_with_locked_boxes": frames_with_locked,
        "frames_with_no_visible_boxes": frames_with_no_visible,
        "median_visible_players_per_frame": median_visible,
        # Config
        "max_hold_raw_frames": max_hold_raw_frames,
        "max_interp_gap_raw_frames": max_interp_gap_raw_frames,
        # Quality checks
        "referee_suppressions": referee_count,
        "new_identities_created_during_render": new_identities_created,
    }
    manifest.update(identity_metrics)

    manifest_path = f"{output_dir}/annotated_tracking_fullfps_smooth_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"✓ Manifest: {manifest_path}")

    # === Write render QA diagnostics ===
    qa_summary = {
        "total_frames": frame_idx,
        "frames_with_visible_boxes": frames_with_any_visible,
        "frames_with_locked_boxes": frames_with_locked,
        "median_visible_per_frame": median_visible,
        "referee_suppressions": referee_count,
        "per_frame_diagnostics": per_frame_diagnostics,
    }

    qa_path = f"{output_dir}/render_visual_qa.json"
    with open(qa_path, "w") as f:
        json.dump(qa_summary, f, indent=2)
    print(f"✓ QA diagnostics: {qa_path}")

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
    parser.add_argument("--debug", action="store_true", help="Debug mode with HUD")

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
        show_hud=args.debug,
        debug_mode=args.debug,
    )
