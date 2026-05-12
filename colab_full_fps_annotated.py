#!/usr/bin/env python3
"""
PASTE THIS INTO COLAB.
Full-FPS annotated tracking video with stride=5 identity locking.

Pipeline: stride=5 tracking/identity → stride=1 visual render via interpolation/holding.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

print("=" * 80)
print("FULL-FPS ANNOTATED TRACKING (stride=5 identity → stride=1 render)")
print("=" * 80)

# Fetch repo if needed
if not os.path.exists('/content/athlink-cv-service'):
    print("Downloading repo...")
    subprocess.run(['curl', '-L',
                    'https://github.com/liys58/athlink-cv-service/archive/refs/heads/main.zip',
                    '-o', '/content/repo.zip'], capture_output=True)
    subprocess.run(['unzip', '-q', '/content/repo.zip', '-d', '/content'], capture_output=True)
    subprocess.run(['mv', '/content/athlink-cv-service-main', '/content/athlink-cv-service'],
                   capture_output=True)
    print("✓ Repo extracted")
else:
    print("✓ Repo already present")

os.chdir('/content/athlink-cv-service')
sys.path.insert(0, '/content/athlink-cv-service')

# Purge modules
for module_name in list(sys.modules):
    if module_name == "services" or module_name.startswith("services."):
        del sys.modules[module_name]

job_id = "full_villa_psg"
video_path = "/content/aston villa vs psg 1.mov"
frame_stride = 5

print(f"\nJob: {job_id}")
print(f"Video: {video_path}")
print(f"Identity stride: {frame_stride}, Render stride: 1")

# Check video exists
if not os.path.exists(video_path):
    alt_paths = ['/content/Aston villa vs Psg clip 1.mov', '/content/1b16c594_villa_psg_40s_new.mp4']
    for alt in alt_paths:
        if os.path.exists(alt):
            video_path = alt
            print(f"Using: {video_path}")
            break

# ============================================================================
# STEP 1: RUN TRACKING AT STRIDE=5
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: TRACKING (stride=5)")
print("=" * 80)

from services.tracker_core import run_tracking
import io
from contextlib import redirect_stdout

output_buffer = io.StringIO()
t0 = time.time()

try:
    with redirect_stdout(output_buffer):
        tracking_results = run_tracking(
            video_path=video_path,
            job_id=job_id,
            frame_stride=frame_stride,
            max_frames=None,
            device="cuda"
        )
    elapsed_track = time.time() - t0
    full_output = output_buffer.getvalue()

    # Print last 100 lines
    lines = full_output.split('\n')
    print('\n'.join(lines[-100:]))

    print(f"\n✓ Tracking complete in {elapsed_track:.1f}s ({len(tracking_results)} sampled frames)")

except Exception as e:
    print(f"✗ Tracking failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Extract identity metrics
identity_metrics = {}
for line in full_output.split('\n'):
    if any(key in line for key in ['collapse_lock_creations', 'locks_created', 'lock_retention_rate', 'valid_id_coverage', 'stable_locked_count', 'locks_live_at_end']):
        parts = line.split('=')
        if len(parts) >= 2:
            key = parts[0].strip()
            value_part = parts[1].strip().split()[0]
            try:
                value = float(value_part) if '.' in value_part else int(value_part)
                identity_metrics[key] = value
            except:
                pass

metrics_json_path = f"temp/{job_id}/tracking/identity_metrics.json"
if os.path.exists(metrics_json_path):
    try:
        with open(metrics_json_path) as f:
            metrics_data = json.load(f)
        identity_metrics.update(metrics_data)
    except:
        pass

print(f"\nIdentity metrics:")
for k, v in sorted(identity_metrics.items()):
    print(f"  {k}: {v}")

# ============================================================================
# STEP 2: BUILD STRIDE-5 TRACKING MAP
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: BUILD STRIDE-5 TRACKING MAP")
print("=" * 80)

# Map: frame_index -> {track_id -> (bbox, p_id, state, team_id, confidence)}
sampled_frame_tracking = {}
track_id_to_pid = {}  # track_id -> p_id for identity
track_id_state = {}   # track_id -> "LOCKED" or "REVIVED" or "PROVISIONAL"

# Parse tracking results (format: list of frames, each with 'players' array)
for frame_idx, frame_data in enumerate(tracking_results):
    raw_frame_idx = frame_idx * frame_stride
    sampled_frame_tracking[raw_frame_idx] = {}

    if 'players' in frame_data:
        for player in frame_data['players']:
            tid = player.get('rawTrackId')
            bbox = player.get('bbox', [0, 0, 0, 0])
            pid = player.get('playerId')  # "P1", "P2", etc. or None
            source = player.get('assignment_source', 'unassigned')  # "locked", "revived", "provisional", "unassigned"
            team_id = player.get('team_id')
            confidence = player.get('identity_confidence', player.get('confidence', 0.0))

            if tid is not None:
                # Map assignment_source to display state
                if source == 'locked':
                    state = 'LOCKED'
                elif source == 'revived':
                    state = 'REVIVED'
                elif source == 'provisional':
                    state = 'PROVISIONAL'
                else:
                    state = 'UNKNOWN'

                sampled_frame_tracking[raw_frame_idx][tid] = {
                    'bbox': bbox,
                    'p_id': pid,
                    'state': state,
                    'team_id': team_id,
                    'confidence': confidence,
                }

                # Track identity assignments (only LOCKED and REVIVED are valid)
                if pid and state in ['LOCKED', 'REVIVED']:
                    track_id_to_pid[tid] = pid
                    track_id_state[tid] = state

print(f"Built map for {len(sampled_frame_tracking)} sampled frames")
print(f"Identity assignments: {len(track_id_to_pid)} tracks with locked/revived P-IDs")

# ============================================================================
# STEP 3: RENDER FULL-FPS ANNOTATED VIDEO
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: RENDER FULL-FPS ANNOTATED VIDEO")
print("=" * 80)

cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Input: {w}x{h} @ {fps:.1f} fps, {total_frames} frames")

output_dir = f"temp/{job_id}"
os.makedirs(output_dir, exist_ok=True)
output_video = f"{output_dir}/annotated_tracking_fullfps.mp4"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

max_hold_frames = 4
interpolated_count = 0
held_count = 0
hidden_count = 0
frames_with_pid = 0

# Track state for interpolation
last_track_states = {}  # track_id -> {'bbox', 'frame_idx', 'p_id', 'state', ...}
frame_idx = 0

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Find surrounding sampled frames
        sampled_before_idx = (frame_idx // frame_stride) * frame_stride
        sampled_after_idx = sampled_before_idx + frame_stride

        # Get tracking data
        is_sampled = (frame_idx == sampled_before_idx)

        if is_sampled and sampled_before_idx in sampled_frame_tracking:
            # Use exact sampled data
            current_tracks = sampled_frame_tracking[sampled_before_idx]
            source_frame = sampled_before_idx
            render_mode = "sampled"

            # Update last_track_states
            for tid, data in current_tracks.items():
                last_track_states[tid] = {
                    'bbox': data['bbox'],
                    'frame_idx': frame_idx,
                    'p_id': data['p_id'],
                    'state': data['state'],
                    'team_id': data['team_id'],
                    'confidence': data['confidence'],
                }
        else:
            # Interpolate or hold
            current_tracks = {}
            source_frame = sampled_before_idx
            render_mode = "interpolated"

            tracks_before = sampled_frame_tracking.get(sampled_before_idx, {})
            tracks_after = sampled_frame_tracking.get(sampled_after_idx, {})

            # Collect all track IDs visible in surrounding frames
            all_tids = set(tracks_before.keys()) | set(tracks_after.keys())

            for tid in all_tids:
                before_data = tracks_before.get(tid)
                after_data = tracks_after.get(tid)

                if before_data and after_data:
                    # Both frames have track → interpolate
                    bbox_before = before_data['bbox']
                    bbox_after = after_data['bbox']
                    progress = (frame_idx - sampled_before_idx) / frame_stride
                    bbox = [
                        int(bbox_before[i] + (bbox_after[i] - bbox_before[i]) * progress)
                        for i in range(4)
                    ]

                    current_tracks[tid] = {
                        'bbox': bbox,
                        'p_id': before_data.get('p_id'),  # Use p_id from before frame
                        'state': before_data.get('state', 'UNKNOWN'),
                        'team_id': before_data.get('team_id'),
                        'confidence': before_data.get('confidence', 0.0),
                        'interpolated': True,
                    }
                    interpolated_count += 1
                    last_track_states[tid] = current_tracks[tid]

                elif before_data:
                    # Only before → hold if within max_hold_frames
                    frames_since = frame_idx - sampled_before_idx
                    if frames_since <= max_hold_frames:
                        current_tracks[tid] = before_data.copy()
                        current_tracks[tid]['held'] = True
                        held_count += 1
                        last_track_states[tid] = current_tracks[tid]
                    else:
                        hidden_count += 1

                elif after_data:
                    # Only after → hold if within max_hold_frames
                    frames_until = sampled_after_idx - frame_idx
                    if frames_until <= max_hold_frames:
                        current_tracks[tid] = after_data.copy()
                        current_tracks[tid]['held'] = True
                        held_count += 1
                        last_track_states[tid] = current_tracks[tid]
                    else:
                        hidden_count += 1

        # Draw bounding boxes
        for tid, data in current_tracks.items():
            bbox = data['bbox']
            p_id = data.get('p_id', 'NONE')
            state = data.get('state', 'UNKNOWN')
            team_id = data.get('team_id', -1)
            confidence = data.get('confidence', 0.0)

            x1, y1, x2, y2 = [int(v) for v in bbox]

            # Choose color by team or state
            if state == 'LOCKED':
                color = (0, 255, 0)  # Green
            elif state == 'REVIVED':
                color = (0, 165, 255)  # Orange
            elif state == 'PROVISIONAL':
                color = (255, 255, 0)  # Cyan
            else:
                color = (128, 128, 128)  # Gray

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label
            label = f"T{tid} {state[:3]}"
            if p_id and p_id != 'NONE':
                label = f"{p_id} {state[:3]}"
                frames_with_pid += 1

            if team_id >= 0:
                label += f" [T{team_id}]"

            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Frame info
        info = f"Frame {frame_idx} (stride-{frame_stride}: {source_frame}) | Mode: {render_mode}"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  Rendered {frame_idx}/{total_frames} frames")

except KeyboardInterrupt:
    print("Render interrupted")

out.release()
cap.release()

print(f"✓ Rendered {frame_idx} frames to {output_video}")
print(f"  Interpolated: {interpolated_count}, Held: {held_count}, Hidden: {hidden_count}")
print(f"  Frames with visible P-ID: {frames_with_pid}")

# ============================================================================
# STEP 4: VERIFY OUTPUT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: VERIFY OUTPUT")
print("=" * 80)

if os.path.exists(output_video):
    file_size_mb = os.path.getsize(output_video) / (1024 * 1024)
    print(f"✓ Video: {output_video}")
    print(f"  Size: {file_size_mb:.1f} MB")
    print(f"  FPS: {fps:.1f}")
    print(f"  Duration: {frame_idx / fps:.1f}s")
    print(f"  Total frames: {frame_idx}")
else:
    print(f"✗ Video not created")
    sys.exit(1)

# ============================================================================
# STEP 5: WRITE MANIFEST
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: WRITE MANIFEST")
print("=" * 80)

manifest = {
    "source_video": str(video_path),
    "output_video": output_video,
    "source_fps": fps,
    "output_fps": fps,
    "total_raw_frames": frame_idx,
    "sampled_tracking_frames": len(tracking_results),
    "identity_frame_stride": frame_stride,
    "render_frame_stride": 1,
    "interpolated_frames_count": interpolated_count,
    "held_frames_count": held_count,
    "hidden_frames_count": hidden_count,
    "max_hold_frames": max_hold_frames,
    "frames_with_visible_pid": frames_with_pid,
}

# Add identity metrics
manifest.update(identity_metrics)

manifest_path = f"{output_dir}/annotated_tracking_fullfps_manifest.json"
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)

print(f"✓ Manifest: {manifest_path}")
print(json.dumps(manifest, indent=2))

# ============================================================================
# STEP 6: GATES & VERDICT
# ============================================================================
print("\n" + "=" * 80)
print("FINAL VERDICT")
print("=" * 80)

locks_created = int(identity_metrics.get('locks_created', 0))
lock_retention = identity_metrics.get('lock_retention_rate', 0.0)
collapse_lock_creations = int(identity_metrics.get('collapse_lock_creations', 0))

gate_1 = locks_created >= 5
gate_2 = lock_retention >= 0.50 if locks_created > 0 else True
gate_3 = collapse_lock_creations == 0

all_pass = gate_1 and gate_2 and gate_3

print(f"\nGate 1 (locks_created >= 5): {locks_created} {'✓' if gate_1 else '✗'}")
print(f"Gate 2 (lock_retention >= 0.50): {lock_retention:.3f} {'✓' if gate_2 else '✗'}")
print(f"Gate 3 (collapse_lock_creations == 0): {collapse_lock_creations} {'✓' if gate_3 else '✗'}")

if all_pass:
    print(f"\n✓✓✓ FULL-FPS RENDER: PASSED ✓✓✓")
else:
    print(f"\n⚠ FULL-FPS RENDER: INCOMPLETE")

print(f"\nOutput: {output_video}")
print(f"Manifest: {manifest_path}")

# ============================================================================
# STEP 7: DOWNLOAD
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: DOWNLOAD RESULTS")
print("=" * 80)

try:
    from google.colab import files

    print("Downloading...")
    files.download(output_video)
    files.download(manifest_path)
    print("✓ Files downloaded")

except ImportError:
    print("⚠ Not in Colab")

print("\n" + "=" * 80)
