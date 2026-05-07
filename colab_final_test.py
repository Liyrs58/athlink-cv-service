#!/usr/bin/env python3
"""
Final VLM-integrated tracking test for Colab.
Run this directly in Colab with: exec(open('/content/athlink-cv-service/colab_final_test.py').read())
"""

import torch
import time
import sys
import os
import cv2
import json
import subprocess
from pathlib import Path

print("=" * 70)
print("BROADCAST TRACKING WITH VLM STATE MACHINE - FINAL TEST")
print("=" * 70)

# Verify GPU
assert torch.cuda.is_available(), "GPU not available"
device = "cuda"
print(f"\n✓ Device: {device}")

# Set up path
os.chdir('/content/athlink-cv-service')
sys.path.insert(0, '/content/athlink-cv-service')

print("✓ Working directory: /content/athlink-cv-service")
print("✓ Python path configured\n")

# Colab keeps imported modules alive across cells. Purge service modules so a
# git pull/checkout actually takes effect without a full runtime restart.
for module_name in list(sys.modules):
    if module_name == "services" or module_name.startswith("services."):
        del sys.modules[module_name]

try:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except Exception as exc:
    head = f"unknown ({exc})"
print(f"✓ Git HEAD: {head}")

# Import tracker after the cache purge.
from services.tracker_core import run_tracking
print(f"✓ Tracker source: {run_tracking.__code__.co_filename}")

print("=" * 70)
print("RUNNING TRACKING PIPELINE")
print("=" * 70)

t0 = time.time()
results = run_tracking(
    video_path="/content/input_video.mp4",
    job_id="colab_vlm_final",
    frame_stride=1,
    max_frames=None,
    device=device,
)
elapsed = time.time() - t0

print(f"\n{'='*70}")
print(f"✓ Processing time: {elapsed:.1f}s")
print(f"✓ Total frames processed: {len(results)}")
print(f"{'='*70}\n")

# Check key frames
print("KEY FRAME ANALYSIS:")
print(f"{'Frame':<10} {'Tracks':<10} {'Detections':<12} {'Game State':<15}")
print("-" * 50)
for frame_idx in [0, 359, 549, 819, 999]:
    if frame_idx < len(results):
        frame_data = results[frame_idx]
        tracks = frame_data['track_count']
        state = frame_data.get('gameState', 'unknown')
        dets = frame_data['detection_count']
        print(f"{frame_idx:<10} {tracks:<10} {dets:<12} {state:<15}")

# Count unique IDs
all_ids = set()
for frame in results:
    for player in frame.get('players', []):
        all_ids.add(player['trackId'])

print(f"\n{'='*70}")
print(f"✓ UNIQUE TRACK IDs: {len(all_ids)}")
print(f"{'='*70}\n")

# Print state transitions
print("GAME STATE TRANSITIONS:")
print("-" * 50)
prev_state = None
transition_count = 0
for frame_idx, frame in enumerate(results):
    state = frame.get('gameState', 'play')
    if state != prev_state:
        print(f"Frame {frame_idx:4d}: {prev_state or 'START':12s} → {state}")
        transition_count += 1
        prev_state = state

print(f"\nTotal transitions: {transition_count}")

# Render annotated video
print(f"\n{'='*70}")
print("RENDERING ANNOTATED VIDEO")
print(f"{'='*70}\n")

json_path = Path('temp/colab_vlm_final/tracking/track_results.json')
if not json_path.exists():
    print(f"ERROR: JSON not found at {json_path}")
    sys.exit(1)

with open(json_path) as f:
    data = json.load(f)

cap = cv2.VideoCapture('/content/input_video.mp4')
if not cap.isOpened():
    print("ERROR: Can't open video")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video: {w}x{h} @ {fps:.1f}fps")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/content/annotated_broadcast_final.mp4', fourcc, fps, (w, h))

frames_data = {f['frameIndex']: f['players'] for f in data['frames']}

frame_idx = 0
rendered = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    players = frames_data.get(frame_idx, [])
    for p in players:
        bbox = p.get('bbox')
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            identity_valid = bool(p.get('identity_valid', False))
            source = p.get('assignment_source', 'unassigned')
            assignment_pending = bool(p.get('assignment_pending', False))
            display_id = p.get('displayId')
            if isinstance(display_id, str) and display_id.startswith('U T'):
                display_id = None
            if not identity_valid:
                if isinstance(display_id, int) or (isinstance(display_id, str) and display_id.isdigit()):
                    display_id = None
            if not display_id and identity_valid:
                display_id = f"P{p.get('trackId', '?')}"
            elif not display_id and assignment_pending:
                display_id = "?"
            state = p.get('gameState', 'play')

            # Color by identity state, not raw tracker state.
            if state == 'bench_shot' or not identity_valid:
                color = (128, 128, 128)  # Gray
            elif source == 'revived':
                color = (0, 220, 255)  # Yellow
            else:
                color = (0, 255, 0)  # Green

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if display_id:
                cv2.putText(frame, display_id, (x1, max(y1-5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out.write(frame)
    frame_idx += 1
    rendered += 1

cap.release()
out.release()

print(f"✓ Rendered {rendered} frames")
print(f"✓ Saved to /content/annotated_broadcast_final.mp4")

# Download
print(f"\n{'='*70}")
print("DOWNLOADING RESULTS")
print(f"{'='*70}\n")

from google.colab import files
print("Downloading annotated video...")
files.download('/content/annotated_broadcast_final.mp4')

print("\n" + "=" * 70)
print("✓ TEST COMPLETE")
print("=" * 70)
