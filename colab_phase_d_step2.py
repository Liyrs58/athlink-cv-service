#!/usr/bin/env python3
"""
PASTE THIS ENTIRE CELL INTO COLAB.
Phase D Step 2: Identity Lock Creation & Retention Verification.
Focus: Verify locks_created >= 20, lock_retention >= 0.65, collapse_lock_creations = 0.
Includes: Repo clone, full tracking test, annotated video render, and download.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path

# ============================================================================
# STEP 0: CLONE REPO (since cell was restarted)
# ============================================================================
print("=" * 80)
print("STEP 0: CLONE REPO")
print("=" * 80)

# Check if already cloned
if not os.path.exists('/content/athlink-cv-service'):
    print("Cloning athlink-cv-service repo...")
    result = subprocess.run(
        ['git', 'clone', 'https://github.com/liys58/athlink-cv-service.git'],
        cwd='/content',
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Clone failed: {result.stderr}")
        sys.exit(1)
    print("✓ Repo cloned")
else:
    print("✓ Repo already present")

os.chdir('/content/athlink-cv-service')
sys.path.insert(0, '/content/athlink-cv-service')

print(f"Working directory: {os.getcwd()}")

# ============================================================================
# PHASE D STEP 2: IDENTITY LOCK VERIFICATION
# ============================================================================
print("\n" + "=" * 80)
print("PHASE D STEP 2: IDENTITY LOCK CREATION & RETENTION")
print("=" * 80)

# Purge prior modules for clean import
for module_name in list(sys.modules):
    if module_name == "services" or module_name.startswith("services."):
        del sys.modules[module_name]

job_id = "phase_d_step2"
video_path = "/content/1b16c594_villa_psg_40s_new.mp4"
frame_stride = 5
max_frames = None  # Full video

print(f"\nJob: {job_id}")
print(f"Video: {video_path}")
print(f"Frame stride: {frame_stride}, max frames: {max_frames or 'all'}\n")

# Check video exists
if not os.path.exists(video_path):
    print(f"⚠ Video not found at {video_path}")
    print("Checking for alternative video paths...")
    alt_paths = [
        '/content/Aston villa vs Psg clip 1.mov',
        '/content/villa_psg.mp4'
    ]
    for alt in alt_paths:
        if os.path.exists(alt):
            print(f"Found: {alt}")
            video_path = alt
            break
    else:
        print("No video found. Upload to /content/ or mount Google Drive.")
        # Try to use whatever exists
        if os.path.exists('/content/Aston villa vs Psg clip 1.mov'):
            video_path = '/content/Aston villa vs Psg clip 1.mov'
            print(f"Using fallback: {video_path}")

# ============================================================================
# STEP 1: TRACKING WITH IDENTITY (identity_frame_seq fix)
# ============================================================================
print("=" * 80)
print("STEP 1: TRACKING (identity_frame_seq + pending_streak continuity)")
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
            max_frames=max_frames,
            device="cuda"
        )

    elapsed = time.time() - t0

    # Print captured output
    full_output = output_buffer.getvalue()
    print(full_output)

    print(f"\n✓ Tracking complete in {elapsed:.1f}s\n")

except Exception as e:
    print(f"✗ Tracking failed: {e}")
    import traceback
    traceback.print_exc()
    full_output = output_buffer.getvalue()
    print(full_output)
    sys.exit(1)

# ============================================================================
# STEP 2: IDENTITY METRICS EXTRACTION
# ============================================================================
print("=" * 80)
print("STEP 2: IDENTITY METRICS")
print("=" * 80)

identity_metrics = {}

# Parse [IdentityMetrics] section
for line in full_output.split('\n'):
    if any(key in line for key in ['collapse_lock_creations', 'locks_created', 'lock_retention_rate', 'valid_id_coverage']):
        parts = line.split('=')
        if len(parts) >= 2:
            key = parts[0].strip()
            value_part = parts[1].strip().split()[0]
            try:
                value = float(value_part)
                identity_metrics[key] = value
            except:
                pass

print(f"\nExtracted metrics:")
for k, v in identity_metrics.items():
    print(f"  {k}: {v}")

# ============================================================================
# STEP 3: ACCEPTANCE GATES
# ============================================================================
print("\n" + "=" * 80)
print("ACCEPTANCE GATES")
print("=" * 80)

locks_created = int(identity_metrics.get('locks_created', 0))
lock_retention = identity_metrics.get('lock_retention_rate', 0.0)
collapse_lock_creations = int(identity_metrics.get('collapse_lock_creations', 0))
valid_id_coverage = identity_metrics.get('valid_id_coverage', 0.0)

gate_1_pass = locks_created >= 20
gate_2_pass = lock_retention >= 0.65
gate_3_pass = collapse_lock_creations == 0

print(f"\nGATE 1: locks_created >= 20")
print(f"  {locks_created} >= 20: {'✓ PASS' if gate_1_pass else '✗ FAIL'}")

print(f"\nGATE 2: lock_retention >= 0.65")
print(f"  {lock_retention:.3f} >= 0.65: {'✓ PASS' if gate_2_pass else '✗ FAIL'}")

print(f"\nGATE 3: collapse_lock_creations == 0")
print(f"  {collapse_lock_creations} == 0: {'✓ PASS' if gate_3_pass else '✗ FAIL'}")

print(f"\nADDITIONAL METRICS:")
print(f"  valid_id_coverage: {valid_id_coverage:.3f}")

# ============================================================================
# STEP 4: RENDER ANNOTATED VIDEO
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: RENDER ANNOTATED VIDEO")
print("=" * 80)

try:
    from services.tracking_service import render_tracking

    render_path = f"temp/{job_id}/verify_tracking"
    os.makedirs(render_path, exist_ok=True)

    print(f"Rendering annotated frames to {render_path}/...")

    # Render first 100 frames (to keep file size manageable)
    frames_to_render = min(len(tracking_results), 100)

    render_tracking(
        video_path=video_path,
        tracking_results=tracking_results[:frames_to_render],
        output_dir=render_path,
        frame_stride=frame_stride
    )

    print(f"✓ Rendered {frames_to_render} frames")

    # Create video from frames
    import cv2
    import glob

    frame_files = sorted(glob.glob(f"{render_path}/*.png"))
    if frame_files:
        print(f"Creating video from {len(frame_files)} frames...")

        # Read first frame to get dimensions
        img = cv2.imread(frame_files[0])
        h, w = img.shape[:2]

        # Create video writer (mp4 with codec)
        video_out = f"temp/{job_id}/annotated_tracking.mp4"
        os.makedirs(os.path.dirname(video_out), exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_out, fourcc, 30.0, (w, h))

        for frame_file in frame_files:
            img = cv2.imread(frame_file)
            if img is not None:
                out.write(img)

        out.release()

        file_size_mb = os.path.getsize(video_out) / (1024 * 1024)
        print(f"✓ Video created: {video_out} ({file_size_mb:.1f}MB)")
    else:
        print("⚠ No rendered frames found")

except Exception as e:
    print(f"⚠ Video rendering failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# STEP 5: DOWNLOAD RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: DOWNLOAD RESULTS")
print("=" * 80)

try:
    from google.colab import files

    files_to_download = [
        f"temp/{job_id}/annotated_tracking.mp4",
        f"temp/{job_id}/tracking/identity_metrics.json"
    ]

    print("Preparing files for download...")

    for file_path in files_to_download:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ⚠ {file_path} not found")

    print("\nDownloading files...")
    files.download(f"temp/{job_id}/annotated_tracking.mp4")
    files.download(f"temp/{job_id}/tracking/identity_metrics.json")

    print("✓ Files downloaded to local machine")

except ImportError:
    print("⚠ Not running in Colab (google.colab not available)")
    print(f"Results saved to: {os.path.abspath(f'temp/{job_id}')}")

# ============================================================================
# FINAL VERDICT
# ============================================================================
print("\n" + "=" * 80)
print("PHASE D STEP 2 VERDICT")
print("=" * 80)

all_pass = gate_1_pass and gate_2_pass and gate_3_pass

if all_pass:
    print(f"\n✓✓✓ PHASE D STEP 2: PASSED ✓✓✓")
    print(f"    Identity locks building correctly with identity_frame_seq fix")
    print(f"    Ready for deployment")
else:
    print(f"\n⚠ PHASE D STEP 2: INCOMPLETE")
    if not gate_1_pass:
        print(f"    locks_created={locks_created} (target >=20) — identity seeding/lock creation issue")
    if not gate_2_pass:
        print(f"    lock_retention={lock_retention:.3f} (target >=0.65) — locks being dropped too early")
    if not gate_3_pass:
        print(f"    collapse_lock_creations={collapse_lock_creations} (must be 0) — Hungarian locked during collapse")

print(f"\n  Pipeline time: {elapsed:.1f}s")
print(f"  Frames processed: {len(tracking_results)}")
print(f"  Video: {video_path}")
print(f"  Results: temp/{job_id}/")

print("\n" + "=" * 80)
print("END PHASE D STEP 2")
print("=" * 80)
