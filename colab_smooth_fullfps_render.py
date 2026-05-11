#!/usr/bin/env python3
"""
PASTE THIS INTO COLAB.
Smooth full-FPS annotated tracking video.

Pipeline:
1. Run tracking at stride=5 (identity decisions)
2. Render smooth interpolated video at stride=1 (visual output)
3. Download results
"""

import os
import sys
import json
import subprocess
import time

print("=" * 80)
print("SMOOTH FULL-FPS ANNOTATED TRACKING")
print("=" * 80)

# Fetch repo
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

# Purge prior modules
for module_name in list(sys.modules):
    if module_name == "services" or module_name.startswith("services."):
        del sys.modules[module_name]

job_id = "full_villa_psg"
video_path = "/content/1b16c594_villa_psg_40s_new.mp4"
frame_stride = 5

print(f"\nJob: {job_id}")
print(f"Video: {video_path}")
print(f"Identity stride: {frame_stride}, Render stride: 1\n")

# Check video exists
if not os.path.exists(video_path):
    alt_paths = ['/content/Aston villa vs Psg clip 1.mov', '/content/aston villa vs psg 1.mov']
    for alt in alt_paths:
        if os.path.exists(alt):
            video_path = alt
            print(f"Using: {video_path}")
            break

# ============================================================================
# STEP 1: TRACKING (stride=5)
# ============================================================================
print("=" * 80)
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

    # Print last 150 lines
    lines = full_output.split('\n')
    print('\n'.join(lines[-150:]))

    print(f"\n✓ Tracking complete in {elapsed_track:.1f}s ({len(tracking_results)} sampled frames)")

except Exception as e:
    print(f"✗ Tracking failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Extract identity metrics
identity_metrics = {}
for line in full_output.split('\n'):
    if any(key in line for key in ['collapse_lock_creations', 'locks_created', 'lock_retention_rate', 'valid_id_coverage']):
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
    if isinstance(v, float):
        print(f"  {k}: {v:.3f}")
    else:
        print(f"  {k}: {v}")

# ============================================================================
# STEP 2: SMOOTH RENDER (stride=1)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: SMOOTH RENDER (stride=1 with interpolation)")
print("=" * 80)

from render_fullfps_smooth import render_smooth_tracking

try:
    output_video, manifest_path = render_smooth_tracking(
        video_path=video_path,
        tracking_results=tracking_results,
        identity_metrics=identity_metrics,
        job_id=job_id,
        identity_frame_stride=frame_stride,
        render_stride=1,
        max_hold_raw_frames=4,
        max_interp_gap_raw_frames=10,
        ema_alpha=0.25,
        show_unknown=True,
        show_provisional=True,
        show_trails=True,
        trail_length=10,
        show_hud=True,
    )
    print(f"\n✓ Smooth render complete")

except Exception as e:
    print(f"✗ Render failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 3: VERIFY OUTPUT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: VERIFY OUTPUT")
print("=" * 80)

if os.path.exists(output_video):
    file_size_mb = os.path.getsize(output_video) / (1024 * 1024)
    print(f"✓ Video: {output_video}")
    print(f"  Size: {file_size_mb:.1f} MB")
else:
    print(f"✗ Video not created")
    sys.exit(1)

if os.path.exists(manifest_path):
    print(f"✓ Manifest: {manifest_path}")
else:
    print(f"⚠ Manifest not found")

# ============================================================================
# STEP 4: DOWNLOAD RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: DOWNLOAD RESULTS")
print("=" * 80)

try:
    from google.colab import files

    print("Downloading...")
    if os.path.exists(output_video):
        files.download(output_video)
        print(f"✓ {output_video}")
    if os.path.exists(manifest_path):
        files.download(manifest_path)
        print(f"✓ {manifest_path}")

except ImportError:
    print("⚠ Not in Colab")

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
    print(f"\n✓✓✓ SMOOTH FULL-FPS RENDER: PASSED ✓✓✓")
else:
    print(f"\n⚠ SMOOTH FULL-FPS RENDER: INCOMPLETE")

print(f"\nOutput: {output_video}")
print(f"Manifest: {manifest_path}")
print("\n" + "=" * 80)
