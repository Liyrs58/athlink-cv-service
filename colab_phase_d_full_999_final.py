#!/usr/bin/env python3
"""
PASTE THIS INTO COLAB.
Phase D: Full 999-frame pipeline run on villa_psg with identity metrics capture.
Focus: Verify locks_created >= 20, lock_retention >= 0.65, collapse_lock_creations = 0.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path

print("=" * 80)
print("PHASE D: FULL 999-FRAME RUN")
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

# Purge prior modules
for module_name in list(sys.modules):
    if module_name == "services" or module_name.startswith("services."):
        del sys.modules[module_name]

job_id = "full_villa_psg_999"
video_path = "/content/1b16c594_villa_psg_40s_new.mp4"

print(f"\nJob: {job_id}")
print(f"Video: {video_path}")
print(f"Frame stride: 5, max frames: None (full video)")

# Check video exists
if not os.path.exists(video_path):
    alt_paths = ['/content/Aston villa vs Psg clip 1.mov', '/content/villa_psg.mp4']
    for alt in alt_paths:
        if os.path.exists(alt):
            video_path = alt
            print(f"Using: {video_path}")
            break

print("\n" + "=" * 80)
print("STEP 1: TRACKING (frame_stride=5)")
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
            frame_stride=5,
            max_frames=None,
            device="cuda"
        )
    elapsed = time.time() - t0
    full_output = output_buffer.getvalue()

    # Only print last 200 lines to avoid truncation
    lines = full_output.split('\n')
    print("...[earlier output truncated]...\n")
    print('\n'.join(lines[-200:]))

    print(f"\n✓ Tracking complete in {elapsed:.1f}s ({len(tracking_results)} frames)")

except Exception as e:
    print(f"✗ Tracking failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 2: IDENTITY METRICS EXTRACTION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: IDENTITY METRICS")
print("=" * 80)

identity_metrics = {}
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

# Also check for metrics JSON file
metrics_json_path = f"temp/{job_id}/tracking/identity_metrics.json"
if os.path.exists(metrics_json_path):
    try:
        with open(metrics_json_path) as f:
            metrics_data = json.load(f)
        identity_metrics.update(metrics_data)
        print(f"✓ Loaded metrics from {metrics_json_path}")
    except Exception as e:
        print(f"⚠ Could not load JSON: {e}")

print(f"\nExtracted metrics:")
for k, v in sorted(identity_metrics.items()):
    if isinstance(v, float):
        print(f"  {k}: {v:.3f}")
    else:
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
# STEP 4: FINAL VERDICT
# ============================================================================
print("\n" + "=" * 80)
print("PHASE D: FINAL VERDICT")
print("=" * 80)

all_pass = gate_1_pass and gate_2_pass and gate_3_pass

if all_pass:
    print(f"\n✓✓✓ PHASE D: PASSED ✓✓✓")
    print(f"    Ready for deployment")
else:
    print(f"\n⚠ PHASE D: INCOMPLETE")
    if not gate_1_pass:
        print(f"    locks_created={locks_created} (target >=20)")
    if not gate_2_pass:
        print(f"    lock_retention={lock_retention:.3f} (target >=0.65)")
    if not gate_3_pass:
        print(f"    collapse_lock_creations={collapse_lock_creations} (must be 0)")

print(f"\n  Elapsed: {elapsed:.1f}s")
print(f"  Frames processed: {len(tracking_results)}")
print(f"  Results: temp/{job_id}/")

# ============================================================================
# STEP 5: DOWNLOAD RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: DOWNLOAD RESULTS")
print("=" * 80)

try:
    from google.colab import files

    files_to_download = [
        f"temp/{job_id}/tracking/identity_metrics.json",
        f"temp/{job_id}/tracking/track_results.json",
    ]

    print("Downloading files...")
    for file_path in files_to_download:
        if os.path.exists(file_path):
            try:
                files.download(file_path)
                print(f"  ✓ {file_path}")
            except Exception as e:
                print(f"  ⚠ {file_path}: {e}")
        else:
            print(f"  ⚠ {file_path} not found")

except ImportError:
    print("⚠ Not in Colab (results in temp/{job_id}/)")

print("\n" + "=" * 80)
