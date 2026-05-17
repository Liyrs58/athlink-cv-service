#!/usr/bin/env python3
"""
PASTE THIS ENTIRE CELL INTO COLAB.
Phase D Step 2: Identity Lock Creation & Retention Verification (with GPU T4).
Focus: Verify locks_created >= 20, lock_retention >= 0.65, collapse_lock_creations = 0.
"""

import os, sys, shutil, subprocess, json, time
from pathlib import Path

# --- Configuration ---
REPO_URL = "https://github.com/liys58/athlink-cv-service.git"
REPO = Path("/content/athlink-cv-service")
VIDEO = Path("/content/1b16c594_villa_psg_40s_new.mp4")
JOB_ID = "phase_d_step2"

# Environment Controls
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["ATHLINK_FORCE_DEVICE"] = "cuda"
os.environ["ATHLINK_YOLO_HALF"] = "0"
os.environ["ATHLINK_MAX_PLAYER_SLOTS"] = "14"
os.environ["ATHLINK_ALLOW_NEW_PLAYER_SLOTS"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("=" * 80)
print("PHASE D STEP 2: IDENTITY LOCK RETENTION VERIFICATION")
print("=" * 80)

print("\n--- 1. GPU & Repo Setup ---")
import torch
assert torch.cuda.is_available(), "Switch Colab runtime to T4 GPU."
print(f"GPU: {torch.cuda.get_device_name(0)}")

if REPO.exists():
    print("Refreshing repository...")
    subprocess.run(["git", "-C", str(REPO), "fetch", "--all"], check=True)
    subprocess.run(["git", "-C", str(REPO), "reset", "--hard", "origin/main"], check=True)
else:
    subprocess.run(["git", "clone", REPO_URL, str(REPO)], check=True)

os.chdir(REPO)
commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
print(f"HEAD: {commit_hash}")

print("\n--- 2. Dependencies ---")
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "ultralytics", "boxmot", "huggingface_hub"], check=True)
try:
    import torchreid
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "git+https://github.com/KaiyangZhou/deep-person-reid.git"], check=True)
    import torchreid
print("✓ Dependencies OK")

# Link OSNet Weights if present
models_dir = REPO / "models"
models_dir.mkdir(exist_ok=True)
src_weights = Path("/content/osnet_x1_0_msmt17.pt")
dst_weights = models_dir / "osnet_x1_0_msmt17.pt"
if src_weights.exists():
    shutil.copy2(src_weights, dst_weights)
    print("✓ OSNet weights linked")

# Check video
if not VIDEO.exists():
    print(f"⚠ Video not found at {VIDEO}")
    alt = Path("/content/Aston villa vs Psg clip 1.mov")
    if alt.exists():
        VIDEO = alt
        print(f"Using fallback: {VIDEO}")
    else:
        print("ERROR: No video found. Upload 1b16c594_villa_psg_40s_new.mp4 or Aston villa vs Psg clip 1.mov to /content/")
        sys.exit(1)

print(f"Video: {VIDEO}")

print("\n--- 3. Running Tracking Pipeline ---")
print("=" * 80)

import sys
sys.path.insert(0, str(REPO))

# Purge prior modules for clean import
for module_name in list(sys.modules):
    if module_name == "services" or module_name.startswith("services."):
        del sys.modules[module_name]

from services.tracker_core import run_tracking
import io
from contextlib import redirect_stdout

output_buffer = io.StringIO()
frame_stride = 5
max_frames = None

t0 = time.time()
try:
    with redirect_stdout(output_buffer):
        tracking_results = run_tracking(
            video_path=str(VIDEO),
            job_id=JOB_ID,
            frame_stride=frame_stride,
            max_frames=max_frames,
            device="cuda"
        )

    elapsed = time.time() - t0
    full_output = output_buffer.getvalue()

    # Print last 3000 chars of output
    print(full_output[-3000:])
    print(f"\n✓ Tracking complete in {elapsed:.1f}s\n")

except Exception as e:
    print(f"✗ Tracking failed: {e}")
    import traceback
    traceback.print_exc()
    full_output = output_buffer.getvalue()
    print("\nFull output:")
    print(full_output[-5000:])
    sys.exit(1)

print("=" * 80)
print("STEP 2: IDENTITY METRICS EXTRACTION")
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
for k, v in sorted(identity_metrics.items()):
    print(f"  {k}: {v}")

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

print("\n" + "=" * 80)
print("FINAL VERDICT")
print("=" * 80)

all_pass = gate_1_pass and gate_2_pass and gate_3_pass

if all_pass:
    print(f"\n✅✅✅ PHASE D STEP 2: PASSED ✅✅✅")
    print(f"    Identity locks building correctly with dormancy fixes")
    print(f"    Ready for deployment")
else:
    print(f"\n⚠️  PHASE D STEP 2: INCOMPLETE")
    if not gate_1_pass:
        print(f"    ✗ locks_created={locks_created} (target >=20)")
    else:
        print(f"    ✓ locks_created={locks_created}")

    if not gate_2_pass:
        print(f"    ✗ lock_retention={lock_retention:.3f} (target >=0.65)")
    else:
        print(f"    ✓ lock_retention={lock_retention:.3f}")

    if not gate_3_pass:
        print(f"    ✗ collapse_lock_creations={collapse_lock_creations} (must be 0)")
    else:
        print(f"    ✓ collapse_lock_creations={collapse_lock_creations}")

print(f"\nPipeline Statistics:")
print(f"  Time: {elapsed:.1f}s")
print(f"  Frames processed: {len(tracking_results)}")
print(f"  Video: {VIDEO.name}")
print(f"  Results: temp/{JOB_ID}/")

print("\n" + "=" * 80)

# --- DRIFT REPORT ANALYSIS (Optional) ---
try:
    drift_report_path = REPO / f"temp/{JOB_ID}/tracking/drift_report.json"
    if drift_report_path.exists():
        with open(drift_report_path) as f:
            drift_report = json.load(f)

        print("\nDRIFT REPORT SUMMARY")
        print("=" * 80)
        print(f"\nDrift Threshold: {drift_report['config']['drift_threshold']}")
        print(f"Players Tracked: {len(drift_report['players'])}\n")

        high_drift = {pid: data for pid, data in drift_report['players'].items()
                      if data['final_stability'] < 0.70}

        if high_drift:
            print("High Drift Players (stability < 0.70):\n")
            for pid in sorted(high_drift.keys()):
                data = high_drift[pid]
                print(f"  {pid}: stability={data['final_stability']:.3f}, "
                      f"frames={data['total_frames']}, drift_events={data['drift_triggered_count']}")
        else:
            print("✓ No high-drift players detected!")

        print("\n" + "=" * 80)
except Exception as e:
    print(f"\n⚠ Could not load drift report: {e}")

# --- DOWNLOAD RESULTS ---
try:
    from google.colab import files

    print("\nDownloading results...\n")

    metrics_json = REPO / f"temp/{JOB_ID}/tracking/identity_metrics.json"
    if metrics_json.exists():
        files.download(str(metrics_json))

    drift_json = REPO / f"temp/{JOB_ID}/tracking/drift_report.json"
    if drift_json.exists():
        files.download(str(drift_json))

    print("\n✓ Files downloaded to local machine")

except ImportError:
    print("\n⚠ Not running in Colab (google.colab not available)")
    print(f"Results saved to: {REPO}/temp/{JOB_ID}/")

print("\nPHASE D STEP 2 COMPLETE")
print("=" * 80)
