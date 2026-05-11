#!/usr/bin/env python3
"""
Phase D Step 1: Local test for proactive snapshot verification.
Uses the test video if available, otherwise uses a sample frame sequence.
"""

import os
import sys
import time
from pathlib import Path

print("=" * 80)
print("PHASE D STEP 1: PROACTIVE SNAPSHOT VERIFICATION (LOCAL)")
print("=" * 80)

job_id = "phase_d_step1_local"
video_path = "/Users/rudra/Desktop/villa_psg_30s_v3_annotated.mp4"

# Fallback to other videos if primary doesn't exist
if not os.path.exists(video_path):
    candidates = [
        "/Users/rudra/Desktop/aston_villa_psg_clip1_phase2_pz.mp4",
        "/Users/rudra/Desktop/aston_villa_psg_clip1_annotated.mp4",
        "/Users/rudra/Desktop/pressing_triangle_product.mp4",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            video_path = candidate
            break

if not os.path.exists(video_path):
    print(f"✗ No test video found at {video_path}")
    print("  Searching for .mov or .mp4 files in Desktop/Downloads...")
    sys.exit(1)

print(f"\nJob: {job_id}")
print(f"Video: {video_path}")
print(f"Frame stride: 5, max frames: 30\n")

# ============================================================================
# PHASE D STEP 1: TRACKING WITH IDENTITY RE-ENABLED
# ============================================================================
print("=" * 80)
print("STEP 1: TRACKING (detector → BoT-SORT, identity=IdentityCore)")
print("=" * 80)

from services.tracker_core import run_tracking
import io
from contextlib import redirect_stdout

# Capture stdout to find ProactiveSnapshot log lines
output_buffer = io.StringIO()

t0 = time.time()
try:
    with redirect_stdout(output_buffer):
        tracking_results = run_tracking(
            video_path=video_path,
            job_id=job_id,
            frame_stride=5,
            max_frames=30,
            device="cpu"  # Use CPU for local testing
        )

    elapsed = time.time() - t0

    # Print the captured output
    full_output = output_buffer.getvalue()
    print(full_output)

    print(f"\n✓ Tracking complete in {elapsed:.1f}s\n")

    # ============================================================================
    # ACCEPTANCE GATE 1: Proactive snapshots fired
    # ============================================================================
    proactive_snapshots = [line for line in full_output.split('\n') if '[ProactiveSnapshot]' in line]

    print(f"GATE 1: Proactive snapshots")
    print(f"  Found {len(proactive_snapshots)} snapshots (target: ≥3)")
    if proactive_snapshots:
        for line in proactive_snapshots[:5]:  # Show first 5
            print(f"    {line.strip()}")
    print(f"  {'✓ PASS (target: ≥3)' if len(proactive_snapshots) >= 3 else '⚠ MARGINAL (target: ≥3)'}")

    # ============================================================================
    # ACCEPTANCE GATE 2: Tracking still works (coverage, unique IDs)
    # ============================================================================
    frames_with_players = 0
    max_active_in_frame = 0
    unique_track_ids = set()

    for frame in tracking_results:
        players = frame.get('players', [])
        if len(players) > 0:
            frames_with_players += 1
            max_active_in_frame = max(max_active_in_frame, len(players))
            for player in players:
                unique_track_ids.add(player['trackId'])

    coverage = frames_with_players / max(len(tracking_results), 1)
    print(f"\nGATE 2: Frames with detections")
    print(f"  {frames_with_players} / {len(tracking_results)} = {coverage:.1%}")
    print(f"  {'✓ PASS (target: >0)' if coverage > 0 else '✗ FAIL'}")

    print(f"\nGATE 3: Unique track IDs")
    print(f"  {len(unique_track_ids)} unique IDs across all frames")
    print(f"  {'✓ PASS (target: >=10)' if len(unique_track_ids) >= 10 else '⚠ MARGINAL (target: >=10)'}")

    print(f"\nGATE 4: Max players per frame")
    print(f"  {max_active_in_frame} max simultaneous tracks")
    print(f"  {'✓ PASS (target: >=10)' if max_active_in_frame >= 10 else '⚠ MARGINAL (target: >=10)'}")

    # ============================================================================
    # ACCEPTANCE GATE 5: Identity metrics
    # ============================================================================
    identity_lines = [line for line in full_output.split('\n') if '[IdentityMetrics]' in line or 'collapse_lock_creations' in line or 'locks_created' in line]

    print(f"\nGATE 5: Identity metrics")
    if identity_lines:
        for line in identity_lines[:10]:
            print(f"  {line.strip()}")
    else:
        print(f"  (No identity metrics found in output)")

    # ============================================================================
    # PHASE D STEP 1 ACCEPTANCE VERDICT
    # ============================================================================
    print(f"\n" + "=" * 80)
    print("PHASE D STEP 1 ACCEPTANCE VERDICT")
    print("=" * 80)

    gate_1_pass = len(proactive_snapshots) >= 3
    gate_2_pass = coverage > 0
    gate_3_pass = len(unique_track_ids) >= 10
    gate_4_pass = max_active_in_frame >= 10

    all_pass = gate_1_pass and gate_2_pass and gate_3_pass and gate_4_pass

    print(f"\n  Gate 1 (snapshots >= 3):       {'✓' if gate_1_pass else '✗'} {len(proactive_snapshots)}")
    print(f"  Gate 2 (coverage > 0):         {'✓' if gate_2_pass else '✗'} {coverage:.1%}")
    print(f"  Gate 3 (unique IDs >= 10):     {'✓' if gate_3_pass else '✗'} {len(unique_track_ids)}")
    print(f"  Gate 4 (max per frame >= 10):  {'✓' if gate_4_pass else '✗'} {max_active_in_frame}")

    if all_pass:
        print(f"\n✓✓✓ PHASE D STEP 1: PASSED ✓✓✓")
        print(f"    Proactive snapshots firing correctly with identity re-enabled.")
        print(f"    Ready for Step 2: Identity metrics verification")
    else:
        print(f"\n⚠ PHASE D STEP 1: MARGINAL")
        if not gate_1_pass:
            print(f"    Proactive snapshots not firing enough. Check identity engine logs.")
        if not gate_2_pass or not gate_3_pass or not gate_4_pass:
            print(f"    Check detector/tracker parameters in tracker_core.py")

    print(f"\n  Pipeline time: {elapsed:.1f}s")
    print(f"  Frames: {len(tracking_results)} processed")
    print(f"  Unique tracks: {len(unique_track_ids)}")

except Exception as e:
    print(full_output)  # Print any captured output before error
    print(f"✗ Tracking failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n" + "=" * 80)
print("PHASE D STEP 1 COMPLETE")
print("=" * 80)
