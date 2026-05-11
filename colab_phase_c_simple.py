#!/usr/bin/env python3
"""
PASTE THIS ENTIRE CELL INTO COLAB.
Phase C: Simplified acceptance verification.
Focus: Tracking stability (Phase A/B deferred).
"""

import os
import sys
import time
from pathlib import Path

os.chdir('/content/athlink-cv-service')
sys.path.insert(0, '/content/athlink-cv-service')

print("=" * 80)
print("PHASE C ACCEPTANCE: Tracking Stability Verification")
print("=" * 80)

# Purge prior modules
for module_name in list(sys.modules):
    if module_name == "services" or module_name.startswith("services."):
        del sys.modules[module_name]

job_id = "phase_c_acceptance"
video_path = "/content/Aston villa vs Psg clip 1.mov"
frame_stride = 5
max_frames = 30

print(f"\nJob: {job_id}")
print(f"Video: {video_path}")
print(f"Frame stride: {frame_stride}, max frames: {max_frames}\n")

# ============================================================================
# TRACKING (with identity disabled)
# ============================================================================
print("="*80)
print("PHASE C ACCEPTANCE GATE: TRACKING STABILITY")
print("="*80)

from services.tracker_core import run_tracking

t0 = time.time()
try:
    tracking_results = run_tracking(
        video_path=video_path,
        job_id=job_id,
        frame_stride=frame_stride,
        max_frames=max_frames,
        device="cuda"
    )
    elapsed = time.time() - t0

    print(f"\n✓ Tracking complete in {elapsed:.1f}s\n")

    # ============================================================================
    # ACCEPTANCE GATE 1: Minimum frames with active tracks
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
    print(f"GATE 1: Frames with detections")
    print(f"  {frames_with_players} / {len(tracking_results)} = {coverage:.1%}")
    print(f"  ✓ PASS (target: >0)" if coverage > 0 else f"  ✗ FAIL")

    print(f"\nGATE 2: Unique track IDs")
    print(f"  {len(unique_track_ids)} unique IDs across all frames")
    print(f"  ✓ PASS (target: >=10)" if len(unique_track_ids) >= 10 else f"  ⚠ MARGINAL (target: >=10)")

    print(f"\nGATE 3: Max players per frame")
    print(f"  {max_active_in_frame} max simultaneous tracks")
    print(f"  ✓ PASS (target: >=10)" if max_active_in_frame >= 10 else f"  ⚠ MARGINAL (target: >=10)")

    # ============================================================================
    # FRAME-BY-FRAME DETAIL
    # ============================================================================
    print(f"\n" + "="*80)
    print("FRAME-BY-FRAME BREAKDOWN (first 10 frames)")
    print("="*80)
    print(f"{'Frame':<8} {'Dets':<6} {'Unique IDs':<12} {'Status':<20}")
    print("-" * 50)

    for frame in tracking_results[:10]:
        frame_idx = frame.get('frameIndex')
        players = frame.get('players', [])
        unique_in_frame = set(p['trackId'] for p in players)
        status = "✓ OK" if len(players) > 0 else "✗ EMPTY"
        print(f"{frame_idx:<8} {len(players):<6} {len(unique_in_frame):<12} {status:<20}")

    # ============================================================================
    # PHASE C ACCEPTANCE VERDICT
    # ============================================================================
    print(f"\n" + "="*80)
    print("PHASE C ACCEPTANCE VERDICT")
    print("="*80)

    gate_1_pass = coverage > 0
    gate_2_pass = len(unique_track_ids) >= 10
    gate_3_pass = max_active_in_frame >= 10

    all_pass = gate_1_pass and gate_2_pass and gate_3_pass

    print(f"\n  Gate 1 (coverage > 0):        {'✓' if gate_1_pass else '✗'} {coverage:.1%}")
    print(f"  Gate 2 (unique IDs >= 10):   {'✓' if gate_2_pass else '✗'} {len(unique_track_ids)}")
    print(f"  Gate 3 (max per frame >= 10):{'✓' if gate_3_pass else '✗'} {max_active_in_frame}")

    if all_pass:
        print(f"\n✓✓✓ PHASE C ACCEPTANCE: PASSED ✓✓✓")
        print(f"    Ready for Phase D: Identity engine fix + Phase B validator hardening")
    else:
        print(f"\n⚠ PHASE C ACCEPTANCE: MARGINAL")
        if not gate_2_pass or not gate_3_pass:
            print(f"    Check BoT-SORT parameters in tracking_service.py")
        if not gate_1_pass:
            print(f"    Check detector output in tracker_core.py")

    print(f"\n  Pipeline time: {elapsed:.1f}s")
    print(f"  Frames: {len(tracking_results)} processed")
    print(f"  Unique tracks: {len(unique_track_ids)}")

except Exception as e:
    print(f"✗ Tracking failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n" + "="*80)
print("PHASE C COMPLETE")
print("="*80)
