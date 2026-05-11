#!/usr/bin/env python3
"""
Re-run tracking with FULL identity debug output.
This will show exactly where soft snapshot / locks fail.
"""

import os
import sys
import time
os.chdir('/content/athlink-cv-service')
sys.path.insert(0, '/content/athlink-cv-service')

print("=" * 70)
print("TRACKING WITH IDENTITY DEBUG")
print("=" * 70)

# Purge modules
for module_name in list(sys.modules):
    if module_name == "services" or module_name.startswith("services."):
        del sys.modules[module_name]

from services.tracker_core import run_tracking

print("\n⏳ Running tracking pipeline...")
t0 = time.time()

results = run_tracking(
    video_path="/content/Aston villa vs Psg clip 1.mov",
    job_id="identity_debug_run",
    frame_stride=5,
    max_frames=30,
    device="cuda"
)

elapsed = time.time() - t0

print(f"\n{'='*70}")
print(f"✓ Tracking complete: {elapsed:.1f}s")
print(f"✓ Frames processed: {len(results)}")
print(f"✓ Unique track IDs: {len(set(p['trackId'] for f in results for p in f.get('players', [])))}")
print(f"{'='*70}\n")

# Parse manifest for identity metrics
import json
from pathlib import Path

manifest_path = Path('temp/identity_debug_run/tracking/track_results.json')
if manifest_path.exists():
    with open(manifest_path) as f:
        data = json.load(f)

    print("MANIFEST SUMMARY:")
    print(f"  jobId: {data.get('jobId')}")
    print(f"  trackCount: {data.get('trackCount')}")
    print(f"  framesProcessed: {data.get('framesProcessed')}")

    # Check for identity metrics
    identity_summary = data.get('identity_summary', {})
    print(f"\nIDENTITY METRICS:")
    for key, val in identity_summary.items():
        print(f"  {key}: {val}")

    # Print first 5 frames' player details
    print(f"\nFIRST 5 FRAMES - PLAYER DETAILS:")
    print(f"{'Frame':<8} {'Players':<10} {'P-IDs':<10} {'Unknown':<10} {'State':<20}")
    print("-" * 60)

    frames_data = data.get('frames', [])
    for frame in frames_data[:5]:
        frame_idx = frame['frameIndex']
        players = frame.get('players', [])

        p_ids = sum(1 for p in players if p.get('displayId') and isinstance(p.get('displayId'), int))
        unknowns = sum(1 for p in players if p.get('displayId') == '?')
        state = frame.get('gameState', 'unknown')

        print(f"{frame_idx:<8} {len(players):<10} {p_ids:<10} {unknowns:<10} {state:<20}")
else:
    print(f"ERROR: Manifest not found at {manifest_path}")
