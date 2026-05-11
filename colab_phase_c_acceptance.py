#!/usr/bin/env python3
"""
Phase C Acceptance: Full pipeline with manifest verification.
Run in Colab AFTER identity engine is disabled in tracker_core.py:228

Sequence:
1. Tracking (detector → BoT-SORT, identity disabled)
2. Pitch mapping (homography)
3. Team assignment (color clustering)
4. Render story (tactical overlay + manifest)

Outputs:
- temp/phase_c_run/tracking/track_results.json
- temp/phase_c_run/pitch/pitch_map.json
- temp/phase_c_run/tracking/team_results.json
- temp/phase_c_run/render/render_manifest.json
- temp/phase_c_run/render/annotated_*.mp4 (visual output)
"""

import os
import sys
import json
import time
from pathlib import Path

os.chdir('/content/athlink-cv-service')
sys.path.insert(0, '/content/athlink-cv-service')

print("=" * 80)
print("PHASE C ACCEPTANCE: Full Pipeline Execution")
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
# STEP 1: TRACKING (with identity disabled)
# ============================================================================
print("="*80)
print("STEP 1: TRACKING (detector → BoT-SORT, identity=NoOp)")
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
    elapsed_track = time.time() - t0

    print(f"\n✓ Tracking complete in {elapsed_track:.1f}s")
    print(f"  Frames processed: {len(tracking_results)}")
    print(f"  Unique track IDs: {len(set(p['trackId'] for f in tracking_results for p in f.get('players', [])))}")

    # Detailed frame-by-frame stats
    print(f"\n  Frame-by-frame player count:")
    for frame in tracking_results[:10]:  # First 10 frames
        frame_idx = frame.get('frameIndex')
        players = frame.get('players', [])
        track_ids = set(p['trackId'] for p in players)
        print(f"    Frame {frame_idx}: {len(players)} detections, {len(track_ids)} unique tracks")

except Exception as e:
    print(f"✗ Tracking failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 2: PITCH MAPPING
# ============================================================================
print("\n" + "="*80)
print("STEP 2: PITCH MAPPING (homography detection)")
print("="*80)

from services.pitch_service import map_pitch

t0 = time.time()
try:
    pitch_result = map_pitch(
        video_path=video_path,
        job_id=job_id,
        device="cuda"
    )
    elapsed_pitch = time.time() - t0

    print(f"\n✓ Pitch mapping complete in {elapsed_pitch:.1f}s")
    print(f"  Homography found: {pitch_result.get('homographyFound', False)}")
    print(f"  Pitch corners detected: {len(pitch_result.get('cornerPoints', []))}")
    if pitch_result.get('homographyConfidence'):
        print(f"  Homography confidence: {pitch_result['homographyConfidence']:.3f}")

except Exception as e:
    print(f"✗ Pitch mapping failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 3: TEAM ASSIGNMENT
# ============================================================================
print("\n" + "="*80)
print("STEP 3: TEAM ASSIGNMENT (color clustering)")
print("="*80)

from services.team_service import assign_teams

t0 = time.time()
try:
    team_result = assign_teams(
        job_id=job_id,
        video_path=video_path
    )
    elapsed_teams = time.time() - t0

    print(f"\n✓ Team assignment complete in {elapsed_teams:.1f}s")

    # Summarize team distribution
    if team_result and 'tracks' in team_result:
        tracks = team_result['tracks']
        team_counts = {}
        for track in tracks:
            team_id = track.get('teamId', -1)
            team_counts[team_id] = team_counts.get(team_id, 0) + 1

        print(f"  Total tracks: {len(tracks)}")
        print(f"  Team distribution:")
        for team_id in sorted(team_counts.keys()):
            count = team_counts[team_id]
            label = f"Team {team_id}" if team_id >= 0 else "Unassigned"
            print(f"    {label}: {count} tracks")

except Exception as e:
    print(f"✗ Team assignment failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 4: RENDER STORY (tactical overlay + manifest verification)
# ============================================================================
print("\n" + "="*80)
print("STEP 4: RENDER STORY (tactical overlay)")
print("="*80)

from services.render_performance_zone import render_story

t0 = time.time()
try:
    render_result = render_story(
        job_id=job_id,
        video_path=video_path
    )
    elapsed_render = time.time() - t0

    print(f"\n✓ Render story complete in {elapsed_render:.1f}s")

    # Parse manifest for verification
    manifest_path = Path('temp') / job_id / 'render' / 'render_manifest.json'
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

        print(f"\n  MANIFEST VERIFICATION:")
        print(f"    jobId: {manifest.get('jobId')}")
        print(f"    framesProcessed: {manifest.get('framesProcessed')}")
        print(f"    storyType: {manifest.get('storyType', 'NONE')}")

        if manifest.get('storyType') != 'NONE':
            overlay_frames = manifest.get('framesWithOverlay', [])
            total_frames = manifest.get('framesProcessed', 1)
            overlay_ratio = len(overlay_frames) / max(total_frames, 1)

            print(f"    overlayDrawnRatio: {overlay_ratio:.2%}")
            print(f"    ACCEPTANCE GATE: overlay_ratio >= 0.80? {overlay_ratio >= 0.80}")

            # Check per-frame validity
            per_frame = manifest.get('perFrameValidity', [])
            if per_frame:
                valid_count = sum(1 for pf in per_frame if pf.get('valid'))
                print(f"    validFrames / total: {valid_count} / {len(per_frame)}")

                # Check for failure reasons
                failures = {}
                for pf in per_frame:
                    if not pf.get('valid'):
                        for reason in pf.get('reasons', []):
                            failures[reason] = failures.get(reason, 0) + 1

                if failures:
                    print(f"    failureReasons:")
                    for reason, count in sorted(failures.items(), key=lambda x: -x[1]):
                        print(f"      {reason}: {count} frames")

            # Check story outcome
            story_outcome = manifest.get('storyOutcome', 'UNKNOWN')
            print(f"    storyOutcome: {story_outcome}")

            # Ball-to-carrier distances (should be <= 1.8m)
            if 'perFrameEvidence' in manifest:
                evidence = manifest['perFrameEvidence']
                distances = [e.get('ballToCarrierM') for e in evidence if e.get('ballToCarrierM')]
                if distances:
                    max_dist = max(distances)
                    avg_dist = sum(distances) / len(distances)
                    print(f"    ballToCarrier stats:")
                    print(f"      max: {max_dist:.2f}m (threshold: 1.8m)")
                    print(f"      avg: {avg_dist:.2f}m")

        else:
            print(f"    storyType: NONE (no tactical pattern detected)")

    else:
        print(f"  ⚠ Manifest not found at {manifest_path}")

except Exception as e:
    print(f"✗ Render story failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PHASE C ACCEPTANCE SUMMARY")
print("="*80)

total_elapsed = elapsed_track + elapsed_pitch + elapsed_teams + elapsed_render
print(f"\nTotal pipeline time: {total_elapsed:.1f}s")
print(f"  Tracking:  {elapsed_track:.1f}s")
print(f"  Pitch:     {elapsed_pitch:.1f}s")
print(f"  Teams:     {elapsed_teams:.1f}s")
print(f"  Render:    {elapsed_render:.1f}s")

print(f"\n✓ PHASE C COMPLETE")
print(f"  Next: Download annotated video and verify visual output")
print(f"  Output location: /content/athlink-cv-service/temp/{job_id}/render/")

# List output files
render_dir = Path('temp') / job_id / 'render'
if render_dir.exists():
    print(f"\n  Generated files:")
    for f in sorted(render_dir.glob('*')):
        size_mb = f.stat().st_size / (1024*1024)
        print(f"    {f.name} ({size_mb:.1f} MB)")
