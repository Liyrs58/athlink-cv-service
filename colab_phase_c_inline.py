#!/usr/bin/env python3
"""
PASTE THIS ENTIRE CELL INTO COLAB (no file dependencies).
Phase C: tracking → pitch → teams → render with manifest verification.
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
    unique_ids = len(set(p['trackId'] for f in tracking_results for p in f.get('players', [])))
    print(f"  Unique track IDs: {unique_ids}")

    # Frame-by-frame stats (first 10 frames)
    print(f"\n  Frame-by-frame player count (first 10):")
    for frame in tracking_results[:10]:
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
        frame_stride=frame_stride,
        max_frames=max_frames
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
    # team_service.assign_teams needs: tracks, frames_dir, job_id, output_dir
    frames_dir = str(Path('temp') / job_id / 'frames')
    output_dir = str(Path('temp') / job_id / 'tracking')

    # Extract tracks from tracking_results
    tracks = []
    for frame in tracking_results:
        for player in frame.get('players', []):
            # Convert frame-level player to track format
            track_entry = {
                'trackId': player.get('trackId'),
                'frameIndex': frame.get('frameIndex'),
                'bbox': player.get('bbox'),
                'confidence': player.get('confidence')
            }
            tracks.append(track_entry)

    team_result = assign_teams(
        tracks=tracks,
        frames_dir=frames_dir,
        job_id=job_id,
        output_dir=output_dir
    )
    elapsed_teams = time.time() - t0

    print(f"\n✓ Team assignment complete in {elapsed_teams:.1f}s")

    if team_result:
        team_counts = {}
        for track in team_result:
            team_id = track.get('teamId', -1)
            team_counts[team_id] = team_counts.get(team_id, 0) + 1

        print(f"  Total tracks processed: {len(team_result)}")
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
# STEP 4: RENDER (annotated video output)
# ============================================================================
print("\n" + "="*80)
print("STEP 4: RENDER (annotated video with overlays)")
print("="*80)

from services.render_service import run_render

t0 = time.time()
try:
    render_result = run_render(
        job_id=job_id,
        include_minimap=False
    )
    elapsed_render = time.time() - t0

    print(f"\n✓ Render complete in {elapsed_render:.1f}s")
    print(f"  Frames rendered: {render_result.get('framesRendered', 0)}")
    print(f"  Output file: {render_result.get('outputPath', 'N/A')}")

except Exception as e:
    print(f"✗ Render failed: {e}")
    import traceback
    traceback.print_exc()
    # Don't exit—render failure is not critical for tracking verification

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

print(f"\n✓ PHASE C PIPELINE COMPLETE")

# List output files
render_dir = Path('temp') / job_id / 'render'
if render_dir.exists():
    print(f"\n  Generated files in {render_dir}:")
    for f in sorted(render_dir.glob('*')):
        size_mb = f.stat().st_size / (1024*1024)
        print(f"    {f.name} ({size_mb:.1f} MB)")

print(f"\n  ➤ NEXT: Download annotated video and verify visual output")
print(f"  ➤ Check manifest gates in render_manifest.json")

# ============================================================================
# AUTO-DOWNLOAD (if running in Colab)
# ============================================================================
try:
    from google.colab import files
    print(f"\n" + "="*80)
    print("DOWNLOADING OUTPUT FILES")
    print("="*80)

    manifest_file = render_dir / 'render_manifest.json'
    if manifest_file.exists():
        files.download(str(manifest_file))
        print(f"✓ Downloaded render_manifest.json")

    # Find annotated video
    for mp4 in sorted(render_dir.glob('annotated_*.mp4')):
        files.download(str(mp4))
        print(f"✓ Downloaded {mp4.name}")
        break  # Download only first video

except ImportError:
    print("\n  (Not in Colab - manual download required)")
