#!/usr/bin/env python3
"""
CLI — Detect identity conflicts in tracking output.

Usage:
    python3 scripts/detect_identity_conflicts.py \
        --job-id full_villa_psg \
        --track-results temp/full_villa_psg/tracking/track_results.json \
        --camera-motion temp/full_villa_psg/tracking/camera_motion.json

Writes:
    temp/{job_id}/identity_conflicts.json
"""

import argparse
import json
import sys
import os

# Ensure the repo root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.identity_conflict_detector import IdentityConflictDetector
from services.tracklet_graph_resolver import TrackletGraphResolver
from services.vlm_casefile_builder import VLMCaseFileBuilder


def main():
    p = argparse.ArgumentParser(description="Detect identity conflicts in tracking output")
    p.add_argument("--job-id", required=True, help="Job identifier")
    p.add_argument("--track-results", required=True, help="Path to track_results.json")
    p.add_argument("--camera-motion", default=None, help="Path to camera_motion.json")
    p.add_argument("--identity-metrics", default=None, help="Path to identity_metrics.json")
    p.add_argument("--video", default=None, help="Path to source video for contact sheets")
    p.add_argument("--build-casefiles", action="store_true", help="Build VLM case files")
    p.add_argument("--resolve", action="store_true",
                    help="Also run tracklet resolver and write patch plan")
    args = p.parse_args()

    job_id = args.job_id
    out_dir = f"temp/{job_id}"

    # Load inputs
    print(f"[CLI] Loading track results: {args.track_results}")
    with open(args.track_results) as f:
        track_results = json.load(f)

    camera_motion = None
    if args.camera_motion and os.path.exists(args.camera_motion):
        print(f"[CLI] Loading camera motion: {args.camera_motion}")
        with open(args.camera_motion) as f:
            camera_motion = json.load(f)

    identity_metrics = None
    if args.identity_metrics and os.path.exists(args.identity_metrics):
        with open(args.identity_metrics) as f:
            identity_metrics = json.load(f)

    # Step 1: Detect conflicts
    print("[CLI] Running conflict detection...")
    detector = IdentityConflictDetector()
    conflicts_path = f"{out_dir}/identity_conflicts.json"
    manifest = detector.detect_and_save(
        track_results, conflicts_path,
        camera_motion=camera_motion,
        identity_metrics=identity_metrics,
    )

    print(f"\n{'='*60}")
    print(f"CONFLICT DETECTION RESULTS")
    print(f"{'='*60}")
    print(f"  Total conflict windows:   {manifest['conflict_windows_total']}")
    print(f"  High severity:            {manifest['high_severity_conflicts']}")
    print(f"  Medium severity:          {manifest['medium_severity_conflicts']}")
    print(f"  Team flip conflicts:      {manifest['team_flip_conflicts']}")
    print(f"  Role flip conflicts:      {manifest['role_flip_conflicts']}")
    print(f"  Duplicate PID conflicts:  {manifest['duplicate_pid_conflicts']}")
    print(f"  Written to: {conflicts_path}")
    print(f"{'='*60}\n")

    # Step 2: Optionally resolve tracklets and propose corrections
    if args.resolve:
        print("[CLI] Running tracklet resolver...")
        resolver = TrackletGraphResolver()
        patch_plan = resolver.resolve(track_results, manifest["windows"])
        patch_plan_path = f"{out_dir}/identity_patch_plan.json"
        os.makedirs(out_dir, exist_ok=True)
        with open(patch_plan_path, "w") as f:
            json.dump(patch_plan, f, indent=2)

        print(f"  Tracklets built:          {len(patch_plan['tracklets'])}")
        print(f"  Corrections proposed:     {patch_plan['patches_proposed']}")
        print(f"  Written to: {patch_plan_path}")

    # Step 3: Optionally build VLM case files
    if args.build_casefiles:
        print("[CLI] Building VLM case files...")
        builder = VLMCaseFileBuilder(video_path=args.video)
        corrections = patch_plan["corrections"] if args.resolve else []
        casefile_dir = f"{out_dir}/vlm_casefiles"
        case_paths = builder.build_case_files(
            conflict_windows=manifest["windows"],
            track_results=track_results,
            corrections=corrections,
            output_dir=casefile_dir,
        )
        print(f"  Case files created:       {len(case_paths)}")
        for cp in case_paths[:5]:
            print(f"    - {cp}")
        if len(case_paths) > 5:
            print(f"    ... and {len(case_paths) - 5} more")


if __name__ == "__main__":
    main()
