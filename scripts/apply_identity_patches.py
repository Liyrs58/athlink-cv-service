#!/usr/bin/env python3
"""
CLI — Apply identity patches to tracking output.

Usage:
    python3 scripts/apply_identity_patches.py \
        --job-id full_villa_psg \
        --track-results temp/full_villa_psg/tracking/track_results.json \
        --patch-plan temp/full_villa_psg/identity_patch_plan.json \
        --out temp/full_villa_psg/tracking/track_results_patched.json

Reads the patch plan and applies validated corrections to produce
track_results_patched.json. The renderer then points at this file.
"""

import argparse
import json
import sys
import os

# Ensure the repo root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.identity_patch_service import IdentityPatchService, PatchValidator


def main():
    p = argparse.ArgumentParser(description="Apply identity patches to tracking output")
    p.add_argument("--job-id", required=True, help="Job identifier")
    p.add_argument("--track-results", required=True, help="Path to track_results.json")
    p.add_argument("--patch-plan", required=True, help="Path to identity_patch_plan.json")
    p.add_argument("--out", required=True, help="Output path for track_results_patched.json")
    p.add_argument("--min-confidence", type=float, default=0.75,
                    help="Minimum confidence for auto-applying patches (default: 0.75)")
    args = p.parse_args()

    # Load inputs
    print(f"[CLI] Loading track results: {args.track_results}")
    with open(args.track_results) as f:
        track_results = json.load(f)

    print(f"[CLI] Loading patch plan: {args.patch_plan}")
    with open(args.patch_plan) as f:
        patch_plan = json.load(f)

    # Apply patches
    print("[CLI] Applying patches...")
    validator = PatchValidator(min_confidence=args.min_confidence)
    service = IdentityPatchService(validator=validator)
    result = service.apply_and_save(track_results, patch_plan, args.out)

    manifest = result["manifest"]
    print(f"\n{'='*60}")
    print(f"IDENTITY PATCH RESULTS")
    print(f"{'='*60}")
    print(f"  Patches proposed:         {manifest['patches_proposed']}")
    print(f"  Patches applied:          {manifest['patches_applied']}")
    print(f"  Patches rejected:         {manifest['patches_rejected']}")
    print(f"  Written to: {args.out}")
    print(f"{'='*60}\n")

    if result["rejected"]:
        print("Rejected patches:")
        for rej in result["rejected"]:
            print(f"  [{rej['reason']}] {rej['window_id']} — {rej.get('detail', '')}")

    if result["applied"]:
        print("Applied patches:")
        for ap in result["applied"]:
            print(f"  [{ap['action']}] {ap['window_id']} — PIDs: {ap.get('pid_a', '?')}, {ap.get('pid_b', '?')}")


if __name__ == "__main__":
    main()
