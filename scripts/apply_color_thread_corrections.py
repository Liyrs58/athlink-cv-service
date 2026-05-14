#!/usr/bin/env python3
"""Apply human color-thread sidecar corrections to tracking output."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from services.color_thread_service import apply_and_save


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply color-thread review corrections")
    p.add_argument("--track-results", required=True, help="Path to track_results.json")
    p.add_argument("--color-threads", required=True, help="Path to color_threads.json")
    p.add_argument("--review-file", required=True, help="Edited thread_review.csv or JSON corrections file")
    p.add_argument("--out", required=True, help="Output corrected track_results JSON path")
    p.add_argument("--threads-out", help="Optional output corrected color_threads.json path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    summary = apply_and_save(
        track_results_path=args.track_results,
        color_threads_path=args.color_threads,
        review_file_path=args.review_file,
        out_path=args.out,
        threads_out_path=args.threads_out,
    )
    print("=" * 60)
    print("COLOR THREAD CORRECTION RESULTS")
    print("=" * 60)
    print(f"  actions_requested: {summary.get('actions_requested', 0)}")
    print(f"  actions_applied:   {summary.get('actions_applied', 0)}")
    print(f"  actions_skipped:   {summary.get('actions_skipped', 0)}")
    print(f"  players_annotated: {summary.get('players_annotated', 0)}")
    print(f"  written:           {args.out}")
    if args.threads_out:
        print(f"  threads_written:   {args.threads_out}")
    if summary.get("skipped"):
        print("  skipped:")
        for item in summary["skipped"]:
            print(f"    - {item.get('action')}: {item.get('reason')}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
