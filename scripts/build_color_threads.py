#!/usr/bin/env python3
"""Build post-tracking color threads and a review sidecar."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from services.color_thread_service import build_and_save, write_review_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build color-thread identity sidecars from tracking output")
    p.add_argument("--job-id", required=True, help="Job ID used for default temp paths")
    p.add_argument("--track-results", help="Path to track_results.json")
    p.add_argument("--camera-motion", help="Path to camera_motion.json")
    p.add_argument("--out", help="Output color_threads.json path")
    p.add_argument("--review-out", help="Output thread_review.csv path")
    p.add_argument("--max-segment-gap", type=int, default=8)
    p.add_argument("--max-segment-jump", type=float, default=180.0)
    p.add_argument("--max-reconnect-gap", type=int, default=45)
    p.add_argument("--max-reconnect-distance", type=float, default=260.0)
    p.add_argument("--min-reconnect-confidence", type=float, default=0.42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    job_id = args.job_id
    track_results = args.track_results or f"temp/{job_id}/tracking/track_results.json"
    camera_motion = args.camera_motion or f"temp/{job_id}/tracking/camera_motion.json"
    out = args.out or f"temp/{job_id}/tracking/color_threads.json"
    review_out = args.review_out or f"temp/{job_id}/thread_review.csv"

    color_threads = build_and_save(
        track_results_path=track_results,
        camera_motion_path=camera_motion,
        color_threads_path=out,
        max_segment_gap=args.max_segment_gap,
        max_segment_jump=args.max_segment_jump,
        max_reconnect_gap=args.max_reconnect_gap,
        max_reconnect_distance=args.max_reconnect_distance,
        min_reconnect_confidence=args.min_reconnect_confidence,
    )
    review_rows = write_review_csv(color_threads, review_out)
    stats = color_threads.get("stats", {})
    print("=" * 60)
    print("COLOR THREADS")
    print("=" * 60)
    print(f"  threads:       {stats.get('threads', 0)}")
    print(f"  segments:      {stats.get('segments', 0)}")
    print(f"  review_events: {review_rows}")
    print(f"  written:       {out}")
    print(f"  review_file:   {review_out}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
