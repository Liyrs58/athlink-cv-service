#!/usr/bin/env python3
"""CLI for the pan-safe full-FPS tracking renderer."""

from __future__ import annotations

import argparse
import os
import sys

# Allow running as a script from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from services.fullfps_tracking_renderer import render_video


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pan-safe full-FPS tracking renderer (identity stride=5, render stride=1)"
    )
    p.add_argument("--video", required=True, help="Source video path")
    p.add_argument("--job-id", required=True, help="Job ID (used for default paths)")
    p.add_argument("--track-results", help="Path to track_results.json")
    p.add_argument("--camera-motion", help="Path to camera_motion.json")
    p.add_argument("--identity-metrics", help="Path to identity_metrics.json")
    p.add_argument("--out", help="Output video path")
    p.add_argument("--identity-stride", type=int, default=5)
    p.add_argument("--render-stride", type=int, default=1)
    p.add_argument("--show-unknown", action="store_true",
                   help="Render UNKNOWN/PROVISIONAL entities as TID:* (debug)")
    p.add_argument("--debug-officials", action="store_true",
                   help="Render officials as gray REF labels (debug)")
    p.add_argument("--debug", action="store_true",
                   help="Add HUD and verbose labels")
    p.add_argument("--write-contact-sheet", action="store_true")
    p.add_argument("--no-qa-json", action="store_true",
                   help="Disable per-frame QA JSON output (default is on)")
    p.add_argument("--render-mode", choices=["production", "audit", "casefile"], default="production",
                   help="Render mode: production (clean), audit (show all boxes/IDs), casefile (VLM debug)")
    p.add_argument("--show-officials", action="store_true",
                   help="Show officials even if normally suppressed (default: false)")
    p.add_argument("--show-raw-id", action="store_true",
                   help="Show raw track ID in labels")
    p.add_argument("--show-confidence", action="store_true",
                   help="Show identity confidence in labels")
    p.add_argument("--strict", action="store_true",
                   help="Fail on missing camera_motion/identity_metrics, "
                        "dimension mismatch, duplicate PID, or hungarian/provisional render.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    job_id = args.job_id
    track_results = args.track_results or f"temp/{job_id}/tracking/track_results.json"
    camera_motion = args.camera_motion or f"temp/{job_id}/tracking/camera_motion.json"
    identity_metrics = args.identity_metrics or f"temp/{job_id}/tracking/identity_metrics.json"
    output_path = args.out or f"temp/{job_id}/annotated_tracking_fullfps_game_smooth.mp4"

    if args.render_stride != 1:
        print(f"[CLI] WARNING: --render-stride {args.render_stride} ignored; renderer always uses 1.")
    if args.identity_stride != 5:
        print(f"[CLI] WARNING: --identity-stride {args.identity_stride} is a label; identity is read as-is from track_results.")

    manifest = render_video(
        video_path=args.video,
        job_id=job_id,
        track_results_path=track_results,
        camera_motion_path=camera_motion,
        identity_metrics_path=identity_metrics,
        output_path=output_path,
        debug=args.debug,
        debug_unknown=args.show_unknown,
        debug_officials=args.debug_officials,
        write_contact_sheet=args.write_contact_sheet,
        write_qa_json=(not args.no_qa_json),
        strict=args.strict,
        render_mode=args.render_mode,
        show_officials=args.show_officials,
        show_raw_id=args.show_raw_id,
        show_confidence=args.show_confidence,
    )

    print("=" * 60)
    print(f"  rendered_frames={manifest['rendered_frames']}/{manifest['total_raw_frames']}")
    print(f"  tracked_range={manifest.get('tracked_first_frame')}..{manifest.get('tracked_last_frame')}")
    print(f"  tracking_coverage={manifest.get('tracking_coverage_ratio', 0.0):.1%}")
    print(f"  render_untracked_tail_frames={manifest.get('render_untracked_tail_frames', 0)}")
    print(f"  camera_motion_present={manifest.get('camera_motion_present')} samples={manifest.get('camera_motion_samples')}")
    print(f"  visible_object_frames={manifest['visible_object_frames']}")
    print(f"  locked_object_frames={manifest['locked_object_frames']}")
    print(f"  revived_object_frames={manifest['revived_object_frames']}")
    print(f"  duplicate_pid_suppressed={manifest['duplicate_pid_suppressed_object_frames']}")
    print(f"  official_suppressed={manifest['official_suppressed_object_frames']}")
    print(f"  fast_pan_frames={manifest['fast_pan_frames']} cut_frames={manifest['cut_frames']}")
    print(f"  identity_sources_rendered={manifest['identity_sources_rendered']}")
    if manifest.get("warnings"):
        print(f"  warnings={','.join(manifest['warnings'])}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
