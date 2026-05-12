import os
import sys
import argparse
from pathlib import Path

# Add root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.tracker_core import run_tracking

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"Running tracker on {args.video} (job: {args.job_id})...")
    results = run_tracking(
        video_path=args.video,
        job_id=args.job_id,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        device=args.device
    )
    print(f"Tracking complete. Results saved to temp/{args.job_id}/tracking/track_results.json")

if __name__ == "__main__":
    main()
