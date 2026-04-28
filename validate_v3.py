#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from collections import defaultdict


def validate(job_id, stitched=False):
    suffix = "_stitched" if stitched else ""
    track_path = Path(f"temp/{job_id}/tracking/track_results{suffix}.json")

    if not track_path.exists():
        print(f"❌ File not found: {track_path}")
        return

    with open(track_path) as f:
        data = json.load(f)

    frames = data.get("frames", [])
    all_ids = set()
    id_lifespans = defaultdict(list)
    frame_counts = []

    for frame in frames:
        idx = frame.get("frameIndex", 0)
        players = frame.get("players", [])
        frame_counts.append(len(players))
        for p in players:
            tid = p.get("trackId")
            if tid is not None:
                all_ids.add(tid)
                id_lifespans[tid].append(idx)

    lengths = [len(fr) for fr in id_lifespans.values()]
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    short = sum(1 for l in lengths if l < 10)
    long = sum(1 for l in lengths if l > 100)

    drops = sum(1 for i in range(1, len(frame_counts)) if frame_counts[i] < frame_counts[i - 1] - 5)

    label = "STITCHED" if stitched else "RAW"
    print(f"\n{'='*60}")
    print(f"VALIDATION: {job_id} ({label})")
    print(f"{'='*60}")
    print(f"Unique IDs     : {len(all_ids)}  (target: 22)")
    print(f"Avg track len  : {avg_len:.1f} frames")
    print(f"Short (<10)    : {short}")
    print(f"Long (>100)    : {long}")
    print(f"Sudden drops   : {drops}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    job = sys.argv[1] if len(sys.argv) > 1 else "fix_test"
    validate(job, stitched=False)
    stitched = Path(f"temp/{job}/tracking/track_results_stitched.json")
    if stitched.exists():
        validate(job, stitched=True)
