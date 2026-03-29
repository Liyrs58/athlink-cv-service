"""
test_pose.py

Standalone script to test Phase 1 pose extraction.
Finds the most recent completed job, hits the whatif test endpoint,
and prints a readable summary.

Exit code 0 = pass, 1 = fail.
"""

import json
import os
import sys

import requests

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8001")


def find_most_recent_job_id() -> str:
    """Find the most recent completed job from the jobs list endpoint."""
    resp = requests.get(f"{BASE_URL}/api/v1/jobs/list", timeout=10)
    if resp.status_code != 200:
        # Fallback: scan disk
        jobs_dir = os.environ.get("JOBS_DIR", "/tmp/athlink_jobs")
        if not os.path.isdir(jobs_dir):
            print(f"ERROR: No jobs found (jobs endpoint returned {resp.status_code}, "
                  f"and {jobs_dir} does not exist)")
            sys.exit(1)

        best_id = None
        best_time = 0
        for fname in os.listdir(jobs_dir):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(jobs_dir, fname)) as f:
                    job = json.load(f)
                if job.get("status") == "completed" and job.get("completedAt", 0) > best_time:
                    best_time = job["completedAt"]
                    best_id = job["jobId"]
            except Exception:
                continue

        if not best_id:
            print("ERROR: No completed jobs found on disk")
            sys.exit(1)
        return best_id

    jobs = resp.json()
    completed = [j for j in jobs if j.get("status") == "completed"]
    if not completed:
        print("ERROR: No completed jobs found")
        sys.exit(1)

    # Most recent by completedAt
    completed.sort(key=lambda j: j.get("completedAt", 0), reverse=True)
    return completed[0]["jobId"]


def main():
    job_id = find_most_recent_job_id()
    print(f"Testing with job_id: {job_id}")

    # Quick summary check first
    print("Checking job data...")
    summary = requests.get(f"{BASE_URL}/api/v1/whatif/test/{job_id}/summary", timeout=10)
    if summary.status_code == 200:
        s = summary.json()
        print(f"  Status: {s.get('status')}")
        print(f"  Video: {s.get('video_path', 'not found')}")
        print(f"  Tracks: {s.get('total_tracks')}")
        print(f"  Frames: {s.get('frames_processed')}")
        if s.get("status") == "no_video":
            print("\nERROR: Video file not found. Re-submit a video first.")
            sys.exit(1)
    else:
        print(f"  Summary check failed: {summary.status_code}")

    print("\nRunning RTMPose + MotionBERT (this may take a few minutes)...")
    response = requests.get(
        f"{BASE_URL}/api/v1/whatif/test/{job_id}",
        params={"start_frame": 100, "end_frame": 225},
        timeout=600,
    )

    if response.status_code != 200:
        print(f"FAILED: {response.status_code}")
        try:
            print(response.json().get("detail", response.text[:500]))
        except Exception:
            print(response.text[:500])
        sys.exit(1)

    data = response.json()
    sanity = data["sanity_checks"]

    print(f"\n{'='*50}")
    print(f"  PHASE 1 RESULTS")
    print(f"{'='*50}")
    print(f"  Frames processed:    {data['frames_processed']}")
    print(f"  Frames with players: {sanity['frames_with_players']}")
    print(f"  Avg players/frame:   {sanity['avg_players_per_frame']}")
    print(f"  Players with 3D:     {sanity['players_with_3d']}")
    print(f"  Head above hips:     {sanity['head_above_hips']}")

    if sanity.get("sample_player"):
        sp = sanity["sample_player"]
        print(f"\n  Sample player (track #{sp['track_id']}, team {sp['team_id']}):")
        print(f"  First 5 keypoints (3D, root-relative):")
        names = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]
        for i, kp in enumerate(sp.get("keypoints_3d_sample", [])):
            print(f"    {names[i]:12s}: x={kp[0]:+.4f}  y={kp[1]:+.4f}  z={kp[2]:+.4f}")

    all_passed = (
        data["frames_processed"] > 20
        and sanity["avg_players_per_frame"] >= 2
        and sanity["players_with_3d"] >= 5
        and sanity["head_above_hips"]
    )

    if all_passed:
        print(f"\n  ✓ PHASE 1 COMPLETE — All checks passed")
        print(f"  Ready for Phase 2 (What-If physics engine)")
        sys.exit(0)
    else:
        print(f"\n  ✗ PHASE 1 INCOMPLETE — Some checks failed")
        if data["frames_processed"] <= 20:
            print("    - Not enough frames processed")
        if sanity["avg_players_per_frame"] < 2:
            print("    - Too few players per frame")
        if sanity["players_with_3d"] < 5:
            print("    - Too few players with 3D poses")
        if not sanity["head_above_hips"]:
            print("    - Head-above-hips check failed (inverted poses)")
        sys.exit(1)


if __name__ == "__main__":
    main()
