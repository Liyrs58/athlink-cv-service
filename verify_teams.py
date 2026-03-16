#!/usr/bin/env python3
"""
verify_teams.py
Usage:
    python3 verify_teams.py <jobId>
    python3 verify_teams.py <jobId> --render
"""

import json, os, sys, argparse
import cv2

TEAM_COLOURS = {
    0: (255, 100, 0),   # Blue
    1: (0, 100, 255),   # Red
   -1: (128, 128, 128), # Grey = unassigned
}

def verify_teams(job_id, render=False):
    team_path = os.path.join("temp", job_id, "tracking", "team_results.json")
    track_path = os.path.join("temp", job_id, "tracking", "track_results.json")

    if not os.path.exists(team_path):
        print(f"[ERROR] Not found: {team_path}")
        sys.exit(1)

    with open(team_path) as f:
        raw = json.load(f)
        teams = raw["tracks"] if isinstance(raw, dict) else raw

    if not teams:
        print("[WARNING] No tracks in team_results.json")
        sys.exit(1)

    team0 = [t for t in teams if t["teamId"] == 0]
    team1 = [t for t in teams if t["teamId"] == 1]
    skipped = [t for t in teams if t["teamId"] == -1]

    print(f"\nTeam Summary — Job: {job_id}")
    print("=" * 50)
    print(f"  Total tracks : {len(teams)}")
    print(f"  Team 0 (blue): {len(team0)}")
    print(f"  Team 1 (red) : {len(team1)}")
    print(f"  Unassigned   : {len(skipped)}")
    print()

    print(f"{'TrackID':>8}  {'TeamID':>6}  {'Hits':>5}  {'FirstSeen':>9}  {'LastSeen':>8}")
    print("-" * 50)
    for t in sorted(teams, key=lambda x: (x["teamId"], x["trackId"])):
        print(f"{t['trackId']:>8}  {t['teamId']:>6}  {t['hits']:>5}  {t['firstSeen']:>9}  {t['lastSeen']:>8}")

    if not render:
        return

    if not os.path.exists(track_path):
        print(f"\n[WARN] track_results.json not found, cannot render.")
        return

    with open(track_path) as f:
        track_data = json.load(f)

    video_path = track_data["videoPath"]
    if not os.path.exists(video_path):
        print(f"\n[WARN] Video not found: {video_path}")
        return

    # Build teamId lookup
    team_map = {t["trackId"]: t["teamId"] for t in teams}

    # Build frame → [(trackId, bbox, teamId)]
    frame_lookup = {}
    for track in track_data["tracks"]:
        tid = track["trackId"]
        team_id = team_map.get(tid, -1)
        for point in track["trajectory"]:
            fi = point["frameIndex"]
            if fi not in frame_lookup:
                frame_lookup[fi] = []
            frame_lookup[fi].append((tid, point["bbox"], team_id))

    out_dir = os.path.join("temp", job_id, "verify_teams")
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    raw_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    raw_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    needs_rotation = raw_h > raw_w   # portrait-mode phone recording
    rendered = 0

    for frame_idx in sorted(frame_lookup.keys()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, img = cap.read()
        if not ret or img is None:
            continue
        if needs_rotation:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        h, w = img.shape[:2]
        for tid, bbox, team_id in frame_lookup[frame_idx]:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            colour = TEAM_COLOURS.get(team_id, (128, 128, 128))
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
            label = f"T{tid} Team{team_id}"
            cv2.putText(img, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)

        ts = round(frame_idx / fps, 2)
        cv2.putText(img, f"frame {frame_idx} | t={ts}s",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 0), 1, cv2.LINE_AA)

        out_path = os.path.join(out_dir, f"team_{frame_idx:06d}.jpg")
        cv2.imwrite(out_path, img)
        rendered += 1

    cap.release()
    print(f"\nRendered {rendered} frames → temp/{job_id}/verify_teams/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("job_id")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    verify_teams(args.job_id, render=args.render)
