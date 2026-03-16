#!/usr/bin/env python3
"""
verify_pitch.py
Usage:
    python3 verify_pitch.py <jobId>
"""

import json, os, sys, argparse
import cv2
import numpy as np

TEAM_COLOURS = {
    0:  (255, 100,   0),   # Blue  (BGR)
    1:  (  0, 100, 255),   # Red
    2:  (  0, 200, 100),   # Green (goalkeeper)
   -1:  (128, 128, 128),   # Grey  (unassigned)
}

PITCH_WIDTH  = 105.0
PITCH_HEIGHT = 68.0
CANVAS_W = 1050
CANVAS_H = 680
MARGIN   = 50
TRAIL_LEN = 12


def pitch_to_canvas(x: float, y: float):
    inner_w = CANVAS_W - 2 * MARGIN
    inner_h = CANVAS_H - 2 * MARGIN
    cx = int(MARGIN + (x / PITCH_WIDTH)  * inner_w)
    cy = int(MARGIN + (y / PITCH_HEIGHT) * inner_h)
    return cx, cy


def draw_pitch_background() -> np.ndarray:
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    canvas[:] = (34, 139, 34)   # grass green

    iw = CANVAS_W - 2 * MARGIN
    ih = CANVAS_H - 2 * MARGIN
    W  = (CANVAS_W, CANVAS_H)

    # Pitch boundary
    cv2.rectangle(canvas, (MARGIN, MARGIN), (CANVAS_W - MARGIN, CANVAS_H - MARGIN), (255, 255, 255), 2)
    # Centre line
    cv2.line(canvas, (CANVAS_W // 2, MARGIN), (CANVAS_W // 2, CANVAS_H - MARGIN), (255, 255, 255), 1)
    # Centre circle (radius ≈ 9.15 m → 9.15/68 of canvas height)
    r = int(ih * (9.15 / PITCH_HEIGHT))
    cv2.circle(canvas, (CANVAS_W // 2, CANVAS_H // 2), r, (255, 255, 255), 1)
    cv2.circle(canvas, (CANVAS_W // 2, CANVAS_H // 2), 4, (255, 255, 255), -1)

    # Penalty areas (16.5 m deep, 40.3 m wide → centred on goal line)
    pa_depth = int(iw * (16.5 / PITCH_WIDTH))
    pa_half  = int(ih * (20.15 / PITCH_HEIGHT))
    cy_mid   = CANVAS_H // 2
    # Left PA
    cv2.rectangle(canvas,
                  (MARGIN, cy_mid - pa_half),
                  (MARGIN + pa_depth, cy_mid + pa_half),
                  (255, 255, 255), 1)
    # Right PA
    cv2.rectangle(canvas,
                  (CANVAS_W - MARGIN - pa_depth, cy_mid - pa_half),
                  (CANVAS_W - MARGIN, cy_mid + pa_half),
                  (255, 255, 255), 1)

    # Goal areas (5.5 m deep, 18.3 m wide)
    ga_depth = int(iw * (5.5 / PITCH_WIDTH))
    ga_half  = int(ih * (9.15 / PITCH_HEIGHT))
    cv2.rectangle(canvas,
                  (MARGIN, cy_mid - ga_half),
                  (MARGIN + ga_depth, cy_mid + ga_half),
                  (255, 255, 255), 1)
    cv2.rectangle(canvas,
                  (CANVAS_W - MARGIN - ga_depth, cy_mid - ga_half),
                  (CANVAS_W - MARGIN, cy_mid + ga_half),
                  (255, 255, 255), 1)

    return canvas


def verify_pitch(job_id: str):
    pitch_path = os.path.join("temp", job_id, "pitch", "pitch_map.json")

    if not os.path.exists(pitch_path):
        print(f"[ERROR] Not found: {pitch_path}")
        sys.exit(1)

    with open(pitch_path) as f:
        data = json.load(f)

    players = data["players"]

    print(f"\nPitch Map — Job: {job_id}")
    print("=" * 50)
    print(f"  Frames processed : {data['framesProcessed']}")
    print(f"  Homography found : {data['homographyFound']}")
    print(f"  Players          : {len(players)}")
    print(f"  Team 0 (blue)    : {sum(1 for p in players if p['teamId'] == 0)}")
    print(f"  Team 1 (red)     : {sum(1 for p in players if p['teamId'] == 1)}")

    all_frames = set()
    for p in players:
        for pt in p["trajectory2d"]:
            all_frames.add(pt["frameIndex"])

    if not all_frames:
        print("[WARNING] No trajectory data to render.")
        return

    out_dir = os.path.join("temp", job_id, "verify_pitch")
    os.makedirs(out_dir, exist_ok=True)

    # frame → [(trackId, teamId, x, y)]
    frame_lookup: dict = {}
    for p in players:
        tid    = p["trackId"]
        team_id = p["teamId"]
        for pt in p["trajectory2d"]:
            fi = pt["frameIndex"]
            frame_lookup.setdefault(fi, []).append((tid, team_id, pt["x"], pt["y"]))

    # Build team_id lookup for trail drawing
    tid_to_team = {p["trackId"]: p["teamId"] for p in players}

    trails: dict = {}   # trackId → list of (x, y)
    rendered = 0

    for frame_idx in sorted(frame_lookup.keys()):
        canvas = draw_pitch_background()

        # Update trails
        for tid, team_id, x, y in frame_lookup[frame_idx]:
            trails.setdefault(tid, []).append((x, y))
            if len(trails[tid]) > TRAIL_LEN:
                trails[tid].pop(0)

        # Draw trails
        for tid, pts_list in trails.items():
            if len(pts_list) < 2:
                continue
            colour = TEAM_COLOURS.get(tid_to_team.get(tid, -1), (128, 128, 128))
            for i in range(1, len(pts_list)):
                p1 = pitch_to_canvas(pts_list[i-1][0], pts_list[i-1][1])
                p2 = pitch_to_canvas(pts_list[i][0], pts_list[i][1])
                cv2.line(canvas, p1, p2, colour, 1)

        # Draw players
        for tid, team_id, x, y in frame_lookup[frame_idx]:
            colour = TEAM_COLOURS.get(team_id, (128, 128, 128))
            cx, cy = pitch_to_canvas(x, y)
            cv2.circle(canvas, (cx, cy), 8, colour, -1)
            cv2.circle(canvas, (cx, cy), 8, (255, 255, 255), 1)
            cv2.putText(canvas, str(tid), (cx + 10, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(canvas, f"frame {frame_idx}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        out_path = os.path.join(out_dir, f"pitch_frame_{frame_idx:06d}.jpg")
        cv2.imwrite(out_path, canvas)
        rendered += 1

    print(f"\nRendered {rendered} frames → temp/{job_id}/verify_pitch/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("job_id")
    args = parser.parse_args()
    verify_pitch(args.job_id)
