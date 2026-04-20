"""Aggregates tracking, pitch, and tactics data into mobile-friendly JSON.
"""

import json
import os
from pathlib import Path
import cv2

TEAM_COLORS = {0: "#0064FF", 1: "#FF3200"}


def _fill_missing_arcs(ball_frames, fps, hallucinate_fn, np):
    """Post-process: any gap between two grounded ball positions >5m apart
    that has all-zero Z gets a hallucinated parabolic arc."""
    i = 0
    while i < len(ball_frames) - 1:
        # Find next segment where Z is all zero
        j = i + 1
        while j < len(ball_frames) and ball_frames[j].get("pitchZ", 0) == 0:
            j += 1
        if j >= len(ball_frames):
            j = len(ball_frames) - 1
        a, b = ball_frames[i], ball_frames[j if j < len(ball_frames) else len(ball_frames) - 1]
        fi_a, fi_b = a["frameIndex"], b["frameIndex"]
        if fi_b - fi_a >= 3:
            xy_a = (a["pitchX"], a["pitchY"])
            xy_b = (b["pitchX"], b["pitchY"])
            zs = hallucinate_fn(fi_a, fi_b, xy_a, xy_b, fps)
            if zs.max() > 0:
                # Map frame indices to ball_frames indices
                fi_to_idx = {bf["frameIndex"]: k for k, bf in enumerate(ball_frames)}
                for off, z_val in enumerate(zs):
                    fi = fi_a + off
                    if fi in fi_to_idx and z_val > 0:
                        ball_frames[fi_to_idx[fi]]["pitchZ"] = round(float(z_val), 3)
        i = j
        if i == len(ball_frames) - 1:
            break


def build_export(job_id: str) -> dict:
    base = Path("temp") / job_id
    track_path = base / "tracking" / "track_results.json"
    team_path  = base / "tracking" / "team_results.json"

    if not track_path.exists():
        raise ValueError(f"track_results.json not found for job '{job_id}'")
    if not team_path.exists():
        raise ValueError(f"team_results.json not found for job '{job_id}'")

    with open(track_path) as f:
        track_data = json.load(f)
    with open(team_path) as f:
        raw_team = json.load(f)

    team_list = raw_team["tracks"] if isinstance(raw_team, dict) else raw_team

    pitch_path   = base / "pitch"   / "pitch_map.json"
    tactics_path = base / "tactics" / "tactics_results.json"
    pitch_data   = None
    tactics_data = None

    if pitch_path.exists():
        with open(pitch_path) as f:
            pitch_data = json.load(f)
    if tactics_path.exists():
        with open(tactics_path) as f:
            tactics_data = json.load(f)

    video_path = track_data.get("videoPath", "")
    video_meta = _get_video_meta(video_path)

    track_to_team = {}
    for entry in team_list:
        tid  = entry.get("trackId")
        team = entry.get("teamId")
        if tid is not None and team is not None:
            track_to_team[int(tid)] = int(team)

    team0_count = sum(1 for v in track_to_team.values() if v == 0)
    team1_count = sum(1 for v in track_to_team.values() if v == 1)
    teams = {
        "team0": {"color": TEAM_COLORS[0], "playerCount": team0_count},
        "team1": {"color": TEAM_COLORS[1], "playerCount": team1_count},
    }

    pitch_lookup = {}
    ball_frames: list = []
    if pitch_data:
        for player in pitch_data.get("players", []):
            # Ball entry lives under trackId=-1 with is_ball=True — extract separately.
            if player.get("is_ball"):
                for pt in player.get("trajectory2d", []):
                    ball_frames.append({
                        "frameIndex": int(pt["frameIndex"]),
                        "pitchX": float(pt["x"]),
                        "pitchY": float(pt["y"]),
                        # Z in metres above pitch plane. Populated when pitch
                        # calibration_valid=True and the point is inside a
                        # detected flight arc; otherwise 0.
                        "pitchZ": float(pt.get("z", 0.0)),
                        "source": pt.get("source", "unknown"),
                        "confidence": float(pt.get("confidence", 0.0)),
                        "reliable": str(pt.get("reliable", "False")),
                    })
                continue
            tid = int(player["trackId"])
            for pt in player.get("trajectory2d", []):
                key = (tid, int(pt["frameIndex"]))
                pitch_lookup[key] = (float(pt["x"]), float(pt["y"]))
    ball_frames.sort(key=lambda b: b["frameIndex"])

    fps = video_meta["fps"] if video_meta["fps"] > 0 else 30.0

    # Hallucinate arcs for likely aerial passes only
    if len(ball_frames) >= 2:
        try:
            from services.ball_tracking_service import hallucinate_ball_arc
            import numpy as np
            _fill_missing_arcs(ball_frames, fps, hallucinate_ball_arc, np)
        except ImportError:
            pass
    frame_map = {}
    for track in track_data.get("tracks", []):
        track_id = int(track["trackId"])
        team_id  = track_to_team.get(track_id, -1)
        for pt in track.get("trajectory", []):
            fi   = int(pt["frameIndex"])
            bbox = [int(pt["bbox"][0]), int(pt["bbox"][1]),
                    int(pt["bbox"][2]), int(pt["bbox"][3])]
            ts   = float(pt.get("timestampSeconds", fi / fps))
            pitch_x, pitch_y = None, None
            if (track_id, fi) in pitch_lookup:
                pitch_x, pitch_y = pitch_lookup[(track_id, fi)]
            player_entry = {
                "trackId": track_id, "teamId": team_id,
                "bbox": bbox, "pitchX": pitch_x, "pitchY": pitch_y,
            }
            if fi not in frame_map:
                frame_map[fi] = {"frameIndex": fi, "timestampSeconds": ts, "players": []}
            frame_map[fi]["players"].append(player_entry)

    # Fill all frames from 0 to frameCount-1 so the mobile app has a complete timeline
    total_fc = video_meta["frameCount"]
    for fi in range(total_fc):
        if fi not in frame_map:
            frame_map[fi] = {"frameIndex": fi, "timestampSeconds": fi / fps, "players": []}

    frames = sorted(frame_map.values(), key=lambda x: x["frameIndex"])

    tactics = {
        "team0Formation": None, "team1Formation": None,
        "team0Heatmap": None,   "team1Heatmap": None,
    }
    if tactics_data:
        t0 = tactics_data.get("team0", {})
        t1 = tactics_data.get("team1", {})
        tactics["team0Formation"] = t0.get("formation")
        tactics["team1Formation"] = t1.get("formation")
        tactics["team0Heatmap"]   = t0.get("heatmap")
        tactics["team1Heatmap"]   = t1.get("heatmap")

    space_occupation = None
    events = None
    if tactics_data:
        space_occupation = tactics_data.get("spaceOccupation")
        events = tactics_data.get("events")

    return {
        "jobId": job_id, "videoMeta": video_meta,
        "teams": teams, "frames": frames, "ball": ball_frames,
        "tactics": tactics,
        "spaceOccupation": space_occupation, "events": events,
    }

def _get_video_meta(video_path: str) -> dict:
    default = {"width": 0, "height": 0, "fps": 0.0, "durationSeconds": 0.0, "frameCount": 0}
    if not video_path or not os.path.exists(video_path):
        return default
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return default
    raw_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps      = float(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    width, height = (raw_h, raw_w) if raw_h > raw_w else (raw_w, raw_h)
    duration = (n_frames / fps) if fps > 0 else 0.0
    return {"width": width, "height": height, "fps": fps,
            "durationSeconds": round(duration, 3), "frameCount": n_frames}
