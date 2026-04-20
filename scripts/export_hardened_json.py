"""
FC26 Hardened JSON Exporter — Google Colab / CLI
=================================================
Takes raw fc26_pro_backend.json and produces fc26_hardened.json with:
  - speed_kmh and vector [vx, vy] derived per frame per player
  - heading derived from velocity when missing
  - is_possession flag (nearest player to ball within 2m)
  - is_active flag (filters stationary duds)
  - All tracks clipped to consistent frame count
  - NaN gaps linearly interpolated (up to 10 frames)

Usage:
  python export_hardened_json.py input.json output.json
  # or in Colab: just run the cell after uploading your JSON
"""

import json
import math
import sys
from pathlib import Path


def interpolate_gaps(values: list[float | None], max_gap: int = 10) -> list[float]:
    """Linear interpolation across None/NaN gaps up to max_gap frames."""
    out = values[:]
    n = len(out)

    # Find first valid
    first = None
    for i in range(n):
        if out[i] is not None and not math.isnan(out[i]):
            first = i
            break
    if first is None:
        return [0.0] * n  # entirely missing — zero fill

    # Backfill before first valid
    for i in range(first):
        out[i] = out[first]

    # Forward pass — interpolate or hold
    i = first
    while i < n:
        if out[i] is not None and not math.isnan(out[i]):
            i += 1
            continue
        # Found gap start
        gap_start = i
        while i < n and (out[i] is None or math.isnan(out[i])):
            i += 1
        gap_end = i  # first valid after gap, or n

        if gap_end >= n:
            # Tail — hold last
            for k in range(gap_start, n):
                out[k] = out[gap_start - 1]
            break

        gap_len = gap_end - gap_start
        if gap_len <= max_gap:
            v0 = out[gap_start - 1]
            v1 = out[gap_end]
            for k in range(gap_len):
                t = (k + 1) / (gap_len + 1)
                out[gap_start + k] = v0 + (v1 - v0) * t
        else:
            # Hold last across long gap
            for k in range(gap_start, gap_end):
                out[k] = out[gap_start - 1]

    return out


def harden(raw: dict) -> dict:
    fps = raw.get("metadata", {}).get("fps", 30)
    tracks = raw.get("tracks", [])
    ball_data = raw.get("ball", [])

    # 1. Clip to consistent frame count
    min_frames = min(len(t["trajectory2d"]) for t in tracks) if tracks else 0
    if ball_data:
        min_frames = min(min_frames, len(ball_data))
    print(f"Clipping to {min_frames} stable frames (fps={fps})")

    # Build ball lookup by frameIndex
    ball_by_frame: dict[int, dict] = {}
    for b in ball_data[:min_frames]:
        ball_by_frame[b["frameIndex"]] = b

    # 2. Process each track
    for t in tracks:
        traj = t["trajectory2d"][:min_frames]

        # Extract raw positions
        xs = [s.get("x") for s in traj]
        ys = [s.get("y") for s in traj]
        headings = [s.get("heading") for s in traj]

        # Interpolate positions
        xs = interpolate_gaps(xs)
        ys = interpolate_gaps(ys)

        # Derive velocity + heading from position deltas
        total_movement = 0.0
        for i in range(len(traj)):
            s = traj[i]
            s["x"] = xs[i]
            s["y"] = ys[i]

            # Possession distance for all frames including 0
            fi = s["frameIndex"]
            ball = ball_by_frame.get(fi)
            if ball:
                bdist = math.sqrt(
                    (xs[i] - ball["x"]) ** 2 + (ys[i] - ball["y"]) ** 2
                )
                s["ball_dist"] = round(bdist, 2)
            else:
                s["ball_dist"] = 999.0

            if i == 0:
                s["speed_kmh"] = 0.0
                s["vector"] = [0.0, 0.0]
                continue

            dx = xs[i] - xs[i - 1]
            dy = ys[i] - ys[i - 1]
            dist_m = math.sqrt(dx * dx + dy * dy)
            speed_ms = dist_m * fps  # metres per second
            speed_kmh = speed_ms * 3.6

            s["speed_kmh"] = round(speed_kmh, 2)
            s["vector"] = [round(dx * fps, 3), round(dy * fps, 3)]
            total_movement += dist_m

            # Derive heading from velocity if missing
            if (headings[i] is None or math.isnan(headings[i])) and dist_m > 0.01:
                s["heading"] = round(math.degrees(math.atan2(dy, dx)) % 360, 2)
            elif headings[i] is not None and not math.isnan(headings[i]):
                s["heading"] = round(headings[i], 2)

            # ball_dist already set above

        # Copy frame 1 velocity to frame 0
        if len(traj) > 1:
            traj[0]["speed_kmh"] = traj[1]["speed_kmh"]
            traj[0]["vector"] = traj[1]["vector"]

        # Interpolate heading gaps
        raw_headings = [s.get("heading") for s in traj]
        filled_headings = interpolate_gaps(
            [h if h is not None else float("nan") for h in raw_headings]
        )
        for i, s in enumerate(traj):
            s["heading"] = round(filled_headings[i], 2)

        # Mark active/inactive
        t["is_active"] = total_movement > 0.5  # moved more than 0.5m total
        t["trajectory2d"] = traj

    # 3. Assign possession per frame (nearest player to ball within 2m)
    for fi in range(min_frames):
        best_track = -1
        best_dist = 999.0
        for t in tracks:
            s = t["trajectory2d"][fi]
            if s.get("ball_dist", 999) < best_dist:
                best_dist = s["ball_dist"]
                best_track = t["trackId"]

        for t in tracks:
            s = t["trajectory2d"][fi]
            s["is_possession"] = (
                t["trackId"] == best_track and best_dist < 2.0
            )
            # Clean up intermediate field
            del s["ball_dist"]

    # 4. Build output
    result = {
        "metadata": {
            **raw.get("metadata", {}),
            "hardened": True,
            "frame_count": min_frames,
        },
        "tracks": tracks,
        "ball": ball_data[:min_frames],
    }

    # Copy decision_points if present
    if "decision_points" in raw:
        result["decision_points"] = raw["decision_points"]

    # Stats
    active = sum(1 for t in tracks if t.get("is_active"))
    inactive = len(tracks) - active
    print(f"Tracks: {len(tracks)} total, {active} active, {inactive} duds")
    poss_frames = sum(
        1 for t in tracks
        for s in t["trajectory2d"]
        if s.get("is_possession")
    )
    print(f"Possession frames: {poss_frames}/{min_frames * len(tracks)}")

    return result


def main():
    if len(sys.argv) >= 3:
        in_path = Path(sys.argv[1])
        out_path = Path(sys.argv[2])
    elif len(sys.argv) == 2:
        in_path = Path(sys.argv[1])
        out_path = in_path.with_stem(in_path.stem + "_hardened")
    else:
        # Default paths for Colab
        in_path = Path("fc26_pro_backend.json")
        out_path = Path("fc26_hardened.json")

    print(f"Reading: {in_path}")
    raw = json.loads(in_path.read_text())

    result = harden(raw)

    out_path.write_text(json.dumps(result, separators=(",", ":")))
    size_kb = out_path.stat().st_size / 1024
    print(f"Written: {out_path} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
