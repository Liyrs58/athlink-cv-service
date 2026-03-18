import math
from typing import Optional, Dict, Any, List
from statistics import median

def to_scalar(v):
    """Convert numpy scalars to Python native types for JSON serialization."""
    if hasattr(v, 'item'):
        return v.item()
    if hasattr(v, '__float__'):
        return float(v)
    return v

PIXELS_PER_METRE = 15.5
SPRINT_MS = 5.5  # 19.8 km/h
HIGH_INTENSITY_MS = 5.5
WALKING_MS = 2.0
MAX_REALISTIC_SPEED_MS = 10.0  # 36 km/h cap

# Part 3c: improved sprint params
SPRINT_COOLDOWN_BEFORE = 1.5   # must be below 4.0 m/s for 1.5s before sprint
SPRINT_PRE_SPEED = 4.0
SPRINT_POST_SPEED = 4.5
MAX_SPRINTS_PER_PLAYER = 3     # max realistic in 40s clip

# Part 3b: noise floor
MIN_DISPLACEMENT_M = 0.3  # per-frame displacement below this is noise


def get_centre(bbox):
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _smooth_trajectory(positions: List[tuple]) -> List[tuple]:
    """Apply 3-point moving average smoothing to trajectory positions."""
    if len(positions) < 3:
        return positions
    smoothed = [positions[0]]
    for i in range(1, len(positions) - 1):
        sx = (positions[i-1][0] + positions[i][0] + positions[i+1][0]) / 3.0
        sy = (positions[i-1][1] + positions[i][1] + positions[i+1][1]) / 3.0
        smoothed.append((sx, sy))
    smoothed.append(positions[-1])
    return smoothed


def _rolling_median_speed(speeds_raw: List[float], window: int = 5) -> List[float]:
    """Rolling median over speeds to eliminate single-frame jitter."""
    if len(speeds_raw) < window:
        return speeds_raw
    result = []
    half = window // 2
    for i in range(len(speeds_raw)):
        lo = max(0, i - half)
        hi = min(len(speeds_raw), i + half + 1)
        result.append(median(speeds_raw[lo:hi]))
    return result


def compute_player_velocity(track, calibration: Optional[Dict[str, Any]] = None):
    ppm = PIXELS_PER_METRE
    if calibration and isinstance(calibration.get("pixels_per_metre"), (int, float)):
        ppm = calibration["pixels_per_metre"]

    vis_frac = calibration.get("visible_fraction", 0.55) if calibration else 0.55

    traj = track.get("trajectory", [])
    if len(traj) < 2:
        return {
            "track_id": track["trackId"],
            "distance_metres": 0.0,
            "sprint_count": 0,
            "max_speed_ms": 0.0,
            "avg_speed_ms": 0.0,
            "high_intensity_runs": 0,
        }

    # Part 3d: filter positions to on-pitch only
    valid_entries = []
    for entry in traj:
        cx, cy = get_centre(entry["bbox"])
        world_x = (cx / 1920) * (105.0 * vis_frac)
        world_y = (cy / 1080) * (68.0 * vis_frac)
        margin = 5.0
        if (-margin <= world_x <= 105.0 + margin and
            -margin <= world_y <= 68.0 + margin):
            # Skip crowd/bench area (top 15% and bottom 15% of frame)
            frame_y_frac = cy / 1080.0
            if 0.15 <= frame_y_frac <= 0.85:
                valid_entries.append(entry)

    if len(valid_entries) < 2:
        return {
            "track_id": track["trackId"],
            "distance_metres": 0.0,
            "sprint_count": 0,
            "max_speed_ms": 0.0,
            "avg_speed_ms": 0.0,
            "high_intensity_runs": 0,
        }

    # Part 3d: position jump filter — remove entries where position jumps >8m per frame gap
    filtered = [valid_entries[0]]
    for i in range(1, len(valid_entries)):
        prev = filtered[-1]
        curr = valid_entries[i]
        cx_p, cy_p = get_centre(prev["bbox"])
        cx_c, cy_c = get_centre(curr["bbox"])
        dist_px = math.sqrt((cx_c - cx_p)**2 + (cy_c - cy_p)**2)
        dist_m = dist_px / ppm
        frame_gap = abs(curr["frameIndex"] - prev["frameIndex"])
        if dist_m <= 8.0 * max(frame_gap, 1):
            filtered.append(curr)
    valid_entries = filtered

    if len(valid_entries) < 2:
        return {
            "track_id": track["trackId"],
            "distance_metres": 0.0,
            "sprint_count": 0,
            "max_speed_ms": 0.0,
            "avg_speed_ms": 0.0,
            "high_intensity_runs": 0,
        }

    # Part 3b: smooth trajectory before distance computation
    positions = [get_centre(e["bbox"]) for e in valid_entries]
    smoothed = _smooth_trajectory(positions)

    # Compute raw frame-to-frame speeds and distances
    raw_speeds = []
    timestamps = []
    total_distance_m = 0.0

    for i in range(1, len(valid_entries)):
        dt = valid_entries[i]["timestampSeconds"] - valid_entries[i-1]["timestampSeconds"]
        if dt <= 0:
            raw_speeds.append(0.0)
            timestamps.append(valid_entries[i]["timestampSeconds"])
            continue

        dx = smoothed[i][0] - smoothed[i-1][0]
        dy = smoothed[i][1] - smoothed[i-1][1]
        dist_px = math.sqrt(dx*dx + dy*dy)
        dist_m = dist_px / ppm

        # Only count displacement above noise floor
        if dist_m >= MIN_DISPLACEMENT_M:
            total_distance_m += dist_m

        speed_ms = dist_m / dt
        if speed_ms > MAX_REALISTIC_SPEED_MS:
            speed_ms = 0.0  # discard artifact
        raw_speeds.append(speed_ms)
        timestamps.append(valid_entries[i]["timestampSeconds"])

    # Part 3a: rolling median smoothing on speeds
    speeds = _rolling_median_speed(raw_speeds)

    # Part 3c: improved sprint detection with pre/post cooldown
    sprint_intervals = []
    in_sprint = False
    sprint_start_idx = None

    for i, speed in enumerate(speeds):
        if speed >= SPRINT_MS:
            if not in_sprint:
                # Check pre-sprint cooldown
                t_now = timestamps[i]
                pre_ok = True
                for j in range(i - 1, -1, -1):
                    if t_now - timestamps[j] > SPRINT_COOLDOWN_BEFORE:
                        break
                    if speeds[j] >= SPRINT_PRE_SPEED:
                        pre_ok = False
                        break
                if pre_ok:
                    sprint_start_idx = i
                    in_sprint = True
        else:
            if in_sprint and sprint_start_idx is not None:
                if speed < SPRINT_POST_SPEED:
                    duration = timestamps[i] - timestamps[sprint_start_idx]
                    if duration >= 0.4:
                        peak = max(speeds[sprint_start_idx:i+1])
                        sprint_intervals.append({
                            "start": timestamps[sprint_start_idx],
                            "end": timestamps[i],
                            "peak_speed": peak,
                        })
                    in_sprint = False
                    sprint_start_idx = None

    # Handle final sprint
    if in_sprint and sprint_start_idx is not None:
        last_t = timestamps[-1] if timestamps else 0
        duration = last_t - timestamps[sprint_start_idx]
        if duration >= 0.4:
            peak = max(speeds[sprint_start_idx:])
            sprint_intervals.append({
                "start": timestamps[sprint_start_idx],
                "end": last_t,
                "peak_speed": peak,
            })

    # Merge sprints with <2s gap
    if sprint_intervals:
        merged = [sprint_intervals[0]]
        for s in sprint_intervals[1:]:
            if s["start"] - merged[-1]["end"] < 2.0:
                merged[-1]["end"] = max(s["end"], merged[-1]["end"])
                merged[-1]["peak_speed"] = max(s["peak_speed"], merged[-1]["peak_speed"])
            else:
                merged.append(s)
        sprint_intervals = merged

    # Cap at MAX_SPRINTS_PER_PLAYER, keep highest peak speed
    if len(sprint_intervals) > MAX_SPRINTS_PER_PLAYER:
        sprint_intervals.sort(key=lambda s: s["peak_speed"], reverse=True)
        sprint_intervals = sprint_intervals[:MAX_SPRINTS_PER_PLAYER]

    sprint_count = len(sprint_intervals)

    # High intensity runs
    high_intensity_runs = 0
    in_high = False
    for speed in speeds:
        if speed >= HIGH_INTENSITY_MS:
            if not in_high:
                high_intensity_runs += 1
                in_high = True
        else:
            in_high = False

    max_speed = max(speeds) if speeds else 0.0
    avg_speed = sum(speeds) / len(speeds) if speeds else 0.0

    return {
        "track_id": track["trackId"],
        "distance_metres": round(to_scalar(total_distance_m), 1),
        "sprint_count": sprint_count,
        "max_speed_ms": round(to_scalar(max_speed), 2),
        "avg_speed_ms": round(to_scalar(avg_speed), 2),
        "high_intensity_runs": high_intensity_runs,
    }


def compute_all_velocities(tracks, calibration: Optional[Dict[str, Any]] = None):
    results = []
    for track in tracks:
        if (track.get("confirmed_detections", 0) >= 5 and
            not track.get("is_staff", False)):
            results.append(compute_player_velocity(track, calibration=calibration))
    results.sort(key=lambda x: x["distance_metres"], reverse=True)
    return results


def get_team_velocity_summary(velocity_results):
    if not velocity_results:
        return {}
    valid = [v for v in velocity_results if isinstance(v.get("distance_metres"), (int, float)) and v["distance_metres"] > 0]
    if not valid:
        return {}
    total_distance = sum(v["distance_metres"] for v in valid)
    total_sprints = sum(v["sprint_count"] for v in valid)
    max_speed = max(v["max_speed_ms"] for v in valid)
    top_sprinter = max(valid, key=lambda x: x["sprint_count"])
    top_runner = max(valid, key=lambda x: x["distance_metres"])
    return {
        "total_distance_metres": round(to_scalar(total_distance), 1),
        "total_sprints": int(to_scalar(total_sprints)),
        "max_speed_ms": round(to_scalar(max_speed), 2),
        "max_speed_kmh": round(to_scalar(max_speed) * 3.6, 1),
        "players_analysed": len(valid),
        "top_sprinter_id": top_sprinter["track_id"],
        "top_runner_id": top_runner["track_id"],
        "top_runner_distance": top_runner["distance_metres"],
    }
