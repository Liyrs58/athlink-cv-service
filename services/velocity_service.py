"""Player speed and acceleration calculations.
"""

import math
import logging
from typing import Optional, Dict, Any, List
from statistics import median

logger = logging.getLogger(__name__)

def to_scalar(v):
    """Convert numpy scalars to Python native types for JSON serialization."""
    if hasattr(v, 'item'):
        return v.item()
    if hasattr(v, '__float__'):
        return float(v)
    return v

PIXELS_PER_METRE = 15.5
SPRINT_MS = 7.0  # FIX 6: sprint = >7 m/s sustained >1s (25.2 km/h)
HIGH_INTENSITY_MS = 5.5
WALKING_MS = 2.0
MAX_REALISTIC_SPEED_MS = 12.0  # FIX 6: 12 m/s cap (43.2 km/h — Mbappé territory)

# Part 3c: improved sprint params
SPRINT_COOLDOWN_BEFORE = 1.0   # FIX 6: 1s pre-cooldown
SPRINT_PRE_SPEED = 4.0
SPRINT_POST_SPEED = 4.5
SPRINT_MIN_DURATION = 1.0      # FIX 6: must sustain >7 m/s for ≥1s
MAX_SPRINTS_PER_PLAYER = 5     # slightly more generous with higher threshold

# Part 3b: noise floor
MIN_DISPLACEMENT_M = 0.3  # per-frame displacement below this is noise
MAX_FRAME_DISTANCE_M = 0.48  # FIX 6: cap per-frame distance (12 m/s at 25fps)


def get_centre(bbox):
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _smooth_trajectory_world(positions: List[tuple]) -> List[tuple]:
    """Apply 3-point moving average smoothing to world coordinate positions."""
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


def _optical_flow_speed_check(
    track: Dict[str, Any],
    video_path: Optional[str],
    ppm: float,
) -> Dict[str, Any]:
    """
    Estimate per-frame speeds using Lucas-Kanade optical flow on bbox centres,
    then compare against homography-based speeds.

    Returns:
      {
        "unreliable_speed_frames": int,
        "total_frames_checked": int,
        "speed_confidence_downgraded": bool,
      }
    """
    result = {
        "unreliable_speed_frames": 0,
        "total_frames_checked": 0,
        "speed_confidence_downgraded": False,
    }

    if video_path is None:
        return result

    traj = track.get("trajectory", [])
    if len(traj) < 4:
        return result

    try:
        import cv2
        import numpy as np
    except ImportError:
        return result

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return result

        raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        needs_rotation = raw_h > raw_w

        FLOW_DISAGREEMENT = 0.20   # 20% threshold

        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        unreliable = 0
        checked = 0

        for i in range(1, len(traj)):
            prev_entry = traj[i - 1]
            curr_entry = traj[i]

            dt = curr_entry.get("timestampSeconds", 0) - prev_entry.get("timestampSeconds", 0)
            if dt <= 0:
                continue

            # Homography-based speed (from smoothed positions, recomputed simply)
            cx_p, cy_p = get_centre(prev_entry["bbox"])
            cx_c, cy_c = get_centre(curr_entry["bbox"])
            dist_px_hom = math.sqrt((cx_c - cx_p) ** 2 + (cy_c - cy_p) ** 2)
            speed_hom = (dist_px_hom / ppm) / dt
            if speed_hom > MAX_REALISTIC_SPEED_MS:
                continue  # already flagged elsewhere

            # Read the two frames for optical flow
            fi_prev = prev_entry.get("frameIndex", 0)
            fi_curr = curr_entry.get("frameIndex", 0)

            cap.set(cv2.CAP_PROP_POS_FRAMES, fi_prev)
            ret_p, frame_p = cap.read()
            if not ret_p:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi_curr)
            ret_c, frame_c = cap.read()
            if not ret_c:
                continue

            if needs_rotation:
                frame_p = cv2.rotate(frame_p, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frame_c = cv2.rotate(frame_c, cv2.ROTATE_90_COUNTERCLOCKWISE)

            gray_p = cv2.cvtColor(frame_p, cv2.COLOR_BGR2GRAY)
            gray_c = cv2.cvtColor(frame_c, cv2.COLOR_BGR2GRAY)

            # Use bbox centre as the point to track
            pt = np.array([[[float(cx_p), float(cy_p)]]], dtype=np.float32)

            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(gray_p, gray_c, pt, None, **lk_params)

            if status is None or status[0][0] == 0:
                continue

            nx, ny = float(next_pts[0][0][0]), float(next_pts[0][0][1])
            dist_px_flow = math.sqrt((nx - cx_p) ** 2 + (ny - cy_p) ** 2)
            speed_flow = (dist_px_flow / ppm) / dt

            checked += 1

            # Compare
            denom = max(speed_hom, speed_flow, 1e-9)
            if abs(speed_hom - speed_flow) / denom > FLOW_DISAGREEMENT:
                unreliable += 1

        cap.release()

        result["unreliable_speed_frames"] = unreliable
        result["total_frames_checked"] = checked

        # Downgrade if >30% of checked frames are unreliable
        if checked > 0 and (unreliable / checked) > 0.30:
            result["speed_confidence_downgraded"] = True

    except Exception as e:
        logger.debug(f"Optical flow check failed for track {track.get('trackId')}: {e}")

    return result


def compute_player_velocity(track, calibration: Optional[Dict[str, Any]] = None, video_path: Optional[str] = None):
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

    # Part 3b: use world coordinates (from physics_corrector) for distance
    # Extract world_x, world_y from trajectory entries set by physics_corrector
    world_positions = []
    for entry in valid_entries:
        wx = entry.get("world_x")
        wy = entry.get("world_y")
        if wx is not None and wy is not None:
            world_positions.append((float(wx), float(wy)))
        else:
            # Fallback: compute from pixel coords if world coords unavailable
            cx, cy = get_centre(entry["bbox"])
            wx_fb = (cx / 1920) * (105.0 * vis_frac)
            wy_fb = (cy / 1080) * (68.0 * vis_frac)
            world_positions.append((wx_fb, wy_fb))

    # Smooth world coordinate positions
    smoothed = _smooth_trajectory_world(world_positions)

    # Compute raw frame-to-frame speeds and distances using world coordinates
    raw_speeds = []
    timestamps = []
    total_distance_m = 0.0

    for i in range(1, len(valid_entries)):
        dt = valid_entries[i]["timestampSeconds"] - valid_entries[i-1]["timestampSeconds"]
        if dt <= 0:
            raw_speeds.append(0.0)
            timestamps.append(valid_entries[i]["timestampSeconds"])
            continue

        # World coordinates are already in metres
        dx = smoothed[i][0] - smoothed[i-1][0]
        dy = smoothed[i][1] - smoothed[i-1][1]
        dist_m = math.sqrt(dx*dx + dy*dy)

        # FIX 6: cap per-frame distance at MAX_REALISTIC_SPEED_MS / fps
        dist_m = min(dist_m, MAX_FRAME_DISTANCE_M)

        # Low homography confidence: apply a conservative distance discount
        is_approx = valid_entries[i].get("metric_quality") == "approximate"
        if is_approx:
            dist_m *= 0.85  # treat as approximate — discount by 15%

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
                    if duration >= SPRINT_MIN_DURATION:
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

    # FIX 6: 1s rolling max for top speed (smooths out single-frame spikes)
    max_speed = 0.0
    if speeds and timestamps:
        for i in range(len(speeds)):
            t_end = timestamps[i]
            window_max = speeds[i]
            for j in range(i - 1, -1, -1):
                if t_end - timestamps[j] > 1.0:
                    break
                if speeds[j] > window_max:
                    window_max = speeds[j]
            if window_max > max_speed:
                max_speed = window_max
    max_speed = min(max_speed, MAX_REALISTIC_SPEED_MS)
    avg_speed = sum(speeds) / len(speeds) if speeds else 0.0

    # Optical flow cross-check
    flow_check = _optical_flow_speed_check(track, video_path, ppm)

    return {
        "track_id": track["trackId"],
        "distance_metres": round(to_scalar(total_distance_m), 1),
        "sprint_count": sprint_count,
        "max_speed_ms": round(to_scalar(max_speed), 2),
        "avg_speed_ms": round(to_scalar(avg_speed), 2),
        "high_intensity_runs": high_intensity_runs,
        "unreliable_speed_frames": flow_check["unreliable_speed_frames"],
        "speed_confidence_downgraded": flow_check["speed_confidence_downgraded"],
    }


def compute_all_velocities(tracks, calibration: Optional[Dict[str, Any]] = None, video_path: Optional[str] = None):
    results = []
    for track in tracks:
        if (track.get("confirmed_detections", 0) >= 5 and
            not track.get("is_staff", False)):
            results.append(compute_player_velocity(track, calibration=calibration, video_path=video_path))
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
