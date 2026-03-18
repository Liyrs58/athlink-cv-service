import math

PIXELS_PER_METRE = 15.5   # 1920px frame, pitch ~85% of width, 105m wide
SPRINT_MS = 6.94  # 25 km/h threshold (25 / 3.6 = 6.944 m/s)
HIGH_INTENSITY_MS = 5.5
WALKING_MS = 2.0
MAX_REALISTIC_SPEED_MS = 10.0  # 36 km/h — cap to remove tracking artifacts

def get_centre(bbox):
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

def compute_player_velocity(track):
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

    speeds = []
    total_distance_px = 0.0
    sprint_count = 0
    high_intensity_runs = 0

    # Track sprint intervals: (start_time, end_time) for each sustained sprint
    sprint_intervals = []
    in_sprint = False
    sprint_start_time = None
    in_high = False

    for i in range(1, len(traj)):
        prev = traj[i-1]
        curr = traj[i]

        # Skip duplicate frames
        if curr["frameIndex"] == prev["frameIndex"]:
            continue

        dt = curr["timestampSeconds"] - prev["timestampSeconds"]
        if dt <= 0:
            continue

        cx_prev, cy_prev = get_centre(prev["bbox"])
        cx_curr, cy_curr = get_centre(curr["bbox"])

        # FIX 1c: Only count sprints/distance if player is on pitch
        world_x, world_y = (cx_curr / 1920) * 105.0, (cy_curr / 1080) * 68.0  # Approximate scaling
        margin = 5.0
        if not (-margin <= world_x <= 105.0 + margin and
                -margin <= world_y <= 68.0 + margin):
            continue  # Skip off-pitch positions

        dx = cx_curr - cx_prev
        dy = cy_curr - cy_prev
        dist_px = math.sqrt(dx*dx + dy*dy)
        speed_px_s = dist_px / dt
        speed_ms = speed_px_s / PIXELS_PER_METRE

        # Sanity cap — no human runs faster than 10 m/s (36 km/h)
        if speed_ms > MAX_REALISTIC_SPEED_MS:
            continue

        total_distance_px += dist_px
        speeds.append(speed_ms)

        # Sprint detection: require 1.0s sustained >25 km/h + 3s cooldown between sprints
        if speed_ms >= SPRINT_MS:
            if not in_sprint:
                # Start new potential sprint
                sprint_start_time = curr["timestampSeconds"]
                in_sprint = True
        else:
            # Exited sprint zone
            if in_sprint and sprint_start_time is not None:
                sprint_duration = curr["timestampSeconds"] - sprint_start_time
                # Only count if sustained for >= 1.0 second
                if sprint_duration >= 1.0:
                    sprint_intervals.append((sprint_start_time, curr["timestampSeconds"]))
            in_sprint = False
            sprint_start_time = None

        if speed_ms >= HIGH_INTENSITY_MS:
            if not in_high:
                high_intensity_runs += 1
                in_high = True
        else:
            in_high = False

    # Handle final sprint if still active
    if in_sprint and sprint_start_time is not None:
        sprint_duration = traj[-1]["timestampSeconds"] - sprint_start_time
        if sprint_duration >= 1.0:
            sprint_intervals.append((sprint_start_time, traj[-1]["timestampSeconds"]))

    # Apply 3-second cooldown: merge sprints separated by less than 3 seconds
    if sprint_intervals:
        merged_intervals = [sprint_intervals[0]]
        for start, end in sprint_intervals[1:]:
            last_start, last_end = merged_intervals[-1]
            time_since_last = start - last_end
            if time_since_last < 3.0:
                # Merge with previous sprint (extend end time)
                merged_intervals[-1] = (last_start, max(end, last_end))
            else:
                # Add as separate sprint
                merged_intervals.append((start, end))
        sprint_count = len(merged_intervals)

    total_distance_m = total_distance_px / PIXELS_PER_METRE

    return {
        "track_id": track["trackId"],
        "distance_metres": round(total_distance_m, 1),
        "sprint_count": sprint_count,
        "max_speed_ms": round(max(speeds), 2) if speeds else 0.0,
        "avg_speed_ms": round(sum(speeds) / len(speeds), 2) if speeds else 0.0,
        "high_intensity_runs": high_intensity_runs,
    }

def compute_all_velocities(tracks):
    results = []
    for track in tracks:
        # FIX 1c: Only compute velocities for non-staff tracks
        if (track.get("confirmed_detections", 0) >= 5 and
            not track.get("is_staff", False)):
            results.append(compute_player_velocity(track))
    results.sort(key=lambda x: x["distance_metres"], reverse=True)
    return results

def get_team_velocity_summary(velocity_results):
    if not velocity_results:
        return {}
    valid = [v for v in velocity_results if v["distance_metres"] > 0]
    if not valid:
        return {}
    total_distance = sum(v["distance_metres"] for v in valid)
    total_sprints = sum(v["sprint_count"] for v in valid)
    max_speed = max(v["max_speed_ms"] for v in valid)
    top_sprinter = max(valid, key=lambda x: x["sprint_count"])
    top_runner = max(valid, key=lambda x: x["distance_metres"])
    return {
        "total_distance_metres": round(total_distance, 1),
        "total_sprints": total_sprints,
        "max_speed_ms": round(max_speed, 2),
        "max_speed_kmh": round(max_speed * 3.6, 1),
        "players_analysed": len(valid),
        "top_sprinter_id": top_sprinter["track_id"],
        "top_runner_id": top_runner["track_id"],
        "top_runner_distance": top_runner["distance_metres"],
    }
