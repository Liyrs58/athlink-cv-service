import math

PIXELS_PER_METRE = 15.5   # 1920px frame, pitch ~85% of width, 105m wide
SPRINT_MS = 7.0
HIGH_INTENSITY_MS = 5.5
WALKING_MS = 2.0
MAX_REALISTIC_SPEED_MS = 12.0  # 43km/h — fastest humans ever recorded

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
    in_sprint = False
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

        # Sanity cap — no human runs faster than 12m/s
        if speed_ms > MAX_REALISTIC_SPEED_MS:
            continue

        total_distance_px += dist_px
        speeds.append(speed_ms)

        if speed_ms >= SPRINT_MS:
            if not in_sprint:
                sprint_count += 1
                in_sprint = True
        else:
            in_sprint = False

        if speed_ms >= HIGH_INTENSITY_MS:
            if not in_high:
                high_intensity_runs += 1
                in_high = True
        else:
            in_high = False

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
