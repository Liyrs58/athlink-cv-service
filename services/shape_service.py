import math

def get_centre(bbox):
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

def compute_team_shape(tracks, frame_idx):
    """
    Computes team shape metrics for a given frame.
    Returns width, depth, compactness, centroid per team.
    """
    active = [
        t for t in tracks
        if (t.get("firstSeen", 0) <= frame_idx <= t.get("lastSeen", 0) and
            not t.get("is_staff", False))  # Filter out staff tracks
    ]
    if len(active) < 4:
        return None

    # Split by team
    team_0_positions = []
    team_1_positions = []
    
    for t in active:
        traj = t.get("trajectory", [])
        closest = min(traj, key=lambda e: abs(e["frameIndex"] - frame_idx), default=None)
        if closest:
            cx, cy = get_centre(closest["bbox"])
            # Additional pitch boundary check for current position
            world_x, world_y = (cx / 1920) * 105.0, (cy / 1080) * 68.0  # Approximate scaling
            margin = 5.0
            if (-margin <= world_x <= 105.0 + margin and
                -margin <= world_y <= 68.0 + margin):
                if t.get("team", 0) == 0:
                    team_0_positions.append((cx, cy))
                else:
                    team_1_positions.append((cx, cy))

    if len(team_0_positions) < 2 and len(team_1_positions) < 2:
        return None

    def calculate_team_metrics(positions, team_name):
        if len(positions) < 2:
            return None
            
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        width_px = max(xs) - min(xs)
        depth_px = max(ys) - min(ys)
        centroid_x = sum(xs) / len(xs)
        centroid_y = sum(ys) / len(ys)

        # Compactness: average distance from centroid
        distances = [math.sqrt((x-centroid_x)**2+(y-centroid_y)**2) for x,y in positions]
        compactness = sum(distances) / len(distances)

        PIXELS_PER_METRE = 15.5
        width_m = round(width_px / PIXELS_PER_METRE, 1)
        depth_m = round(depth_px / PIXELS_PER_METRE, 1)
        compactness_m = round(compactness / PIXELS_PER_METRE, 1)

        # FIX 1: Assertion to catch impossible values
        assert width_m < 80.0, f"Team {team_name} width {width_m}m exceeds physical limits"
        assert depth_m < 120.0, f"Team {team_name} depth {depth_m}m exceeds physical limits"
        
        # FIX 3: Clamp to physical limits
        width_m = max(0, min(68, width_m))  # Pitch is 68m wide
        depth_m = max(0, min(105, depth_m))  # Pitch is 105m long
        compactness_m = max(0, min(50, compactness_m))

        return {
            "width_metres": width_m,
            "depth_metres": depth_m,
            "compactness_metres": compactness_m,
            "centroid": [round(centroid_x, 1), round(centroid_y, 1)],
            "player_count": len(positions)
        }

    team_0_metrics = calculate_team_metrics(team_0_positions, "team_0")
    team_1_metrics = calculate_team_metrics(team_1_positions, "team_1")

    # FIX 2: Combined width is max of both teams (how spread the game is)
    combined_width = max(
        team_0_metrics["width_metres"] if team_0_metrics else 0,
        team_1_metrics["width_metres"] if team_1_metrics else 0
    )
    
    return {
        "frame_idx": frame_idx,
        "team_0": team_0_metrics,
        "team_1": team_1_metrics,
        "combined_width_metres": combined_width,
        "total_players": len(team_0_positions) + len(team_1_positions),
    }

def compute_shape_summary(tracks, frame_metadata):
    """Compute average shape metrics across all valid frames."""
    shapes = []
    for meta in frame_metadata:
        frame_idx = meta.get("frameIndex", 0)
        shape = compute_team_shape(tracks, frame_idx)
        if shape:
            shapes.append(shape)

    if not shapes:
        return {}

    # Collect metrics for each team separately
    team_0_widths = [s["team_0"]["width_metres"] for s in shapes if s["team_0"]]
    team_1_widths = [s["team_1"]["width_metres"] for s in shapes if s["team_1"]]
    team_0_depths = [s["team_0"]["depth_metres"] for s in shapes if s["team_0"]]
    team_1_depths = [s["team_1"]["depth_metres"] for s in shapes if s["team_1"]]
    team_0_compactness = [s["team_0"]["compactness_metres"] for s in shapes if s["team_0"]]
    team_1_compactness = [s["team_1"]["compactness_metres"] for s in shapes if s["team_1"]]
    combined_widths = [s["combined_width_metres"] for s in shapes]

    def safe_avg(values):
        return round(sum(values) / len(values), 1) if values else 0.0

    def safe_min(values):
        return round(min(values), 1) if values else 0.0

    def safe_max(values):
        return round(max(values), 1) if values else 0.0

    result = {
        "frames_analysed": len(shapes),
        # FIX 2: Per-team metrics
        "team_0": {
            "avg_width_metres": safe_avg(team_0_widths),
            "avg_depth_metres": safe_avg(team_0_depths),
            "avg_compactness_metres": safe_avg(team_0_compactness),
            "min_width_metres": safe_min(team_0_widths),
            "max_width_metres": safe_max(team_0_widths),
        },
        "team_1": {
            "avg_width_metres": safe_avg(team_1_widths),
            "avg_depth_metres": safe_avg(team_1_depths),
            "avg_compactness_metres": safe_avg(team_1_compactness),
            "min_width_metres": safe_min(team_1_widths),
            "max_width_metres": safe_max(team_1_widths),
        },
        # Combined spread (how wide the game is overall)
        "combined_width_metres": safe_avg(combined_widths),
        "min_combined_width_metres": safe_min(combined_widths),
        "max_combined_width_metres": safe_max(combined_widths),
    }

    # Legacy fields for backward compatibility
    result["avg_width_metres"] = result["combined_width_metres"]
    result["avg_depth_metres"] = safe_avg(team_0_depths + team_1_depths)
    result["avg_compactness_metres"] = safe_avg(team_0_compactness + team_1_compactness)
    result["min_width_metres"] = result["min_combined_width_metres"]
    result["max_width_metres"] = result["max_combined_width_metres"]

    # FIX 1: Validation check
    if result["avg_width_metres"] > 80:
        print(f"WARNING: Average width {result['avg_width_metres']}m exceeds physical limits")

    return result
