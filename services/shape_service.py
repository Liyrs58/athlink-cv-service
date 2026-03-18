import math
import logging

logger = logging.getLogger(__name__)

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

        # Soft handling: set metrics to None if they exceed physical limits
        # (instead of crashing with assertions)
        if width_m > 68.0:
            logger.warning(f"Team {team_name} width {width_m}m exceeds pitch width (68m), marking as unreliable")
            width_m = None

        if depth_m > 105.0:
            logger.warning(f"Team {team_name} depth {depth_m}m exceeds pitch length (105m), marking as unreliable")
            depth_m = None

        if compactness_m > 80.0:
            logger.warning(f"Team {team_name} compactness {compactness_m}m exceeds limits, marking as unreliable")
            compactness_m = None

        return {
            "width_metres": width_m,
            "depth_metres": depth_m,
            "compactness_metres": compactness_m,
            "centroid": [round(centroid_x, 1), round(centroid_y, 1)],
            "player_count": len(positions)
        } if (width_m is not None and depth_m is not None) else None

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
        # Return complete dict with None values instead of empty dict
        logger.warning("No valid shape data available for any frame (all geometry invalid)")
        return {
            "frames_analysed": 0,
            "team_0": {
                "avg_width_metres": None,
                "avg_depth_metres": None,
                "avg_compactness_metres": None,
                "min_width_metres": None,
                "max_width_metres": None,
            },
            "team_1": {
                "avg_width_metres": None,
                "avg_depth_metres": None,
                "avg_compactness_metres": None,
                "min_width_metres": None,
                "max_width_metres": None,
            },
            "combined_width_metres": None,
            "min_combined_width_metres": None,
            "max_combined_width_metres": None,
            "avg_width_metres": None,
            "avg_depth_metres": None,
            "avg_compactness_metres": None,
            "min_width_metres": None,
            "max_width_metres": None,
        }

    # Collect metrics for each team separately (filtering out None values)
    team_0_widths = [s["team_0"]["width_metres"] for s in shapes if s["team_0"] and s["team_0"]["width_metres"] is not None]
    team_1_widths = [s["team_1"]["width_metres"] for s in shapes if s["team_1"] and s["team_1"]["width_metres"] is not None]
    team_0_depths = [s["team_0"]["depth_metres"] for s in shapes if s["team_0"] and s["team_0"]["depth_metres"] is not None]
    team_1_depths = [s["team_1"]["depth_metres"] for s in shapes if s["team_1"] and s["team_1"]["depth_metres"] is not None]
    team_0_compactness = [s["team_0"]["compactness_metres"] for s in shapes if s["team_0"] and s["team_0"]["compactness_metres"] is not None]
    team_1_compactness = [s["team_1"]["compactness_metres"] for s in shapes if s["team_1"] and s["team_1"]["compactness_metres"] is not None]
    combined_widths = [s["combined_width_metres"] for s in shapes if s["combined_width_metres"]]

    def safe_avg(values):
        """Return average if values exist, else return 'data unavailable'"""
        return round(sum(values) / len(values), 1) if values else "data unavailable"

    def safe_min(values):
        """Return min if values exist, else return 'data unavailable'"""
        return round(min(values), 1) if values else "data unavailable"

    def safe_max(values):
        """Return max if values exist, else return 'data unavailable'"""
        return round(max(values), 1) if values else "data unavailable"

    result = {
        "frames_analysed": len(shapes),
        # Per-team metrics
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

    # Log if width data is unavailable
    if result["avg_width_metres"] == "data unavailable":
        logger.warning("Shape data unavailable for this clip (geometry values out of physical limits)")

    return result
