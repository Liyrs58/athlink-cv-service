"""Team shape and formation shape analysis.
"""

import math
import logging

logger = logging.getLogger(__name__)

def to_scalar(v):
    """Convert numpy scalars to Python native types for JSON serialization."""
    if hasattr(v, 'item'):
        return v.item()
    if hasattr(v, '__float__'):
        return float(v)
    return v

_FALLBACK = {
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
    "data_quality": "unavailable",
}

def _safe_shape_fallback():
    """
    Return a safe fallback dict when shape computation fails.
    All numeric fields are None, ensures no type errors downstream.
    """
    return dict(_FALLBACK)

def get_centre(bbox):
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

def compute_team_shape(tracks, frame_idx, calibration=None):
    """
    Computes team shape metrics for a given frame.
    Returns width, depth, compactness, centroid per team.

    Handles both list of tracks and dict with 'tracks' key.
    """
    # Defensive: handle both list and dict inputs
    if isinstance(tracks, dict):
        tracks_list = tracks.get("tracks", [])
    else:
        tracks_list = tracks if isinstance(tracks, list) else []

    if not tracks_list:
        return None

    active = [
        t for t in tracks_list
        if (isinstance(t, dict) and
            t.get("firstSeen", 0) <= frame_idx <= t.get("lastSeen", 0) and
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
            # Convert to world coordinates (metres) using calibration
            vis_frac = 0.55  # default
            if calibration and isinstance(calibration.get("visible_fraction"), (int, float)):
                vis_frac = calibration["visible_fraction"]
            world_x = (cx / 1920) * (105.0 * vis_frac)
            world_y = (cy / 1080) * (68.0 * vis_frac)
            margin = 5.0
            if (-margin <= world_x <= 105.0 + margin and
                -margin <= world_y <= 68.0 + margin):
                tid = t.get("teamId", t.get("team", -1))
                if tid == 0:
                    team_0_positions.append((world_x, world_y))
                elif tid == 1:
                    team_1_positions.append((world_x, world_y))
                # Skip unassigned (-1) and goalkeepers (2) / officials (-2)

    if len(team_0_positions) < 2 and len(team_1_positions) < 2:
        return None

    def calculate_team_metrics(positions, team_name):
        """Positions are already in world coordinates (metres)."""
        if len(positions) < 2:
            return None

        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        width_m = round(to_scalar(max(xs)) - to_scalar(min(xs)), 1)
        depth_m = round(to_scalar(max(ys)) - to_scalar(min(ys)), 1)
        centroid_x = sum(xs) / len(xs)
        centroid_y = sum(ys) / len(ys)

        # Compactness: average distance from centroid (already in metres)
        distances = [math.sqrt((x-centroid_x)**2+(y-centroid_y)**2) for x,y in positions]
        compactness_m = round(to_scalar(sum(distances)) / len(distances), 1)

        # Soft handling: set metrics to None if they exceed physical limits
        # With calibration, max visible width is vis_frac * 68m
        max_visible_width = 68.0
        max_visible_depth = 105.0
        if isinstance(width_m, (int, float)) and width_m > max_visible_width:
            logger.warning(f"Team {team_name} width {width_m}m exceeds pitch width (68m), marking as unreliable")
            width_m = None

        if isinstance(depth_m, (int, float)) and depth_m > 105.0:
            logger.warning(f"Team {team_name} depth {depth_m}m exceeds pitch length (105m), marking as unreliable")
            depth_m = None

        if isinstance(compactness_m, (int, float)) and compactness_m > 80.0:
            logger.warning(f"Team {team_name} compactness {compactness_m}m exceeds limits, marking as unreliable")
            compactness_m = None

        return {
            "width_metres": width_m,
            "depth_metres": depth_m,
            "compactness_metres": compactness_m,
            "centroid": [round(to_scalar(centroid_x), 1), round(to_scalar(centroid_y), 1)],
            "player_count": len(positions)
        } if (width_m is not None and depth_m is not None) else None

    team_0_metrics = calculate_team_metrics(team_0_positions, "team_0")
    team_1_metrics = calculate_team_metrics(team_1_positions, "team_1")

    # Combined width is max of both teams (how spread the game is)
    t0_w = team_0_metrics["width_metres"] if team_0_metrics and isinstance(team_0_metrics.get("width_metres"), (int, float)) else 0
    t1_w = team_1_metrics["width_metres"] if team_1_metrics and isinstance(team_1_metrics.get("width_metres"), (int, float)) else 0
    combined_width = max(t0_w, t1_w) if (t0_w or t1_w) else None

    return {
        "frame_idx": frame_idx,
        "team_0": team_0_metrics,
        "team_1": team_1_metrics,
        "combined_width_metres": combined_width,
        "total_players": len(team_0_positions) + len(team_1_positions),
    }

def compute_shape_summary(tracks, frame_metadata, calibration=None):
    """
    Compute average shape metrics across all valid frames.

    Handles any exception gracefully and returns safe fallback dict.
    Never crashes the pipeline due to shape data issues.
    """
    try:
        # Defensive: handle both list and dict inputs
        if isinstance(tracks, dict):
            tracks_list = tracks.get("tracks", [])
        else:
            tracks_list = tracks if isinstance(tracks, list) else []

        if not tracks_list or not frame_metadata:
            logger.warning("No tracks or frame_metadata provided to compute_shape_summary")
            return _safe_shape_fallback()

        shapes = []
        for meta in frame_metadata:
            if not isinstance(meta, dict):
                continue
            frame_idx = meta.get("frameIndex", 0)
            shape = compute_team_shape(tracks_list, frame_idx, calibration=calibration)
            if shape:
                shapes.append(shape)

        if not shapes:
            logger.warning("No valid shape data available for any frame (all geometry invalid)")
            return _safe_shape_fallback()

        # Collect metrics for each team separately (filtering out None values)
        team_0_widths = [s["team_0"]["width_metres"] for s in shapes if s["team_0"] and isinstance(s["team_0"].get("width_metres"), (int, float))]
        team_1_widths = [s["team_1"]["width_metres"] for s in shapes if s["team_1"] and isinstance(s["team_1"].get("width_metres"), (int, float))]
        team_0_depths = [s["team_0"]["depth_metres"] for s in shapes if s["team_0"] and isinstance(s["team_0"].get("depth_metres"), (int, float))]
        team_1_depths = [s["team_1"]["depth_metres"] for s in shapes if s["team_1"] and isinstance(s["team_1"].get("depth_metres"), (int, float))]
        team_0_compactness = [s["team_0"]["compactness_metres"] for s in shapes if s["team_0"] and isinstance(s["team_0"].get("compactness_metres"), (int, float))]
        team_1_compactness = [s["team_1"]["compactness_metres"] for s in shapes if s["team_1"] and isinstance(s["team_1"].get("compactness_metres"), (int, float))]
        combined_widths = [s["combined_width_metres"] for s in shapes if isinstance(s.get("combined_width_metres"), (int, float))]

        def safe_avg(values):
            """Return average if values exist, else None (never a string)."""
            return round(to_scalar(sum(values)) / len(values), 1) if values else None

        def safe_min(values):
            """Return min if values exist, else None (never a string)."""
            return round(to_scalar(min(values)), 1) if values else None

        def safe_max(values):
            """Return max if values exist, else None (never a string)."""
            return round(to_scalar(max(values)), 1) if values else None

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

        # Determine data quality
        if result["avg_width_metres"] is None:
            logger.warning("Shape data unavailable for this clip (geometry values out of physical limits)")
            result["data_quality"] = "unavailable"
        else:
            result["data_quality"] = "ok"

        return result

    except Exception as e:
        logger.error(f"Exception in compute_shape_summary: {type(e).__name__}: {e}", exc_info=True)
        return _safe_shape_fallback()
