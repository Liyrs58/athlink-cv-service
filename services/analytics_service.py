import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Service registry: (key, module_path, function_name, required_files)
_SERVICE_REGISTRY = [
    ("pass_network", "services.pass_network_service", "compute_pass_network",
     ["tracking/track_results.json", "pitch/pitch_map.json"]),
    ("xg", "services.xg_service", "compute_xg",
     ["tracking/track_results.json"]),
    ("heatmaps", "services.heatmap_service", "compute_heatmaps",
     ["pitch/pitch_map.json"]),
    ("pressing", "services.pressing_service", "compute_pressing",
     ["tracking/track_results.json", "pitch/pitch_map.json"]),
    ("formations", "services.formation_service", "compute_formations",
     ["pitch/pitch_map.json"]),
    ("tactics", "services.tactics_service", "analyze_tactics",
     ["tracking/track_results.json", "pitch/pitch_map.json"]),
    ("events", "services.event_service", "detect_events",
     ["tracking/track_results.json", "tracking/team_results.json", "pitch/pitch_map.json"]),
    ("defensive_lines", "services.defensive_line_service", "compute_defensive_lines",
     ["tracking/track_results.json", "tracking/team_results.json", "pitch/pitch_map.json"]),
    ("counter_press", "services.counter_press_service", "compute_counter_press",
     ["tracking/track_results.json", "tracking/team_results.json", "pitch/pitch_map.json"]),
    ("set_pieces", "services.set_piece_service", "detect_set_pieces",
     ["tracking/track_results.json", "tracking/team_results.json", "pitch/pitch_map.json"]),
]


def _load_json(path):
    # type: (Path) -> Optional[Dict]
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def get_available_services(job_id):
    # type: (str) -> List[str]
    """Return service names whose required input files exist for this job."""
    base = Path("temp") / job_id
    available = []  # type: List[str]
    for key, _mod, _fn, required_files in _SERVICE_REGISTRY:
        all_present = True
        for rf in required_files:
            if not (base / rf).exists():
                all_present = False
                break
        if all_present:
            available.append(key)
    return available


def build_analytics_report(job_id):
    # type: (str) -> Dict[str, Any]
    """Run all available analytics services and return a unified EPL match report."""
    base = Path("temp") / job_id

    # Load track_results for metadata
    track_path = base / "tracking" / "track_results.json"
    if not track_path.exists():
        raise FileNotFoundError(
            "track_results.json not found for job '{}'".format(job_id)
        )
    track_data = _load_json(track_path)

    metadata = track_data.get("metadata", {}) if track_data else {}
    fps = float(metadata.get("fps", 25))
    if fps <= 0:
        fps = 25.0
    frame_count = int(track_data.get("framesProcessed", 0)) if track_data else 0
    video_path = track_data.get("videoPath", "") if track_data else ""
    duration_seconds = round(frame_count / fps, 2) if fps > 0 else 0.0

    teams_detected = (base / "tracking" / "team_results.json").exists()
    pitch_mapped = (base / "pitch" / "pitch_map.json").exists()

    generated_at = datetime.now(timezone.utc).isoformat()

    # Run each service independently
    results = {}   # type: Dict[str, Any]
    errors = {}    # type: Dict[str, str]
    available = [] # type: List[str]

    for key, module_path, fn_name, required_files in _SERVICE_REGISTRY:
        # Check required files
        missing = False
        for rf in required_files:
            if not (base / rf).exists():
                missing = True
                break
        if missing:
            results[key] = None
            continue

        available.append(key)
        try:
            import importlib
            mod = importlib.import_module(module_path)
            fn = getattr(mod, fn_name)
            results[key] = fn(job_id)
        except Exception as e:
            logger.exception("Analytics service '%s' failed for job %s", key, job_id)
            results[key] = None
            errors[key] = str(e)

    # Build match summary from available results
    match_summary = _build_match_summary(results)
    
    # Check for analytics highlight video
    highlight_video_path = None
    highlight_video_url = None
    
    highlight_path = base / "highlights" / "analytics_highlight.mp4"
    if highlight_path.exists():
        highlight_video_path = str(highlight_path)
        # TODO: Get URL from storage service when available
    
    return {
        "job_id": job_id,
        "generated_at": generated_at,
        "video_path": video_path,
        "duration_seconds": duration_seconds,
        "fps": fps,
        "frame_count": frame_count,
        "teams_detected": teams_detected,
        "pitch_mapped": pitch_mapped,
        "match_summary": match_summary,
        "pass_network": results.get("pass_network"),
        "xg": results.get("xg"),
        "heatmaps": results.get("heatmaps"),
        "pressing": results.get("pressing"),
        "formations": results.get("formations"),
        "tactics": results.get("tactics"),
        "player_stats": None,  # stats_service does not exist yet
        "highlight_video_path": highlight_video_path,
        "highlight_video_url": highlight_video_url,
        "errors": errors,
        "available_services": available,
    }


def _build_match_summary(results):
    # type: (Dict[str, Any]) -> Dict[str, Any]
    summary = {
        "total_passes": None,
        "possession_pct": None,
        "xg_team_0": None,
        "xg_team_1": None,
        "shots_team_0": None,
        "shots_team_1": None,
        "total_shots": None,
        "dribble_success_rate_team_0": None,
        "dribble_success_rate_team_1": None,
        "avg_def_line_depth_team_0": None,
        "avg_def_line_depth_team_1": None,
        "counter_press_success_rate_team_0": None,
        "counter_press_success_rate_team_1": None,
        "corners_team_0": None,
        "corners_team_1": None,
        "free_kicks_team_0": None,
        "free_kicks_team_1": None,
        "ppda_team_0": None,
        "ppda_team_1": None,
        "dominant_formation_team_0": None,
        "dominant_formation_team_1": None,
        "top_distance_player": None,
        "top_speed_player": None,
    }  # type: Dict[str, Any]

    pn = results.get("pass_network")
    if pn is not None:
        summary["total_passes"] = pn.get("total_passes")
        pbt = pn.get("passes_by_team")
        if pbt and isinstance(pbt, dict):
            t0 = pbt.get("0", pbt.get(0, 0))
            t1 = pbt.get("1", pbt.get(1, 0))
            total = t0 + t1
            if total > 0:
                summary["possession_pct"] = {
                    "team_0": round(t0 / total * 100, 1),
                    "team_1": round(t1 / total * 100, 1),
                }

    xg = results.get("xg")
    if xg is not None:
        summary["xg_team_0"] = xg.get("xg_team_0")
        summary["xg_team_1"] = xg.get("xg_team_1")
        summary["shots_team_0"] = xg.get("shots_team_0")
        summary["shots_team_1"] = xg.get("shots_team_1")

    # Add event-based statistics
    events = results.get("events")
    if events is not None:
        summary["total_shots"] = events.get("summary", {}).get("team_0", {}).get("shot_count", 0) + events.get("summary", {}).get("team_1", {}).get("shot_count", 0)
        summary["dribble_success_rate_team_0"] = events.get("summary", {}).get("team_0", {}).get("dribble_success_rate")
        summary["dribble_success_rate_team_1"] = events.get("summary", {}).get("team_1", {}).get("dribble_success_rate")

    # Add defensive line statistics
    defensive_lines = results.get("defensive_lines")
    if defensive_lines is not None:
        summary["avg_def_line_depth_team_0"] = defensive_lines.get("team_0", {}).get("avg_defensive_line_depth_m")
        summary["avg_def_line_depth_team_1"] = defensive_lines.get("team_1", {}).get("avg_defensive_line_depth_m")

    # Add counter-press statistics
    counter_press = results.get("counter_press")
    if counter_press is not None:
        summary["counter_press_success_rate_team_0"] = counter_press.get("team_0", {}).get("success_rate")
        summary["counter_press_success_rate_team_1"] = counter_press.get("team_1", {}).get("success_rate")

    # Add set piece statistics
    set_pieces = results.get("set_pieces")
    if set_pieces is not None:
        summary["corners_team_0"] = set_pieces.get("summary", {}).get("team_0", {}).get("corners")
        summary["corners_team_1"] = set_pieces.get("summary", {}).get("team_1", {}).get("corners")
        summary["free_kicks_team_0"] = set_pieces.get("summary", {}).get("team_0", {}).get("free_kicks")
        summary["free_kicks_team_1"] = set_pieces.get("summary", {}).get("team_1", {}).get("free_kicks")

    pressing = results.get("pressing")
    if pressing is not None:
        t0 = pressing.get("team_0", {})
        t1 = pressing.get("team_1", {})
        summary["ppda_team_0"] = t0.get("ppda")
        summary["ppda_team_1"] = t1.get("ppda")

    formations = results.get("formations")
    if formations is not None:
        t0 = formations.get("team_0", {})
        t1 = formations.get("team_1", {})
        summary["dominant_formation_team_0"] = t0.get("dominant_formation")
        summary["dominant_formation_team_1"] = t1.get("dominant_formation")

    heatmaps = results.get("heatmaps")
    if heatmaps is not None:
        players = heatmaps.get("players", {})
        if players:
            best_dist = None  # type: Optional[Dict[str, Any]]
            best_speed = None  # type: Optional[Dict[str, Any]]
            for tid_str, pdata in players.items():
                tid = int(tid_str) if isinstance(tid_str, str) else tid_str
                team = pdata.get("team", -1)
                dist = pdata.get("total_distance_m", 0)
                speed = pdata.get("top_speed_ms", 0)
                if best_dist is None or dist > best_dist["distance_m"]:
                    best_dist = {"track_id": tid, "team": team, "distance_m": round(dist, 1)}
                if best_speed is None or speed > best_speed["speed_ms"]:
                    best_speed = {"track_id": tid, "team": team, "speed_ms": round(speed, 2)}
            summary["top_distance_player"] = best_dist
            summary["top_speed_player"] = best_speed

    return summary
