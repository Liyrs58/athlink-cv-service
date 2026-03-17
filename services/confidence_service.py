import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Minimum frames for each service to produce reliable metrics
MINIMUM_FRAMES = {
    "pass_network":    150,
    "pressing":        150,
    "xg":              75,
    "formations":      75,
    "heatmaps":        50,
    "events":          150,
    "counter_press":   150,
    "set_pieces":      500,
    "defensive_lines": 50,
    "tactics":         75,
}


def _load_json(path):
    # type: (Path) -> Optional[Dict]
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def assess_confidence(job_id, service_name):
    # type: (str, str) -> Dict[str, Any]
    """
    Assess data quality and confidence for a given service and job.

    Returns:
        {
          "confidence": "high" | "medium" | "low" | "unavailable",
          "frames_analysed": int,
          "frames_required": int,
          "calibration_valid": bool,
          "ball_available": bool,
          "reasons": [str],
          "usable": bool
        }
    """
    base = Path("temp") / job_id
    reasons = []  # type: List[str]
    frames_required = MINIMUM_FRAMES.get(service_name, 75)

    # --- Frames analysed ---
    track_data = _load_json(base / "tracking" / "track_results.json")
    frames_analysed = 0
    valid_frames_pct = 100.0  # FIX 5: assume valid if no metadata
    if track_data is not None:
        frames_analysed = int(track_data.get("framesProcessed", 0))
        valid_frames_count = int(track_data.get("validFramesCount", frames_analysed))
        valid_frames_pct = (valid_frames_count / frames_analysed * 100.0) if frames_analysed > 0 else 100.0
    else:
        reasons.append("track_results.json not found")

    # FIX 5: Check if too many non-pitch frames
    if valid_frames_pct < 50.0:
        reasons.append(
            "only {:.1f}% of frames were valid pitch views — many cutaways detected".format(valid_frames_pct)
        )

    if frames_analysed == 0:
        return {
            "confidence": "unavailable",
            "frames_analysed": 0,
            "frames_required": frames_required,
            "calibration_valid": False,
            "ball_available": False,
            "reasons": reasons or ["no frames analysed"],
            "usable": False,
        }

    if frames_analysed < frames_required:
        pct = frames_analysed / frames_required
        reasons.append(
            "Only {} frames analysed, {} recommended".format(frames_analysed, frames_required)
        )

    # --- Calibration ---
    pitch_data = _load_json(base / "pitch" / "pitch_map.json")
    calibration_valid = False
    ball_available = False

    if pitch_data is None:
        reasons.append("pitch_map.json not found")
    else:
        calibration_valid = pitch_data.get("calibration_valid", False)
        if not calibration_valid:
            reasons.append("homography_calibration_failed")

        # Ball available = trackId -1 entry with trajectory
        for player in pitch_data.get("players", []):
            if player.get("trackId") == -1:
                traj = player.get("trajectory2d", [])
                coverage = len(traj) / max(frames_analysed, 1)
                if coverage >= 0.05:
                    ball_available = True
                else:
                    reasons.append("ball tracking coverage too low ({:.1%})".format(coverage))
                break
        if not ball_available and calibration_valid:
            reasons.append("ball not tracked in pitch_map")

    # --- Team balance ---
    team_data = _load_json(base / "tracking" / "team_results.json")
    if team_data is not None:
        tracks = team_data.get("tracks", team_data) if isinstance(team_data, dict) else team_data
        counts = {0: 0, 1: 0}
        for t in tracks:
            tid = t.get("teamId", -1)
            if tid in counts:
                counts[tid] += 1
        total = counts[0] + counts[1]
        if total > 0:
            ratio = max(counts[0], counts[1]) / total
            if ratio > 0.70:
                reasons.append(
                    "team split imbalanced ({}/{})".format(counts[0], counts[1])
                )

    # --- Services that require world coords ---
    world_coord_services = {
        "pass_network", "pressing", "events", "counter_press",
        "set_pieces", "defensive_lines", "xg",
    }
    needs_world_coords = service_name in world_coord_services

    # --- Determine confidence level ---
    # FIX 5: Downgrade confidence if too many non-pitch frames
    if valid_frames_pct < 50.0:
        confidence = "low"
        usable = True if frames_analysed >= frames_required * 0.5 else False
    elif needs_world_coords and not calibration_valid:
        confidence = "unavailable"
        usable = False
    elif needs_world_coords and not ball_available:
        confidence = "unavailable"
        usable = False
    elif frames_analysed < frames_required * 0.5:
        confidence = "unavailable"
        usable = False
    elif frames_analysed < frames_required:
        confidence = "low"
        usable = True
    elif reasons:
        confidence = "medium"
        usable = True
    else:
        confidence = "high"
        usable = True

    return {
        "confidence": confidence,
        "frames_analysed": frames_analysed,
        "valid_frames_pct": round(valid_frames_pct, 1),  # FIX 5
        "frames_required": frames_required,
        "calibration_valid": calibration_valid,
        "ball_available": ball_available,
        "reasons": reasons,
        "usable": usable,
    }


def assess_data_quality(job_id):
    # type: (str) -> Dict[str, Any]
    """
    Assess overall data quality for the job.
    Returns a data_quality section suitable for inclusion in the analytics report.
    """
    base = Path("temp") / job_id

    track_data = _load_json(base / "tracking" / "track_results.json")
    frames_analysed = 0
    fps = 25.0
    if track_data is not None:
        frames_analysed = int(track_data.get("framesProcessed", 0))
        fps = float(
            track_data.get("fps") or
            track_data.get("metadata", {}).get("fps") or
            25.0
        )
        if fps <= 0:
            fps = 25.0

    clip_duration = round(frames_analysed / fps, 2) if fps > 0 else 0.0

    pitch_data = _load_json(base / "pitch" / "pitch_map.json")
    calibration_valid = False
    ball_tracking_available = False

    if pitch_data is not None:
        calibration_valid = pitch_data.get("calibration_valid", False)
        for player in pitch_data.get("players", []):
            if player.get("trackId") == -1:
                traj = player.get("trajectory2d", [])
                coverage = len(traj) / max(frames_analysed, 1)
                if coverage >= 0.05:
                    ball_tracking_available = True
                break

    sufficient_for_tactical = (
        calibration_valid
        and ball_tracking_available
        and frames_analysed >= 150
    )

    # Per-service confidence
    services_unavailable = []
    services_low_confidence = []
    for svc in MINIMUM_FRAMES:
        c = assess_confidence(job_id, svc)
        if c["confidence"] == "unavailable":
            services_unavailable.append(svc)
        elif c["confidence"] == "low":
            services_low_confidence.append(svc)

    if not calibration_valid or not ball_tracking_available or frames_analysed < 50:
        overall = "unavailable"
    elif services_unavailable:
        overall = "low"
    elif services_low_confidence:
        overall = "medium"
    else:
        overall = "high"

    return {
        "overall": overall,
        "calibration_valid": calibration_valid,
        "ball_tracking_available": ball_tracking_available,
        "frames_analysed": frames_analysed,
        "clip_duration_seconds": clip_duration,
        "sufficient_for_tactical_analysis": sufficient_for_tactical,
        "services_unavailable": services_unavailable,
        "services_low_confidence": services_low_confidence,
    }
