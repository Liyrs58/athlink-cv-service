"""
Confidence scoring system for honest metric reporting.

Every metric carries a confidence rating so no number is reported
without the system knowing how much to trust it.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# ── Legacy: per-service confidence (used by analytics_service) ────

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
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def assess_confidence(job_id, service_name):
    base = Path("temp") / job_id
    reasons = []
    frames_required = MINIMUM_FRAMES.get(service_name, 75)

    track_data = _load_json(base / "tracking" / "track_results.json")
    frames_analysed = 0
    valid_frames_pct = 100.0
    if track_data is not None:
        frames_analysed = int(track_data.get("framesProcessed", 0))
        valid_frames_count = int(track_data.get("validFramesCount", frames_analysed))
        valid_frames_pct = (valid_frames_count / frames_analysed * 100.0) if frames_analysed > 0 else 100.0
    else:
        reasons.append("track_results.json not found")

    if valid_frames_pct < 50.0:
        reasons.append("only {:.1f}% of frames were valid pitch views".format(valid_frames_pct))

    tq = track_data.get("tracking_quality", {}) if track_data else {}
    id_switches = tq.get("id_switches_total", 0)
    avg_track_len = tq.get("avg_track_length_frames", 999)
    stable_tracks = tq.get("tracks_with_5plus_detections", 999)

    if id_switches > 50:
        reasons.append("high ID churn: {} switches".format(id_switches))
    if avg_track_len < 10:
        reasons.append("short avg track length: {:.1f} frames".format(avg_track_len))
    if stable_tracks < 10:
        reasons.append("insufficient stable tracks: {}".format(stable_tracks))

    if frames_analysed == 0:
        return {
            "confidence": "unavailable", "frames_analysed": 0,
            "frames_required": frames_required, "calibration_valid": False,
            "ball_available": False, "reasons": reasons or ["no frames analysed"],
            "usable": False,
        }

    if frames_analysed < frames_required:
        reasons.append("Only {} frames analysed, {} recommended".format(frames_analysed, frames_required))

    pitch_data = _load_json(base / "pitch" / "pitch_map.json")
    calibration_valid = False
    ball_available = False

    if pitch_data is None:
        reasons.append("pitch_map.json not found")
    else:
        calibration_valid = pitch_data.get("calibration_valid", False)
        if not calibration_valid:
            reasons.append("homography_calibration_failed")
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

    team_data = _load_json(base / "tracking" / "team_results.json")
    if team_data is not None:
        tracks = team_data.get("tracks", team_data) if isinstance(team_data, dict) else team_data
        counts = {0: 0, 1: 0}
        for t in tracks:
            tid = t.get("teamId", -1)
            if tid in counts:
                counts[tid] += 1
        total = counts[0] + counts[1]
        if total > 0 and max(counts[0], counts[1]) / total > 0.70:
            reasons.append("team split imbalanced ({}/{})".format(counts[0], counts[1]))

    world_coord_services = {"pass_network", "pressing", "events", "counter_press", "set_pieces", "defensive_lines", "xg"}
    needs_world_coords = service_name in world_coord_services

    if valid_frames_pct < 50.0:
        confidence = "low"
        usable = frames_analysed >= frames_required * 0.5
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
        "confidence": confidence, "frames_analysed": frames_analysed,
        "valid_frames_pct": round(valid_frames_pct, 1),
        "frames_required": frames_required, "calibration_valid": calibration_valid,
        "ball_available": ball_available, "reasons": reasons, "usable": usable,
    }


def assess_data_quality(job_id):
    base = Path("temp") / job_id
    track_data = _load_json(base / "tracking" / "track_results.json")
    frames_analysed = 0
    fps = 25.0
    if track_data is not None:
        frames_analysed = int(track_data.get("framesProcessed", 0))
        fps = float(track_data.get("fps") or track_data.get("metadata", {}).get("fps") or 25.0)
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
                if len(traj) / max(frames_analysed, 1) >= 0.05:
                    ball_tracking_available = True
                break

    sufficient_for_tactical = calibration_valid and ball_tracking_available and frames_analysed >= 150

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
        "overall": overall, "calibration_valid": calibration_valid,
        "ball_tracking_available": ball_tracking_available,
        "frames_analysed": frames_analysed, "clip_duration_seconds": clip_duration,
        "sufficient_for_tactical_analysis": sufficient_for_tactical,
        "services_unavailable": services_unavailable,
        "services_low_confidence": services_low_confidence,
    }


# ── Part 1.1: Track-level confidence ─────────────────────────────

def score_track_confidence(track: dict) -> dict:
    """Score how much to trust a single track's data."""
    det_count = track.get("confirmed_detections", 0) or 0
    reid_merges = track.get("reid_merges", 0) or 0

    traj = track.get("trajectory", [])
    first = track.get("firstSeen", 0)
    last = track.get("lastSeen", 0)
    expected_span = max(last - first, 1)
    continuity = len(traj) / expected_span if expected_span > 0 else 0.0
    continuity = min(continuity, 1.0)

    reasons: List[str] = []
    score = 0.5  # start neutral

    # Detection count
    if det_count >= 20:
        score += 0.25
    elif det_count >= 10:
        score += 0.10
    else:
        score -= 0.20
        reasons.append(f"only {det_count} detections")

    # ReID merges
    if reid_merges <= 1:
        score += 0.15
    elif reid_merges <= 3:
        score += 0.05
        reasons.append(f"{reid_merges} ReID merges")
    else:
        score -= 0.15
        reasons.append(f"{reid_merges} ReID merges (high fragmentation)")

    # Continuity
    if continuity >= 0.70:
        score += 0.10
    elif continuity >= 0.40:
        pass
    else:
        score -= 0.10
        reasons.append(f"visible in only {continuity:.0%} of expected frames")

    score = max(0.0, min(1.0, score))

    if score >= 0.80:
        level = "high"
    elif score >= 0.50:
        level = "medium"
    else:
        level = "low"

    return {
        "level": level,
        "score": round(score, 2),
        "reasons": reasons,
        "reid_merges": reid_merges,
        "detection_count": det_count,
        "track_continuity": round(continuity, 2),
    }


# ── Part 1.2: Physical metric confidence ─────────────────────────

_MARGIN_TABLE = {
    "speed":    {"high": 10, "medium": 20, "low": 35},
    "distance": {"high": 12, "medium": 25, "low": 40},
}

_SPRINT_MARGIN = {"high": 1, "medium": 3, "low": None}  # None → range


def score_physical_metric(value, track_confidence: str, metric_type: str) -> dict:
    """Attach confidence and margin of error to a single metric value."""
    if value is None or value == 0:
        return {
            "value": value,
            "confidence": "low",
            "margin_of_error": "N/A",
            "display_value": "data insufficient",
        }

    if metric_type == "sprint_count":
        margin = _SPRINT_MARGIN.get(track_confidence)
        if margin is None:
            lo = max(0, int(value) - 3)
            hi = int(value) + 3
            display = f"{lo}-{hi} sprints"
        else:
            display = f"{int(value)} ±{margin} sprints"
        return {
            "value": int(value),
            "confidence": track_confidence,
            "margin_of_error": f"±{margin}" if margin is not None else "±3 (range)",
            "display_value": display,
        }

    pct = _MARGIN_TABLE.get(metric_type, {}).get(track_confidence, 30)
    unit = "km/h" if metric_type == "speed" else "m"
    display = f"{value} {unit} ±{pct}%"

    return {
        "value": value,
        "confidence": track_confidence,
        "margin_of_error": f"±{pct}%",
        "display_value": display,
    }


# ── Part 1.3: Shape confidence ───────────────────────────────────

def score_shape_confidence(shape_summary: dict, track_list: list) -> dict:
    """Score how much to trust formation/shape metrics."""
    if not shape_summary or shape_summary.get("data_quality") == "unavailable":
        return {
            "overall": "low",
            "reason": "shape data unavailable",
            "high_confidence_contributors": 0,
            "frames_with_valid_shape": 0,
        }

    high_count = sum(
        1 for t in track_list
        if score_track_confidence(t)["level"] == "high"
        and not t.get("is_staff", False)
    )
    frames_analysed = shape_summary.get("frames_analysed", 0)

    if high_count >= 10 and frames_analysed >= 5:
        level = "high"
    elif high_count >= 5 and frames_analysed >= 3:
        level = "medium"
    else:
        level = "low"

    return {
        "overall": level,
        "high_confidence_contributors": high_count,
        "frames_with_valid_shape": frames_analysed,
    }


# ── Part 1.4: Top-level data confidence summary ──────────────────

def build_data_confidence_summary(
    tracks: list,
    vel_summary: dict,
    shape_summary: dict,
) -> dict:
    """Build an honest top-level summary of data quality."""

    confidences = []
    for t in tracks:
        if t.get("is_staff", False):
            continue
        if (t.get("confirmed_detections", 0) or 0) < 5:
            continue
        confidences.append(score_track_confidence(t))

    high = sum(1 for c in confidences if c["level"] == "high")
    medium = sum(1 for c in confidences if c["level"] == "medium")
    low = sum(1 for c in confidences if c["level"] == "low")
    total = high + medium + low

    # Overall grade
    if total == 0:
        grade = "D"
        explanation = "Insufficient tracking data for reliable analysis."
    elif high / total >= 0.6:
        grade = "A"
        explanation = "Majority of players tracked with high confidence. Team-level and individual metrics are reliable."
    elif high / total >= 0.3:
        grade = "B"
        explanation = "Team-level metrics reliable. Individual player stats have medium confidence due to broadcast tracking limitations."
    elif medium / total >= 0.4:
        grade = "C"
        explanation = "Team-level patterns visible but individual stats are approximate. Use for relative comparison only."
    else:
        grade = "D"
        explanation = "Low tracking quality. Only broad team-level observations are usable."

    what_to_trust = []
    what_approximate = []
    what_unavailable = [
        "Ball possession",
        "Pass accuracy",
        "Exact positions during dead ball",
    ]

    if grade in ("A", "B"):
        what_to_trust.extend([
            "Situation timeline",
            "Team shape and spacing",
            "Relative player workload comparison",
        ])
    if grade == "A":
        what_to_trust.append("Individual sprint counts")

    what_approximate.extend([
        f"Individual sprint counts (±2-3 sprints)",
        f"Individual distances (±20%)",
        f"Speed figures (±15%)",
    ])

    if grade in ("C", "D"):
        what_approximate.append("Team shape metrics")

    if grade in ("A", "B"):
        rec = "Use team-level insights for tactical decisions. Use player physical data for relative comparison only."
    elif grade == "C":
        rec = "Focus on team shape and rhythm. Individual player numbers are rough estimates."
    else:
        rec = "Data quality is too low for confident analysis. Consider re-recording with a steadier camera angle."

    return {
        "overall_grade": grade,
        "grade_explanation": explanation,
        "high_confidence_players": high,
        "medium_confidence_players": medium,
        "low_confidence_players": low,
        "what_to_trust": what_to_trust,
        "what_to_treat_as_approximate": what_approximate,
        "what_is_unavailable": what_unavailable,
        "coach_recommendation": rec,
    }
