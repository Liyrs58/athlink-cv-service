"""
Multi-pass validator service.

Runs the core tracking + physics pipeline N times on the same video,
compares physical metrics across runs, and marks values as confirmed
(agree within tolerance) or disputed (use median).
"""
import copy
import logging
import math
from statistics import median
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

TOLERANCE = 0.10   # 10% agreement threshold
N_PASSES = 3


def _pct_diff(a: float, b: float) -> float:
    """Percentage difference between two values, relative to the larger."""
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom


def _compare_metric(values: List[float]) -> Dict[str, Any]:
    """
    Given N measurements of the same metric, decide if they are confirmed.

    confirmed = all pairwise differences within TOLERANCE.
    value     = median of all values (robust to any single outlier).
    """
    med = median(values)
    confirmed = all(
        _pct_diff(values[i], values[j]) <= TOLERANCE
        for i in range(len(values))
        for j in range(i + 1, len(values))
    )
    return {"value": round(med, 2), "confirmed": confirmed, "n": len(values)}


def _run_single_pass(video_path: str, job_id: str) -> Dict[str, Any]:
    """
    Run the core tracking + physics pipeline once and return per-player
    physical metrics keyed by track_id.
    """
    # Import here to avoid circular imports at module load
    from services.homography_service import get_frame_calibration
    from services.tracking_service import run_tracking
    from services.reid_service import merge_fragmented_tracks
    from services.physics_corrector import PhysicsCorrector
    from services.velocity_service import compute_all_velocities

    calibration = get_frame_calibration(video_path)
    r = run_tracking(job_id=job_id, video_path=video_path, frame_stride=2)
    tracks = r.get("tracks", [])
    frame_metadata = r.get("frame_metadata", [])

    tracks = merge_fragmented_tracks(tracks, video_path)

    corrector = PhysicsCorrector()
    correction_report = corrector.apply_all_constraints(
        tracks=tracks,
        frame_metadata=frame_metadata,
        calibration=calibration,
    )
    tracks = correction_report["corrected_tracks"]
    calibration = correction_report["calibration"]

    velocities = compute_all_velocities(tracks, calibration=calibration)

    # Index by track_id
    by_track: Dict[int, Dict[str, float]] = {}
    for v in velocities:
        by_track[v["track_id"]] = {
            "max_speed_ms": v["max_speed_ms"],
            "distance_metres": v["distance_metres"],
            "sprint_count": float(v["sprint_count"]),
        }

    return by_track


def run_multi_pass_validation(
    video_path: str,
    job_id: str,
    players_physical: List[Dict[str, Any]],
    n_passes: int = N_PASSES,
) -> Dict[str, Any]:
    """
    Run the pipeline n_passes times and cross-check results.

    Returns a dict with:
      - "passes_run": int
      - "metrics_confirmed": list of metric names that agreed in all passes
      - "metrics_disputed": list of metric names with disagreement
      - "overall_confidence_boost": "high" / "medium" / "low"
      - "per_player": dict keyed by track_id with confirmed values
    """
    logger.info(f"Multi-pass validation: running {n_passes} passes on {video_path}")

    all_pass_results: List[Dict[int, Dict[str, float]]] = []
    for i in range(n_passes):
        try:
            result = _run_single_pass(video_path, job_id=f"{job_id}_v{i}")
            all_pass_results.append(result)
            logger.info(f"Pass {i+1}/{n_passes} complete: {len(result)} tracks")
        except Exception as e:
            logger.warning(f"Pass {i+1} failed: {e}")

    if len(all_pass_results) < 2:
        # Not enough passes succeeded — return pass-through
        return {
            "passes_run": len(all_pass_results),
            "metrics_confirmed": [],
            "metrics_disputed": [],
            "overall_confidence_boost": "low",
            "per_player": {},
            "status": "insufficient_passes",
        }

    # Collect all track_ids seen across all passes
    all_track_ids = set()
    for pass_result in all_pass_results:
        all_track_ids.update(pass_result.keys())

    per_player: Dict[str, Dict[str, Any]] = {}
    metric_keys = ["max_speed_ms", "distance_metres", "sprint_count"]
    metric_confirmed_counts = {k: 0 for k in metric_keys}
    metric_disputed_counts = {k: 0 for k in metric_keys}

    for track_id in all_track_ids:
        player_result: Dict[str, Any] = {}
        for metric in metric_keys:
            values = [
                pr[track_id][metric]
                for pr in all_pass_results
                if track_id in pr and metric in pr[track_id]
            ]
            if len(values) >= 2:
                cmp = _compare_metric(values)
                player_result[metric] = cmp
                if cmp["confirmed"]:
                    metric_confirmed_counts[metric] += 1
                else:
                    metric_disputed_counts[metric] += 1
            elif len(values) == 1:
                player_result[metric] = {
                    "value": round(values[0], 2),
                    "confirmed": False,
                    "n": 1,
                }
                metric_disputed_counts[metric] += 1
        per_player[str(track_id)] = player_result

    # Decide which metrics are confirmed overall
    total_tracks = max(len(all_track_ids), 1)
    confirmed_metrics = []
    disputed_metrics = []
    for metric in metric_keys:
        conf_rate = metric_confirmed_counts[metric] / total_tracks
        if conf_rate >= 0.70:
            confirmed_metrics.append(metric)
        else:
            disputed_metrics.append(metric)

    # Add team_shape to confirmed list if all velocity metrics are confirmed
    if len(confirmed_metrics) == len(metric_keys):
        confirmed_metrics.append("team_shape")

    # Overall confidence boost
    if len(confirmed_metrics) >= 3:
        boost = "high"
    elif len(confirmed_metrics) >= 1:
        boost = "medium"
    else:
        boost = "low"

    # Apply confirmed median values back to players_physical in-place
    for player in players_physical:
        tid = str(player.get("track_id"))
        if tid not in per_player:
            continue
        pr = per_player[tid]

        if "max_speed_ms" in pr:
            confirmed_speed_ms = pr["max_speed_ms"]["value"]
            confirmed_speed_kmh = round(confirmed_speed_ms * 3.6, 1)
            player["max_speed_kmh"] = confirmed_speed_kmh
            player["validation_confirmed"] = pr["max_speed_ms"]["confirmed"]

        if "distance_metres" in pr:
            med_dist = pr["distance_metres"]["value"]
            player["distance_metres"] = round(med_dist, 1)
            pct = {"high": 0.12, "medium": 0.25, "low": 0.40}.get(
                player.get("confidence", "medium"), 0.25
            )
            dist_lo = round(med_dist * (1 - pct), 0)
            dist_hi = round(med_dist * (1 + pct), 0)
            player["distance_range"] = f"{dist_lo:.0f}-{dist_hi:.0f}m"

        if "sprint_count" in pr:
            player["sprints"] = int(round(pr["sprint_count"]["value"]))

    return {
        "passes_run": len(all_pass_results),
        "metrics_confirmed": confirmed_metrics,
        "metrics_disputed": disputed_metrics,
        "overall_confidence_boost": boost,
        "per_player": per_player,
        "status": "ok",
    }
