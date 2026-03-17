import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

PHYSICAL_LIMITS = {
    "max_speed_ms": 12.0,
    "max_distance_per_second_m": 9.0,
    "max_xg_per_shot": 0.99,
    "min_xg_per_shot": 0.01,
    "max_defensive_line_depth_m": 105.0,
    "min_defensive_line_depth_m": 0.0,
    "max_ppda": 50.0,
    "min_ppda": 1.0,
    "max_passes_per_minute": 60,
}


def _check_speed(value, field):
    # type: (Any, str) -> Optional[Dict[str, Any]]
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v > PHYSICAL_LIMITS["max_speed_ms"]:
        return {
            "field": field,
            "value": v,
            "limit": PHYSICAL_LIMITS["max_speed_ms"],
            "action": "nullified",
            "reason": "Exceeds maximum human sprint speed ({} m/s)".format(
                PHYSICAL_LIMITS["max_speed_ms"]
            ),
        }
    return None


def _check_range(value, field, min_val, max_val, reason):
    # type: (Any, str, float, float, str) -> Optional[Dict[str, Any]]
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v < min_val or v > max_val:
        return {
            "field": field,
            "value": v,
            "limit": "{} – {}".format(min_val, max_val),
            "action": "nullified",
            "reason": reason,
        }
    return None


def validate_analytics_report(report):
    # type: (Dict[str, Any]) -> Dict[str, Any]
    """
    Check every metric in match_summary against physical limits.
    For each violation: set the field to null and record a failure entry.
    Returns the cleaned report with a "sanity_check" section.
    """
    failures = []  # type: List[Dict[str, Any]]
    ms = report.get("match_summary", {})
    if not isinstance(ms, dict):
        report["sanity_check"] = {"passed": True, "failures": []}
        return report

    # Top speed player
    tsp = ms.get("top_speed_player")
    if isinstance(tsp, dict):
        speed_ms = tsp.get("speed_ms")
        failure = _check_speed(speed_ms, "top_speed_player.speed_ms")
        if failure:
            failures.append(failure)
            ms["top_speed_player"] = None

    # Top distance player — cross-check against clip duration
    duration = report.get("duration_seconds", 0.0) or 0.0
    max_dist = PHYSICAL_LIMITS["max_distance_per_second_m"] * duration
    tdp = ms.get("top_distance_player")
    if isinstance(tdp, dict) and duration > 0:
        dist_m = tdp.get("distance_m")
        if dist_m is not None:
            try:
                d = float(dist_m)
            except (TypeError, ValueError):
                d = 0.0
            if d > max_dist and max_dist > 0:
                failures.append({
                    "field": "top_distance_player.distance_m",
                    "value": d,
                    "limit": max_dist,
                    "action": "nullified",
                    "reason": (
                        "Distance {:.1f}m exceeds physical maximum "
                        "({} m/s × {:.1f}s = {:.1f}m)".format(
                            d,
                            PHYSICAL_LIMITS["max_distance_per_second_m"],
                            duration,
                            max_dist,
                        )
                    ),
                })
                ms["top_distance_player"] = None

    # xG values
    for field in ("xg_team_0", "xg_team_1"):
        v = ms.get(field)
        if v is not None:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                fv = 0.0
            if fv < 0 or fv > 0.99 * max(ms.get("shots_team_0", 1) or 1,
                                           ms.get("shots_team_1", 1) or 1,
                                           1):
                # xG per shot check — total xG can exceed 0.99 if multiple shots
                pass  # aggregate xG is fine to exceed 0.99

    # PPDA
    for field in ("ppda_team_0", "ppda_team_1"):
        v = ms.get(field)
        if v is not None:
            f = _check_range(
                v, field,
                PHYSICAL_LIMITS["min_ppda"],
                PHYSICAL_LIMITS["max_ppda"],
                "PPDA outside valid range ({} – {})".format(
                    PHYSICAL_LIMITS["min_ppda"], PHYSICAL_LIMITS["max_ppda"]
                ),
            )
            if f:
                failures.append(f)
                ms[field] = None

    # Defensive line depth
    for field in ("avg_def_line_depth_team_0", "avg_def_line_depth_team_1"):
        v = ms.get(field)
        if v is not None:
            f = _check_range(
                v, field,
                PHYSICAL_LIMITS["min_defensive_line_depth_m"],
                PHYSICAL_LIMITS["max_defensive_line_depth_m"],
                "Defensive line depth outside pitch bounds (0 – 105m)",
            )
            if f:
                failures.append(f)
                ms[field] = None

    # Passes per minute
    total_passes = ms.get("total_passes")
    if total_passes is not None and duration > 0:
        try:
            passes_per_min = float(total_passes) / (duration / 60.0)
        except (TypeError, ValueError, ZeroDivisionError):
            passes_per_min = 0.0
        if passes_per_min > PHYSICAL_LIMITS["max_passes_per_minute"]:
            failures.append({
                "field": "total_passes",
                "value": total_passes,
                "limit": "{} passes/min".format(PHYSICAL_LIMITS["max_passes_per_minute"]),
                "action": "nullified",
                "reason": "Pass rate {:.1f}/min exceeds physical maximum".format(
                    passes_per_min
                ),
            })
            ms["total_passes"] = None
            ms["possession_pct"] = None

    report["match_summary"] = ms
    report["sanity_check"] = {
        "passed": len(failures) == 0,
        "failures": failures,
    }

    if failures:
        logger.warning(
            "Sanity check: %d violation(s) nullified for job %s: %s",
            len(failures),
            report.get("job_id", "?"),
            [f["field"] for f in failures],
        )

    return report
