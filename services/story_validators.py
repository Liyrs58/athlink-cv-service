"""Story-validity gates for tactical overlay rendering.

A clip should not carry a `PRESSING TRAP / TOUCHLINE 2V1` title unless the
broadcast geometry actually shows a pressing trap. This module checks the
preconditions and recommends an honest story-type when they're not met.

Pure functions over pre-computed world_pos data — no I/O, no dependency on
the renderer.
"""
from __future__ import annotations

from math import atan2, degrees
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# (title, subtitle) per honest story type. Used by the renderer when it
# downgrades a story.
STORY_TYPE_TITLE: Dict[str, Tuple[str, str]] = {
    "PRESSING_TRAP":        ("PRESSING TRAP",       "TOUCHLINE 2V1"),
    "PRESSING_TRIANGLE":    ("PRESSING TRIANGLE",   "MID-PITCH SQUEEZE"),
    "WIDE_CHANNEL_CHASE":   ("WIDE CHANNEL",        "CHASE TO TOUCHLINE"),
    "TRANSITION_PRESSURE":  ("TRANSITION",          "PRESSURE ON CARRIER"),
    "TRANSITION_CARRY":     ("TRANSITION CARRY",    "DRIVING INTO SPACE"),
    "GOALMOUTH_PRESSURE":   ("GOALMOUTH PRESSURE",  "LAST DITCH DEFENDING"),
    "RECOVERY_RUN":         ("RECOVERY RUN",        "TRACKING BACK"),
    "1V1_PRESSURE":         ("1V1 PRESSURE",        "ONE-ON-ONE DUEL"),
}


# Geometry thresholds (metres + degrees). These are intentionally strict so a
# story has to actually look like the tactic it claims to illustrate.
PRESSING_TRAP_TOUCHLINE_M = 6.0
PRESSING_TRAP_PRESSER_RADIUS_M = 8.0
PRESSING_TRAP_MIN_PRESSERS = 2
PRESSING_TRAP_MIN_ANGLE_SPREAD_DEG = 45.0
PRESSING_TRAP_MAX_ZONE_AREA_M2 = 80.0


def _polygon_area(points: Sequence[Tuple[float, float]]) -> float:
    """Shoelace area of a 2D polygon. Returns 0 for <3 points."""
    if len(points) < 3:
        return 0.0
    a = 0.0
    n = len(points)
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        a += x0 * y1 - x1 * y0
    return abs(a) / 2.0


def _convex_hull(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Andrew's monotone chain. Returns CCW hull, no duplicate endpoint."""
    pts = sorted(set((float(x), float(y)) for x, y in points))
    if len(pts) <= 2:
        return pts
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _angular_spread(carrier: Tuple[float, float], pressers: Sequence[Tuple[float, float]]) -> float:
    """Maximum pairwise angle (deg) between presser->carrier vectors as seen
    from the carrier. 0 = colinear chase. 90 = perpendicular pressure.
    Returns 0 if <2 pressers."""
    if len(pressers) < 2:
        return 0.0
    cx, cy = carrier
    angles: List[float] = []
    for px, py in pressers:
        a = degrees(atan2(py - cy, px - cx))
        angles.append(a)
    angles.sort()
    # Best pairwise spread = max gap on a circle, where the spread is
    # (180 - largest_gap) when one of the gaps spans the "outside" of the
    # cluster. Simpler: sort and find the largest |a_i - a_j| under modular
    # arithmetic.
    best = 0.0
    for i in range(len(angles)):
        for j in range(i + 1, len(angles)):
            d = abs(angles[i] - angles[j])
            d = min(d, 360.0 - d)
            best = max(best, d)
    return best


def _carrier_pitch_third(carrier_x: float, pitch_w: float = 105.0) -> str:
    if carrier_x < pitch_w / 3.0:
        return "def"
    if carrier_x < 2 * pitch_w / 3.0:
        return "mid"
    return "att"


def validate_pressing_trap(
    world_pos: Dict[Tuple[int, int], Tuple[float, float]],
    anchor_frame: int,
    carrier_tid: int,
    defender_tids: Iterable[int],
    *,
    pitch_w: float = 105.0,
    pitch_h: float = 68.0,
) -> Tuple[bool, str, dict]:
    """Returns (is_valid, recommended_story_type, geometry_dict).

    A pressing trap requires (all must hold):
      - carrier within `PRESSING_TRAP_TOUCHLINE_M` metres of nearest touchline
      - >= `PRESSING_TRAP_MIN_PRESSERS` defenders within
        `PRESSING_TRAP_PRESSER_RADIUS_M` metres of the carrier
      - presser angle spread (around the carrier) >= `PRESSING_TRAP_MIN_ANGLE_SPREAD_DEG`
      - convex hull of (carrier + active pressers) has area
        <= `PRESSING_TRAP_MAX_ZONE_AREA_M2` square metres

    If invalid, recommends a downgraded story type. See docstring of
    `_recommend_downgrade` for the decision tree.
    """
    cw = world_pos.get((int(carrier_tid), int(anchor_frame)))
    if cw is None:
        # No world position — can't validate. Fall back to a non-trap default.
        return False, "1V1_PRESSURE", {
            "carrier_touchline_distance_m": None,
            "pressers_active": 0,
            "presser_angle_spread_deg": 0.0,
            "trap_zone_area_m2": 0.0,
            "carrier_pitch_third": "?",
        }

    cx, cy = cw
    # Distance to nearest touchline (top y=0 or bottom y=pitch_h)
    touchline_dist = min(abs(cy), abs(pitch_h - cy))

    # Active pressers: defenders within radius
    active_pressers: List[Tuple[int, Tuple[float, float], float]] = []
    for dtid in defender_tids:
        dw = world_pos.get((int(dtid), int(anchor_frame)))
        if dw is None:
            continue
        d = float(np.hypot(dw[0] - cx, dw[1] - cy))
        if d <= PRESSING_TRAP_PRESSER_RADIUS_M:
            active_pressers.append((int(dtid), dw, d))

    presser_pts = [p[1] for p in active_pressers]
    spread = _angular_spread(cw, presser_pts)

    # Convex-hull area of carrier + active pressers
    hull_pts = _convex_hull([cw] + list(presser_pts))
    zone_area = _polygon_area(hull_pts)

    geom = {
        "carrier_touchline_distance_m": round(float(touchline_dist), 2),
        "pressers_active": len(active_pressers),
        "presser_angle_spread_deg": round(float(spread), 1),
        "trap_zone_area_m2": round(float(zone_area), 1),
        "carrier_pitch_third": _carrier_pitch_third(cx, pitch_w),
    }

    near_touchline = touchline_dist <= PRESSING_TRAP_TOUCHLINE_M
    enough_pressers = len(active_pressers) >= PRESSING_TRAP_MIN_PRESSERS
    good_spread = spread >= PRESSING_TRAP_MIN_ANGLE_SPREAD_DEG
    tight_zone = zone_area <= PRESSING_TRAP_MAX_ZONE_AREA_M2

    is_trap = near_touchline and enough_pressers and good_spread and tight_zone
    if is_trap:
        return True, "PRESSING_TRAP", geom

    return False, _recommend_downgrade(
        near_touchline=near_touchline,
        n_pressers=len(active_pressers),
        good_spread=good_spread,
        tight_zone=tight_zone,
    ), geom


def _recommend_downgrade(*, near_touchline: bool, n_pressers: int,
                         good_spread: bool, tight_zone: bool) -> str:
    """Pick an honest story_type label given which preconditions failed."""
    if n_pressers >= 2 and good_spread and tight_zone and not near_touchline:
        return "PRESSING_TRIANGLE"
    if near_touchline and n_pressers == 1:
        return "WIDE_CHANNEL_CHASE"
    if n_pressers >= 2 and not good_spread:
        # Defenders chasing in the same line behind the carrier
        return "TRANSITION_PRESSURE"
    if n_pressers == 1 and not near_touchline:
        return "RECOVERY_RUN"
    return "1V1_PRESSURE"


def soft_score_pressing_trap(geom: dict) -> float:
    """Used by the relocator to rank candidate frames when no frame is fully
    valid. Higher = closer to a real pressing trap."""
    if geom is None:
        return -1.0
    pressers = float(geom.get("pressers_active") or 0)
    touchline = geom.get("carrier_touchline_distance_m")
    if touchline is None:
        touchline = 999.0
    spread = float(geom.get("presser_angle_spread_deg") or 0.0)
    area = float(geom.get("trap_zone_area_m2") or 9999.0)
    # Strongly prefer 2+ pressers, near touchline, good spread, tight zone
    score = (
        pressers * 10.0
        + max(0.0, (PRESSING_TRAP_TOUCHLINE_M * 2 - float(touchline))) * 0.8
        + min(spread, 90.0) / 9.0
        - max(0.0, area - PRESSING_TRAP_MAX_ZONE_AREA_M2) * 0.05
    )
    return score


# ── Additional story-type validators ─────────────────────────────────────

PRESSING_TRIANGLE_MIN_PRESSERS = 2
PRESSING_TRIANGLE_MIN_ANGLE_DEG = 45.0
PRESSING_TRIANGLE_MAX_ZONE_AREA_M2 = 80.0


def validate_pressing_triangle(
    world_pos: Dict[Tuple[int, int], Tuple[float, float]],
    anchor_frame: int,
    carrier_tid: int,
    defender_tids: Iterable[int],
    *,
    pitch_w: float = 105.0,
    pitch_h: float = 68.0,
) -> Tuple[bool, str, dict]:
    """Mid-pitch three-man press — carrier is NOT near a touchline.
    Requires >=2 defenders within 8 m with >=45 deg angle spread.
    Returns (is_valid, story_type, geom_dict)."""
    cw = world_pos.get((int(carrier_tid), int(anchor_frame)))
    if cw is None:
        return False, "1V1_PRESSURE", {}
    cx, cy = cw
    touchline_dist = min(abs(cy), abs(pitch_h - cy))

    active: list = []
    for dtid in defender_tids:
        dw = world_pos.get((int(dtid), int(anchor_frame)))
        if dw is None:
            continue
        d = float(np.hypot(dw[0] - cx, dw[1] - cy))
        if d <= PRESSING_TRAP_PRESSER_RADIUS_M:
            active.append(dw)

    spread = _angular_spread(cw, active)
    hull_pts = _convex_hull([cw] + active)
    zone_area = _polygon_area(hull_pts)

    geom = {
        "carrier_touchline_distance_m": round(float(touchline_dist), 2),
        "pressers_active": len(active),
        "presser_angle_spread_deg": round(float(spread), 1),
        "trap_zone_area_m2": round(float(zone_area), 1),
        "carrier_pitch_third": _carrier_pitch_third(cx, pitch_w),
    }

    is_triangle = (
        len(active) >= PRESSING_TRIANGLE_MIN_PRESSERS
        and spread >= PRESSING_TRIANGLE_MIN_ANGLE_DEG
        and zone_area <= PRESSING_TRIANGLE_MAX_ZONE_AREA_M2
        and touchline_dist > PRESSING_TRAP_TOUCHLINE_M  # must be mid-pitch
    )
    return (is_triangle, "PRESSING_TRIANGLE" if is_triangle else "1V1_PRESSURE", geom)


GOALMOUTH_ATT_THIRD_FRAC = 2.0 / 3.0
GOALMOUTH_TOUCHLINE_M = 20.0  # goal area is within 20 m of a touchline
GOALMOUTH_MIN_PRESSERS = 1


def validate_goalmouth_pressure(
    world_pos: Dict[Tuple[int, int], Tuple[float, float]],
    anchor_frame: int,
    carrier_tid: int,
    defender_tids: Iterable[int],
    *,
    pitch_w: float = 105.0,
    pitch_h: float = 68.0,
) -> Tuple[bool, str, dict]:
    """Carrier inside the attacking third, goal-mouth zone, with at least
    one tight defender. Returns (is_valid, story_type, geom_dict)."""
    cw = world_pos.get((int(carrier_tid), int(anchor_frame)))
    if cw is None:
        return False, "1V1_PRESSURE", {}
    cx, cy = cw
    touchline_dist = min(abs(cy), abs(pitch_h - cy))
    in_att_third = cx >= pitch_w * GOALMOUTH_ATT_THIRD_FRAC

    active: list = []
    for dtid in defender_tids:
        dw = world_pos.get((int(dtid), int(anchor_frame)))
        if dw is None:
            continue
        d = float(np.hypot(dw[0] - cx, dw[1] - cy))
        if d <= PRESSING_TRAP_PRESSER_RADIUS_M:
            active.append(dw)

    geom = {
        "carrier_touchline_distance_m": round(float(touchline_dist), 2),
        "pressers_active": len(active),
        "presser_angle_spread_deg": 0.0,
        "trap_zone_area_m2": 0.0,
        "carrier_pitch_third": _carrier_pitch_third(cx, pitch_w),
    }

    is_gm = (
        in_att_third
        and len(active) >= GOALMOUTH_MIN_PRESSERS
        and touchline_dist <= GOALMOUTH_TOUCHLINE_M
    )
    return (is_gm, "GOALMOUTH_PRESSURE" if is_gm else "1V1_PRESSURE", geom)


TRANSITION_CARRY_MIN_CARRY_DIST_M = 5.0  # carrier must have run >=5 m in window
TRANSITION_CARRY_MAX_PRESSERS = 1        # at most 1 close defender (open space carry)


def validate_transition_carry(
    world_pos: Dict[Tuple[int, int], Tuple[float, float]],
    anchor_frame: int,
    carrier_tid: int,
    defender_tids: Iterable[int],
    *,
    look_back_frames: int = 25,
    pitch_w: float = 105.0,
    pitch_h: float = 68.0,
) -> Tuple[bool, str, dict]:
    """Carrier driving into space — few defenders close, significant carry distance.
    Returns (is_valid, story_type, geom_dict)."""
    cw = world_pos.get((int(carrier_tid), int(anchor_frame)))
    if cw is None:
        return False, "1V1_PRESSURE", {}
    cx, cy = cw

    # Estimate carry distance: compare to position look_back_frames ago
    past = world_pos.get((int(carrier_tid), max(0, int(anchor_frame) - look_back_frames)))
    carry_dist = float(np.hypot(cx - past[0], cy - past[1])) if past else 0.0

    close_defenders = sum(
        1 for dtid in defender_tids
        if (dw := world_pos.get((int(dtid), int(anchor_frame)))) is not None
        and float(np.hypot(dw[0] - cx, dw[1] - cy)) <= PRESSING_TRAP_PRESSER_RADIUS_M
    )

    geom = {
        "carrier_touchline_distance_m": round(float(min(abs(cy), abs(pitch_h - cy))), 2),
        "pressers_active": close_defenders,
        "carry_dist_m": round(carry_dist, 2),
        "presser_angle_spread_deg": 0.0,
        "trap_zone_area_m2": 0.0,
        "carrier_pitch_third": _carrier_pitch_third(cx, pitch_w),
    }

    is_carry = (
        carry_dist >= TRANSITION_CARRY_MIN_CARRY_DIST_M
        and close_defenders <= TRANSITION_CARRY_MAX_PRESSERS
    )
    return (is_carry, "TRANSITION_CARRY" if is_carry else "1V1_PRESSURE", geom)
