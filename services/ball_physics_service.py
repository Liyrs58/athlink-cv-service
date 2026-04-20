"""
Ball Z recovery from 2D pitch trajectory.

The ball is observed only in pitch coordinates (x, y). Height Z is unobserved
and must be inferred from physics. We segment the 2D trajectory into arcs
between contact events (kicks, bounces, header deflections) and for each
arc with high enough 2D speed to qualify as flight, compute Z analytically
under the assumption Z(t_a)=Z(t_b)=0. This reduces the Cd/Magnus fit to a
closed-form parabola: Z(t) = (g/2)(t - t_a)(t_b - t).

Hard-gated on calibration_valid=True — without a validated homography, pitch
coords are proportional-fallback estimates and any Z derived from them is
noise.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List

logger = logging.getLogger(__name__)

# Physics
GRAVITY_MS2 = 9.81

# Arc-segmentation / flight-classification thresholds. These are conservative
# on purpose: false positives (showing ball in flight on a ground roll) look
# worse than false negatives (keeping a real flight grounded).
MIN_ARC_DURATION_SEC = 0.3
MAX_ARC_DURATION_SEC = 4.0
MIN_FLIGHT_SPEED_MS = 8.0        # below this → ground roll, not flight
DIRECTION_CHANGE_DEG = 45.0      # angle between successive velocities to flag a contact
SPEED_JUMP_RATIO = 0.6           # |Δs|/s_prev threshold for a contact
SMOOTHING_WINDOW = 3             # moving average for speed smoothing


def _smooth(xs: List[float], window: int) -> List[float]:
    """Symmetric moving average; skips NaN entries in the mean."""
    if window <= 1 or len(xs) <= window:
        return list(xs)
    half = window // 2
    out = []
    for i in range(len(xs)):
        lo = max(0, i - half)
        hi = min(len(xs), i + half + 1)
        seg = [v for v in xs[lo:hi] if not math.isnan(v)]
        out.append(sum(seg) / len(seg) if seg else float("nan"))
    return out


def _detect_contacts(
    xs: List[float],
    ys: List[float],
    frame_indices: List[int],
    fps: float,
) -> List[int]:
    """
    Indices (into the arrays) where a contact event occurs — sharp direction
    change or speed jump. Endpoints are implicit boundaries; do not include.
    """
    n = len(xs)
    if n < 3:
        return []

    vxs: List[float] = []
    vys: List[float] = []
    speeds: List[float] = []
    for i in range(n - 1):
        dt = (frame_indices[i + 1] - frame_indices[i]) / fps
        if dt <= 0:
            vxs.append(0.0)
            vys.append(0.0)
            speeds.append(0.0)
            continue
        vx = (xs[i + 1] - xs[i]) / dt
        vy = (ys[i + 1] - ys[i]) / dt
        vxs.append(vx)
        vys.append(vy)
        speeds.append(math.hypot(vx, vy))

    speeds_s = _smooth(speeds, SMOOTHING_WINDOW)
    cos_thresh = math.cos(math.radians(DIRECTION_CHANGE_DEG))

    contacts: List[int] = []
    for i in range(1, n - 1):
        s_prev = speeds_s[i - 1]
        s_curr = speeds_s[i]
        if s_prev < 1e-3 or s_curr < 1e-3:
            continue
        dot = vxs[i - 1] * vxs[i] + vys[i - 1] * vys[i]
        cos_angle = max(-1.0, min(1.0, dot / (s_prev * s_curr)))
        if cos_angle < cos_thresh:
            contacts.append(i)
            continue
        if abs(s_curr - s_prev) / max(s_prev, 1e-3) > SPEED_JUMP_RATIO:
            contacts.append(i)
    return contacts


def _arc_mean_speed(
    xs: List[float],
    ys: List[float],
    frame_indices: List[int],
    fps: float,
    a: int,
    b: int,
) -> float:
    """Mean 2D speed (m/s) over indices [a, b] inclusive."""
    if b <= a:
        return 0.0
    total_dist = 0.0
    for i in range(a, b):
        total_dist += math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])
    duration = (frame_indices[b] - frame_indices[a]) / fps
    if duration <= 0:
        return 0.0
    return total_dist / duration


def compute_ball_heights(
    ball_trajectory: List[Dict],
    fps: float,
    calibration_valid: bool,
) -> Dict[int, float]:
    """
    Return {frameIndex -> Z_metres} for every entry in ball_trajectory.
    Non-flight frames get 0. Empty dict if calibration is not valid.
    """
    if not calibration_valid:
        logger.info("[ball_physics] skipped — calibration_valid=False")
        return {}

    items = [
        b for b in ball_trajectory
        if b.get("frameIndex") is not None
        and b.get("x") is not None
        and b.get("y") is not None
    ]
    items = sorted(items, key=lambda b: int(b["frameIndex"]))
    if len(items) < 3 or fps <= 0:
        return {int(b["frameIndex"]): 0.0 for b in items}

    frame_indices = [int(b["frameIndex"]) for b in items]
    xs = [float(b["x"]) for b in items]
    ys = [float(b["y"]) for b in items]

    contacts = _detect_contacts(xs, ys, frame_indices, fps)
    n = len(items)
    boundaries = sorted(set([0] + contacts + [n - 1]))

    z_by_frame: Dict[int, float] = {int(fi): 0.0 for fi in frame_indices}
    flight_arcs = 0

    for a_idx, b_idx in zip(boundaries[:-1], boundaries[1:]):
        if b_idx - a_idx < 2:
            continue
        t_a = frame_indices[a_idx] / fps
        t_b = frame_indices[b_idx] / fps
        duration = t_b - t_a
        if duration < MIN_ARC_DURATION_SEC or duration > MAX_ARC_DURATION_SEC:
            continue

        mean_speed = _arc_mean_speed(xs, ys, frame_indices, fps, a_idx, b_idx)
        if mean_speed < MIN_FLIGHT_SPEED_MS:
            continue

        # Z(t) = (g/2)(t - t_a)(t_b - t). Endpoints stay at 0 (contact moments).
        for i in range(a_idx, b_idx + 1):
            if i == a_idx or i == b_idx:
                z_by_frame[frame_indices[i]] = 0.0
                continue
            t = frame_indices[i] / fps
            z = 0.5 * GRAVITY_MS2 * (t - t_a) * (t_b - t)
            z_by_frame[frame_indices[i]] = round(max(0.0, z), 3)
        flight_arcs += 1

    logger.info(
        "[ball_physics] %d ball samples, %d flight arcs identified", n, flight_arcs
    )
    return z_by_frame
