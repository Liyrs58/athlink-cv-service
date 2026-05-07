"""Project world-coord points (metres on 105x68 pitch) back to image space.

Used by render_performance_zone.py to draw foot rings, arrows, and tactical
markers anchored to pitch coordinates rather than raw bbox pixels.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def invert_homography(H: np.ndarray) -> Optional[np.ndarray]:
    """Return H_inv (world -> image) given H (image -> world). None if degenerate."""
    if H is None:
        return None
    arr = np.asarray(H, dtype=np.float64)
    if arr.shape != (3, 3):
        return None
    if not np.isfinite(arr).all():
        return None
    try:
        ok, inv = cv2.invert(arr)
        if not ok or not np.isfinite(inv).all():
            return None
        return inv
    except cv2.error:
        return None


def project_world_point(world_xy: Tuple[float, float], H_inv: np.ndarray) -> Optional[Tuple[int, int]]:
    """Project a single world point (metres) to image pixel (px, py).
    Returns None if H_inv is missing or projection lands at infinity."""
    if H_inv is None:
        return None
    pt = np.array([[[float(world_xy[0]), float(world_xy[1])]]], dtype=np.float32)
    try:
        out = cv2.perspectiveTransform(pt, H_inv)
    except cv2.error:
        return None
    if out is None or not np.isfinite(out).all():
        return None
    x, y = out[0][0]
    return int(round(x)), int(round(y))


def project_world_radius(
    world_radius: float,
    world_xy: Tuple[float, float],
    H_inv: np.ndarray,
) -> Optional[int]:
    """Approximate the image-space radius of a circle of `world_radius` metres
    centred at `world_xy`. Computed as the average pixel distance from the
    centre to four cardinal points on the world-space circle."""
    if H_inv is None:
        return None
    cx, cy = float(world_xy[0]), float(world_xy[1])
    centre = project_world_point((cx, cy), H_inv)
    if centre is None:
        return None
    samples = [
        (cx + world_radius, cy),
        (cx - world_radius, cy),
        (cx, cy + world_radius),
        (cx, cy - world_radius),
    ]
    dists = []
    for s in samples:
        p = project_world_point(s, H_inv)
        if p is None:
            continue
        dx, dy = p[0] - centre[0], p[1] - centre[1]
        dists.append(float(np.hypot(dx, dy)))
    if not dists:
        return None
    r = int(round(sum(dists) / len(dists)))
    return max(2, r)
