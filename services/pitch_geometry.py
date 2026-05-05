"""
Pitch geometry utilities — image-space to pitch-space conversion
and motion gate for identity assignment.

No homography required. Falls back to normalized image coordinates.
When a homography matrix H is provided (3x3 perspective transform from
image pixels → pitch metres), uses that for accurate pitch positions.

Pitch dimensions: 105m x 68m (FIFA standard).
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


# Max realistic player speed for gate calculation
_MAX_SPEED_M_PER_FRAME = 1.2   # ~30 km/h at 25 fps ≈ 0.33 m/frame; 1.2 = generous 3x headroom
_MAX_SPEED_IMG_NORM = 0.08     # max normalised image displacement per frame (image-space fallback)


def image_to_pitch(
    px: float,
    py: float,
    frame_w: int,
    frame_h: int,
    H: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Convert image pixel (px, py) → normalised pitch coordinates (x, y) ∈ [0,1]².

    If H is provided (3×3 homography, image→pitch metres), use perspective transform.
    Otherwise fall back to proportional image coordinates.
    """
    if H is not None:
        pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
        mapped = np.array(pt)
        try:
            import cv2
            mapped = cv2.perspectiveTransform(pt, H)
            mx, my = float(mapped[0, 0, 0]), float(mapped[0, 0, 1])
            # Clamp to pitch bounds and normalise
            return (float(np.clip(mx / 105.0, 0.0, 1.0)),
                    float(np.clip(my / 68.0, 0.0, 1.0)))
        except Exception:
            pass
    # Proportional fallback
    return (
        float(np.clip(px / max(frame_w, 1), 0.0, 1.0)),
        float(np.clip(py / max(frame_h, 1), 0.0, 1.0)),
    )


def pitch_distance(
    a: Tuple[float, float],
    b: Tuple[float, float],
    using_homography: bool = False,
) -> float:
    """
    Euclidean distance between two pitch positions.

    With homography: coords are in metres → distance in metres.
    Without: coords are normalised [0,1] → distance in normalised units.
    """
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    if using_homography:
        # Scale back: normalised → metres
        dx *= 105.0
        dy *= 68.0
    return float((dx * dx + dy * dy) ** 0.5)


def max_speed_gate(
    last_pitch: Optional[Tuple[float, float]],
    curr_pitch: Tuple[float, float],
    frame_delta: int,
    using_homography: bool = False,
) -> bool:
    """
    Return True if the displacement from last_pitch to curr_pitch is
    physically plausible (within max player speed × frame_delta).

    If last_pitch is None (first appearance), always passes.
    """
    if last_pitch is None or frame_delta <= 0:
        return True

    dist = pitch_distance(last_pitch, curr_pitch, using_homography)

    if using_homography:
        max_allowed = _MAX_SPEED_M_PER_FRAME * frame_delta
    else:
        max_allowed = _MAX_SPEED_IMG_NORM * frame_delta

    return dist <= max_allowed


def assignment_position_cost(
    last_pitch: Optional[Tuple[float, float]],
    curr_pitch: Tuple[float, float],
    using_homography: bool = False,
) -> float:
    """
    Normalised position cost [0,1] for use in assignment cost matrix.
    Returns 1.0 if motion gate fails (impossible displacement).
    """
    if last_pitch is None:
        return 0.5   # neutral — no prior

    if using_homography:
        max_expected = 105.0 * 0.3   # ~31m = 30% of pitch width
        dist = pitch_distance(last_pitch, curr_pitch, using_homography=True)
    else:
        max_expected = 0.4           # 40% of normalised pitch
        dist = pitch_distance(last_pitch, curr_pitch, using_homography=False)

    return float(min(dist / max_expected, 1.0))
