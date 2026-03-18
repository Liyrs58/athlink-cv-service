"""
Pitch homography estimation service.

Estimates the visible fraction of the pitch from broadcast camera footage
and provides accurate pixel-to-metre conversion.
"""
import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

def to_scalar(v):
    """Convert numpy scalars to Python native types for JSON serialization."""
    if hasattr(v, 'item'):
        return v.item()
    if hasattr(v, '__float__'):
        return float(v)
    return v

PITCH_LENGTH = 105.0  # metres
PITCH_WIDTH = 68.0    # metres


def detect_pitch_lines(frame: np.ndarray) -> list:
    """
    Detect white pitch line markings using HSV thresholding + Hough transform.
    Returns list of (rho, theta) line parameters.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # White lines: high value, low saturation
    white_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 60, 255]))

    # Only keep white pixels on green areas (pitch lines, not advertising boards)
    green_mask = cv2.inRange(hsv, np.array([30, 30, 30]), np.array([90, 255, 255]))
    # Dilate green to include lines adjacent to grass
    green_dilated = cv2.dilate(green_mask, np.ones((25, 25), np.uint8))
    line_mask = cv2.bitwise_and(white_mask, green_dilated)

    # Clean up
    kernel = np.ones((3, 3), np.uint8)
    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel)

    # Hough line detection
    lines = cv2.HoughLines(line_mask, 1, np.pi / 180, threshold=120)
    if lines is None:
        return []

    return [(float(to_scalar(l[0][0])), float(to_scalar(l[0][1]))) for l in lines]


def _classify_lines(lines: list) -> Tuple[list, list]:
    """Split lines into horizontal (near 0 or pi) and vertical (near pi/2)."""
    horizontal = []
    vertical = []
    for rho, theta in lines:
        angle_deg = np.degrees(theta)
        if angle_deg < 30 or angle_deg > 150:
            horizontal.append((rho, theta))
        elif 60 < angle_deg < 120:
            vertical.append((rho, theta))
    return horizontal, vertical


def _cluster_lines(lines: list, threshold: float = 50.0) -> list:
    """Cluster nearby lines by rho, return cluster representatives."""
    if not lines:
        return []
    sorted_lines = sorted(lines, key=lambda l: abs(l[0]))
    clusters = [[sorted_lines[0]]]
    for line in sorted_lines[1:]:
        if abs(abs(line[0]) - abs(clusters[-1][0][0])) < threshold:
            clusters[-1].append(line)
        else:
            clusters.append([line])
    # Return median of each cluster
    result = []
    for cluster in clusters:
        rhos = [l[0] for l in cluster]
        thetas = [l[1] for l in cluster]
        result.append((float(to_scalar(np.median(rhos))), float(to_scalar(np.median(thetas)))))
    return result


def detect_pitch_keypoints(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect visible pitch boundary corners from line intersections.
    Returns (4, 2) float32 array of corners in TL, TR, BR, BL order, or None.
    """
    lines = detect_pitch_lines(frame)
    if len(lines) < 4:
        return None

    horizontal, vertical = _classify_lines(lines)
    h_clustered = _cluster_lines(horizontal, threshold=40)
    v_clustered = _cluster_lines(vertical, threshold=40)

    if len(h_clustered) < 2 or len(v_clustered) < 2:
        return None

    # Take outermost horizontal and vertical lines
    h_sorted = sorted(h_clustered, key=lambda l: abs(l[0]))
    v_sorted = sorted(v_clustered, key=lambda l: abs(l[0]))

    h_top = h_sorted[0]
    h_bottom = h_sorted[-1]
    v_left = v_sorted[0]
    v_right = v_sorted[-1]

    def line_intersection(l1, l2):
        rho1, theta1 = l1
        rho2, theta2 = l2
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([rho1, rho2])
        det = np.linalg.det(A)
        if abs(det) < 1e-6:
            return None
        x, y = np.linalg.solve(A, b)
        return (float(to_scalar(x)), float(to_scalar(y)))

    corners = []
    for h_line, v_line in [(h_top, v_left), (h_top, v_right),
                           (h_bottom, v_right), (h_bottom, v_left)]:
        pt = line_intersection(h_line, v_line)
        if pt is None:
            return None
        corners.append(pt)

    corners = np.array(corners, dtype=np.float32)

    # Validate: corners should span a reasonable portion of the frame
    h, w = frame.shape[:2]
    xs = corners[:, 0]
    ys = corners[:, 1]
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)

    if span_x < w * 0.2 or span_y < h * 0.2:
        return None
    if span_x > w * 1.5 or span_y > h * 1.5:
        return None

    return corners


def estimate_visible_fraction(frame: np.ndarray) -> float:
    """
    Estimate what fraction of the pitch width is visible in the frame
    based on the ratio of green pixels to total frame area.

    Broadcast cameras typically show 40-70% of the pitch width.
    A frame that is ~45% green likely shows about 55% of the pitch
    (accounting for players, stands visible at edges).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, np.array([30, 30, 30]), np.array([90, 255, 255]))
    green_pct = np.count_nonzero(green_mask) / green_mask.size

    # Map green percentage to visible pitch fraction
    # Broadcast cameras typically show 40-65% of pitch width.
    # Green percentage is high because stands/sky are minimal in tight shots.
    # Key insight: even when ~60% of frame is green, the camera only sees
    # about half the pitch because of perspective compression.
    if green_pct < 0.10:
        return 0.25
    elif green_pct < 0.30:
        return 0.30 + (green_pct - 0.10) * 0.5   # 0.30 - 0.40
    elif green_pct < 0.50:
        return 0.40 + (green_pct - 0.30) * 0.5   # 0.40 - 0.50
    elif green_pct < 0.65:
        return 0.50 + (green_pct - 0.50) * 0.33  # 0.50 - 0.55
    else:
        return min(0.60, 0.55 + (green_pct - 0.65) * 0.25)


def estimate_homography(frame: np.ndarray) -> Dict[str, Any]:
    """
    Estimate pitch calibration from a single frame.

    Returns dict with:
    - 'method': 'keypoints' or 'green_fraction'
    - 'homography': 3x3 matrix (if keypoints method)
    - 'visible_fraction': float (always present)
    - 'pixels_per_metre': float (always present)
    """
    h, w = frame.shape[:2]

    # Try keypoint-based homography first
    corners = detect_pitch_keypoints(frame)
    if corners is not None:
        # Map detected corners to pitch coordinates
        # Assume corners bound the visible pitch area
        dst = np.array([
            [0, 0],
            [PITCH_LENGTH, 0],
            [PITCH_LENGTH, PITCH_WIDTH],
            [0, PITCH_WIDTH],
        ], dtype=np.float32)

        H, _ = cv2.findHomography(corners, dst, cv2.RANSAC)
        if H is not None:
            # Validate by checking centre point
            centre = np.array([[[w / 2, h / 2]]], dtype=np.float32)
            world_pt = cv2.perspectiveTransform(centre, H)
            wx, wy = float(to_scalar(world_pt[0][0][0])), float(to_scalar(world_pt[0][0][1]))
            if 10 < wx < 95 and 10 < wy < 58:
                # Estimate visible fraction from corner spread
                span_x = max(corners[:, 0]) - min(corners[:, 0])
                visible_frac = min(0.85, max(0.30, span_x / w))
                ppm = w * visible_frac / PITCH_LENGTH
                return {
                    'method': 'keypoints',
                    'homography': H.tolist(),
                    'visible_fraction': round(to_scalar(visible_frac), 3),
                    'pixels_per_metre': round(to_scalar(ppm), 2),
                }

    # Fallback: estimate from green pixel fraction
    visible_frac = estimate_visible_fraction(frame)
    # pixels_per_metre: how many pixels correspond to 1 metre of pitch
    # If we see `visible_frac` of the 105m pitch in `w` pixels:
    ppm = w / (PITCH_LENGTH * visible_frac)
    # But this is for the length axis. For width:
    # visible_frac of 68m pitch in h pixels → ppm_y = h / (68 * visible_frac)
    # Use the average for isotropic scaling
    ppm_x = w / (PITCH_LENGTH * visible_frac)
    ppm_y = h / (PITCH_WIDTH * visible_frac)
    ppm = (ppm_x + ppm_y) / 2.0

    return {
        'method': 'green_fraction',
        'homography': None,
        'visible_fraction': round(to_scalar(visible_frac), 3),
        'pixels_per_metre': round(to_scalar(ppm), 2),
    }


def get_frame_calibration(video_path: str, sample_frames: int = 5) -> Dict[str, Any]:
    """
    Sample frames from the video, estimate calibration on each,
    return the median calibration (robust to outliers).

    Result is cached per video_path within the returned dict.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Cannot open video for calibration: {video_path}")
        return {
            'method': 'default',
            'visible_fraction': 0.55,
            'pixels_per_metre': 15.5,
            'homography': None,
            'frames_sampled': 0,
        }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    needs_rotation = raw_h > raw_w

    if total_frames < 1:
        cap.release()
        return {
            'method': 'default',
            'visible_fraction': 0.55,
            'pixels_per_metre': 15.5,
            'homography': None,
            'frames_sampled': 0,
        }

    # Sample evenly spaced frames (skip first and last 10%)
    start = int(total_frames * 0.1)
    end = int(total_frames * 0.9)
    if end <= start:
        start, end = 0, total_frames
    step = max(1, (end - start) // sample_frames)
    sample_indices = list(range(start, end, step))[:sample_frames]

    fractions = []
    ppms = []
    best_homography = None

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        if needs_rotation:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        cal = estimate_homography(frame)
        fractions.append(cal['visible_fraction'])
        ppms.append(cal['pixels_per_metre'])
        if cal['method'] == 'keypoints' and cal['homography'] is not None:
            best_homography = cal['homography']

    cap.release()

    if not fractions:
        return {
            'method': 'default',
            'visible_fraction': 0.55,
            'pixels_per_metre': 15.5,
            'homography': None,
            'frames_sampled': 0,
        }

    median_frac = float(to_scalar(np.median(fractions)))
    median_ppm = float(to_scalar(np.median(ppms)))

    method = 'keypoints' if best_homography is not None else 'green_fraction'

    logger.info(
        f"Calibration: visible_fraction={median_frac:.3f}, "
        f"pixels_per_metre={median_ppm:.2f}, method={method}, "
        f"frames_sampled={len(fractions)}"
    )

    return {
        'method': method,
        'visible_fraction': round(median_frac, 3),
        'pixels_per_metre': round(median_ppm, 2),
        'homography': best_homography,
        'frames_sampled': len(fractions),
    }


def pixels_to_metres(px_x: float, px_y: float, calibration: Dict[str, Any],
                     frame_w: int = 1920, frame_h: int = 1080) -> Tuple[float, float]:
    """
    Convert pixel coordinates to real-world metres using calibration.

    If calibration has a homography matrix, use perspective transform.
    Otherwise, use visible_fraction-adjusted scaling.
    """
    H = calibration.get('homography')
    if H is not None and not isinstance(H, list):
        # H is a numpy array — use perspective transform
        pt = np.array([[[px_x, px_y]]], dtype=np.float32)
        world = cv2.perspectiveTransform(pt, np.array(H, dtype=np.float32))
        return float(to_scalar(world[0][0][0])), float(to_scalar(world[0][0][1]))

    if H is not None and isinstance(H, list):
        H_arr = np.array(H, dtype=np.float32)
        pt = np.array([[[px_x, px_y]]], dtype=np.float32)
        world = cv2.perspectiveTransform(pt, H_arr)
        return float(to_scalar(world[0][0][0])), float(to_scalar(world[0][0][1]))

    # Fallback: use visible_fraction scaling
    visible_frac = calibration.get('visible_fraction', 0.55)
    # The camera shows visible_frac of the pitch
    # So frame_w pixels = visible_frac * PITCH_LENGTH metres
    world_x = (px_x / frame_w) * (PITCH_LENGTH * visible_frac)
    world_y = (px_y / frame_h) * (PITCH_WIDTH * visible_frac)
    return world_x, world_y
