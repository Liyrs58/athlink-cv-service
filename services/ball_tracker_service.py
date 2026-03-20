"""
Color-based ball tracking service.
Detects and tracks the ball using HSV color thresholding.
"""

import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class BallTracker:
    """Track ball across frames using color detection."""

    def __init__(self, h_range: Tuple[int, int] = (0, 255),
                 s_range: Tuple[int, int] = (0, 255),
                 v_range: Tuple[int, int] = (150, 255),
                 min_ball_area: int = 20,
                 max_ball_area: int = 10000):
        """
        Args:
            h_range: Hue range for ball color (0-180 in OpenCV)
            s_range: Saturation range
            v_range: Value/brightness range
            min_ball_area: Minimum contour area to consider as ball
            max_ball_area: Maximum contour area
        """
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range
        self.min_ball_area = min_ball_area
        self.max_ball_area = max_ball_area

        self.ball_track = []  # List of (frame_id, center_x, center_y, bbox, confidence)
        self.last_ball_center = None
        self.last_frame_id = -1
        self.max_ball_distance = 150  # Max pixels ball can move between frames

    def _detect_ball(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detect ball using HSV color thresholding.

        Args:
            frame: Input frame (H, W, 3) BGR

        Returns:
            Dict with bbox, center, area, confidence or None
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask for ball color (white/yellow typically)
        lower = np.array([self.h_range[0], self.s_range[0], self.v_range[0]])
        upper = np.array([self.h_range[1], self.s_range[1], self.v_range[1]])
        mask = cv2.inRange(hsv, lower, upper)

        # Morphology to clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find largest contour in valid size range
        valid_contours = [c for c in contours
                         if self.min_ball_area <= cv2.contourArea(c) <= self.max_ball_area]

        if not valid_contours:
            return None

        # Get largest valid contour (most likely the ball)
        largest = max(valid_contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        # Get bounding rect
        x, y, w, h = cv2.boundingRect(largest)
        bbox = {"x1": x, "y1": y, "x2": x + w, "y2": y + h}

        # Center of mass (more accurate than bbox center)
        m = cv2.moments(largest)
        if m["m00"] > 0:
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2

        # Confidence based on contour circularity
        perimeter = cv2.arcLength(largest, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            confidence = min(circularity, 1.0)  # More circular = higher confidence
        else:
            confidence = 0.5

        return {
            "mask": mask,
            "bbox": bbox,
            "center": (cx, cy),
            "confidence": float(confidence),
            "area": int(area)
        }

    def track_frame(self, frame: np.ndarray, frame_id: int) -> Optional[Dict]:
        """
        Track ball in frame.

        Args:
            frame: Input frame
            frame_id: Frame number

        Returns:
            Tracking result or None if ball not found
        """
        # Detect ball
        result = self._detect_ball(frame)

        if result is not None:
            center = result["center"]

            # Validate based on motion (check if movement is reasonable)
            if self.last_ball_center is not None:
                dist = np.sqrt((center[0] - self.last_ball_center[0]) ** 2 +
                              (center[1] - self.last_ball_center[1]) ** 2)

                # Reject if ball moved too far (likely false positive)
                if dist > self.max_ball_distance:
                    logger.debug(f"Frame {frame_id}: Ball moved {dist:.0f}px, rejecting (likely false positive)")
                    return None

            self.last_ball_center = center
            self.last_frame_id = frame_id
            self.ball_track.append((frame_id, center[0], center[1],
                                   result["bbox"], result["confidence"]))

            logger.debug(f"Frame {frame_id}: Ball at {center} conf={result['confidence']:.2f}")
            return result

        return None

    def set_color_range(self, h_range: Tuple[int, int] = None,
                       s_range: Tuple[int, int] = None,
                       v_range: Tuple[int, int] = None):
        """Update HSV color range (for fine-tuning detection)."""
        if h_range:
            self.h_range = h_range
        if s_range:
            self.s_range = s_range
        if v_range:
            self.v_range = v_range

    def get_track(self) -> List[Tuple]:
        """Get full ball track history."""
        return self.ball_track.copy()

    def reset(self):
        """Reset tracking state."""
        self.ball_track = []
        self.last_ball_center = None
        self.last_frame_id = -1


# Preset HSV ranges for different ball colors
BALL_COLOR_PRESETS = {
    "white": {"h": (0, 180), "s": (0, 30), "v": (200, 255)},    # White ball
    "yellow": {"h": (15, 35), "s": (50, 255), "v": (150, 255)},  # Yellow ball
    "orange": {"h": (5, 20), "s": (100, 255), "v": (150, 255)},  # Orange ball
    "bright": {"h": (0, 180), "s": (0, 100), "v": (180, 255)},   # Any bright object
}
