"""
Scene classifier for football video analysis.

Classifies each frame into one of:
  pitch_wide       — full/wide pitch view, valid for player tracking and detection
  pitch_close      — tight close-up on 1-3 players, valid for tracking only
  cutaway          — dugout, crowd, tunnel, referee close-up — skip entirely
  graphic_overlay  — scoreboard/replay graphic covering majority of frame — skip entirely

Uses OpenCV only. Target: <5ms per frame on CPU.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# HSV green range for football pitch grass
_GREEN_H_LO, _GREEN_H_HI = 25, 95
_GREEN_S_LO, _GREEN_S_HI = 20, 255
_GREEN_V_LO, _GREEN_V_HI = 20, 255

# Thresholds
_PITCH_GREEN_THRESHOLD = 0.40       # >40% green → pitch frame
_CLOSE_GREEN_THRESHOLD = 0.15       # 15-40% green → pitch_close
_GRAPHIC_GREEN_THRESHOLD = 0.05     # <5% green AND high white → graphic_overlay
_WHITE_LINE_THRESHOLD = 0.04        # >4% pitch-line white in lower 60% → supports pitch_wide


def _green_ratio(hsv: np.ndarray) -> float:
    """Return fraction of pixels that are pitch-green."""
    mask = cv2.inRange(
        hsv,
        np.array([_GREEN_H_LO, _GREEN_S_LO, _GREEN_V_LO], dtype=np.uint8),
        np.array([_GREEN_H_HI, _GREEN_S_HI, _GREEN_V_HI], dtype=np.uint8),
    )
    return float(np.count_nonzero(mask)) / mask.size


def _white_line_ratio(hsv: np.ndarray) -> float:
    """Return fraction of pixels in lower 60% that are bright white (pitch lines)."""
    h = hsv.shape[0]
    lower_region = hsv[int(h * 0.40):, :]
    # White: low saturation, high value
    mask = cv2.inRange(
        lower_region,
        np.array([0, 0, 200], dtype=np.uint8),
        np.array([180, 40, 255], dtype=np.uint8),
    )
    return float(np.count_nonzero(mask)) / mask.size


def _graphic_score(hsv: np.ndarray, frame_bgr: np.ndarray) -> float:
    """
    Estimate probability of a graphic overlay.
    Signals: very low green, high uniform-white region, high saturation in upper strip.
    Returns 0-1 where 1 = very likely graphic.
    """
    green = _green_ratio(hsv)
    if green > 0.15:
        return 0.0  # Too much green to be a graphic

    # Check for large flat-coloured band at top or bottom (scoreboard bar)
    h, w = frame_bgr.shape[:2]
    top_strip = frame_bgr[:int(h * 0.15), :]
    bot_strip = frame_bgr[int(h * 0.85):, :]

    def _strip_uniformity(strip):
        """Return 0-1 how uniform the strip colours are (high = likely graphic bar)."""
        if strip.size == 0:
            return 0.0
        std = float(np.std(strip.reshape(-1, 3), axis=0).mean())
        # Low std = very uniform colour = graphic-like
        return max(0.0, 1.0 - std / 60.0)

    top_score = _strip_uniformity(top_strip)
    bot_score = _strip_uniformity(bot_strip)
    strip_score = max(top_score, bot_score)

    # Overall white ratio
    white_mask = cv2.inRange(hsv, np.array([0, 0, 210], dtype=np.uint8),
                              np.array([180, 30, 255], dtype=np.uint8))
    white_ratio = float(np.count_nonzero(white_mask)) / white_mask.size

    graphic = (strip_score * 0.5) + (white_ratio * 2.0) + (1.0 - green) * 0.2
    return min(1.0, graphic)


class SceneClassifier:
    """
    Classifies video frames by scene type using HSV colour analysis.
    OpenCV only — no ML models required.
    """

    def classify_frame(self, frame: np.ndarray) -> tuple[str, float]:
        """
        Classify a BGR frame.

        Returns
        -------
        (scene_class, confidence) where scene_class is one of:
            'pitch_wide', 'pitch_close', 'cutaway', 'graphic_overlay'
        and confidence is 0-1.
        """
        # Downsample to 320×180 for speed — sufficient for colour stats
        small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        green = _green_ratio(hsv)
        white_lines = _white_line_ratio(hsv)

        # --- graphic_overlay ---
        if green < _GRAPHIC_GREEN_THRESHOLD:
            g_score = _graphic_score(hsv, small)
            if g_score > 0.5:
                conf = min(1.0, g_score)
                return "graphic_overlay", round(conf, 3)

        # --- cutaway (not enough green for any pitch view) ---
        if green < _CLOSE_GREEN_THRESHOLD:
            # Could still be a dark indoor cutaway or transition
            conf = 1.0 - (green / _CLOSE_GREEN_THRESHOLD)
            return "cutaway", round(min(1.0, conf), 3)

        # --- pitch frames ---
        if green >= _PITCH_GREEN_THRESHOLD:
            # Wide pitch: substantial green AND pitch lines visible
            if white_lines >= _WHITE_LINE_THRESHOLD:
                # Strong pitch_wide signal
                conf = min(1.0, (green - _PITCH_GREEN_THRESHOLD) / 0.3 * 0.5
                           + (white_lines / 0.15) * 0.5)
                return "pitch_wide", round(min(1.0, conf + 0.5), 3)
            else:
                # Lots of green but no clear lines — still wide-ish, lower confidence
                conf = min(1.0, (green - _PITCH_GREEN_THRESHOLD) / 0.4 + 0.4)
                return "pitch_wide", round(conf, 3)

        # --- pitch_close vs cutaway (15-40% green) ---
        # If white_lines are essentially absent, this is likely a cutaway with
        # incidental green (advertising boards, jackets, seats). Pitch close-ups
        # always retain some line/texture visibility.
        if white_lines < 0.005:
            # No pitch lines at all → cutaway with incidental green
            conf = 1.0 - (white_lines / 0.005)
            return "cutaway", round(min(1.0, conf * 0.7 + 0.3), 3)

        # Enough grass visible to be on the pitch — tight framing
        conf = (green - _CLOSE_GREEN_THRESHOLD) / (_PITCH_GREEN_THRESHOLD - _CLOSE_GREEN_THRESHOLD)
        return "pitch_close", round(min(1.0, conf * 0.8 + 0.2), 3)
