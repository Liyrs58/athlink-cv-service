"""
Camera Motion Detection Service.

Detects global camera motion between processed frames using:
1. ORB feature matching + RANSAC affine transform
2. ECC (Enhanced Correlation Coefficient) fallback
3. Motion classification: stable | pan | fast_pan | cut | unknown
"""

import numpy as np
import cv2
from typing import Optional, Dict, Tuple


class CameraMotionDetector:
    """Estimate global camera motion between frames."""

    def __init__(self):
        self.prev_gray = None
        self.prev_frame_idx = None
        self.orb = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def estimate(self, frame: np.ndarray, frame_idx: int) -> Dict:
        """
        Estimate camera motion from current frame.

        Args:
            frame: BGR frame
            frame_idx: frame index in video

        Returns:
            {
              "frameIndex": int,
              "prevFrameIndex": Optional[int],
              "dx": float,
              "dy": float,
              "motion_px": float,
              "affine": Optional[list[list[float]]],
              "confidence": float,
              "num_matches": int,
              "num_inliers": int,
              "motion_class": "stable|pan|fast_pan|cut|unknown"
            }
        """
        result = {
            "frameIndex": frame_idx,
            "prevFrameIndex": self.prev_frame_idx,
            "dx": 0.0,
            "dy": 0.0,
            "motion_px": 0.0,
            "affine": None,
            "confidence": 0.0,
            "num_matches": 0,
            "num_inliers": 0,
            "motion_class": "unknown",
        }

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # First frame
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_frame_idx = frame_idx
            result["motion_class"] = "stable"
            return result

        # Phase 1: ORB + RANSAC
        motion = self._detect_orb_ransac(self.prev_gray, gray)

        if motion["confidence"] >= 0.4 and motion["num_inliers"] >= 20:
            # ORB succeeded
            result.update(motion)
        else:
            # Phase 2: ECC fallback
            motion = self._detect_ecc(self.prev_gray, gray)
            result.update(motion)

        # Classify motion
        motion_px = result["motion_px"]
        confidence = result["confidence"]
        inliers = result["num_inliers"]

        if motion_px >= 180 or confidence < 0.25 or inliers < 10:
            result["motion_class"] = "cut"
        elif motion_px >= 80:
            result["motion_class"] = "fast_pan"
        elif motion_px >= 20:
            result["motion_class"] = "pan"
        else:
            result["motion_class"] = "stable"

        self.prev_gray = gray
        self.prev_frame_idx = frame_idx

        return result

    def _detect_orb_ransac(self, prev_gray: np.ndarray, gray: np.ndarray) -> Dict:
        """ORB feature matching with RANSAC affine."""
        result = {
            "dx": 0.0,
            "dy": 0.0,
            "motion_px": 0.0,
            "affine": None,
            "confidence": 0.0,
            "num_matches": 0,
            "num_inliers": 0,
        }

        # Detect keypoints
        kp1, des1 = self.orb.detectAndCompute(prev_gray, None)
        kp2, des2 = self.orb.detectAndCompute(gray, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return result

        # Match descriptors
        matches = self.matcher.knnMatch(des1, des2, k=2)

        # Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

        result["num_matches"] = len(good_matches)

        if len(good_matches) < 20:
            return result

        # Extract matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        # RANSAC affine
        M, mask = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC,
                                               ransacReprojThreshold=5.0,
                                               maxIters=1000, confidence=0.95)

        if M is None:
            return result

        # Extract translation
        dx, dy = M[0, 2], M[1, 2]
        motion_px = np.sqrt(dx**2 + dy**2)

        # Inlier ratio
        inliers = np.sum(mask)
        conf = min(inliers / max(len(good_matches), 1), 1.0)

        result["dx"] = float(dx)
        result["dy"] = float(dy)
        result["motion_px"] = float(motion_px)
        result["affine"] = M.tolist()
        result["confidence"] = float(conf)
        result["num_inliers"] = int(inliers)

        return result

    def _detect_ecc(self, prev_gray: np.ndarray, gray: np.ndarray) -> Dict:
        """ECC (Enhanced Correlation Coefficient) fallback."""
        result = {
            "dx": 0.0,
            "dy": 0.0,
            "motion_px": 0.0,
            "affine": None,
            "confidence": 0.0,
            "num_matches": 0,
            "num_inliers": 0,
        }

        # ECC motion model (translation)
        motion = np.eye(2, 3, dtype=np.float32)

        try:
            cc, M = cv2.findTransformECC(
                prev_gray, gray, motion,
                cv2.MOTION_TRANSLATION,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.001),
                inputMask=None, gaussFiltSize=5
            )

            dx, dy = M[0, 2], M[1, 2]
            motion_px = np.sqrt(dx**2 + dy**2)

            # Confidence from correlation
            conf = max(0.0, min(cc, 1.0))

            result["dx"] = float(dx)
            result["dy"] = float(dy)
            result["motion_px"] = float(motion_px)
            result["affine"] = M.tolist()
            result["confidence"] = float(conf)
            result["num_inliers"] = 1  # ECC is a single estimate

        except Exception:
            pass

        return result


def compensate_point(point: Tuple[float, float], motion: Dict) -> Tuple[float, float]:
    """Apply camera motion compensation to a point."""
    if motion["affine"] is not None:
        M = np.array(motion["affine"], dtype=np.float32)
        pt = np.array([[point[0], point[1]]], dtype=np.float32).reshape(-1, 1, 2)
        pt_warped = cv2.transform(pt, M)
        return tuple(pt_warped[0, 0])
    else:
        return (point[0] + motion["dx"], point[1] + motion["dy"])


def compensate_bbox(bbox: Tuple[float, float, float, float], motion: Dict) -> Tuple[float, float, float, float]:
    """Apply camera motion compensation to a bounding box."""
    x1, y1, x2, y2 = bbox

    if motion["affine"] is not None:
        M = np.array(motion["affine"], dtype=np.float32)
        # Transform corners
        corners = np.array([[x1, y1], [x2, y2]], dtype=np.float32).reshape(-1, 1, 2)
        corners_warped = cv2.transform(corners, M)
        x1_w, y1_w = corners_warped[0, 0]
        x2_w, y2_w = corners_warped[1, 0]
        return (x1_w, y1_w, x2_w, y2_w)
    else:
        dx, dy = motion["dx"], motion["dy"]
        return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def log_camera_motion(frame_idx: int, motion: Dict) -> str:
    """Format camera motion log line."""
    return (
        f"[CameraMotion] frame={motion['frameIndex']} "
        f"prev={motion['prevFrameIndex']} "
        f"dx={motion['dx']:.1f} dy={motion['dy']:.1f} "
        f"motion={motion['motion_px']:.1f} "
        f"class={motion['motion_class']} "
        f"conf={motion['confidence']:.2f} "
        f"matches={motion['num_matches']} "
        f"inliers={motion['num_inliers']}"
    )
