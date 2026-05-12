"""
Camera motion compensation for Kalman tracking.

During camera pans, all players move together in the frame. The Kalman filter
predicts each player's motion independently, causing divergence when the global
scene motion dominates. This module detects global camera motion via optical flow
and compensates Kalman predictions.

Flow: frame → compute optical flow on sparse grid → median motion vector →
      supply to tracker for prediction adjustment.
"""

import numpy as np
import cv2


class CameraMotionEstimator:
    """Estimate global camera motion from optical flow on sparse grid."""

    def __init__(self, grid_size=8):
        """
        grid_size: divide frame into grid_size x grid_size cells, compute flow in each
        """
        self.grid_size = grid_size
        self.prev_gray = None
        self.motion_history = []  # ring buffer of (dx, dy) vectors
        self.max_history = 5

    def estimate(self, frame):
        """
        Compute global camera motion vector (median of flows in grid cells).
        Returns (dx, dy) in pixels, or (0, 0) if flow is too weak.
        """
        if frame is None:
            return (0.0, 0.0)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        if self.prev_gray is None:
            self.prev_gray = gray
            return (0.0, 0.0)

        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, n8=True, poly_n=5, poly_sigma=1.2, flags=0
        )

        # Sample flows at grid centers
        cell_h = h // self.grid_size
        cell_w = w // self.grid_size
        flows = []

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                y = (i + 0.5) * cell_h
                x = (j + 0.5) * cell_w
                y_int, x_int = int(y), int(x)

                # Clip to valid range
                y_int = np.clip(y_int, 0, h - 1)
                x_int = np.clip(x_int, 0, w - 1)

                fx, fy = flow[y_int, x_int]
                magnitude = np.sqrt(fx*fx + fy*fy)

                # Only include non-zero flows (ignore static regions)
                if magnitude > 0.5:
                    flows.append((fx, fy))

        # Estimate global motion as median of flows
        if not flows:
            motion = (0.0, 0.0)
        else:
            flows = np.array(flows)
            motion_x = np.median(flows[:, 0])
            motion_y = np.median(flows[:, 1])
            motion = (float(motion_x), float(motion_y))

        # Reject outliers: if magnitude > 50 pixels/frame, likely tracking failure
        mag = np.sqrt(motion[0]**2 + motion[1]**2)
        if mag > 50:
            motion = (0.0, 0.0)

        # Keep history for smoothing
        self.motion_history.append(motion)
        if len(self.motion_history) > self.max_history:
            self.motion_history.pop(0)

        # Smooth: median of history
        if self.motion_history:
            hist = np.array(self.motion_history)
            smoothed_x = np.median(hist[:, 0])
            smoothed_y = np.median(hist[:, 1])
            motion = (float(smoothed_x), float(smoothed_y))

        self.prev_gray = gray.copy()
        return motion

    def reset(self):
        """Reset on scene cuts or major changes."""
        self.prev_gray = None
        self.motion_history = []


class KalmanBoxWithMotionCompensation:
    """
    Kalman filter that subtracts camera motion from predictions.

    Instead of predicting the player moves in their direction,
    we predict (player_velocity - camera_motion).
    """

    def __init__(self, bbox_xyxy: np.ndarray):
        """Initialize from detection bbox [x1, y1, x2, y2]."""
        cx = (bbox_xyxy[0] + bbox_xyxy[2]) / 2
        cy = (bbox_xyxy[1] + bbox_xyxy[3]) / 2
        w  = bbox_xyxy[2] - bbox_xyxy[0]
        h  = bbox_xyxy[3] - bbox_xyxy[1]
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=float)
        self.P = np.diag([w*w, h*h, w*w, h*h, 100., 100., 10., 10.])

        # Motion model: x[t+1] = F * x[t]
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i+4] = 1.0

        # Observation model: z = H * x
        self.H = np.zeros((4, 8))
        for i in range(4):
            self.H[i, i] = 1.0

        self.Q = np.diag([1., 1., 1., 1., 1., 1., 0.1, 0.1])
        self.R = np.diag([1., 1., 10., 10.])
        self._I = np.eye(8)

        # Camera motion compensation
        self.last_camera_motion = (0.0, 0.0)

    def predict(self, camera_motion=(0.0, 0.0)):
        """
        Predict next state, compensating for camera motion.

        Args:
            camera_motion: (dx, dy) global scene motion in pixels
        """
        self.last_camera_motion = camera_motion

        # Standard Kalman prediction
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Compensate: subtract camera motion from position prediction
        # Player's predicted position -= camera motion
        # This makes them "stable" relative to the field
        self.x[0] -= camera_motion[0]  # cx
        self.x[1] -= camera_motion[1]  # cy

        return self._to_xyxy()

    def update(self, bbox_xyxy: np.ndarray):
        """Update with observation."""
        cx = (bbox_xyxy[0] + bbox_xyxy[2]) / 2
        cy = (bbox_xyxy[1] + bbox_xyxy[3]) / 2
        w  = bbox_xyxy[2] - bbox_xyxy[0]
        h  = bbox_xyxy[3] - bbox_xyxy[1]
        z = np.array([cx, cy, w, h])

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (self._I - K @ self.H) @ self.P

        return self._to_xyxy()

    def _to_xyxy(self) -> np.ndarray:
        cx, cy, w, h = self.x[:4]
        w = max(w, 1.0)
        h = max(h, 1.0)
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

    @property
    def mean_xyxy(self) -> np.ndarray:
        return self._to_xyxy()
