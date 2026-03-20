"""
Camera movement estimation and position compensation.
Adapted directly from Abdullah Tarek's CameraMovementEstimator (MIT).

Uses Lucas-Kanade optical flow on edge strips to estimate
global camera pan/tilt per frame.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def _measure_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def _measure_xy_distance(p1, p2):
    return p1[0] - p2[0], p1[1] - p2[1]


class CameraCompensator:
    """
    Estimates camera movement per frame using Lucas-Kanade optical flow.
    Adapted from camera_movement_estimator.py — same params, same logic.
    """

    def __init__(self, first_frame: np.ndarray):
        self.minimum_distance = 5

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10,
                0.03,
            ),
        )

        first_frame_grayscale = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        # Use left and right edge strips for feature detection
        # (same as Abdullah's approach — avoids players in center)
        h, w = first_frame_grayscale.shape
        mask_features[:, 0:20] = 1
        mask_features[:, max(0, w - 150):w] = 1

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features,
        )

    def get_camera_movement(self, frames: list) -> list:
        """
        Compute camera movement [dx, dy] per frame.
        Adapted from get_camera_movement() — stub logic removed.

        Returns list of [dx, dy] per frame.
        """
        camera_movement = [[0, 0]] * len(frames)

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        if old_features is None:
            logger.warning("No features found in first frame for camera estimation")
            return camera_movement

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, st, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features, None, **self.lk_params
            )

            if new_features is None or old_features is None:
                old_gray = frame_gray.copy()
                continue

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                distance = _measure_distance(new_features_point, old_features_point)
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = _measure_xy_distance(
                        old_features_point, new_features_point
                    )

            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            old_gray = frame_gray.copy()

        return camera_movement

    def adjust_positions(self, tracks: dict, camera_movement: list) -> dict:
        """
        Apply camera compensation to all track positions.
        Adapted from add_adjust_positions_to_tracks().

        Subtracts camera movement from player/ball positions to get
        camera-compensated coordinates.
        """
        for object_key in tracks:
            object_tracks = tracks[object_key]
            for frame_num, track in enumerate(object_tracks):
                if frame_num >= len(camera_movement):
                    break
                for track_id, track_info in track.items():
                    bbox = track_info["bbox"]
                    # Compute foot position for players, center for ball
                    if object_key == "ball":
                        position = (
                            int((bbox[0] + bbox[2]) / 2),
                            int((bbox[1] + bbox[3]) / 2),
                        )
                    else:
                        position = (
                            int((bbox[0] + bbox[2]) / 2),
                            int(bbox[3]),
                        )

                    cam = camera_movement[frame_num]
                    position_adjusted = (
                        position[0] - cam[0],
                        position[1] - cam[1],
                    )
                    tracks[object_key][frame_num][track_id]["position"] = position
                    tracks[object_key][frame_num][track_id]["position_adjusted"] = position_adjusted

        return tracks
