"""
Speed and distance estimation service.
Adapted directly from Abdullah Tarek's SpeedAndDistance_Estimator (MIT).

Computes per-player speed (km/h) and cumulative distance (metres)
using camera-adjusted positions over a sliding frame window.
"""

import logging

logger = logging.getLogger(__name__)


def _measure_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


class SpeedEstimator:
    """
    Calculates speed and distance for all tracked players.
    Adapted from speed_and_distance_estimator.py add_speed_and_distance_to_tracks().
    """

    def __init__(self, frame_window: int = 5, frame_rate: float = 24.0):
        self.frame_window = frame_window
        self.frame_rate = frame_rate

    def calculate(self, tracks: dict, fps: float) -> dict:
        """
        Add speed (km/h) and distance (metres) to each player track entry.
        Uses position_transformed if available, else position_adjusted.

        Args:
            tracks: Abdullah-format tracks dict with "players", "ball" keys
            fps: video frame rate

        Returns:
            Updated tracks dict with speed and distance fields.
        """
        self.frame_rate = fps if fps > 0 else 24.0
        total_distance = {}

        for object_key in tracks:
            if object_key in ("ball", "referees"):
                continue

            object_tracks = tracks[object_key]
            number_of_frames = len(object_tracks)

            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

                for track_id in object_tracks[frame_num]:
                    if track_id not in object_tracks[last_frame]:
                        continue

                    # Use best available position
                    start_info = object_tracks[frame_num][track_id]
                    end_info = object_tracks[last_frame][track_id]

                    start_position = (
                        start_info.get("position_transformed")
                        or start_info.get("position_adjusted")
                    )
                    end_position = (
                        end_info.get("position_transformed")
                        or end_info.get("position_adjusted")
                    )

                    if start_position is None or end_position is None:
                        continue

                    distance_covered = _measure_distance(start_position, end_position)
                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    if time_elapsed <= 0:
                        continue

                    speed_ms = distance_covered / time_elapsed
                    speed_kmh = speed_ms * 3.6

                    if object_key not in total_distance:
                        total_distance[object_key] = {}

                    if track_id not in total_distance[object_key]:
                        total_distance[object_key][track_id] = 0

                    total_distance[object_key][track_id] += distance_covered

                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object_key][frame_num_batch]:
                            continue
                        tracks[object_key][frame_num_batch][track_id]["speed"] = speed_kmh
                        tracks[object_key][frame_num_batch][track_id]["distance"] = (
                            total_distance[object_key][track_id]
                        )

        return tracks
