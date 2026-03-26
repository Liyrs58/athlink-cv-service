"""
Video annotation service — produces annotated match video.
Adapted from Abdullah Tarek's football_analysis draw pipeline (MIT).
"""

import cv2
import numpy as np
import logging
import subprocess
import shutil
import os
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# IMPROVEMENT 2: Supervision ellipse annotators — broadcast-style player tracking [2026-03-25]
# Falls back gracefully to existing OpenCV drawing if supervision unavailable.
try:
    import supervision as sv
    _sv_ellipse = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(['#FF1744', '#00E676']),
        thickness=2,
    )
    _sv_label = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(['#FF1744', '#00E676']),
        text_color=sv.Color.WHITE,
        text_scale=0.4,
        text_thickness=1,
        text_padding=3,
    )
    _sv_triangle = sv.TriangleAnnotator(
        color=sv.Color.from_hex('#FFD700'),
        base=12,
        height=10,
    )
    _SV_AVAILABLE = True
    logger.info("supervision available — using broadcast-style ellipse annotations")
except Exception:
    _SV_AVAILABLE = False
    logger.info("supervision not available — using OpenCV ellipse annotations")


def _get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def _get_bbox_width(bbox):
    return bbox[2] - bbox[0]


def _get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)


def _measure_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


class VideoAnnotator:
    """
    Produces fully annotated video with player ellipses, ball indicator,
    speed/distance overlays, camera movement, and ball control percentages.
    Adapted directly from Abdullah Tarek's tracker.py draw methods.
    """

    def __init__(self):
        self.team_colors = {
            0: (0, 0, 255),      # Red for team 0 (BGR)
            1: (0, 255, 0),      # Green for team 1
            -1: (128, 128, 128),  # Grey for unknown
        }
        self.max_player_ball_distance = 70  # pixels — from Abdullah's PlayerBallAssigner

    # ------------------------------------------------------------------
    # Drawing primitives — adapted from tracker.py
    # ------------------------------------------------------------------

    def _draw_ellipse(self, frame, bbox, color, track_id=None):
        """Draw ellipse at player feet with ID label. From tracker.py draw_ellipse()."""
        y2 = int(bbox[3])
        x_center, _ = _get_center_of_bbox(bbox)
        width = _get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED,
            )

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        return frame

    def _draw_triangle(self, frame, bbox, color):
        """Draw triangle above player (ball possession indicator). From tracker.py draw_traingle()."""
        y = int(bbox[1])
        x, _ = _get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def _draw_ball(self, frame, bbox):
        """Draw ball as filled yellow triangle. From tracker.py draw_annotations() ball section."""
        return self._draw_triangle(frame, bbox, (0, 255, 255))

    def _draw_team_ball_control(self, frame, frame_num, team_ball_control):
        """
        Draw semi-transparent ball control overlay bottom-right.
        From tracker.py draw_team_ball_control().
        """
        h, w = frame.shape[:2]
        # Adaptive positioning based on frame size
        rect_w = min(550, int(w * 0.35))
        rect_h = 120
        x1 = w - rect_w - 20
        y1 = h - rect_h - 20

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (w - 20, h - 20), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        arr = np.array(team_ball_control_till_frame)
        team_0_frames = np.sum(arr == 0)
        team_1_frames = np.sum(arr == 1)
        total = team_0_frames + team_1_frames

        font_scale = min(1.0, rect_w / 550)

        if total > 0:
            t0_pct = team_0_frames / total * 100
            t1_pct = team_1_frames / total * 100
            cv2.putText(
                frame,
                f"Team 1 Ball Control: {t0_pct:.1f}%",
                (x1 + 10, y1 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Team 2 Ball Control: {t1_pct:.1f}%",
                (x1 + 10, y1 + 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                2,
            )
        else:
            cv2.putText(
                frame,
                "Possession: insufficient data",
                (x1 + 10, y1 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                2,
            )

        return frame

    def _draw_camera_movement(self, frame, camera_movement_xy):
        """
        Draw camera movement indicator top-left.
        From camera_movement_estimator.py draw_camera_movement().
        """
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        x_movement, y_movement = camera_movement_xy
        cv2.putText(
            frame,
            f"Camera Movement X: {x_movement:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            frame,
            f"Camera Movement Y: {y_movement:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3,
        )

        return frame

    def _draw_speed_and_distance(self, frame, bbox, speed, distance):
        """
        Draw speed and distance below player.
        From speed_and_distance_estimator.py draw_speed_and_distance().
        """
        position = list(_get_foot_position(bbox))
        position[1] += 40
        position = tuple(map(int, position))

        cv2.putText(
            frame,
            f"{speed:.1f} km/h",
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            frame,
            f"{distance:.1f} m",
            (position[0], position[1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )

        return frame

    # ------------------------------------------------------------------
    # Ball possession assignment
    # ------------------------------------------------------------------

    def assign_ball_possession(self, tracks: dict) -> Tuple[dict, list]:
        """
        For each frame determine which player has ball.
        Adapted from player_ball_assigner.py assign_ball_to_player().

        Uses closest foot-to-ball distance within 70px threshold.

        Returns:
        - Updated tracks with has_ball flags
        - team_ball_control list (team id per frame, -1 if no one)
        """
        team_ball_control = []
        num_frames = len(tracks["players"])

        for frame_num in range(num_frames):
            ball_dict = tracks["ball"][frame_num]
            player_dict = tracks["players"][frame_num]

            if not ball_dict or 1 not in ball_dict:
                # No ball detected — carry forward last possession
                if team_ball_control:
                    team_ball_control.append(team_ball_control[-1])
                else:
                    team_ball_control.append(-1)
                continue

            ball_bbox = ball_dict[1]["bbox"]
            ball_position = _get_center_of_bbox(ball_bbox)

            minimum_distance = 99999
            assigned_player = -1

            for player_id, player in player_dict.items():
                player_bbox = player["bbox"]

                # Measure from both feet corners to ball (Abdullah's approach)
                distance_left = _measure_distance(
                    (player_bbox[0], player_bbox[3]), ball_position
                )
                distance_right = _measure_distance(
                    (player_bbox[2], player_bbox[3]), ball_position
                )
                distance = min(distance_left, distance_right)

                if distance < self.max_player_ball_distance:
                    if distance < minimum_distance:
                        minimum_distance = distance
                        assigned_player = player_id

            if assigned_player != -1:
                tracks["players"][frame_num][assigned_player]["has_ball"] = True
                team_id = tracks["players"][frame_num][assigned_player].get("team", -1)
                team_ball_control.append(team_id)
            else:
                if team_ball_control:
                    team_ball_control.append(team_ball_control[-1])
                else:
                    team_ball_control.append(-1)

        return tracks, team_ball_control

    # ------------------------------------------------------------------
    # Possession percentage
    # ------------------------------------------------------------------

    def get_team_ball_control_pct(self, team_ball_control: list) -> dict:
        """
        Calculate possession percentage per team.
        Returns {0: 62.3, 1: 37.7}
        """
        arr = np.array(team_ball_control)
        t0 = np.sum(arr == 0)
        t1 = np.sum(arr == 1)
        total = t0 + t1
        if total == 0:
            return {0: None, 1: None}
        return {
            0: round(t0 / total * 100, 1),
            1: round(t1 / total * 100, 1),
        }

    # ------------------------------------------------------------------
    # Main annotation pipeline
    # ------------------------------------------------------------------

    def annotate_video(
        self,
        video_path: str,
        tracks: dict,
        output_path: str,
        camera_movement: list,
        team_ball_control: list,
    ) -> str:
        """
        Produce fully annotated video.

        tracks structure (Abdullah Tarek format):
        {
            "players": [
                {player_id: {"bbox": [], "team": int,
                             "speed": float, "distance": float,
                             "has_ball": bool}},
                ... one dict per frame
            ],
            "ball": [
                {1: {"bbox": []}},
                ... one dict per frame
            ]
        }

        Returns output_path on success.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Use mp4v directly — ffmpeg re-encodes to H.264 afterwards anyway.
        # Skipping avc1/H264/X264 avoids noisy hardware-encoder errors on RunPod.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("Failed to open VideoWriter with mp4v codec")
        logger.info("VideoWriter opened with codec: mp4v")

        num_frames = len(tracks["players"])
        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame_num >= num_frames:
                break

            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # IMPROVEMENT 2: Draw players with supervision ellipses if available [2026-03-25]
            if _SV_AVAILABLE and player_dict:
                try:
                    boxes, ids, class_ids, labels = [], [], [], []
                    for track_id, player in player_dict.items():
                        bbox = player.get("bbox")
                        team = player.get("team", -1)
                        if bbox and len(bbox) >= 4:
                            boxes.append(bbox[:4])
                            ids.append(int(track_id))
                            class_ids.append(0 if team == 0 else 1 if team == 1 else 0)
                            labels.append(str(track_id))
                    if boxes:
                        det = sv.Detections(
                            xyxy=np.array(boxes, dtype=np.float32),
                            tracker_id=np.array(ids),
                            class_id=np.array(class_ids),
                        )
                        frame = _sv_ellipse.annotate(frame, det)
                        frame = _sv_label.annotate(frame, det, labels=labels)
                except Exception as _sv_err:
                    logger.debug("supervision annotation failed: %s", _sv_err)
                    # fall through to OpenCV path
                    for track_id, player in player_dict.items():
                        team = player.get("team", -1)
                        color = self.team_colors.get(team, (128, 128, 128))
                        frame = self._draw_ellipse(frame, player["bbox"], color, track_id)
            else:
                for track_id, player in player_dict.items():
                    team = player.get("team", -1)
                    color = self.team_colors.get(team, (128, 128, 128))
                    frame = self._draw_ellipse(frame, player["bbox"], color, track_id)

            # Ball possession triangle + speed/distance (always OpenCV)
            for track_id, player in player_dict.items():
                if player.get("has_ball", False):
                    frame = self._draw_triangle(frame, player["bbox"], (0, 0, 255))
                speed = player.get("speed", 0)
                distance = player.get("distance", 0)
                if speed > 0 or distance > 0:
                    frame = self._draw_speed_and_distance(
                        frame, player["bbox"], speed, distance
                    )

            # Draw ball
            if _SV_AVAILABLE and ball_dict:
                try:
                    for _, ball in ball_dict.items():
                        bbox = ball.get("bbox", [])
                        if len(bbox) >= 4:
                            cx = (bbox[0] + bbox[2]) / 2
                            cy = (bbox[1] + bbox[3]) / 2
                            ball_det = sv.Detections(
                                xyxy=np.array([[cx - 10, cy - 20, cx + 10, cy]], dtype=np.float32)
                            )
                            frame = _sv_triangle.annotate(frame, ball_det)
                except Exception:
                    for _, ball in ball_dict.items():
                        frame = self._draw_ball(frame, ball["bbox"])
            else:
                for _, ball in ball_dict.items():
                    frame = self._draw_ball(frame, ball["bbox"])

            # Camera movement overlay removed — debug info not for end users

            # Draw ball control
            frame = self._draw_team_ball_control(frame, frame_num, team_ball_control)

            writer.write(frame)
            frame_num += 1

        cap.release()
        writer.release()

        # Re-encode to H.264 with ffmpeg for browser compatibility.
        # OpenCV's mp4v codec (MPEG-4 Part 2) won't play in browsers.
        output_path = self._remux_to_h264(output_path, fps)

        logger.info(
            "Annotated video written: %s (%d frames)", output_path, frame_num
        )
        return output_path

    @staticmethod
    def _remux_to_h264(path: str, fps: float) -> str:
        """Re-encode video to H.264/AAC mp4 if ffmpeg is available."""
        if not shutil.which("ffmpeg"):
            logger.warning("ffmpeg not found — skipping H.264 re-encode")
            return path

        tmp = path + ".h264.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-i", path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-r", str(fps),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-vf", "format=yuv420p",
            "-an",
            tmp,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=300)
            os.replace(tmp, path)
            logger.info("Re-encoded to H.264: %s", path)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning("ffmpeg re-encode failed: %s — keeping original", e)
            if os.path.exists(tmp):
                os.remove(tmp)
        return path
