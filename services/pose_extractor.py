"""
services/pose_extractor.py

Extracts 17-point COCO body keypoints from broadcast football footage
using Ultralytics YOLO11-Pose. Works on individual player crops
identified by the existing tracking pipeline.

COCO 17 keypoints:
0:nose 1:left_eye 2:right_eye 3:left_ear 4:right_ear
5:left_shoulder 6:right_shoulder 7:left_elbow 8:right_elbow
9:left_wrist 10:right_wrist 11:left_hip 12:right_hip
13:left_knee 14:right_knee 15:left_ankle 16:right_ankle
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Lazy-loaded singleton
_pose_model = None


def get_pose_model():
    """Load YOLO11-Pose model once and cache globally."""
    global _pose_model
    if _pose_model is not None:
        return _pose_model
    try:
        from ultralytics import YOLO
        import torch

        device = "mps" if torch.backends.mps.is_available() else (
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        _pose_model = YOLO("yolo11n-pose.pt")
        _pose_model.to(device)
        logger.info("YOLO11-Pose loaded on %s", device)
        return _pose_model
    except Exception as e:
        logger.error("Failed to load YOLO11-Pose: %s", e)
        raise


def _crop_with_padding(frame: np.ndarray, bbox: list, pad_ratio: float = 0.15) -> tuple:
    """
    Crop a player from the frame with padding.
    Returns (crop, offset_x, offset_y) where offsets map crop coords back to frame coords.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)

    cx1 = max(0, int(x1) - pad_x)
    cy1 = max(0, int(y1) - pad_y)
    cx2 = min(w, int(x2) + pad_x)
    cy2 = min(h, int(y2) + pad_y)

    crop = frame[cy1:cy2, cx1:cx2]
    return crop, cx1, cy1


def extract_poses_from_video(
    video_path: str,
    frame_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Extract 2D body keypoints for all tracked players across specified frames.

    Args:
        video_path: Path to the original broadcast video.
        frame_data: List of frame dicts from tracking pipeline.
            Each has 'frame_index' and 'players' list with
            bbox, track_id, team_id, world_x, world_y.

    Returns:
        Same structure with 'keypoints_2d' added to each player.
        keypoints_2d: list of [x, y, confidence] for 17 COCO joints
        (coordinates are in original frame pixel space).
    """
    model = get_pose_model()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(
        "Pose extraction: %d frames requested, video has %d total frames",
        len(frame_data), total_frames,
    )

    results = []

    for frame_info in frame_data:
        frame_index = frame_info["frame_index"]
        players = frame_info.get("players", [])

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()

        if not ret:
            logger.warning("Could not read frame %d", frame_index)
            results.append({
                "frame_index": frame_index,
                "players": [{**p, "keypoints_2d": []} for p in players],
            })
            continue

        processed_players = []

        for player in players:
            bbox = player.get("bbox")
            if not bbox or len(bbox) < 4:
                processed_players.append({**player, "keypoints_2d": []})
                continue

            try:
                crop, off_x, off_y = _crop_with_padding(frame, bbox)
                if crop.size == 0:
                    processed_players.append({**player, "keypoints_2d": []})
                    continue

                # Run pose on the crop — single person expected
                pose_results = model(crop, verbose=False)

                if (
                    pose_results
                    and pose_results[0].keypoints is not None
                    and len(pose_results[0].keypoints.data) > 0
                ):
                    # Pick the detection with highest confidence
                    # (should be the single player in the crop)
                    kp_data = pose_results[0].keypoints.data  # [N, 17, 3]
                    if len(kp_data) > 1:
                        # Multiple detections — pick one closest to crop centre
                        crop_h, crop_w = crop.shape[:2]
                        cx, cy = crop_w / 2, crop_h / 2
                        best_idx = 0
                        best_dist = float("inf")
                        for idx in range(len(kp_data)):
                            kps = kp_data[idx]
                            valid = kps[:, 2] > 0.3
                            if valid.any():
                                mx = kps[valid, 0].mean().item()
                                my = kps[valid, 1].mean().item()
                                d = (mx - cx) ** 2 + (my - cy) ** 2
                                if d < best_dist:
                                    best_dist = d
                                    best_idx = idx
                        kps = kp_data[best_idx]
                    else:
                        kps = kp_data[0]

                    # Convert crop coords back to frame coords
                    kp_with_conf = []
                    for j in range(kps.shape[0]):
                        x_frame = float(kps[j, 0].item()) + off_x
                        y_frame = float(kps[j, 1].item()) + off_y
                        conf = float(kps[j, 2].item())
                        kp_with_conf.append([x_frame, y_frame, conf])

                    processed_players.append({
                        **player,
                        "keypoints_2d": kp_with_conf,
                    })
                else:
                    processed_players.append({**player, "keypoints_2d": []})

            except Exception as e:
                logger.warning(
                    "Pose extraction failed for player %s frame %d: %s",
                    player.get("track_id"), frame_index, e,
                )
                processed_players.append({**player, "keypoints_2d": []})

        results.append({
            "frame_index": frame_index,
            "players": processed_players,
        })

        if frame_index % 50 == 0:
            logger.info(
                "Pose: frame %d — %d players processed",
                frame_index, len(processed_players),
            )

    cap.release()
    logger.info("Pose extraction complete: %d frames", len(results))
    return results
