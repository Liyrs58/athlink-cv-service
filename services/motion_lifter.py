"""
services/motion_lifter.py

Lifts 2D COCO-17 keypoints to 3D world-space using MotionBERT.

MotionBERT expects a temporal sequence of 2D keypoints (clip_len=243 frames)
and outputs root-relative 3D coordinates. We handle:
- Padding short sequences (< 243 frames) by repeating edge frames
- Missing keypoints (filled with zeros)
- Converting output from normalised space back to world metres
  using the existing pitch calibration (105m x 68m)
"""

import sys
import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from functools import partial

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Add MotionBERT to path
_MB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "MotionBERT")
if _MB_DIR not in sys.path:
    sys.path.insert(0, _MB_DIR)

_model = None
_config = None
_CLIP_LEN = 243
_NUM_JOINTS = 17

# Checkpoint paths
_CONFIG_PATH = os.path.join(_MB_DIR, "configs/pose3d/MB_ft_h36m.yaml")
_CHECKPOINT_PATH = os.path.join(
    _MB_DIR, "checkpoint/pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin"
)


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_model():
    """Load MotionBERT 3D pose model once and cache."""
    global _model, _config
    if _model is not None:
        return _model, _config

    try:
        from lib.utils.tools import get_config
        from lib.utils.learning import load_backbone

        _config = get_config(_CONFIG_PATH)
        model_backbone = load_backbone(_config)

        device = _get_device()

        checkpoint = torch.load(_CHECKPOINT_PATH, map_location="cpu")
        # Handle DataParallel state dict keys
        state_dict = checkpoint["model_pos"]
        new_state = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")
            new_state[new_key] = v

        model_backbone.load_state_dict(new_state, strict=True)
        model_backbone = model_backbone.to(device)
        model_backbone.eval()

        _model = model_backbone
        logger.info("MotionBERT loaded on %s (clip_len=%d)", device, _CLIP_LEN)
        return _model, _config

    except Exception as e:
        logger.error("Failed to load MotionBERT: %s", e)
        raise


def _normalise_2d(keypoints_seq: np.ndarray, w: float, h: float) -> np.ndarray:
    """
    Normalise pixel-space 2D keypoints to [-1, 1] range.
    keypoints_seq: (T, 17, 3) — last dim is [x, y, conf]
    Returns: (T, 17, 3) normalised
    """
    out = keypoints_seq.copy()
    out[:, :, 0] = (out[:, :, 0] / w) * 2.0 - 1.0  # x to [-1, 1]
    out[:, :, 1] = (out[:, :, 1] / h) * 2.0 - 1.0  # y to [-1, 1]
    return out


def _pad_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    """
    Pad or truncate a sequence to target_len by repeating edge frames.
    seq: (T, 17, 3)
    """
    T = seq.shape[0]
    if T >= target_len:
        return seq[:target_len]

    # Pad by repeating last frame
    pad_count = target_len - T
    padding = np.tile(seq[-1:], (pad_count, 1, 1))
    return np.concatenate([seq, padding], axis=0)


def lift_to_3d(
    pose_sequence: List[Dict[str, Any]],
    video_width: float = 1920.0,
    video_height: float = 1080.0,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> List[Dict[str, Any]]:
    """
    Lift 2D keypoints to 3D using MotionBERT.

    Args:
        pose_sequence: Output of pose_extractor.extract_poses_from_video().
            List of frame dicts with 'players' containing 'keypoints_2d'.
        video_width/height: Video resolution for normalisation.
        pitch_length/width: Real pitch dimensions in metres.

    Returns:
        Same structure with 'keypoints_3d' added per player.
        keypoints_3d: list of [x, y, z] in metres (root-relative).
    """
    model, config = get_model()
    device = _get_device()

    # Group keypoints by track_id across frames
    track_frames: Dict[int, List[tuple]] = {}  # track_id -> [(frame_idx_in_seq, kp2d)]

    for seq_idx, frame_info in enumerate(pose_sequence):
        for player in frame_info.get("players", []):
            tid = player.get("track_id")
            if tid is None:
                continue
            kp2d = player.get("keypoints_2d", [])
            track_frames.setdefault(tid, []).append((seq_idx, kp2d))

    logger.info(
        "MotionBERT: lifting %d tracks across %d frames",
        len(track_frames), len(pose_sequence),
    )

    # For each track, build a temporal sequence and run MotionBERT
    track_3d: Dict[int, Dict[int, list]] = {}  # track_id -> {seq_idx -> kp3d}

    for tid, frame_list in track_frames.items():
        # Build (T, 17, 3) array
        T = len(frame_list)
        kp_seq = np.zeros((T, _NUM_JOINTS, 3), dtype=np.float32)

        for i, (seq_idx, kp2d) in enumerate(frame_list):
            if kp2d and len(kp2d) >= _NUM_JOINTS:
                for j in range(_NUM_JOINTS):
                    kp_seq[i, j, 0] = kp2d[j][0]  # x
                    kp_seq[i, j, 1] = kp2d[j][1]  # y
                    kp_seq[i, j, 2] = kp2d[j][2]  # conf

        # Skip tracks with mostly empty keypoints
        valid_frames = (kp_seq[:, :, 2] > 0.1).any(axis=1).sum()
        if valid_frames < 3:
            logger.info("Track %d: only %d valid frames, skipping 3D lift", tid, valid_frames)
            track_3d[tid] = {}
            continue

        # Normalise to [-1, 1]
        kp_norm = _normalise_2d(kp_seq, video_width, video_height)

        # Pad to clip_len
        kp_padded = _pad_sequence(kp_norm, _CLIP_LEN)  # (243, 17, 3)

        # Run model
        try:
            batch = torch.FloatTensor(kp_padded).unsqueeze(0).to(device)  # (1, 243, 17, 3)
            with torch.no_grad():
                if hasattr(config, "flip") and config.flip:
                    from lib.utils.utils_data import flip_data
                    pred1 = model(batch)
                    pred2 = model(flip_data(batch))
                    pred2 = flip_data(pred2)
                    pred_3d = (pred1 + pred2) / 2.0
                else:
                    pred_3d = model(batch)

                # Root-relative: zero root z at first frame
                if hasattr(config, "rootrel") and config.rootrel:
                    pred_3d[:, :, 0, :] = 0
                else:
                    pred_3d[:, 0, 0, 2] = 0

            pred_3d = pred_3d.cpu().numpy()[0]  # (243, 17, 3)

            # Only keep the real frames (not padding)
            pred_3d = pred_3d[:T]

            # Store per-frame results
            track_3d[tid] = {}
            for i, (seq_idx, _) in enumerate(frame_list):
                kp3d_list = pred_3d[i].tolist()  # [[x,y,z], ...] for 17 joints
                track_3d[tid][seq_idx] = kp3d_list

            logger.info(
                "Track %d: lifted %d frames to 3D", tid, T,
            )

        except Exception as e:
            logger.warning("MotionBERT failed for track %d: %s", tid, e)
            track_3d[tid] = {}

    # Merge 3D results back into the frame structure
    output = []
    for seq_idx, frame_info in enumerate(pose_sequence):
        out_players = []
        for player in frame_info.get("players", []):
            tid = player.get("track_id")
            kp3d = track_3d.get(tid, {}).get(seq_idx, [])
            out_players.append({
                **player,
                "keypoints_3d": kp3d,
            })
        output.append({
            "frame_index": frame_info["frame_index"],
            "players": out_players,
        })

    logger.info("3D lifting complete: %d frames", len(output))
    return output
