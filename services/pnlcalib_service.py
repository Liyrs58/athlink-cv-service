"""
PnLCalib wrapper for broadcast-grade pitch homography estimation.

Uses HRNet-W48 keypoint + line detection models from PnLCalib
(github.com/mguti97/PnLCalib) to estimate camera parameters,
then extracts a 3x3 homography matrix for pixel-to-world conversion.

Falls back gracefully: returns None if models unavailable or calibration fails.
"""

import os
import sys
import cv2
import numpy as np
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Paths — models downloaded at Docker build time
PNLCALIB_DIR = os.environ.get("PNLCALIB_DIR", "/app/pnlcalib")
PNLCALIB_WEIGHTS_DIR = os.environ.get("PNLCALIB_WEIGHTS_DIR", "/app/models/pnlcalib")
KP_WEIGHTS = os.path.join(PNLCALIB_WEIGHTS_DIR, "SV_kp")
LINE_WEIGHTS = os.path.join(PNLCALIB_WEIGHTS_DIR, "SV_lines")
KP_CONFIG = os.path.join(PNLCALIB_DIR, "config", "hrnetv2_w48.yaml")
LINE_CONFIG = os.path.join(PNLCALIB_DIR, "config", "hrnetv2_w48_l.yaml")

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

# Lazy-loaded singletons
_kp_model = None
_line_model = None
_available = None  # None = not checked, True/False = checked


def _check_available() -> bool:
    """Check if PnLCalib models and code are present."""
    global _available
    if _available is not None:
        return _available

    missing = []
    for path, name in [(KP_WEIGHTS, "KP weights"), (LINE_WEIGHTS, "Line weights"),
                       (KP_CONFIG, "KP config"), (LINE_CONFIG, "Line config")]:
        if not os.path.exists(path):
            missing.append(f"{name}: {path}")

    if not os.path.isdir(PNLCALIB_DIR):
        missing.append(f"PnLCalib dir: {PNLCALIB_DIR}")

    if missing:
        logger.info("PnLCalib not available (missing: %s) — will use fallback", ", ".join(missing))
        _available = False
    else:
        _available = True
        logger.info("PnLCalib models found, available for use")

    return _available


def _load_models():
    """Load both HRNet models (keypoint + line). Called once, cached."""
    global _kp_model, _line_model

    if _kp_model is not None:
        return _kp_model, _line_model

    import torch
    import yaml

    # Add PnLCalib to path so its internal imports work
    if PNLCALIB_DIR not in sys.path:
        sys.path.insert(0, PNLCALIB_DIR)

    from model.cls_hrnet import get_cls_net
    from model.cls_hrnet_l import get_cls_net as get_cls_net_l

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Keypoint model
    with open(KP_CONFIG, 'r') as f:
        cfg = yaml.safe_load(f)
    kp_state = torch.load(KP_WEIGHTS, map_location=device)
    _kp_model = get_cls_net(cfg)
    _kp_model.load_state_dict(kp_state)
    _kp_model.to(device)
    _kp_model.eval()

    # Line model
    with open(LINE_CONFIG, 'r') as f:
        cfg_l = yaml.safe_load(f)
    line_state = torch.load(LINE_WEIGHTS, map_location=device)
    _line_model = get_cls_net_l(cfg_l)
    _line_model.load_state_dict(line_state)
    _line_model.to(device)
    _line_model.eval()

    logger.info("PnLCalib models loaded on %s", device)
    return _kp_model, _line_model


def estimate_homography_pnlcalib(frame: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Estimate pitch homography from a single BGR frame using PnLCalib.

    Returns dict matching homography_service.estimate_homography() format:
        {
            'method': 'pnlcalib',
            'homography': 3x3 list (pixel→world, ground plane),
            'visible_fraction': float,
            'pixels_per_metre': float,
        }

    Returns None if PnLCalib is unavailable or calibration fails.
    """
    if not _check_available():
        return None

    try:
        import torch
        from PIL import Image
        import torchvision.transforms.functional as tvf

        # Add PnLCalib to path
        if PNLCALIB_DIR not in sys.path:
            sys.path.insert(0, PNLCALIB_DIR)

        from utils.utils_calib import FramebyFrameCalib
        from utils.utils_heatmap import (
            get_keypoints_from_heatmap_batch_maxpool,
            get_keypoints_from_heatmap_batch_maxpool_l,
            complete_keypoints,
            coords_to_dict,
        )

        kp_model, line_model = _load_models()
        device = next(kp_model.parameters()).device

        h_orig, w_orig = frame.shape[:2]

        # Convert BGR→RGB, to tensor, resize to 960x540
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        tensor = tvf.to_tensor(img).float().unsqueeze(0)

        target_w, target_h = 960, 540
        if tensor.size(-1) != target_w or tensor.size(-2) != target_h:
            tensor = torch.nn.functional.interpolate(
                tensor, size=(target_h, target_w), mode='bilinear', align_corners=False
            )
        tensor = tensor.to(device)

        # Run both models
        with torch.no_grad():
            heatmaps = kp_model(tensor)
            heatmaps_l = line_model(tensor)

        # Extract keypoints and lines from heatmaps
        kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:, :-1, :, :])
        line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:, :-1, :, :])

        kp_dict = coords_to_dict(kp_coords, threshold=0.05)
        lines_dict = coords_to_dict(line_coords, threshold=0.05)

        if not kp_dict or not kp_dict[0]:
            logger.debug("PnLCalib: no keypoints detected")
            return None

        # Complete missing keypoints from line intersections
        kp_completed, lines_completed = complete_keypoints(
            kp_dict[0], lines_dict[0], w=target_w, h=target_h, normalize=True
        )

        # Calibrate camera
        cam = FramebyFrameCalib(iwidth=target_w, iheight=target_h, denormalize=True)
        cam.update(kp_completed, lines_completed)

        params = cam.heuristic_voting(refine_lines=True)
        if params is None:
            logger.debug("PnLCalib: heuristic_voting returned None")
            return None

        cam_params = params.get("cam_params")
        if cam_params is None:
            logger.debug("PnLCalib: no cam_params in result")
            return None

        # Build projection matrix P (3x4) from camera parameters
        fx = cam_params['x_focal_length']
        fy = cam_params['y_focal_length']
        pp = np.array(cam_params['principal_point'])
        pos = np.array(cam_params['position_meters'])
        R = np.array(cam_params['rotation_matrix'])

        K = np.array([
            [fx, 0, pp[0]],
            [0, fy, pp[1]],
            [0, 0, 1],
        ], dtype=np.float64)

        It = np.eye(4, dtype=np.float64)[:3]  # 3x4
        It[:, 3] = -pos
        P = K @ (R @ It)  # 3x4 projection matrix

        # Extract 3x3 homography for ground plane (Z=0):
        # World point [X, Y, 0, 1] → pixel. Drop the Z column from P.
        # H_world_to_pixel = P[:, [0, 1, 3]]
        H_w2p = P[:, [0, 1, 3]]  # 3x3: world(X,Y) → pixel

        # We need pixel → world, so invert
        H_p2w = np.linalg.inv(H_w2p)
        H_p2w = H_p2w / H_p2w[2, 2]  # normalise

        # The P matrix was built for 960x540. Scale H to original frame size.
        sx = w_orig / target_w
        sy = h_orig / target_h
        S_inv = np.array([
            [1.0 / sx, 0, 0],
            [0, 1.0 / sy, 0],
            [0, 0, 1],
        ], dtype=np.float64)
        # H_p2w expects 960x540 pixel coords. To accept original-size pixels:
        # world = H_p2w @ S_inv @ [px_orig, py_orig, 1]
        H_final = H_p2w @ S_inv
        H_final = H_final / H_final[2, 2]

        # Validate: transform frame centre → should land on pitch
        cx, cy = w_orig / 2.0, h_orig / 2.0
        pt = np.array([cx, cy, 1.0])
        world = H_final @ pt
        wx, wy = world[0] / world[2], world[1] / world[2]

        if not (0 < wx < PITCH_LENGTH and 0 < wy < PITCH_WIDTH):
            logger.debug(
                "PnLCalib: centre mapped to (%.1f, %.1f) — outside pitch, rejecting",
                wx, wy,
            )
            return None

        # Estimate visible fraction from how much pitch width is spanned
        # Transform left and right edges of frame
        left_pt = H_final @ np.array([0, h_orig / 2.0, 1.0])
        right_pt = H_final @ np.array([w_orig, h_orig / 2.0, 1.0])
        left_x = left_pt[0] / left_pt[2]
        right_x = right_pt[0] / right_pt[2]
        visible_span = abs(right_x - left_x)
        visible_frac = min(0.95, max(0.20, visible_span / PITCH_LENGTH))

        ppm = w_orig / (PITCH_LENGTH * visible_frac)

        logger.info(
            "PnLCalib: centre→(%.1f, %.1f)m, visible=%.0f%%, ppm=%.1f",
            wx, wy, visible_frac * 100, ppm,
        )

        return {
            'method': 'pnlcalib',
            'homography': H_final.tolist(),
            'visible_fraction': round(float(visible_frac), 3),
            'pixels_per_metre': round(float(ppm), 2),
        }

    except Exception as e:
        logger.warning("PnLCalib estimation failed: %s", e)
        return None
