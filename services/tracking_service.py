"""YOLO11 + BoT-SORT player tracking across frames.
"""

import os
import json
import logging
import math
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import cv2
import numpy as np
from services.pitch_service import _estimate_homography, validate_homography, _transform_point
from services.scene_classifier import SceneClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FIX 2 — Pitch polygon rejection (removes crowd / bench / linesmen)
# ---------------------------------------------------------------------------
_PITCH_POLYGON = None  # lazy Shapely polygon in world coords, 3m margin

def _get_pitch_polygon():
    """Build a Shapely polygon for the pitch with 3m margin (lazy / optional dep)."""
    global _PITCH_POLYGON
    if _PITCH_POLYGON is not None:
        return _PITCH_POLYGON
    try:
        from shapely.geometry import Polygon as _Poly
    except ImportError:
        logger.warning("shapely not installed; polygon-based rejection disabled")
        return None
    # 105 x 68 pitch in world coords, with 3m margin for touchline play
    _PITCH_POLYGON = _Poly([(-3, -3), (108, -3), (108, 71), (-3, 71)])
    return _PITCH_POLYGON


def _foot_point(bbox) -> Tuple[float, float]:
    """Return bottom-centre of bbox (foot-point in pixel space)."""
    return ((bbox[0] + bbox[2]) / 2.0, float(bbox[3]))


def _filter_to_pitch_polygon(
    bboxes_confs: List[Tuple[List[float], float]],
    H: Optional[np.ndarray],
) -> List[Tuple[List[float], float]]:
    """FIX 2: Drop detections whose foot point projects outside the pitch polygon.

    bboxes_confs: list of (bbox, conf). H: pixel→world homography (or None).
    Returns the kept subset. If H or Shapely unavailable, returns input unchanged
    (caller should fall back to the green-mask-based filter).
    """
    if H is None or not bboxes_confs:
        return bboxes_confs
    poly = _get_pitch_polygon()
    if poly is None:
        return bboxes_confs
    try:
        from shapely.geometry import Point as _Point
    except ImportError:
        return bboxes_confs

    pts_pix = np.array([_foot_point(b) for b, _ in bboxes_confs], dtype=np.float32)
    pts_pix = pts_pix.reshape(-1, 1, 2)
    try:
        pts_world = cv2.perspectiveTransform(pts_pix, H).reshape(-1, 2)
    except cv2.error:
        return bboxes_confs

    keep = []
    for (bbox, conf), (wx, wy) in zip(bboxes_confs, pts_world):
        if poly.contains(_Point(float(wx), float(wy))):
            keep.append((bbox, conf))
    return keep


# ---------------------------------------------------------------------------
# FIX 3 — Per-track team vote from torso-region HSV histograms
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# FIX 4 — Dual-Threshold Rescue Detections
# ---------------------------------------------------------------------------
def _compute_iou_single(box1, box2):
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
    return inter / (union + 1e-6)

def _get_detections_with_rescue(model, frame, base_conf=0.10, rescue_conf=0.05):
    """Primary pass at base_conf, rescue pass at rescue_conf for far players."""
    # Robust parsing: YOLO model() or model.predict() might return list or single Result
    results1_raw = model.predict(frame, conf=base_conf, verbose=False)
    
    if isinstance(results1_raw, list) and len(results1_raw) > 0:
        res1 = results1_raw[0]
    else:
        res1 = results1_raw

    if not hasattr(res1, "boxes") or res1.boxes is None or len(res1.boxes) == 0:
        main_dets = np.empty((0, 6))
    else:
        # Vectorized extraction
        xyxy = res1.boxes.xyxy.cpu().numpy()
        conf = res1.boxes.conf.cpu().numpy().reshape(-1, 1)
        cls = res1.boxes.cls.cpu().numpy().reshape(-1, 1)
        main_dets = np.hstack([xyxy, conf, cls])

    if len(main_dets) < 15: # If sparse frame, try rescue
        results2_raw = model.predict(frame, conf=rescue_conf, verbose=False)
        if isinstance(results2_raw, list) and len(results2_raw) > 0:
            res2 = results2_raw[0]
        else:
            res2 = results2_raw
            
        if hasattr(res2, "boxes") and res2.boxes is not None and len(res2.boxes) > len(main_dets):
            # Efficiently pick boxes not already in main_dets
            r_xyxy_all = res2.boxes.xyxy.cpu().numpy()
            r_conf_all = res2.boxes.conf.cpu().numpy().reshape(-1, 1)
            r_cls_all = res2.boxes.cls.cpu().numpy().reshape(-1, 1)
            
            for i in range(len(r_xyxy_all)):
                r_xyxy = r_xyxy_all[i]
                if len(main_dets) > 0 and any(_compute_iou_single(r_xyxy, m[:4]) > 0.5 for m in main_dets):
                    continue
                new_det = np.hstack([r_xyxy.reshape(1, 4), r_conf_all[i:i+1], r_cls_all[i:i+1]])
                if len(main_dets) == 0:
                    main_dets = new_det
                else:
                    main_dets = np.vstack([main_dets, new_det])
    return main_dets

def _compute_torso_histogram(frame, bbox):
    """Refined torso-only histogram (25-65% vertical, 20-80% horizontal)."""
    x1, y1, x2, y2 = map(int, bbox[:4])
    h_frame, w_frame = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    if bw < 5 or bh < 5: return None
    
    # Isolate jersey: skip head/legs
    tx1, tx2 = x1 + int(bw * 0.20), x1 + int(bw * 0.80)
    ty1, ty2 = y1 + int(bh * 0.25), y1 + int(bh * 0.65)
    
    tx1, tx2 = max(0, min(tx1, w_frame-1)), max(0, min(tx2, w_frame-1))
    ty1, ty2 = max(0, min(ty1, h_frame-1)), max(0, min(ty2, h_frame-1))
    
    crop = frame[ty1:ty2, tx1:tx2]
    if crop.size == 0: return None
    
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()
    if tx2 <= tx1 or ty2 <= ty1:
        return None
    torso = frame[ty1:ty2, tx1:tx2]
    if torso.size == 0:
        return None
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten().astype(np.float32)


def _cluster_teams_per_track(tracks: List[Dict[str, Any]], k: int = 3) -> None:
    """FIX 3: Run k-means once across median-per-track torso histograms.

    k=3 clusters: team A, team B, referee/officials. Each track gets ONE teamId
    (0, 1, or 2) written in place. The two clusters with largest member count
    are assigned to teams 0 and 1; the smallest cluster is tagged as officials
    (teamId=2). Tracks without histograms get teamId=-1.
    """
    samples = []
    valid_track_refs = []
    for t in tracks:
        hists = t.get("_torso_hists") or []
        if not hists:
            continue
        median = np.median(np.stack(hists, axis=0), axis=0)
        samples.append(median)
        valid_track_refs.append(t)

    if len(samples) < k:
        for t in tracks:
            t.setdefault("teamId", -1)
        return

    X = np.stack(samples, axis=0).astype(np.float32)
    # cv2.kmeans handles the clustering with no sklearn dependency
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1e-3)
    try:
        _, labels, _centres = cv2.kmeans(
            X, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS
        )
    except cv2.error as e:
        logger.warning("team k-means failed: %s", e)
        for t in tracks:
            t.setdefault("teamId", -1)
        return

    labels = labels.flatten().tolist()
    # Rank clusters by size: largest two → team 0, 1; smallest → officials (2).
    counts: Dict[int, int] = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    sorted_labs = sorted(counts.keys(), key=lambda c: -counts[c])
    label_to_team: Dict[int, int] = {}
    if len(sorted_labs) >= 1:
        label_to_team[sorted_labs[0]] = 0
    if len(sorted_labs) >= 2:
        label_to_team[sorted_labs[1]] = 1
    for c in sorted_labs[2:]:
        # Assign overflow clusters to nearest large team (0 or 1), not a 3rd team
        label_to_team[c] = 1

    for t, lab in zip(valid_track_refs, labels):
        t["teamId"] = int(label_to_team.get(int(lab), 0))
    # Any track without a histogram stays at -1
    for t in tracks:
        t.setdefault("teamId", -1)


# ---------------------------------------------------------------------------
# FIX 4 — Per-track Kalman smoothing of foot-point world coordinates
# ---------------------------------------------------------------------------
def _smooth_track_world_coords(tracks: List[Dict[str, Any]], fps: float) -> None:
    """Apply a constant-velocity Kalman filter per track to the foot-point in
    world coordinates. Writes smoothed world_x / world_y back into each trajectory
    entry. Also gap-fills Kalman predictions across short gaps naturally.

    State: [x, y, vx, vy].  dt = 1/fps.  Q = 0.5*I, R = 1.0*I.
    """
    if not tracks:
        return
    try:
        from filterpy.kalman import KalmanFilter
    except ImportError:
        logger.warning("filterpy not installed; skipping per-track Kalman smoothing")
        return

    dt = 1.0 / max(fps, 1.0)
    for track in tracks:
        traj = track.get("trajectory") or []
        xy = [
            (e.get("world_x"), e.get("world_y"))
            for e in traj
        ]
        if not any(wx is not None and wy is not None for wx, wy in xy):
            continue

        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        kf.Q = np.eye(4) * 0.5
        kf.R = np.eye(2) * 1.0
        kf.P *= 10.0
        initialised = False

        for entry, (wx, wy) in zip(traj, xy):
            if wx is None or wy is None:
                if initialised:
                    kf.predict()
                    entry["world_x"] = round(float(kf.x[0]), 2)
                    entry["world_y"] = round(float(kf.x[1]), 2)
                continue
            if not initialised:
                kf.x = np.array([float(wx), float(wy), 0.0, 0.0], dtype=np.float64)
                initialised = True
                # First sample — emit as-is; no filter lag yet
                entry["world_x"] = round(float(wx), 2)
                entry["world_y"] = round(float(wy), 2)
                continue
            kf.predict()
            kf.update(np.array([float(wx), float(wy)], dtype=np.float64))
            entry["world_x"] = round(float(kf.x[0]), 2)
            entry["world_y"] = round(float(kf.x[1]), 2)

MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolo11n.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONF", "0.25"))
IOU_THRESHOLD = float(os.getenv("YOLO_IOU", "0.45"))
TARGET_CLASSES = [0]  # COCO fallback: person class

# Roboflow sports model classes: {0: ball, 1: goalkeeper, 2: player, 3: referee}
FOOTBALL_PLAYER_CLASSES = [1, 2]  # goalkeeper, player
FOOTBALL_REFEREE_CLASS = 3
FOOTBALL_BALL_CLASS = 0
_using_football_model = False

_pending_rescue_bboxes: list = []

_model = None
_device = None
_use_half = False
_scene_classifier = SceneClassifier()

import threading
_tracking_lock = threading.Lock()


def _detect_device() -> str:
    """Auto-detect the best available device: cuda > mps > cpu."""
    global _device
    try:
        import torch
        # FIX: Resolve MPS synchronization crashes
        if hasattr(torch.backends, "mps"):
            torch.backends.mps.synchronize = lambda: None
        
        if torch.cuda.is_available():
            _device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _device = "mps"
        else:
            _device = "cpu"
    except ImportError:
        _device = "cpu"

    logger.info("Auto-detected device: %s", _device)
    return _device


def _get_model():
    global _model, _use_half, _using_football_model
    if _model is None:
        try:
            from services.model_cache import get_tracking_model
            device = _detect_device()
            _model = get_tracking_model()
            # Auto-detect: Roboflow sports model has classes {0:ball,1:goalkeeper,2:player,3:referee}
            _using_football_model = hasattr(_model, 'names') and 'player' in _model.names.values()
            _model.to(device)
            _use_half = device in ("cuda", "mps")
            logger.info(
                "YOLO model on %s (half=%s, from model_cache)",
                device, _use_half,
            )
        except ImportError:
            raise RuntimeError("ultralytics package not found. Install with: pip install ultralytics")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {str(e)}")
    return _model


def _compute_histogram(frame, bbox):
    """Extract HSV color histogram from upper body region of player crop."""
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    h_frame, w_frame = frame.shape[:2]
    x1 = max(0, min(x1, w_frame - 1))
    x2 = max(0, min(x2, w_frame - 1))
    y1 = max(0, min(y1, h_frame - 1))
    y2 = max(0, min(y2, h_frame - 1))
    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    h = crop.shape[0]
    upper = crop[:int(h * 0.6), :]
    if upper.size == 0:
        return None
    hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def get_pitch_mask(frame_bgr):
    """
    Returns a uint8 mask highlighting the green pitch area.
    Uses HSV green range tuned for grass colour.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([30, 40, 40])
    upper = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def estimate_camera_motion(prev_gray, curr_gray, pitch_mask=None):
    """
    Returns (M, method) where M is a 2x3 affine matrix
    mapping prev coords -> curr coords, and method is
    "orb", "farneback", or "farneback_fallback".
    Returns (None, "none") if motion cannot be estimated.
    """
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(prev_gray, pitch_mask)
    kp2, des2 = orb.detectAndCompute(curr_gray, pitch_mask)

    if des1 is not None and des2 is not None and len(des1) >= 8 and len(des2) >= 8:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda m: m.distance)[:50]

        if len(matches) >= 8:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            inliers = int(mask.sum()) if mask is not None else 0

            if H is not None and inliers >= 6:
                M = H[:2, :]
                return M, "orb"

    # Fallback: Farneback optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    dx = float(np.median(flow[..., 0]))
    dy = float(np.median(flow[..., 1]))
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return M, "farneback"


def _filter_to_pitch(frame_bgr, bboxes_confs):
    """
    Remove detections outside the pitch using green colour mask.
    FootyVision-style bystander removal via activation masks.

    bboxes_confs: list of (bbox, conf) tuples
    Returns filtered list of (bbox, conf).
    Never crashes — returns all detections on any failure.
    """
    if not bboxes_confs:
        return bboxes_confs
    try:
        h, w = frame_bgr.shape[:2]
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(
            hsv,
            np.array([25, 40, 40]),
            np.array([95, 255, 255]),
        )
        kernel = np.ones((60, 60), np.uint8)
        pitch_mask = cv2.dilate(green_mask, kernel)

        keep = []
        for bbox, conf in bboxes_confs:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), \
                int(bbox[2]), int(bbox[3])
            cx = (x1 + x2) // 2
            feet_y = y1 + int((y2 - y1) * 0.75)
            cx = max(0, min(cx, w - 1))
            feet_y = max(0, min(feet_y, h - 1))
            if pitch_mask[feet_y, cx] > 0:
                keep.append((bbox, conf))

        if not keep:
            return bboxes_confs
        return keep
    except Exception as e:
        logger.warning("Pitch filter failed: %s", e)
        return bboxes_confs


def is_pitch_shot(frame, threshold=0.12):
    """Check if frame shows the pitch (enough green) vs closeup/crowd/bench."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    return np.count_nonzero(green) / green.size > threshold


def detect_scene_cut(prev_frame_gray, curr_frame_gray, threshold=45.0) -> bool:
    """
    Detects hard broadcast cuts by measuring mean absolute difference
    between consecutive frames. A cut produces a sudden large intensity change.
    threshold=45.0 catches hard cuts without triggering on fast camera pans
    (which are gradual).
    """
    if prev_frame_gray is None or curr_frame_gray is None:
        return False
    if prev_frame_gray.shape != curr_frame_gray.shape:
        return False
    diff = cv2.absdiff(prev_frame_gray, curr_frame_gray)
    mean_diff = float(diff.mean())
    return mean_diff > threshold


def is_pitch_frame(frame) -> bool:
    """
    Returns True if frame shows the football pitch (>25% green pixels).
    Returns False for cutaway shots (crowd, bench, replay, close-up).

    HSV green range: H 35-85, S 40-255, V 40-255
    Threshold: 25% of frame pixels must be green.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 40, 40])
    upper = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    green_pct = np.count_nonzero(mask) / mask.size
    return green_pct > 0.25


def is_valid_pitch_frame(frame_bgr, min_green_pct=0.25) -> bool:
    """
    Returns True only if frame contains enough green pitch to be a valid
    gameplay view. Bench shots, crowd shots, close-ups fail this test.
    min_green_pct=0.25 means at least 25% of frame must be pitch-green.
    """
    return is_pitch_frame(frame_bgr)


def _is_potential_official(frame_bgr, bbox) -> bool:
    """
    Detects potential referee/linesman by analyzing dominant jersey color.
    Returns True if bbox shows black (ref) or yellow (linesman) kit.
    """
    x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        return False
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    try:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        dominant_v = float(np.median(hsv[:, :, 2]))
        dominant_h = float(np.median(hsv[:, :, 0]))
        dominant_s = float(np.median(hsv[:, :, 1]))
        # Black kit (referee): low Value
        if dominant_v < 50:
            return True
        # Yellow kit (linesman): H 20-35, high saturation
        if 20 < dominant_h < 35 and dominant_s > 100:
            return True
    except Exception:
        return False
    return False


class BallKalmanTracker:
    """
    Dedicated Kalman filter for ball tracking.
    State vector: [x, y, vx, vy] (position + velocity)
    Measurement: [x, y] (pixel position from YOLO)
    """
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array(
            [[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix = np.array(
            [[1,0,1,0],[0,1,0,1],
             [0,0,1,0],[0,0,0,1]], np.float32)
        # FIX 3: Increased process noise — allows faster state changes (less smoothing)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        # FIX 3: Decreased measurement noise — trust YOLO detections more
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.3
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False
        self.frames_since_detection = 0
        # FIX 3: Reduced max prediction frames — stop predicting after 0.8s without detection
        self.max_prediction_frames = 20  # ~0.8s at 25fps (was 45)

    def update(self, x, y):
        """Call when YOLO detects ball. Returns corrected position."""
        measurement = np.array([[x], [y]], dtype=np.float32)
        if not self.initialized:
            self.kf.statePre = np.array(
                [[x],[y],[0],[0]], dtype=np.float32)
            self.kf.statePost = np.array(
                [[x],[y],[0],[0]], dtype=np.float32)
            self.initialized = True
        self.kf.predict()
        corrected = self.kf.correct(measurement).flatten()
        self.frames_since_detection = 0
        return float(corrected[0]), float(corrected[1])

    def predict(self):
        """Call when YOLO does NOT detect ball.
        Returns predicted position if within window."""
        if not self.initialized:
            return None
        if self.frames_since_detection >= self.max_prediction_frames:
            return None
        predicted = self.kf.predict().flatten()
        self.frames_since_detection += 1
        x, y = float(predicted[0]), float(predicted[1])
        return x, y

    def search_region(self, frame_gray, radius=60):
        """
        Returns a bounding box to search for ball candidates
        near the predicted position.
        """
        pred = self.predict()
        if pred is None:
            return None
        px, py = pred
        return (
            max(0, int(px - radius)),
            max(0, int(py - radius)),
            min(frame_gray.shape[1], int(px + radius)),
            min(frame_gray.shape[0], int(py + radius))
        )


def is_on_pitch(world_x, world_y,
                 pitch_w=105.0, pitch_h=68.0,
                 margin=5.0) -> bool:
    """
    Returns True if position is within pitch boundary.
    margin=5.0 allows for players near touchline.
    Returns False for bench, staff, subs.
    """
    return (-margin <= world_x <= pitch_w + margin and
            -margin <= world_y <= pitch_h + margin)


def get_median_position(trajectory, frame_w, frame_h, homography=None):
    """
    Calculate median world position of a track.
    Returns (median_x, median_y) in metres.
    """
    if not trajectory:
        return None, None

    positions = []
    for entry in trajectory:
        bbox = entry["bbox"]
        world_x, world_y = pixel_to_world(bbox, frame_w, frame_h, homography)
        positions.append((world_x, world_y))

    if not positions:
        return None, None

    # Sort by x and y, take medians
    xs = sorted([p[0] for p in positions])
    ys = sorted([p[1] for p in positions])
    median_x = xs[len(xs) // 2]
    median_y = ys[len(ys) // 2]

    return median_x, median_y


def is_on_or_near_pitch(median_x, median_y, pitch_w=105.0, pitch_h=68.0, expand_margin=15.0) -> bool:
    """
    Returns True if median position is within the pitch bounding box expanded
    by expand_margin on each side, OR within 20m of pitch centre.
    Forgiving of homography errors from broadcast cameras.
    """
    # Check 1: within expanded pitch bounding box
    in_expanded_box = (
        -expand_margin <= median_x <= pitch_w + expand_margin and
        -expand_margin <= median_y <= pitch_h + expand_margin
    )
    # Check 2: within 20m of pitch centre
    centre_x, centre_y = pitch_w / 2.0, pitch_h / 2.0
    dist_from_centre = math.sqrt((median_x - centre_x)**2 + (median_y - centre_y)**2)
    near_centre = dist_from_centre <= 20.0

    return in_expanded_box or near_centre


def pixel_to_world(bbox, frame_w, frame_h, homography: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Convert pixel coordinates to world coordinates (metres).

    Uses homography if available (accurate, accounts for camera angle).
    Falls back to proportional scaling if homography is unavailable.
    """
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0

    if homography is not None:
        # Use homography-based transformation (handles camera angle/perspective)
        try:
            pt = np.array([[[cx, cy]]], dtype=np.float32)
            out = cv2.perspectiveTransform(pt, homography)
            world_x, world_y = float(out[0, 0, 0]), float(out[0, 0, 1])
            return world_x, world_y
        except Exception as e:
            logger.warning(f"Homography transform failed: {e}, falling back to proportional scaling")

    # Fallback: proportional scaling (used for dirt pitches or when homography unavailable)
    world_x = (cx / frame_w) * 105.0
    world_y = (cy / frame_h) * 68.0

    return world_x, world_y


def run_tracking(
    video_path: str,
    job_id: str,
    frame_stride: int = 5,
    max_frames: Optional[int] = None,
    max_track_age: int = 90,  # Frames of inactivity before eviction (in processed frame count)
    adaptive_stride: bool = True,  # FIX 2: process every frame during fast pans
    progress_path: Optional[str] = None,  # Path to write progress.json for live streaming
) -> Dict[str, Any]:
    """Run BoT-SORT object tracking on video frames with camera motion compensation."""
    with _tracking_lock:
        return _run_tracking_impl(video_path, job_id, frame_stride, max_frames, max_track_age, adaptive_stride, progress_path)


def _run_tracking_impl(
    video_path: str,
    job_id: str,
    frame_stride: int = 5,
    max_frames: Optional[int] = None,
    max_track_age: int = 90,
    adaptive_stride: bool = True,
    progress_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Internal implementation — must be called under _tracking_lock."""
    logger.info("Tracking started — single loop only")
    model = _get_model()

    # Select class IDs based on which model is loaded
    player_classes = FOOTBALL_PLAYER_CLASSES if _using_football_model else TARGET_CLASSES

    # Reset BoT-SORT tracker state so a previous job's persisted state
    # doesn't bleed into this run (the global model keeps tracker state
    # between calls when persist=True).
    model.predictor = None

    # Sampling multiplier on top of frame_stride. Default 1 = honour frame_stride
    # literally. Set TRACKING_SAMPLE_RATE=5 in the environment to restore the
    # legacy 5× speedup (trades track coverage for speed).
    SAMPLE_RATE = max(1, int(os.getenv("TRACKING_SAMPLE_RATE", "1")))
    effective_stride = frame_stride * SAMPLE_RATE
    logger.info(
        "Tracking: frame_stride=%d, SAMPLE_RATE=%d, effective_stride=%d",
        frame_stride, SAMPLE_RATE, effective_stride,
    )

    active_tracks: Dict[int, Dict[str, Any]] = {}
    completed_tracks: List[Dict[str, Any]] = []
    recently_lost: Dict[int, Dict[str, Any]] = {}  # FIX 1: tracks lost in last 30 frames for ReID recovery

    frame_results: List[Dict[str, Any]] = []
    ball_trajectory: List[Dict[str, Any]] = []
    ball_tracker = BallKalmanTracker()
    frame_metadata: List[Dict[str, Any]] = []  # FIX 5: per-frame validity tracking

    # FIX 3: track quality counters (use list for mutable reference in nested scope)
    id_switches_counter = [0]  # [0] = total_id_switches
    last_rescue_frame = -10  # Track last rescue detection frame
    next_rescue_id = [1000]  # [0] = next ID to allocate for rescue tracks (start at 1000 to avoid collision)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    ret0, _sample = cap.read()
    needs_rotation = (_sample is not None) and (_sample.shape[0] > _sample.shape[1])
    del _sample
    cap.release()

    # Reopen to start from the beginning
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not reopen video: {video_path}")

    # Estimate total frames for progress reporting
    _total_frames_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    output_dir = Path(f"temp/{job_id}/tracking")
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = Path(f"temp/{job_id}/frames")
    frames_dir.mkdir(parents=True, exist_ok=True)

    prev_active_ids: set = set()
    raw_frame_idx = 0
    processed_count = 0

    # FIX 1: Camera motion compensation state
    prev_gray = None
    # FIX 2: Adaptive stride state
    force_next_frame = False
    # FIX 1 & 2: Scene cut and frame validity state
    scene_cut_flag = False
    frame_is_valid = False
    valid_frames_count = 0

    # Estimate homography once from first valid frame
    homography_H = None
    homography_found = False

    # FIX 4: Standalone BoT-SORT tracker for state-managed coasting
    from boxmot.trackers.botsort.botsort import BoTSORT
    # ReID model initialization (OSNet) — FIX 1: Force x1_0 for better discrimination
    reid_model_path = Path("models/osnet_x1_0_msmt17.pt")
    tracker = BoTSORT(
        model_weights=reid_model_path,
        device=_detect_device(),
        half=_use_half,
        per_class=False,
    )
    logger.info(f"ReID model loaded: {reid_model_path}")
    print(f"✓ ReID model initialized: {reid_model_path}")

    # Patch hyperparameters for V2 manually to avoid YAML/Namespace issues
    # FIX 2: Aggressive match_thresh for fast pans (0.95→0.30), proximity (0.9→0.5)
    tracker.track_buffer = 120  # FIX 3: Extended to 24 seconds real-time @ frameStride=5
    tracker.match_thresh = 0.30  # MAX distance (0.30 = can be 30% of frame width apart)
    tracker.proximity_thresh = 0.5  # ReID works even with low IoU
    tracker.appearance_thresh = 0.55
    tracker.track_high_thresh = 0.5
    tracker.track_low_thresh = 0.1
    tracker.new_track_thresh = 0.4
    tracker.frame_rate = 25
    while True:
        print(f"DEBUG: Processing raw_frame_idx {raw_frame_idx}")
        ret, frame = cap.read()
        if not ret:
            print("DEBUG: cap.read() failed, breaking loop.")
            break

        # Skip non-sampled frames entirely (no YOLO inference — pure speed gain)
        if not force_next_frame and raw_frame_idx % effective_stride != 0:
            raw_frame_idx += 1
            continue

        print(f"DEBUG: Frame {raw_frame_idx} sampled")
        force_next_frame = False  # reset after consuming

        if max_frames and processed_count >= max_frames:
            print(f"DEBUG: max_frames {max_frames} reached.")
            break

        current_frame_idx = raw_frame_idx

        if needs_rotation:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        frame_path = frames_dir / f"frame_{current_frame_idx:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)

        # FIX 1: Frame validity gate — skip cutaway frames entirely
        # TODO: Disabled pitch frame check for now to process all frames
        # if not is_pitch_frame(frame):
        #     frame_is_valid = False
        #     logger.info(f"Frame {current_frame_idx}: cutaway frame (crowd/bench/replay), skipping")
        #     curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #     scene_cut_flag = detect_scene_cut(prev_gray, curr_gray, threshold=45.0)
        #     if scene_cut_flag:
        #         logger.info(f"Frame {current_frame_idx}: scene cut detected")
        #     prev_gray = curr_gray
        #     frame_metadata.append({
        #         "frameIndex": current_frame_idx,
        #         "analysis_valid": False,
        #         "scene_cut": scene_cut_flag,
        #         "tracks_active": len(prev_active_ids),
        #         "ball_detected": False,
        #         "ball_source": None,
        #     })
        #     raw_frame_idx += 1
        #     continue

        valid_frames_count += 1

        # FIX 1: Scene cut detection from valid frames
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scene_cut_flag = detect_scene_cut(prev_gray, curr_gray, threshold=45.0)
        if scene_cut_flag:
            logger.info(f"Frame {current_frame_idx}: scene cut detected")

        # Estimate homography from first valid frame (for accurate coordinate conversion)
        if not homography_found:
            # Try PnLCalib first (broadcast-grade calibration)
            try:
                from services.pnlcalib_service import estimate_homography_pnlcalib
                pnl = estimate_homography_pnlcalib(frame)
                if pnl and pnl.get('homography'):
                    homography_H = np.array(pnl['homography'], dtype=np.float32)
                    if validate_homography(homography_H, frame.shape[1], frame.shape[0]):
                        homography_found = True
                        logger.info(f"PnLCalib homography from frame {current_frame_idx}")
            except Exception as e:
                logger.debug(f"PnLCalib unavailable: {e}")

            # Fall back to existing Hough-line approach
            if not homography_found:
                try:
                    homography_H = _estimate_homography(frame)
                    if homography_H is not None and validate_homography(homography_H, frame.shape[1], frame.shape[0]):
                        homography_found = True
                        logger.info(f"Homography estimated from frame {current_frame_idx}")
                    else:
                        logger.info(f"Homography validation failed, will use proportional scaling")
                except Exception as e:
                    logger.warning(f"Homography estimation failed: {e}, will use proportional scaling")

        timestamp = current_frame_idx / fps if fps > 0 else 0.0
        frame_h_px = frame.shape[0]
        frame_w_px = frame.shape[1]

        # Mask broadcast overlay zones before detection
        detection_frame = frame.copy()
        detection_frame[int(frame_h_px * 0.92):, :] = 0   # bottom 8% — lower thirds only

        # Downscale wide frames so YOLO can detect small players
        MAX_WIDTH = 1920
        if frame_w_px > MAX_WIDTH:
            scale = MAX_WIDTH / frame_w_px
            detect_frame = cv2.resize(detection_frame, (MAX_WIDTH, int(frame_h_px * scale)))
        else:
            scale = 1.0
            detect_frame = detection_frame

        # FIX 1: Camera motion compensation via ORB homography (Farneback fallback)
        # curr_gray already computed above for scene cut detection
        detect_frame_gray = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None and prev_gray.shape == detect_frame_gray.shape:
            pitch_mask = get_pitch_mask(detect_frame)
            M, method = estimate_camera_motion(prev_gray, detect_frame_gray, pitch_mask)

            if M is not None:
                dx_cam = float(M[0, 2])
                dy_cam = float(M[1, 2])
                magnitude = np.sqrt(dx_cam ** 2 + dy_cam ** 2)

                # FIX 2: adaptive stride — process next frame if pan is fast
                if adaptive_stride and magnitude > 150.0:
                    force_next_frame = True
                    logger.info(
                        f"Frame {current_frame_idx}: fast pan detected "
                        f"(magnitude={magnitude:.1f}px), forcing next frame [{method}]"
                    )

                if method == "orb":
                    # Full affine: transform bbox corners through the 2x3 matrix
                    H_full = np.vstack([M, [0, 0, 1]])  # 3x3 for perspectiveTransform
                    for _tid, _track in active_tracks.items():
                        if _track["trajectory"]:
                            _b = _track["trajectory"][-1]["bbox"]
                            corners = np.float32([
                                [_b[0] * scale, _b[1] * scale],
                                [_b[2] * scale, _b[1] * scale],
                                [_b[2] * scale, _b[3] * scale],
                                [_b[0] * scale, _b[3] * scale],
                            ]).reshape(-1, 1, 2)
                            transformed = cv2.perspectiveTransform(corners, H_full)
                            transformed = transformed.reshape(-1, 2)
                            new_x1 = float(np.min(transformed[:, 0])) / scale
                            new_y1 = float(np.min(transformed[:, 1])) / scale
                            new_x2 = float(np.max(transformed[:, 0])) / scale
                            new_y2 = float(np.max(transformed[:, 1])) / scale
                            _track["_cam_pred_bbox"] = [new_x1, new_y1, new_x2, new_y2]
                else:
                    # Farneback fallback: simple translation shift
                    dx_orig = dx_cam / scale if scale != 1.0 else dx_cam
                    dy_orig = dy_cam / scale if scale != 1.0 else dy_cam
                    for _tid, _track in active_tracks.items():
                        if _track["trajectory"]:
                            _b = _track["trajectory"][-1]["bbox"]
                            _track["_cam_pred_bbox"] = [
                                _b[0] + dx_orig,
                                _b[1] + dy_orig,
                                _b[2] + dx_orig,
                                _b[3] + dy_orig,
                            ]

        prev_gray = detect_frame_gray  # keep for next iteration

        # --- Scene classification: skip detections on non-pitch frames ---
        scene_class, scene_conf = _scene_classifier.classify_frame(frame)
        frame_is_pitch = scene_class not in ("cutaway", "graphic_overlay")

        # --- DIAGNOSTIC BLOCK (frames 0,5,10,...,50) ---
        _DIAG_FRAMES = {0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50}
        if current_frame_idx in _DIAG_FRAMES:
            # Green pct
            _small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            _hsv = cv2.cvtColor(_small, cv2.COLOR_BGR2HSV)
            _gmask = cv2.inRange(_hsv,
                np.array([25, 20, 20], dtype=np.uint8),
                np.array([95, 255, 255], dtype=np.uint8))
            _green_pct = float(np.count_nonzero(_gmask)) / _gmask.size

            # Ultra-low conf YOLO pass for diagnostics
            _raw_results = model.predict(detect_frame, conf=0.05, verbose=False)
            if isinstance(_raw_results, list) and len(_raw_results) > 0:
                _raw_res = _raw_results[0]
            else:
                _raw_res = _raw_results
            if hasattr(_raw_res, "boxes") and _raw_res.boxes is not None and len(_raw_res.boxes) > 0:
                _raw_classes = _raw_res.boxes.cls.cpu().numpy().astype(int).tolist()
                _raw_confs = [round(float(c), 3) for c in _raw_res.boxes.conf.cpu().numpy()]
                _raw_n = len(_raw_classes)
            else:
                _raw_classes = []
                _raw_confs = []
                _raw_n = 0

            # After class filter (keep 1,2,3)
            _kept_classes = [c for c in _raw_classes if c in (1, 2, 3)]
            _after_class_n = len(_kept_classes)

            # After conf filter (base_conf=0.10)
            _after_conf_confs = [v for c, v in zip(_raw_classes, _raw_confs) if c in (1, 2, 3) and v >= 0.10]
            _after_conf_n = len(_after_conf_confs)

            # Boundary filter preview — compute foot projections for kept dets
            # (just sample up to 3 for brevity)
            _sample_proj = []
            if hasattr(_raw_res, "boxes") and _raw_res.boxes is not None and len(_raw_res.boxes) > 0:
                _xyxy_all = _raw_res.boxes.xyxy.cpu().numpy()
                _cls_all = _raw_res.boxes.cls.cpu().numpy().astype(int)
                _conf_all = _raw_res.boxes.conf.cpu().numpy()
                for _i in range(min(3, len(_xyxy_all))):
                    if _cls_all[_i] in (1, 2, 3) and _conf_all[_i] >= 0.10:
                        _bx1, _by1, _bx2, _by2 = _xyxy_all[_i]
                        _cx = (_bx1 + _bx2) / 2 / scale
                        _cy = _by2 / scale  # foot point
                        _sample_proj.append((round(float(_cx), 1), round(float(_cy), 1)))

            _frame_mean = float(detect_frame.mean())
            _frame_std = float(detect_frame.std())
            print(f"--- DIAGNOSTIC FRAME {current_frame_idx} ---")
            print(f"[FRAME] shape={detect_frame.shape} | mean={_frame_mean:.1f} | std={_frame_std:.1f} | blank={_frame_mean < 5.0}")
            print(f"[YOLO] raw_boxes={_raw_n} | classes={_raw_classes[:20]} | confs={_raw_confs[:20]}")
            print(f"[YOLO] after_class_filter={_after_class_n} | kept_classes={_kept_classes[:20]}")
            print(f"[YOLO] after_conf_filter={_after_conf_n} | min_conf_used=0.10")
            print(f"[VALID] green_pct={_green_pct:.3f} | is_valid={frame_is_pitch} | scene={scene_class}({scene_conf})")
            if _green_pct < 0.15:
                print(f"[VALID] PITCH CHECK FAILED — frame marked invalid (green_pct={_green_pct:.3f} < 0.15)")
            print(f"[BOUND] sample_proj_coords={_sample_proj}")
            print(f"[TRACK] tracker_input_count={_after_conf_n} | scene_class={scene_class}")
            if _after_conf_n == 0 and _raw_n > 0:
                print(f"[TRACK] WARNING: raw_boxes={_raw_n} but after filters=0 — check class IDs and conf thresholds")
            print(f"-----------------------------------")

        # --- Player tracking via manual BoT-SORT update (enables Kalman coasting) ---
        if frame_is_pitch:
            dets_np = _get_detections_with_rescue(model, detect_frame)
            if len(dets_np) > 0:
                # roboflow_players.pt defines: 0=ball, 1=goalkeeper, 2=player, 3=referee
                # Filter to people classes: 1, 2, 3
                dets_np = dets_np[np.isin(dets_np[:, 5], [1, 2, 3])]
        else:
            logger.info("Frame %d: cutaway detected, coasting Kalman filters", current_frame_idx)
            dets_np = np.empty((0, 6))

        # Always update tracker (coasts on empty dets_np)
        bt_tracks = tracker.update(dets_np, detect_frame)
        print(f"DEBUG: Frame {current_frame_idx} BoxMOT update returned {len(bt_tracks) if bt_tracks is not None else 0} tracks.")

        # DEDUPLICATION: If same track ID appears twice in same frame, keep higher confidence
        if bt_tracks is not None and len(bt_tracks) > 0:
            # Group tracks by ID
            tracks_by_id = {}
            for track_idx, track in enumerate(bt_tracks):
                tid = int(track[4])  # track_id at index 4
                if tid not in tracks_by_id:
                    tracks_by_id[tid] = []
                tracks_by_id[tid].append((track_idx, track))

            # Find duplicate IDs
            duplicates = {tid: entries for tid, entries in tracks_by_id.items() if len(entries) > 1}

            if duplicates:
                # Build new track list with deduplicates
                next_new_id = max(int(t[4]) for t in bt_tracks) + 1
                new_bt_tracks = []

                for track_idx, track in enumerate(bt_tracks):
                    tid = int(track[4])

                    if tid in duplicates:
                        # This ID has duplicates
                        entries = duplicates[tid]

                        # Find the entry with highest confidence
                        best_entry = max(entries, key=lambda x: x[1][5])  # conf at index 5

                        if track_idx == best_entry[0]:
                            # This is the best confidence one, keep it
                            new_bt_tracks.append(track)
                        else:
                            # This is a duplicate with lower confidence, assign new ID
                            track_copy = list(track)
                            track_copy[4] = next_new_id
                            new_bt_tracks.append(track_copy)
                            logger.info(f"Frame {current_frame_idx}: DUPLICATE ID {tid} split -> {next_new_id} (conf: {track[5]:.3f})")
                            next_new_id += 1
                    else:
                        # No duplicate, keep as-is
                        new_bt_tracks.append(track)

                bt_tracks = new_bt_tracks

        current_active_ids: set = set()
        frame_track_entries: List[Dict[str, Any]] = []

        # bt_tracks is actually [x1, y1, x2, y2, track_id, conf, cls, ind]
        # or similar depending on BoxMOT version. We assume it's like track results.
        if bt_tracks is not None and len(bt_tracks) > 0:
            bboxes = [t[:4] for t in bt_tracks]
            bt_ids = [int(t[4]) for t in bt_tracks]
            bt_confs = [t[5] for t in bt_tracks]

            # FIX 2: Pitch polygon bystander removal.
            orig_bboxes = [[v / scale for v in b] if scale != 1.0 else b for b in bboxes]
            if homography_H is not None:
                pitch_filtered = _filter_to_pitch_polygon(
                    list(zip(orig_bboxes, bt_confs)), homography_H
                )
                if not pitch_filtered:
                    pitch_filtered = _filter_to_pitch(frame, list(zip(orig_bboxes, bt_confs)))
            else:
                pitch_filtered = _filter_to_pitch(frame, list(zip(orig_bboxes, bt_confs)))
            
            pitch_ok_set = set()
            for fb, fc in pitch_filtered:
                pitch_ok_set.add((round(fb[0], 1), round(fb[1], 1),
                                  round(fb[2], 1), round(fb[3], 1)))

            for bt_id, bbox, conf in zip(bt_ids, bboxes, bt_confs):
                track_id = int(bt_id)
                # Scale bbox back
                if scale != 1.0:
                    bbox = [v / scale for v in bbox]

                bbox_key = (round(bbox[0], 1), round(bbox[1], 1),
                            round(bbox[2], 1), round(bbox[3], 1))
                if bbox_key not in pitch_ok_set:
                    continue

                # Player bounding box constraints
                box_w, box_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                if box_w > frame_w_px * 0.35 or box_h > frame_h_px * 0.60:
                    continue
                if box_h < frame_h_px * 0.03 or bbox[3] < frame_h_px * 0.12 or bbox[1] > frame_h_px * 0.90:
                    continue

                # FIX 2: Track matching gate — only match if within 80px of prediction
                if track_id in active_tracks:
                    pred_bbox = active_tracks[track_id].get("_cam_pred_bbox")
                    if pred_bbox is not None:
                        pred_cx = (pred_bbox[0] + pred_bbox[2]) / 2.0
                        pred_cy = (pred_bbox[1] + pred_bbox[3]) / 2.0
                        curr_cx = (bbox[0] + bbox[2]) / 2.0
                        curr_cy = (bbox[1] + bbox[3]) / 2.0
                        dist = ((pred_cx - curr_cx)**2 + (pred_cy - curr_cy)**2)**0.5
                        if dist > TRACK_MATCHING_GATE:
                            # Detection too far from prediction — treat as new track
                            track_id = -1

                # FIX 1: Suppression gate — if scene is populated (>15 tracks),
                # don't create new tracks for detections close to existing predictions
                if (track_id == -1 and len(active_tracks) > 15 and
                    prev_active_ids):  # Only apply if we have existing tracks
                    det_cx = (bbox[0] + bbox[2]) / 2.0
                    det_cy = (bbox[1] + bbox[3]) / 2.0
                    nearest_tid = None
                    nearest_dist = float('inf')
                    for existing_tid in prev_active_ids:
                        if existing_tid in active_tracks:
                            pred = active_tracks[existing_tid].get("_cam_pred_bbox")
                            if pred is not None:
                                pred_cx = (pred[0] + pred[2]) / 2.0
                                pred_cy = (pred[1] + pred[3]) / 2.0
                                dist = ((det_cx - pred_cx) ** 2 + (det_cy - pred_cy) ** 2) ** 0.5
                                if dist < nearest_dist:
                                    nearest_dist = dist
                                    nearest_tid = existing_tid
                    if nearest_tid is not None and nearest_dist < 120:
                        track_id = nearest_tid  # Assign to nearest existing track

                current_active_ids.add(track_id)

                # FIX 2: Detect if this detection is a potential official
                is_official_detection = _is_potential_official(frame, bbox)

                # Compute world coordinates for this detection
                _wx, _wy = pixel_to_world(bbox, frame_w_px, frame_h_px, homography_H)

                trajectory_entry = {
                    "frameIndex": current_frame_idx,
                    "timestampSeconds": timestamp,
                    "bbox": bbox,
                    "confidence": conf,
                    "world_x": round(_wx, 2),
                    "world_y": round(_wy, 2),
                }

                if track_id not in active_tracks:
                    new_track = {
                        "trackId": track_id,
                        "hits": 1,
                        "firstSeen": current_frame_idx,
                        "lastSeen": current_frame_idx,
                        "trajectory": [trajectory_entry],
                        "_confirmed_detections": 1,
                        "_predicted_frames": 0,
                        "_id_switches": 0,
                        "_official_votes": 1 if is_official_detection else 0,
                        "_first_processed_frame": processed_count,  # Track when first processed
                        "_last_processed_frame": processed_count,
                    }
                    # FIX 1: Check if this is a ReID recovery from recently_lost
                    best_lost_tid = None
                    best_lost_score = float('inf')
                    for lost_tid, lost_track in recently_lost.items():
                        # Check bbox size similarity (±30%)
                        last_bbox = lost_track["trajectory"][-1]["bbox"]
                        last_h = last_bbox[3] - last_bbox[1]
                        curr_h = bbox[3] - bbox[1]
                        size_ratio = curr_h / last_h if last_h > 0 else 1.0
                        if 0.7 <= size_ratio <= 1.3:
                            # Check position proximity to Kalman prediction
                            pred = lost_track.get("_kalman_pred")
                            if pred is not None:
                                pred_cx, pred_cy = pred
                                curr_cx = (bbox[0] + bbox[2]) / 2.0
                                curr_cy = (bbox[1] + bbox[3]) / 2.0
                                dist = ((curr_cx - pred_cx) ** 2 + (curr_cy - pred_cy) ** 2) ** 0.5
                                if dist < 80 and dist < best_lost_score:
                                    best_lost_score = dist
                                    best_lost_tid = lost_tid
                    if best_lost_tid is not None:
                        # Reuse the old track ID
                        old_track = recently_lost.pop(best_lost_tid)
                        new_track["trackId"] = best_lost_tid
                        new_track["trajectory"] = old_track["trajectory"] + [trajectory_entry]
                        new_track["hits"] = old_track["hits"] + 1
                        new_track["firstSeen"] = old_track["firstSeen"]
                        new_track["_confirmed_detections"] = old_track.get("_confirmed_detections", 0) + 1
                        new_track["_predicted_frames"] = old_track.get("_predicted_frames", 0)
                        new_track["_id_switches"] = old_track.get("_id_switches", 0) + 1
                        new_track["_first_processed_frame"] = old_track.get("_first_processed_frame", processed_count)  # Preserve original
                        new_track["_last_processed_frame"] = processed_count
                        id_switches_counter[0] += 1  # FIX 1: track total ID switches
                        track_id = best_lost_tid
                    active_tracks[track_id] = new_track
                else:
                    active_tracks[track_id]["hits"] += 1
                    active_tracks[track_id]["lastSeen"] = current_frame_idx
                    active_tracks[track_id]["_last_processed_frame"] = processed_count
                    active_tracks[track_id]["trajectory"].append(trajectory_entry)
                    active_tracks[track_id]["_confirmed_detections"] += 1
                    active_tracks[track_id]["_official_votes"] += 1 if is_official_detection else 0

                # ReID: update appearance histogram (EMA blend) — for stitching
                hist = _compute_histogram(frame, bbox)
                if hist is not None:
                    prev_hist = active_tracks[track_id].get("_histogram")
                    if prev_hist is None:
                        active_tracks[track_id]["_histogram"] = hist
                    else:
                        blended = 0.3 * hist + 0.7 * prev_hist
                        cv2.normalize(blended, blended, 0, 1, cv2.NORM_MINMAX)
                        active_tracks[track_id]["_histogram"] = blended

                # FIX 3: Sample torso histogram every 5 processed frames for the
                # one-shot end-of-clip team clustering. We store a LIST (not EMA)
                # so k-means can use the median sample per track.
                if processed_count % 5 == 0:
                    torso = _compute_torso_histogram(frame, bbox)
                    if torso is not None:
                        active_tracks[track_id].setdefault("_torso_hists", []).append(torso)

                frame_track_entries.append({
                    "trackId": track_id,
                    "bbox": bbox,
                    "confidence": conf,
                })

        # Store active tracks for this frame
        active_ids = list(active_tracks.keys())
        prev_active_ids = set(active_ids)
        frame_results.append({
            "frameIndex": current_frame_idx,
            "tracks": frame_track_entries,
        })

        # Age out active tracks based on inactivity (processed frame count)
        for tid in list(active_tracks.keys()):
            track = active_tracks[tid]
            inactivity = processed_count - track.get("_last_processed_frame", processed_count)
            if inactivity > max_track_age:
                track = active_tracks.pop(tid)
                if track["trajectory"]:
                    last_bbox = track["trajectory"][-1]["bbox"]
                    track["_kalman_pred"] = (
                        (last_bbox[0] + last_bbox[2]) / 2.0,
                        (last_bbox[1] + last_bbox[3]) / 2.0,
                    )
                recently_lost[tid] = track

        to_complete = [
            tid for tid, track in recently_lost.items()
            if (processed_count - track.get("_last_processed_frame", 0)) > 30
        ]
        for tid in to_complete:
            completed_tracks.append(recently_lost.pop(tid))

        # --- Process rescue detections into actual tracks ---
        # Rescue detections added to frame_track_entries with trackId=-1 need to be
        # converted into real tracks in active_tracks
        rescue_entries = [fte for fte in frame_track_entries if fte.get("trackId") == -1]
        if rescue_entries:
            logger.info(f"Frame {current_frame_idx}: processing {len(rescue_entries)} rescue entries into tracks")
            for entry in rescue_entries:
                r_bbox = entry["bbox"]
                r_conf = entry["confidence"]

                is_official = _is_potential_official(frame, r_bbox)
                hist = _compute_histogram(frame, r_bbox)
                trajectory_entry = {
                    "frameIndex": current_frame_idx,
                    "timestampSeconds": timestamp,
                    "bbox": r_bbox,
                    "confidence": r_conf,
                }

                new_id = next_rescue_id[0]
                new_track = {
                    "trackId": new_id,
                    "hits": 1,
                    "firstSeen": current_frame_idx,
                    "lastSeen": current_frame_idx,
                    "trajectory": [trajectory_entry],
                    "_confirmed_detections": 1,
                    "_predicted_frames": 0,
                    "_id_switches": 0,
                    "_official_votes": 1 if is_official else 0,
                    "_first_processed_frame": processed_count,
                    "_last_processed_frame": processed_count,
                    "_histogram": hist,
                }
                active_tracks[new_id] = new_track
                current_active_ids.add(new_id)
                next_rescue_id[0] += 1

        # FIX 5: Track lifecycle logging
        # Log tracks that disappeared (were active last frame, not active this frame)
        if hasattr(run_tracking, '_prev_active_ids'):
            lost_tracks = run_tracking._prev_active_ids - current_active_ids
            for lost_id in lost_tracks:
                if lost_id in active_tracks:
                    last_pos = active_tracks[lost_id].get("trajectory", [{}])[-1].get("bbox", [0, 0, 0, 0])
                    logger.info(f"TRACK_LOST {lost_id} at frame {current_frame_idx}, pos={last_pos[:2]}")

        # Log tracks that were just born (new tracks appearing)
        for new_id in current_active_ids:
            if not hasattr(run_tracking, '_prev_active_ids') or new_id not in run_tracking._prev_active_ids:
                if new_id in active_tracks:
                    pos = active_tracks[new_id].get("trajectory", [{}])[-1].get("bbox", [0, 0, 0, 0])
                    logger.info(f"TRACK_BORN {new_id} at frame {current_frame_idx}, pos={pos[:2]}")

        # Store current active IDs for next frame comparison
        run_tracking._prev_active_ids = set(current_active_ids)

        prev_active_ids = current_active_ids

        # --- Ball detection with Roboflow dedicated model ---
        ball_detected = False
        try:
            import requests as _requests
            import base64 as _base64
            import os as _os

            _rf_key = _os.environ.get("ROBOFLOW_API_KEY", "uS7ciF51wtr6QxlZrXGJ")
            _, _buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            _b64 = _base64.b64encode(_buf.tobytes()).decode('utf-8')

            _resp = _requests.post(
                'https://detect.roboflow.com/footballs-1trlz/3',
                params={'api_key': _rf_key},
                data=_b64,
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                timeout=2.0
            )

            if _resp.status_code == 200:
                _rf_data = _resp.json()
                _preds = _rf_data.get('predictions', [])
                if _preds:
                    _best = max(_preds, key=lambda p: p.get('confidence', 0))
                    _cx = _best['x']
                    _cy = _best['y']
                    _w = _best['width']
                    _h = _best['height']
                    _bx1 = _cx - _w/2
                    _by1 = _cy - _h/2
                    _bx2 = _cx + _w/2
                    _by2 = _cy + _h/2
                    _cx, _cy = ball_tracker.update(_cx, _cy)
                    ball_trajectory.append({
                        "frameIndex": current_frame_idx,
                        "x": _cx,
                        "y": _cy,
                        "bbox": [_bx1, _by1, _bx2, _by2],
                        "confidence": float(_best.get('confidence', 0)),
                        "source": "roboflow",
                    })
                    ball_detected = True
        except Exception as _ball_err:
            pass

        # Fallback to YOLO if Roboflow failed
        if not ball_detected:
            try:
                ball_cls = [FOOTBALL_BALL_CLASS] if _using_football_model else [32]
                ball_results = model(frame, verbose=False, conf=0.15, classes=ball_cls, half=_use_half)
                if ball_results[0].boxes is not None and len(ball_results[0].boxes) > 0:
                    ball_boxes = ball_results[0].boxes.xyxy.cpu().tolist()
                    ball_confs = ball_results[0].boxes.conf.cpu().tolist()
                    best_idx = int(np.argmax(ball_confs))
                    bx1, by1, bx2, by2 = ball_boxes[best_idx]
                    cx = (bx1 + bx2) / 2.0
                    cy = (by1 + by2) / 2.0
                    cx, cy = ball_tracker.update(cx, cy)
                    ball_trajectory.append({
                        "frameIndex": current_frame_idx,
                        "x": cx,
                        "y": cy,
                        "bbox": [bx1, by1, bx2, by2],
                        "confidence": float(ball_confs[best_idx]),
                        "source": "yolo_fallback",
                    })
                    ball_detected = True
            except Exception:
                pass

        # If YOLO did not detect ball, use Kalman prediction + Hough circles
        if not ball_detected:
            predicted = ball_tracker.predict()
            if predicted is not None:
                px, py = predicted
                region = ball_tracker.search_region(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), radius=50)
                if region is not None:
                    x1, y1, x2, y2 = region
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        circles = cv2.HoughCircles(
                            crop_gray, cv2.HOUGH_GRADIENT, dp=1.2,
                            minDist=20, param1=50, param2=25,
                            minRadius=3, maxRadius=20
                        )
                        if circles is not None:
                            circles = np.uint16(np.around(circles))
                            # Pick circle nearest to prediction centre
                            best_circle = None
                            best_dist = float('inf')
                            pred_in_crop_x = px - x1
                            pred_in_crop_y = py - y1
                            for circle in circles[0]:
                                cx_crop, cy_crop, r = circle
                                # Convert numpy types to Python types for JSON serialization
                                cx_crop = int(cx_crop)
                                cy_crop = int(cy_crop)
                                r = int(r)
                                dist = ((cx_crop - pred_in_crop_x)**2 + (cy_crop - pred_in_crop_y)**2)**0.5
                                if dist < best_dist:
                                    best_dist = dist
                                    best_circle = circle
                            if best_circle is not None:
                                cx_crop, cy_crop, r = best_circle
                                # Convert numpy types to Python types for JSON serialization
                                cx_crop = int(cx_crop)
                                cy_crop = int(cy_crop)
                                r = int(r)
                                cx = x1 + cx_crop
                                cy = y1 + cy_crop
                                ball_tracker.update(cx, cy)
                                ball_trajectory.append({
                                    "frameIndex": current_frame_idx,
                                    "x": cx,
                                    "y": cy,
                                    "confidence": 0.5,
                                    "source": "hough_candidate",
                                })
                            else:
                                # No circle found, use Kalman prediction
                                ball_trajectory.append({
                                    "frameIndex": current_frame_idx,
                                    "x": px,
                                    "y": py,
                                    "confidence": 0.3,
                                    "source": "kalman_prediction",
                                })
                        else:
                            # No circles detected, use Kalman prediction
                            ball_trajectory.append({
                                "frameIndex": current_frame_idx,
                                "x": px,
                                "y": py,
                                "confidence": 0.3,
                                "source": "kalman_prediction",
                            })

        frame_results.append({
            "frameIndex": current_frame_idx,
            "timestampSeconds": timestamp,
            "detectionCount": len(frame_track_entries),
            "tracks": frame_track_entries,
        })

        # Frame is valid if at least 2 tracks are active
        frame_is_valid = len(current_active_ids) >= 2

        # FIX 5: Record per-frame metadata for validity analysis
        frame_metadata.append({
            "frameIndex": current_frame_idx,
            "analysis_valid": frame_is_valid,
            "scene_cut": scene_cut_flag,
            "tracks_active": len(current_active_ids),
            "ball_detected": any(bt["source"] == "yolo" for bt in ball_trajectory if bt.get("frameIndex") == current_frame_idx),
            "ball_source": next((bt.get("source") for bt in ball_trajectory if bt.get("frameIndex") == current_frame_idx), None),
        })

        # Write progress for live streaming (every 10 processed frames)
        if progress_path and processed_count % 10 == 0:
            _progress_tracks = []
            for _tid, _trk in active_tracks.items():
                if _trk["trajectory"]:
                    _last = _trk["trajectory"][-1]
                    _progress_tracks.append({
                        "track_id": _tid,
                        "team_id": _trk.get("teamId", -1),
                        "bbox": _last["bbox"],
                        "confirmed": (_trk.get("_confirmed_detections", 0) or 0) >= 3,
                        "frame_index": _last.get("frameIndex", current_frame_idx),
                    })
            # Find latest ball detection
            _ball_bbox = None
            if ball_trajectory:
                _last_ball = ball_trajectory[-1]
                if _last_ball.get("source") == "yolo":
                    _ball_bbox = _last_ball.get("bbox")
            _progress_data = {
                "frames_processed": raw_frame_idx,
                "total_frames": _total_frames_est,
                "tracks": _progress_tracks,
                "ball_bbox": _ball_bbox,
                "status": "processing",
            }
            try:
                import json as _json
                with open(progress_path, "w") as _pf:
                    _json.dump(_progress_data, _pf)
            except Exception:
                pass

        # FIX 2: Handle scene cut — aggressively reset tracks
        if scene_cut_flag:
            # On scene cut, drastically reduce track lifetimes
            MAX_TRACK_AGE_CUT = 3
            for tid in list(active_tracks.keys()):
                track = active_tracks[tid]
                age = current_frame_idx - track["lastSeen"]
                if age > MAX_TRACK_AGE_CUT:
                    completed_tracks.append(active_tracks.pop(tid))
            # Reset ball Kalman filter on cut
            ball_tracker.initialized = False
            ball_tracker.frames_since_detection = 0

        try:
            logger.info(
                f"Frame {current_frame_idx:6d} | t={timestamp:6.2f}s | "
                f"tracks={len(current_active_ids):3d} | valid={frame_is_valid} | cut={scene_cut_flag}"
            )
        except Exception as log_err:
            logger.error("Logging failed: %s", log_err)

        raw_frame_idx += 1
        processed_count += 1

    cap.release()

    # Merge active + completed
    all_tracks = completed_tracks + list(active_tracks.values())

    # Track stitching: merge tracks that likely represent the same player
    # across a camera pan (one ends within 10 frames of another starting,
    # and their last/first bbox centers are within 200px).
    all_tracks.sort(key=lambda t: t["firstSeen"])
    merged_ids = set()
    for i in range(len(all_tracks)):
        if i in merged_ids:
            continue
        ti = all_tracks[i]
        for j in range(i + 1, len(all_tracks)):
            if j in merged_ids:
                continue
            tj = all_tracks[j]
            gap = tj["firstSeen"] - ti["lastSeen"]
            if gap < 0 or gap > 10 * frame_stride:
                continue
            # Compare last bbox of ti with first bbox of tj
            last_bbox = ti["trajectory"][-1]["bbox"]
            first_bbox = tj["trajectory"][0]["bbox"]
            cx_last = (last_bbox[0] + last_bbox[2]) / 2.0
            cy_last = (last_bbox[1] + last_bbox[3]) / 2.0
            cx_first = (first_bbox[0] + first_bbox[2]) / 2.0
            cy_first = (first_bbox[1] + first_bbox[3]) / 2.0
            dist = ((cx_last - cx_first) ** 2 + (cy_last - cy_first) ** 2) ** 0.5
            if dist < 200.0:
                ti["trajectory"].extend(tj["trajectory"])
                ti["hits"] += tj["hits"]
                ti["lastSeen"] = max(ti["lastSeen"], tj["lastSeen"])
                ti["_last_processed_frame"] = max(ti.get("_last_processed_frame", 0), tj.get("_last_processed_frame", 0))
                merged_ids.add(j)

    all_tracks = [t for i, t in enumerate(all_tracks) if i not in merged_ids]

    # --- ReID stitching: merge tracks with similar appearance across larger gaps ---
    pre_reid_count = len(all_tracks)
    all_tracks.sort(key=lambda t: t["firstSeen"])
    reid_merged = set()
    for i in range(len(all_tracks)):
        if i in reid_merged:
            continue
        ti = all_tracks[i]
        hist_i = ti.get("_histogram")
        if hist_i is None:
            continue
        for j in range(i + 1, len(all_tracks)):
            if j in reid_merged:
                continue
            tj = all_tracks[j]
            hist_j = tj.get("_histogram")
            if hist_j is None:
                continue
            gap = tj["firstSeen"] - ti["lastSeen"]
            if gap < 0 or gap > 60 * frame_stride:
                continue
            dist = cv2.compareHist(hist_i, hist_j, cv2.HISTCMP_BHATTACHARYYA)
            if dist < 0.40:
                ti["trajectory"].extend(tj["trajectory"])
                ti["hits"] += tj["hits"]
                ti["lastSeen"] = max(ti["lastSeen"], tj["lastSeen"])
                ti["_last_processed_frame"] = max(ti.get("_last_processed_frame", 0), tj.get("_last_processed_frame", 0))
                reid_merged.add(j)
                logger.info(
                    f"ReID merged track {tj['trackId']} into {ti['trackId']} "
                    f"(bhattacharyya={dist:.3f})"
                )

    reid_tracks = [t for i, t in enumerate(all_tracks) if i not in reid_merged]
    print(f"DEBUG: Pre-ReID merges: {pre_reid_count}, merged count: {len(reid_merged)}")

    # Safety: revert if ReID collapsed too aggressively
    if len(reid_tracks) < 10 and pre_reid_count >= 10:
        logger.warning(
            f"ReID safety: tracks collapsed {pre_reid_count} -> {len(reid_tracks)}, "
            f"reverting ReID merges"
        )
    else:
        all_tracks = reid_tracks
        logger.info(f"ReID stitching: {pre_reid_count} -> {len(all_tracks)} tracks")
        print(f"DEBUG: After ReID merge logic: {len(all_tracks)} tracks")

    # Strip internal histogram data before serialization
    for t in all_tracks:
        t.pop("_histogram", None)

    # Minimum track length requirement — reduced to 2 for short traces
    MIN_TRACK_DETECTIONS = 2
    filtered = [t for t in all_tracks if t["hits"] >= MIN_TRACK_DETECTIONS]
    print(f"DEBUG: After MIN_TRACK_DETECTIONS filter (min={MIN_TRACK_DETECTIONS}): {len(filtered)} tracks (from {len(all_tracks)})")
    if len(filtered) < 5:
        # Fallback: accept tracks with >= 1 hits if we're too aggressive
        filtered = [t for t in all_tracks if t["hits"] >= 1]
        print(f"DEBUG: Fallback filter applied: {len(filtered)} tracks")

    # Gap-filling: linearly interpolate bbox for gaps of 2-4 frame-strides
    for track in filtered:
        traj = track["trajectory"]
        if len(traj) < 2:
            continue
        filled = []
        predicted_frames_count = 0
        for i in range(len(traj) - 1):
            filled.append(traj[i])
            a = traj[i]
            b = traj[i + 1]
            gap = b["frameIndex"] - a["frameIndex"]
            if 2 * frame_stride <= gap <= 8 * effective_stride:
                steps = gap // frame_stride
                for s in range(1, steps):
                    t_frac = s / steps
                    interp_bbox = [
                        a["bbox"][k] + t_frac * (b["bbox"][k] - a["bbox"][k])
                        for k in range(4)
                    ]
                    interp_fi = a["frameIndex"] + s * frame_stride
                    interp_ts = interp_fi / fps if fps > 0 else 0.0
                    # Interpolate world coords if both endpoints have them
                    interp_entry = {
                        "frameIndex": interp_fi,
                        "timestampSeconds": interp_ts,
                        "bbox": interp_bbox,
                        "confidence": (a["confidence"] + b["confidence"]) / 2.0,
                    }
                    if "world_x" in a and "world_x" in b:
                        interp_entry["world_x"] = round(a["world_x"] + t_frac * (b["world_x"] - a["world_x"]), 2)
                        interp_entry["world_y"] = round(a["world_y"] + t_frac * (b["world_y"] - a["world_y"]), 2)
                    filled.append(interp_entry)
                    predicted_frames_count += 1
        filled.append(traj[-1])
        track["trajectory"] = filled
        # FIX 3: Add quality metadata
        track["_predicted_frames"] = track.get("_predicted_frames", 0) + predicted_frames_count

    # FIX 1a: Filter staff/bench tracks based on pitch boundary presence
    final_tracks = []
    for track in filtered:
        on_pitch_count = 0
        traj = track["trajectory"]
        total_detections = len(traj)

        for entry in traj:
            bbox = entry["bbox"]
            world_x, world_y = pixel_to_world(bbox, frame_w_px, frame_h_px, homography_H)
            if is_on_pitch(world_x, world_y):
                on_pitch_count += 1
            
        on_pitch_pct = (on_pitch_count / total_detections) if total_detections > 0 else 0.0
        track["on_pitch_pct"] = round(on_pitch_pct, 2)
        track["is_staff"] = on_pitch_pct < 0.3

        if on_pitch_pct >= 0.3:
            final_tracks.append(track)

    filtered = final_tracks
    logger.info(f"After pitch boundary filter: {len(filtered)} tracks (from {len(all_tracks)} total)")

    # FIX 4: Additional pitch zone filter
    zone_filtered = []
    for track in filtered:
        median_x, median_y = get_median_position(track["trajectory"], frame_w_px, frame_h_px, homography_H)
        if median_x is not None and median_y is not None:
            if is_on_or_near_pitch(median_x, median_y, expand_margin=15.0):
                zone_filtered.append(track)
                track["median_x"] = round(median_x, 1)
                track["median_y"] = round(median_y, 1)
        else:
            logger.warning(f"Could not compute median position for track {track.get('trackId')}, excluding")

    filtered = zone_filtered
    logger.info(f"After pitch zone filter: {len(filtered)} tracks (from {len(all_tracks)} total)")
    # FIX 3 (v3): Dominant color team assignment — fast, no ML model needed.
    # Cascade: Dominant Color (LAB) → SigLIP (if installed) → HSV histogram (legacy).
    team_assigned = False
    try:
        from services.dominant_color_classifier import classify_teams_dominant_color
        classify_teams_dominant_color(filtered, video_path)
        t0 = sum(1 for t in filtered if t.get("teamId") == 0)
        t1 = sum(1 for t in filtered if t.get("teamId") == 1)
        if t0 >= 3 and t1 >= 3:
            team_assigned = True
            logger.info(f"Dominant color team classification: T0={t0}, T1={t1}")
        else:
            logger.warning(f"Dominant color produced imbalanced teams (T0={t0}, T1={t1}), trying SigLIP...")
    except Exception as e:
        logger.warning(f"Dominant color classifier failed ({e}), trying SigLIP...")

    if not team_assigned:
        try:
            from services.team_classifier import classify_teams_siglip
            classify_teams_siglip(filtered, video_path)
            logger.info(f"SigLIP team classification assigned teamId to {len(filtered)} tracks")
        except (ImportError, Exception) as e:
            logger.warning(f"SigLIP unavailable ({e}), falling back to HSV k-means")
            _cluster_teams_per_track(filtered, k=2)
            logger.info(f"HSV team clustering assigned teamId to {len(filtered)} tracks")

    # FIX 4: Per-track constant-velocity Kalman smoothing on foot-point world
    # coords. Kills vibration from noisy bbox-bottom measurements and gap-fills
    # short occlusions. Must run AFTER team clustering (independent) but BEFORE
    # downstream stats (distance / top speed / sprints).
    _smooth_track_world_coords(filtered, fps=fps)
    logger.info(f"Kalman-smoothed world coords for {len(filtered)} tracks")

    # Strip the raw torso-histogram list — large and not serialisable
    for _t in filtered:
        _t.pop("_torso_hists", None)

    # Clean up internal fields and compute quality metrics
    total_confirmed_detections = 0
    total_predicted_frames = 0
    stable_tracks = 0
    official_tracks_count = 0
    for track in filtered:
        confirmed = track.pop("_confirmed_detections", 0)
        predicted = track.pop("_predicted_frames", 0)
        official_votes = track.pop("_official_votes", 0)
        id_switches = track.pop("_id_switches", 0)
        total_confirmed_detections += confirmed
        total_predicted_frames += predicted
        if confirmed >= 5:
            stable_tracks += 1
        # Mark track as official if ≥30% of frames voted official
        traj_len = len(track["trajectory"])
        is_official = (official_votes >= traj_len * 0.3) if traj_len > 0 else False
        if is_official:
            official_tracks_count += 1
        # Add quality metadata to track output
        track["confirmed_detections"] = confirmed
        track["predicted_frames"] = predicted
        track["id_switches"] = id_switches
        track["is_official"] = is_official
        track["confidence_score"] = (confirmed / traj_len) if traj_len > 0 else 0.0
        # Set world_x/world_y on track dict from median trajectory position
        wx_vals = [e["world_x"] for e in track["trajectory"] if "world_x" in e]
        wy_vals = [e["world_y"] for e in track["trajectory"] if "world_y" in e]
        if wx_vals:
            wx_vals.sort()
            wy_vals.sort()
            track["world_x"] = wx_vals[len(wx_vals) // 2]
            track["world_y"] = wy_vals[len(wy_vals) // 2]
        else:
            track["world_x"] = None
            track["world_y"] = None

        # Clean up internal fields
        track.pop("_kalman_pred", None)
        track.pop("_cam_pred_bbox", None)
        track.pop("_histogram", None)

    avg_track_length = sum(len(t["trajectory"]) for t in filtered) / len(filtered) if filtered else 0.0

    # Convert bool values for JSON serialization
    for frame_meta in frame_metadata:
        for key, value in frame_meta.items():
            if isinstance(value, (bool, np.bool_)):
                frame_meta[key] = bool(value)

    tracking_quality = {
        "id_switches_total": id_switches_counter[0],  # FIX 1: total ID switches
        "avg_track_length_frames": avg_track_length,
        "tracks_with_5plus_detections": stable_tracks,
        "official_tracks": official_tracks_count,
        "valid_frames_pct": (valid_frames_count / len(frame_results) * 100.0) if frame_results else 0.0,
    }

    results_data = {
        "jobId": job_id,
        "videoPath": video_path,
        "frameStride": frame_stride,
        "framesProcessed": len(frame_results),
        "validFramesCount": valid_frames_count,
        "trackCount": len(filtered),
        "tracks": filtered,
        "ballDetections": len(ball_trajectory),
        "ball_trajectory": ball_trajectory,
        "frame_metadata": frame_metadata,
        "tracking_quality": tracking_quality,
    }

    # FIX: Final recursive JSON serialisation safety pass
    def json_safe(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [json_safe(v) for v in obj]
        return obj

    final_results_json = json_safe(results_data)

    results_file = output_dir / "track_results.json"
    with open(results_file, "w") as f:
        json.dump(final_results_json, f, indent=2)

    # Write final progress (completed)
    if progress_path:
        try:
            import json as _json
            with open(progress_path, "w") as _pf:
                _json.dump({"frames_processed": _total_frames_est, "total_frames": _total_frames_est, "tracks": [], "ball_bbox": None, "status": "completed"}, _pf)
        except Exception:
            pass

    logger.info(f"Tracking complete — {len(frame_results)} frames processed in single pass")
    logger.info(f"Tracking complete: {len(filtered)} tracks saved to {results_file}")
    return results_data
