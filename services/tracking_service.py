import os
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import cv2
import numpy as np
from services.pitch_service import _estimate_homography, validate_homography, _transform_point

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8s.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONF", "0.25"))
IOU_THRESHOLD = float(os.getenv("YOLO_IOU", "0.45"))
TARGET_CLASSES = [0]

_pending_rescue_bboxes: list = []

_model = None
_device = None
_use_half = False


def _detect_device() -> str:
    """Auto-detect the best available device: cuda > mps > cpu."""
    global _device
    if _device is not None:
        return _device

    forced = os.getenv("YOLO_DEVICE", "").strip()
    if forced:
        _device = forced
        logger.info("Using forced device from YOLO_DEVICE env: %s", _device)
        return _device

    try:
        import torch
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
    global _model, _use_half
    if _model is None:
        try:
            from ultralytics import YOLO
            device = _detect_device()
            _model = YOLO(MODEL_PATH)
            _model.to(device)
            # FP16 half-precision on GPU for faster inference
            _use_half = device in ("cuda", "mps")
            logger.info(
                "Loaded YOLO model for tracking: %s on %s (half=%s)",
                MODEL_PATH, device, _use_half,
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


def is_valid_pitch_frame(frame_bgr, min_green_pct=0.25) -> bool:
    """
    Returns True only if frame contains enough green pitch to be a valid
    gameplay view. Bench shots, crowd shots, close-ups fail this test.
    min_green_pct=0.25 means at least 25% of frame must be pitch-green.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([30, 40, 40])
    upper = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    green_pct = mask.sum() / 255 / mask.size
    return green_pct >= min_green_pct


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
        corrected = self.kf.correct(measurement)
        self.frames_since_detection = 0
        return float(corrected[0]), float(corrected[1])

    def predict(self):
        """Call when YOLO does NOT detect ball.
        Returns predicted position if within window."""
        if not self.initialized:
            return None
        if self.frames_since_detection >= self.max_prediction_frames:
            return None
        predicted = self.kf.predict()
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


def is_near_pitch_boundary(median_x, median_y, pitch_w=105.0, pitch_h=68.0, boundary_margin=10.0) -> bool:
    """
    Returns True if median position is within boundary_margin of any pitch edge.
    This filters out tracks that are always in crowd/bench area far from play.
    """
    # Check if within boundary_margin of any edge:
    # Left/right edges (x: -boundary_margin to pitch_w+boundary_margin)
    # Top/bottom edges (y: -boundary_margin to pitch_h+boundary_margin)
    near_x_edge = (-boundary_margin <= median_x <= boundary_margin) or (
        (pitch_w - boundary_margin) <= median_x <= (pitch_w + boundary_margin)
    )
    near_y_edge = (-boundary_margin <= median_y <= boundary_margin) or (
        (pitch_h - boundary_margin) <= median_y <= (pitch_h + boundary_margin)
    )
    return near_x_edge or near_y_edge


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
            world_x, world_y = _transform_point(homography, cx, cy)
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
) -> Dict[str, Any]:
    """Run BoT-SORT object tracking on video frames with camera motion compensation."""
    model = _get_model()

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

    # Estimate homography once from first valid frame (for accurate world coordinate conversion)
    homography_H = None
    homography_found = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # FIX 2: override stride if previous frame had fast camera motion
        if not force_next_frame and raw_frame_idx % frame_stride != 0:
            # Feed skipped frames to BoT-SORT silently to keep Kalman state current
            if prev_gray is not None and prev_gray.shape == detect_frame_gray.shape:
                model.track(
                    detect_frame,
                    tracker="tracker_config/botsort_football.yaml",
                    persist=True,
                    verbose=False,
                    conf=0.20,
                    iou=0.40,
                    classes=TARGET_CLASSES,
                    half=_use_half,
                )
            raw_frame_idx += 1
            continue

        force_next_frame = False  # reset after consuming

        if max_frames and processed_count >= max_frames:
            break

        current_frame_idx = raw_frame_idx

        if needs_rotation:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        frame_path = frames_dir / f"frame_{current_frame_idx:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)

        # FIX 1: Frame validity gate — check if frame contains valid pitch view
        frame_is_valid = is_valid_pitch_frame(frame, min_green_pct=0.25)
        if not frame_is_valid:
            logger.info(f"Frame {current_frame_idx}: non-pitch frame, skipping")
            # Still detect scene cuts from grayscale even on invalid frames
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scene_cut_flag = detect_scene_cut(prev_gray, curr_gray, threshold=45.0)
            if scene_cut_flag:
                logger.info(f"Frame {current_frame_idx}: scene cut detected")
            prev_gray = curr_gray
            frame_metadata.append({
                "frameIndex": current_frame_idx,
                "analysis_valid": False,
                "scene_cut": scene_cut_flag,
                "tracks_active": len(prev_active_ids),
                "ball_detected": False,
                "ball_source": None,
            })
            raw_frame_idx += 1
            continue

        valid_frames_count += 1

        # FIX 1: Scene cut detection from valid frames
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scene_cut_flag = detect_scene_cut(prev_gray, curr_gray, threshold=45.0)
        if scene_cut_flag:
            logger.info(f"Frame {current_frame_idx}: scene cut detected")

        # Estimate homography from first valid frame (for accurate coordinate conversion)
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

        # --- Player tracking via BoT-SORT (camera motion compensation) ---
        # FIX 3: BoT-SORT internal Kalman filters also benefit from tighter track lifecycle
        # The botsort_football.yaml config controls coasting frames (default ~30).
        # Combined with our aggressive MAX_TRACK_AGE=30, weak tracks are eliminated quickly.
        if _pending_rescue_bboxes:
            for pb in _pending_rescue_bboxes:
                x1 = int(pb[0] * scale)
                y1 = int(pb[1] * scale)
                x2 = int(pb[2] * scale)
                y2 = int(pb[3] * scale)
                cv2.rectangle(detect_frame, (x1, y1), (x2, y2), (255, 255, 255), -1)
            _pending_rescue_bboxes.clear()
        bt_results = model.track(
            detect_frame,
            tracker="tracker_config/botsort_football.yaml",
            persist=True,
            verbose=False,
            conf=0.20,
            iou=0.40,
            classes=TARGET_CLASSES,
            half=_use_half,
        )

        current_active_ids: set = set()
        frame_track_entries: List[Dict[str, Any]] = []
        TRACK_MATCHING_GATE = 80  # FIX 2: distance threshold for matching detections to tracks

        if bt_results[0].boxes is not None and bt_results[0].boxes.id is not None:
            bt_ids = bt_results[0].boxes.id.cpu().tolist()
            bboxes = bt_results[0].boxes.xyxy.cpu().tolist()
            confs = bt_results[0].boxes.conf.cpu().tolist()

            for bt_id_f, bbox, conf in zip(bt_ids, bboxes, confs):
                track_id = int(bt_id_f)

                # Scale bbox back to original frame coordinates
                if scale != 1.0:
                    bbox = [v / scale for v in bbox]

                # Drop non-player detections
                box_w = bbox[2] - bbox[0]
                box_h = bbox[3] - bbox[1]
                if box_w > frame_w_px * 0.35:
                    continue
                if box_h > frame_h_px * 0.60:
                    continue
                if box_h < frame_h_px * 0.03:
                    continue
                if bbox[3] < frame_h_px * 0.12:
                    continue
                if bbox[1] > frame_h_px * 0.90:
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

                trajectory_entry = {
                    "frameIndex": current_frame_idx,
                    "timestampSeconds": timestamp,
                    "bbox": bbox,
                    "confidence": conf,
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

                # ReID: update appearance histogram (EMA blend)
                hist = _compute_histogram(frame, bbox)
                if hist is not None:
                    prev_hist = active_tracks[track_id].get("_histogram")
                    if prev_hist is None:
                        active_tracks[track_id]["_histogram"] = hist
                    else:
                        blended = 0.3 * hist + 0.7 * prev_hist
                        cv2.normalize(blended, blended, 0, 1, cv2.NORM_MINMAX)
                        active_tracks[track_id]["_histogram"] = blended

                frame_track_entries.append({
                    "trackId": track_id,
                    "bbox": bbox,
                    "confidence": conf,
                })

        # --- Rescue detection when active tracks drop ---
        # Only fire when: tracks < 3, valid pitch frame, no fast pan, and not recently used
        frames_since_rescue = current_frame_idx - last_rescue_frame
        if (len(current_active_ids) < 3 and 
            frame_is_valid and 
            not scene_cut_flag and
            frames_since_rescue >= 10):
            rescue_results = model(detect_frame, verbose=False, conf=0.10, classes=[0], half=_use_half)
            if rescue_results[0].boxes is not None and len(rescue_results[0].boxes) > 0:
                rescue_bboxes = rescue_results[0].boxes.xyxy.cpu().tolist()
                rescue_confs = rescue_results[0].boxes.conf.cpu().tolist()
                for r_bbox, r_conf in zip(rescue_bboxes, rescue_confs):
                    if scale != 1.0:
                        r_bbox = [v / scale for v in r_bbox]
                    if r_bbox[3] < frame_h_px * 0.08:
                        continue
                    # Check if this detection overlaps an existing active track
                    r_cx = (r_bbox[0] + r_bbox[2]) / 2.0
                    r_cy = (r_bbox[1] + r_bbox[3]) / 2.0
                    already_tracked = False
                    matched_tid = None
                    for fte in frame_track_entries:
                        e_bbox = fte["bbox"]
                        e_cx = (e_bbox[0] + e_bbox[2]) / 2.0
                        e_cy = (e_bbox[1] + e_bbox[3]) / 2.0
                        if ((r_cx - e_cx) ** 2 + (r_cy - e_cy) ** 2) ** 0.5 < 50.0:
                            already_tracked = True
                            matched_tid = fte.get("trackId")
                            break
                    # Also check against camera-compensated predicted positions
                    if not already_tracked:
                        for _tid, _track in active_tracks.items():
                            pred = _track.get("_cam_pred_bbox")
                            if pred is None:
                                continue
                            p_cx = (pred[0] + pred[2]) / 2.0
                            p_cy = (pred[1] + pred[3]) / 2.0
                            if ((r_cx - p_cx) ** 2 + (r_cy - p_cy) ** 2) ** 0.5 < 50.0:
                                already_tracked = True
                                matched_tid = _tid
                                break
                    if already_tracked and matched_tid is not None and matched_tid in active_tracks:
                        # Update the existing track's _last_processed_frame to keep it alive
                        active_tracks[matched_tid]["_last_processed_frame"] = processed_count
                    if not already_tracked:
                        _pending_rescue_bboxes.append(r_bbox)
                        frame_track_entries.append({
                            "trackId": -1,
                            "bbox": r_bbox,
                            "confidence": r_conf,
                        })
                logger.info(
                    f"Frame {current_frame_idx}: rescue detection fired, "
                    f"{len(frame_track_entries)} total detections"
                )
                last_rescue_frame = current_frame_idx

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

        prev_active_ids = current_active_ids

        # --- Ball detection with Kalman prediction ---
        ball_detected = False
        ball_results = model(frame, verbose=False, conf=0.15, classes=[32], half=_use_half)
        if ball_results[0].boxes is not None and len(ball_results[0].boxes) > 0:
            ball_boxes = ball_results[0].boxes.xyxy.cpu().tolist()
            ball_confs = ball_results[0].boxes.conf.cpu().tolist()
            best_idx = int(np.argmax(ball_confs))
            bx1, by1, bx2, by2 = ball_boxes[best_idx]
            cx = (bx1 + bx2) / 2.0
            cy = (by1 + by2) / 2.0
            # Update Kalman filter with YOLO detection
            cx, cy = ball_tracker.update(cx, cy)
            ball_trajectory.append({
                "frameIndex": current_frame_idx,
                "x": cx,
                "y": cy,
                "bbox": [bx1, by1, bx2, by2],
                "confidence": float(ball_confs[best_idx]),
                "source": "yolo",
            })
            ball_detected = True

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

        # FIX 5: Record per-frame metadata for validity analysis
        frame_metadata.append({
            "frameIndex": current_frame_idx,
            "analysis_valid": frame_is_valid,
            "scene_cut": scene_cut_flag,
            "tracks_active": len(current_active_ids),
            "ball_detected": any(bt["source"] == "yolo" for bt in ball_trajectory if bt.get("frameIndex") == current_frame_idx),
            "ball_source": next((bt.get("source") for bt in ball_trajectory if bt.get("frameIndex") == current_frame_idx), None),
        })

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

        logger.info(
            f"Frame {current_frame_idx:6d} | t={timestamp:6.2f}s | "
            f"tracks={len(current_active_ids):3d} | valid={frame_is_valid} | cut={scene_cut_flag}"
        )

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
            if dist < 0.25:
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

    # Safety: revert if ReID collapsed too aggressively
    if len(reid_tracks) < 10 and pre_reid_count >= 10:
        logger.warning(
            f"ReID safety: tracks collapsed {pre_reid_count} -> {len(reid_tracks)}, "
            f"reverting ReID merges"
        )
    else:
        all_tracks = reid_tracks
        logger.info(f"ReID stitching: {pre_reid_count} -> {len(all_tracks)} tracks")

    # Strip internal histogram data before serialization
    for t in all_tracks:
        t.pop("_histogram", None)

    # FIX 2: Minimum track length requirement — at least 15 detections (was 3, reduced false positives)
    MIN_TRACK_DETECTIONS = 15
    filtered = [t for t in all_tracks if t["hits"] >= MIN_TRACK_DETECTIONS]
    if len(filtered) < 5:
        # Fallback: accept tracks with >= 5 hits if we're too aggressive
        filtered = [t for t in all_tracks if t["hits"] >= 5]

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
            if 2 * frame_stride <= gap <= 4 * frame_stride:
                steps = gap // frame_stride
                for s in range(1, steps):
                    t_frac = s / steps
                    interp_bbox = [
                        a["bbox"][k] + t_frac * (b["bbox"][k] - a["bbox"][k])
                        for k in range(4)
                    ]
                    interp_fi = a["frameIndex"] + s * frame_stride
                    interp_ts = interp_fi / fps if fps > 0 else 0.0
                    filled.append({
                        "frameIndex": interp_fi,
                        "timestampSeconds": interp_ts,
                        "bbox": interp_bbox,
                        "confidence": (a["confidence"] + b["confidence"]) / 2.0,
                    })
                    predicted_frames_count += 1
        filled.append(traj[-1])
        track["trajectory"] = filled
        # FIX 3: Add quality metadata
        track["_predicted_frames"] = track.get("_predicted_frames", 0) + predicted_frames_count

    # FIX 1a: Filter staff/bench tracks based on pitch boundary presence
    # Also exclude tracks whose average position is outside the pitch
    final_tracks = []
    for track in filtered:
        on_pitch_count = 0
        total_detections = len(track["trajectory"])

        for entry in track["trajectory"]:
            bbox = entry["bbox"]
            world_x, world_y = pixel_to_world(bbox, frame_w_px, frame_h_px, homography_H)
            if is_on_pitch(world_x, world_y):
                on_pitch_count += 1

        on_pitch_pct = (on_pitch_count / total_detections) if total_detections > 0 else 0.0
        track["on_pitch_pct"] = round(on_pitch_pct, 2)
        track["is_staff"] = on_pitch_pct < 0.3  # Less than 30% on pitch = staff/bench

        # FILTER: exclude tracks that are mostly outside the pitch (staff/bench)
        if on_pitch_pct >= 0.3:
            final_tracks.append(track)

    filtered = final_tracks
    logger.info(f"After pitch boundary filter: {len(filtered)} tracks (from {len(all_tracks)} total)")

    # FIX 4: Additional pitch zone filter — keep only tracks whose median position is near pitch edge
    # This removes tracks that stay in crowd/bench area away from active play
    zone_filtered = []
    for track in filtered:
        median_x, median_y = get_median_position(track["trajectory"], frame_w_px, frame_h_px, homography_H)
        if median_x is not None and median_y is not None:
            if is_near_pitch_boundary(median_x, median_y, boundary_margin=10.0):
                zone_filtered.append(track)
                track["median_x"] = round(median_x, 1)
                track["median_y"] = round(median_y, 1)
        else:
            # If we can't compute median, exclude the track
            logger.warning(f"Could not compute median position for track {track.get('trackId')}, excluding")

    filtered = zone_filtered
    logger.info(f"After pitch zone filter: {len(filtered)} tracks (from {len(all_tracks)} total)")

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
        "validFramesCount": valid_frames_count,  # FIX 5: track valid frames
        "trackCount": len(filtered),
        "tracks": filtered,
        "ballDetections": len(ball_trajectory),
        "ball_trajectory": ball_trajectory,
        "frame_metadata": frame_metadata,  # FIX 5: per-frame validity data
        "tracking_quality": tracking_quality,  # FIX 3: quality metrics
    }

    results_file = output_dir / "track_results.json"
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"Tracking complete: {len(filtered)} tracks saved to {results_file}")
    return results_data
