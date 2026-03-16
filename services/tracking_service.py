import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8s.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONF", "0.25"))
IOU_THRESHOLD = float(os.getenv("YOLO_IOU", "0.45"))
TARGET_CLASSES = [0]

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


def is_pitch_shot(frame, threshold=0.12):
    """Check if frame shows the pitch (enough green) vs closeup/crowd/bench."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    return np.count_nonzero(green) / green.size > threshold


def run_tracking(
    video_path: str,
    job_id: str,
    frame_stride: int = 5,
    max_frames: Optional[int] = None,
    max_track_age: int = 50,
    adaptive_stride: bool = True,  # FIX 2: process every frame during fast pans
) -> Dict[str, Any]:
    """Run BoT-SORT object tracking on video frames with camera motion compensation."""
    model = _get_model()

    active_tracks: Dict[int, Dict[str, Any]] = {}
    completed_tracks: List[Dict[str, Any]] = []

    frame_results: List[Dict[str, Any]] = []
    ball_trajectory: List[Dict[str, Any]] = []

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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # FIX 2: override stride if previous frame had fast camera motion
        if not force_next_frame and raw_frame_idx % frame_stride != 0:
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

        # Pitch shot gate — skip YOLO on non-pitch frames (bench/crowd/closeup)
        if not is_pitch_shot(frame):
            logger.info(
                f"Frame {current_frame_idx}: non-pitch shot, skipping detection"
            )
            raw_frame_idx += 1
            continue

        timestamp = current_frame_idx / fps if fps > 0 else 0.0
        frame_h_px = frame.shape[0]
        frame_w_px = frame.shape[1]

        # Mask broadcast overlay zones before detection
        detection_frame = frame.copy()
        detection_frame[:int(frame_h_px * 0.10), :] = 0   # top 10% — scoreboard
        detection_frame[int(frame_h_px * 0.92):, :] = 0   # bottom 8% — lower thirds

        # Downscale wide frames so YOLO can detect small players
        MAX_WIDTH = 1920
        if frame_w_px > MAX_WIDTH:
            scale = MAX_WIDTH / frame_w_px
            detect_frame = cv2.resize(detection_frame, (MAX_WIDTH, int(frame_h_px * scale)))
        else:
            scale = 1.0
            detect_frame = detection_frame

        # FIX 1: Camera motion compensation via homography (ORB keypoints)
        curr_gray = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)
        homography_found = False
        dx_cam, dy_cam = 0.0, 0.0

        if prev_gray is not None and prev_gray.shape == curr_gray.shape:
            # Build green pitch mask for keypoint filtering
            hsv_comp = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2HSV)
            pitch_mask = cv2.inRange(hsv_comp, np.array([35, 40, 40]), np.array([85, 255, 255]))

            # Detect ORB keypoints in pitch region only
            orb = cv2.ORB_create(nfeatures=500)
            kp_prev, des_prev = orb.detectAndCompute(prev_gray, pitch_mask)
            kp_curr, des_curr = orb.detectAndCompute(curr_gray, pitch_mask)

            if des_prev is not None and des_curr is not None and len(des_prev) >= 8 and len(des_curr) >= 8:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des_prev, des_curr)
                matches = sorted(matches, key=lambda m: m.distance)

                if len(matches) >= 8:
                    prev_pts = np.float32([kp_prev[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    curr_pts = np.float32([kp_curr[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    H, mask = cv2.findHomography(prev_pts, curr_pts, cv2.RANSAC, 5.0)

                    if H is not None:
                        homography_found = True
                        # Estimate pan magnitude from homography translation
                        dx_cam = float(H[0, 2])
                        dy_cam = float(H[1, 2])
                        magnitude = np.sqrt(dx_cam ** 2 + dy_cam ** 2)

                        # FIX 2: adaptive stride — process next frame if pan is fast
                        if adaptive_stride and magnitude > 8.0:
                            force_next_frame = True
                            logger.info(
                                f"Frame {current_frame_idx}: fast pan detected "
                                f"(magnitude={magnitude:.1f}px), forcing next frame"
                            )

                        # Transform active track bboxes using homography
                        # H is in detect_frame (scaled) coords — scale back to original
                        for _tid, _track in active_tracks.items():
                            if _track["trajectory"]:
                                _b = _track["trajectory"][-1]["bbox"]
                                # Convert bbox corners to scaled coords, transform, scale back
                                corners = np.float32([
                                    [_b[0] * scale, _b[1] * scale],
                                    [_b[2] * scale, _b[1] * scale],
                                    [_b[2] * scale, _b[3] * scale],
                                    [_b[0] * scale, _b[3] * scale],
                                ]).reshape(-1, 1, 2)
                                transformed = cv2.perspectiveTransform(corners, H)
                                transformed = transformed.reshape(-1, 2)
                                # New bbox from transformed corners, scaled back
                                new_x1 = float(np.min(transformed[:, 0])) / scale
                                new_y1 = float(np.min(transformed[:, 1])) / scale
                                new_x2 = float(np.max(transformed[:, 0])) / scale
                                new_y2 = float(np.max(transformed[:, 1])) / scale
                                _track["_cam_pred_bbox"] = [new_x1, new_y1, new_x2, new_y2]

            # Fallback: median optical flow if homography failed
            if not homography_found:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                dx_cam = float(np.median(flow[..., 0]))
                dy_cam = float(np.median(flow[..., 1]))

                magnitude = np.sqrt(dx_cam ** 2 + dy_cam ** 2)
                if adaptive_stride and magnitude > 8.0:
                    force_next_frame = True
                    logger.info(
                        f"Frame {current_frame_idx}: fast pan detected "
                        f"(magnitude={magnitude:.1f}px), forcing next frame [flow fallback]"
                    )

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

        prev_gray = curr_gray  # keep for next iteration

        # --- Player tracking via BoT-SORT (camera motion compensation) ---
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

                current_active_ids.add(track_id)

                trajectory_entry = {
                    "frameIndex": current_frame_idx,
                    "timestampSeconds": timestamp,
                    "bbox": bbox,
                    "confidence": conf,
                }

                if track_id not in active_tracks:
                    active_tracks[track_id] = {
                        "trackId": track_id,
                        "hits": 1,
                        "firstSeen": current_frame_idx,
                        "lastSeen": current_frame_idx,
                        "trajectory": [trajectory_entry],
                    }
                else:
                    active_tracks[track_id]["hits"] += 1
                    active_tracks[track_id]["lastSeen"] = current_frame_idx
                    active_tracks[track_id]["trajectory"].append(trajectory_entry)

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
        if len(current_active_ids) < 3:
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
                    for fte in frame_track_entries:
                        e_bbox = fte["bbox"]
                        e_cx = (e_bbox[0] + e_bbox[2]) / 2.0
                        e_cy = (e_bbox[1] + e_bbox[3]) / 2.0
                        if ((r_cx - e_cx) ** 2 + (r_cy - e_cy) ** 2) ** 0.5 < 50.0:
                            already_tracked = True
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
                                break
                    if not already_tracked:
                        frame_track_entries.append({
                            "trackId": -1,
                            "bbox": r_bbox,
                            "confidence": r_conf,
                        })
                logger.info(
                    f"Frame {current_frame_idx}: rescue detection fired, "
                    f"{len(frame_track_entries)} total detections"
                )

        # Move disappeared tracks to completed_tracks
        disappeared = prev_active_ids - current_active_ids
        for tid in disappeared:
            if tid in active_tracks:
                completed_tracks.append(active_tracks.pop(tid))

        prev_active_ids = current_active_ids

        # --- Ball detection ---
        ball_results = model(frame, verbose=False, conf=0.15, classes=[32], half=_use_half)
        if ball_results[0].boxes is not None and len(ball_results[0].boxes) > 0:
            ball_boxes = ball_results[0].boxes.xyxy.cpu().tolist()
            ball_confs = ball_results[0].boxes.conf.cpu().tolist()
            best_idx = int(np.argmax(ball_confs))
            bx1, by1, bx2, by2 = ball_boxes[best_idx]
            ball_trajectory.append({
                "frameIndex": current_frame_idx,
                "x": (bx1 + bx2) / 2.0,
                "y": (by1 + by2) / 2.0,
                "bbox": [bx1, by1, bx2, by2],
                "confidence": float(ball_confs[best_idx]),
            })

        frame_results.append({
            "frameIndex": current_frame_idx,
            "timestampSeconds": timestamp,
            "detectionCount": len(frame_track_entries),
            "tracks": frame_track_entries,
        })

        logger.info(
            f"Frame {current_frame_idx:6d} | t={timestamp:6.2f}s | "
            f"tracks={len(current_active_ids):3d}"
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

    filtered = [t for t in all_tracks if t["hits"] >= 2]
    if len(filtered) < 5:
        filtered = [t for t in all_tracks if t["hits"] >= 1]

    # Gap-filling: linearly interpolate bbox for gaps of 2-4 frame-strides
    for track in filtered:
        traj = track["trajectory"]
        if len(traj) < 2:
            continue
        filled = []
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
        filled.append(traj[-1])
        track["trajectory"] = filled

    results_data = {
        "jobId": job_id,
        "videoPath": video_path,
        "frameStride": frame_stride,
        "framesProcessed": len(frame_results),
        "trackCount": len(filtered),
        "tracks": filtered,
        "ballDetections": len(ball_trajectory),
        "ball_trajectory": ball_trajectory,
    }

    results_file = output_dir / "track_results.json"
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"Tracking complete: {len(filtered)} tracks saved to {results_file}")
    return results_data
