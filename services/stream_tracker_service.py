"""
Lightweight per-batch tracker for streaming frame processing.
Optimised for speed — must process 25 frames in under 1500ms.
"""
import os
import base64
import logging
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _iou(boxA: list, boxB: list) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter + 1e-6)


class StreamTrackerService:

    def __init__(self):
        self.model = None
        self._next_track_id = 1

    def load_model(self):
        """Load YOLO model once and cache it."""
        if self.model is not None:
            return
        try:
            from ultralytics import YOLO
            from services.tracking_service import _detect_device
            model_path = os.getenv("YOLO_MODEL", "yolov8n-pose.pt")
            device = _detect_device()
            self.model = YOLO(model_path)
            self.model.to(device)
            logger.info("StreamTracker: loaded %s on %s", model_path, device)
        except Exception as e:
            logger.error("StreamTracker: model load failed: %s", e)
            # Fallback to basic model
            try:
                from ultralytics import YOLO
                self.model = YOLO("yolov8n.pt")
                logger.info("StreamTracker: fallback to yolov8n.pt on cpu")
            except Exception as e2:
                logger.error("StreamTracker: fallback also failed: %s", e2)

    def process_batch(self,
                      frames: list,
                      frame_indices: list,
                      timestamps: list,
                      existing_tracks: dict,
                      confirmed_ids: dict,
                      frame_w: int,
                      frame_h: int) -> dict:
        """Process a batch of frames. Returns updated tracks and new uncertain list."""
        try:
            if self.model is None:
                self.load_model()
            if self.model is None:
                return {"updated_tracks": existing_tracks, "new_uncertain": []}

            new_uncertain = []

            for i, frame in enumerate(frames):
                fi = frame_indices[i] if i < len(frame_indices) else i
                ts = timestamps[i] if i < len(timestamps) else 0.0

                # Resize for speed
                h, w = frame.shape[:2]
                if w > 640:
                    scale = 640.0 / w
                    frame_resized = cv2.resize(frame, (640, int(h * scale)))
                    scale_back = w / 640.0
                else:
                    frame_resized = frame
                    scale_back = 1.0

                # YOLO detection
                results = self.model(frame_resized, verbose=False, conf=0.25, iou=0.45, classes=[0])

                detections = []
                for r in results:
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        cls_id = int(box.cls)
                        conf = float(box.conf)
                        if cls_id != 0:
                            continue
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        # Scale back to original resolution
                        x1 = int(x1 * scale_back)
                        y1 = int(y1 * scale_back)
                        x2 = int(x2 * scale_back)
                        y2 = int(y2 * scale_back)
                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": conf,
                        })

                # Match detections to existing tracks by IoU
                used_tracks = set()
                used_dets = set()

                # Build match pairs sorted by IoU descending
                match_pairs = []
                for det_idx, det in enumerate(detections):
                    for tid, track in existing_tracks.items():
                        iou_val = _iou(det["bbox"], track["bbox"])
                        if iou_val > 0.25:
                            match_pairs.append((iou_val, det_idx, tid))

                match_pairs.sort(reverse=True)

                for iou_val, det_idx, tid in match_pairs:
                    if det_idx in used_dets or tid in used_tracks:
                        continue
                    used_dets.add(det_idx)
                    used_tracks.add(tid)

                    det = detections[det_idx]
                    track = existing_tracks[tid]
                    track["bbox"] = det["bbox"]
                    cx = (det["bbox"][0] + det["bbox"][2]) / 2.0
                    cy = (det["bbox"][1] + det["bbox"][3]) / 2.0
                    track["positions"].append([cx, cy, fi])
                    track["last_frame"] = fi
                    track["detection_count"] += 1
                    n = track["detection_count"]
                    track["avg_confidence"] = (
                        track["avg_confidence"] * (n - 1) + det["confidence"]
                    ) / n

                    # Team colour detection
                    if track["team_id"] is None:
                        team_id = self._detect_team_colour(frame, det["bbox"])
                        if team_id >= 0:
                            track["team_id"] = team_id

                    # Update crop if better
                    if det["confidence"] > 0.6:
                        crop = self.extract_best_crop(frame, det["bbox"])
                        if crop is not None:
                            track["best_crop_b64"] = crop

                    # Mark confirmed if in confirmed_ids
                    if tid in confirmed_ids:
                        track["confirmed"] = True
                        track["confirmed_label"] = confirmed_ids[tid]
                        track["uncertain"] = False

                # Create new tracks for unmatched detections
                for det_idx, det in enumerate(detections):
                    if det_idx in used_dets:
                        continue

                    tid = self._next_track_id
                    self._next_track_id += 1
                    cx = (det["bbox"][0] + det["bbox"][2]) / 2.0
                    cy = (det["bbox"][1] + det["bbox"][3]) / 2.0

                    team_id = self._detect_team_colour(frame, det["bbox"])
                    crop = self.extract_best_crop(frame, det["bbox"])

                    new_track = {
                        "track_id": tid,
                        "team_id": team_id if team_id >= 0 else None,
                        "bbox": det["bbox"],
                        "positions": [[cx, cy, fi]],
                        "confirmed": False,
                        "confirmed_label": None,
                        "uncertain": True,
                        "first_frame": fi,
                        "last_frame": fi,
                        "detection_count": 1,
                        "avg_confidence": det["confidence"],
                        "best_crop_b64": crop,
                    }
                    existing_tracks[tid] = new_track
                    new_uncertain.append({
                        "track_id": tid,
                        "bbox": det["bbox"],
                        "crop_b64": crop,
                        "first_seen_frame": fi,
                    })

            # Flag tracks with low confidence as uncertain
            for tid, track in existing_tracks.items():
                if track["confirmed"]:
                    track["uncertain"] = False
                elif track["detection_count"] < 3 or track["avg_confidence"] < 0.4:
                    track["uncertain"] = True

            return {
                "updated_tracks": existing_tracks,
                "new_uncertain": new_uncertain,
            }

        except Exception as e:
            logger.error("StreamTracker batch processing failed: %s", e)
            return {"updated_tracks": existing_tracks, "new_uncertain": []}

    def _detect_team_colour(self, frame: np.ndarray, bbox: list) -> int:
        """Simplified team colour detection via HSV on torso region."""
        try:
            x1, y1, x2, y2 = bbox
            h_frame, w_frame = frame.shape[:2]
            x1 = max(0, min(x1, w_frame - 1))
            x2 = max(x1 + 1, min(x2, w_frame))
            y1 = max(0, min(y1, h_frame - 1))
            y2 = max(y1 + 1, min(y2, h_frame))

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return -1

            # Top 40% for torso
            ch = crop.shape[0]
            torso = crop[0:int(ch * 0.4), :]
            if torso.size == 0:
                return -1

            hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
            mean_hue = float(hsv[:, :, 0].mean())

            if 90 <= mean_hue <= 130:
                return 0  # blue-ish
            elif mean_hue <= 20 or mean_hue >= 160:
                return 1  # red-ish
            return -1
        except Exception:
            return -1

    def extract_best_crop(self, frame: np.ndarray, bbox: list,
                          min_height: int = 50) -> Optional[str]:
        """Extract torso crop as base64 JPEG if sharp enough."""
        try:
            x1, y1, x2, y2 = bbox
            bbox_h = y2 - y1
            if bbox_h < min_height:
                return None

            h_frame, w_frame = frame.shape[:2]
            x1 = max(0, min(x1, w_frame - 1))
            x2 = max(x1 + 1, min(x2, w_frame))
            y1 = max(0, min(y1, h_frame - 1))
            y2 = max(y1 + 1, min(y2, h_frame))

            # Top 55% for torso
            torso_bottom = y1 + int((y2 - y1) * 0.55)
            crop = frame[y1:torso_bottom, x1:x2]
            if crop.size == 0:
                return None

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if lap_var < 80:
                return None

            _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return base64.b64encode(buf.tobytes()).decode("utf-8")
        except Exception:
            return None

    def incremental_reid_patch(self, confirmed_track_id: int,
                                confirmed_label: str,
                                all_tracks: dict) -> int:
        """Merge fragmented tracks near the confirmed track's start position."""
        try:
            confirmed = all_tracks.get(confirmed_track_id)
            if not confirmed or not confirmed.get("positions"):
                return 0

            first_pos = confirmed["positions"][0]  # [x, y, frame_idx]
            first_frame = confirmed["first_frame"]
            merged = 0

            merge_ids = []
            for tid, track in all_tracks.items():
                if tid == confirmed_track_id:
                    continue
                if track["confirmed"]:
                    continue
                if not track.get("positions"):
                    continue

                # Check if this track was active near the confirmed track's start
                if abs(track["last_frame"] - first_frame) > 10:
                    continue

                # Check spatial proximity
                last_pos = track["positions"][-1]
                dist = ((last_pos[0] - first_pos[0]) ** 2 +
                        (last_pos[1] - first_pos[1]) ** 2) ** 0.5
                if dist < 50:
                    merge_ids.append(tid)

            for tid in merge_ids:
                track = all_tracks[tid]
                # Merge positions into confirmed track
                confirmed["positions"] = (
                    track["positions"] + confirmed["positions"]
                )
                confirmed["positions"].sort(key=lambda p: p[2])
                confirmed["first_frame"] = min(
                    confirmed["first_frame"], track["first_frame"]
                )
                confirmed["detection_count"] += track["detection_count"]
                # Remove merged track
                del all_tracks[tid]
                merged += 1

            return merged
        except Exception as e:
            logger.error("Incremental ReID patch failed: %s", e)
            return 0


# Module-level singleton
_stream_tracker: Optional[StreamTrackerService] = None


def get_stream_tracker() -> StreamTrackerService:
    global _stream_tracker
    if _stream_tracker is None:
        _stream_tracker = StreamTrackerService()
        _stream_tracker.load_model()
    return _stream_tracker
