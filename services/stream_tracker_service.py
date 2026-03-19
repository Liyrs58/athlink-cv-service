"""
Stream tracker using BoxMOT ByteTrack.
Per-session instances — no shared state.

Mirrors the proven approach from tracking_service.py:
1. Camera motion estimation BEFORE matching
2. IoU primary + center distance fallback (100px gate)
3. Track lifecycle: tentative → confirmed → lost → removed
4. ByteTrack (no ReID weights needed, works on Railway CPU)
"""

import os
import logging
import base64
import cv2
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

TENTATIVE_THRESHOLD = 5
MAX_LOST_FRAMES = 20
YOLO_CONF = 0.25
YOLO_IOU = 0.45
MIN_BOX_H = 30
MIN_BOX_W = 10
MAX_ASPECT = 4.0
MOTION_CAP = 80
CENTER_DIST_GATE = 100  # pixels — fallback matching when IoU fails


class Tracker:
    """
    Per-session stateful tracker wrapping BoxMOT ByteTrack.
    One instance per streaming session. No shared state.
    """

    def __init__(self, frame_w: int, frame_h: int, fps: float):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.fps = fps
        self.yolo = None
        self.bytetrack = None

        self._meta = {}
        # track_id -> {
        #   "track_id": int,
        #   "state": "tentative"|"confirmed"|"lost"|"removed",
        #   "detection_count": int,
        #   "frames_since_seen": int,
        #   "positions": [[cx,cy,fi],...],
        #   "first_frame": int,
        #   "last_frame": int,
        #   "team_id": int|None,
        #   "coach_confirmed": bool,
        #   "coach_label": str|None,
        #   "best_crop_b64": str|None,
        #   "surfaced_to_ui": bool,
        #   "bbox": [x1,y1,x2,y2],
        #   "avg_confidence": float,
        # }

        # Camera motion compensation state
        self._prev_gray = None
        self._motion_dx = 0.0
        self._motion_dy = 0.0

        # Team colour clustering
        self._colour_features = []
        self._team_centroids = None

    def load_model(self):
        """
        Load YOLO and ByteTrack once. Never raise.
        Try yolov8n-pose.pt first, fall back to yolov8n.pt.
        """
        try:
            from ultralytics import YOLO
            from services.tracking_service import _detect_device
            model_path = os.getenv("YOLO_MODEL", "yolov8n-pose.pt")
            device = _detect_device()
            self.yolo = YOLO(model_path)
            self.yolo.to(device)
            logger.info("Stream YOLO loaded: %s on %s",
                        model_path, device)
        except Exception as e:
            logger.error("Stream YOLO load failed: %s", e)
            try:
                from ultralytics import YOLO
                self.yolo = YOLO("yolov8n.pt")
                logger.info("Stream YOLO fallback: yolov8n.pt")
            except Exception as e2:
                logger.error("Fallback YOLO failed: %s", e2)

        try:
            from boxmot import ByteTrack
            self.bytetrack = ByteTrack(
                track_high_thresh=0.5,
                track_low_thresh=0.1,
                new_track_thresh=0.6,
                track_buffer=20,
                match_thresh=0.8,
                frame_rate=max(1, int(self.fps)),
            )
            logger.info("ByteTrack initialised (fps=%d, buffer=20)",
                        max(1, int(self.fps)))
        except Exception as e:
            logger.error("ByteTrack init failed: %s", e)

    def _estimate_camera_motion(self, gray: np.ndarray) -> tuple:
        """
        Camera motion compensation using ORB features with
        RANSAC homography, Farneback fallback.
        MUST be called BEFORE matching each frame.
        Returns (dx, dy) in pixels.
        """
        if self._prev_gray is None:
            self._prev_gray = gray
            return (0.0, 0.0)
        try:
            # Try ORB first (matches tracking_service.py approach)
            orb = cv2.ORB_create(nfeatures=300)
            kp1, des1 = orb.detectAndCompute(self._prev_gray, None)
            kp2, des2 = orb.detectAndCompute(gray, None)

            if (des1 is not None and des2 is not None
                    and len(des1) >= 8 and len(des2) >= 8):
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda m: m.distance)[:50]

                if len(matches) >= 8:
                    src_pts = np.float32(
                        [kp1[m.queryIdx].pt for m in matches]
                    ).reshape(-1, 1, 2)
                    dst_pts = np.float32(
                        [kp2[m.trainIdx].pt for m in matches]
                    ).reshape(-1, 1, 2)
                    H, mask = cv2.findHomography(
                        src_pts, dst_pts, cv2.RANSAC, 5.0
                    )
                    inliers = int(mask.sum()) if mask is not None else 0
                    if H is not None and inliers >= 6:
                        dx = float(H[0, 2])
                        dy = float(H[1, 2])
                        if abs(dx) <= MOTION_CAP and abs(dy) <= MOTION_CAP:
                            self._prev_gray = gray
                            return (dx, dy)

            # Farneback fallback
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray, gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            dx = float(np.median(flow[..., 0]))
            dy = float(np.median(flow[..., 1]))
            if abs(dx) > MOTION_CAP or abs(dy) > MOTION_CAP:
                dx, dy = 0.0, 0.0
            self._prev_gray = gray
            return (dx, dy)
        except Exception as e:
            logger.warning("Motion estimation failed: %s", e)
            self._prev_gray = gray
            return (0.0, 0.0)

    def _shift_all_tracks(self, dx: float, dy: float):
        """
        Apply camera motion compensation to ALL track positions.
        This shifts stored bboxes so they align with the new frame.
        Called BEFORE detection matching.
        """
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            return
        for tid, meta in self._meta.items():
            if meta["state"] == "removed":
                continue
            bbox = meta.get("bbox")
            if bbox:
                meta["bbox"] = [
                    bbox[0] + dx,
                    bbox[1] + dy,
                    bbox[2] + dx,
                    bbox[3] + dy,
                ]

    def _filter_to_pitch(self, frame: np.ndarray,
                          detections: np.ndarray) -> np.ndarray:
        """
        Remove detections outside the pitch using green colour mask.
        Never crashes — returns all detections on any failure.
        """
        if len(detections) == 0:
            return detections
        try:
            h, w = frame.shape[:2]
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(
                hsv,
                np.array([25, 40, 40]),
                np.array([95, 255, 255]),
            )
            kernel = np.ones((60, 60), np.uint8)
            pitch_mask = cv2.dilate(green_mask, kernel)

            keep = []
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = int(det[0]), int(det[1]), \
                    int(det[2]), int(det[3])
                cx = (x1 + x2) // 2
                feet_y = y1 + int((y2 - y1) * 0.75)
                cx = max(0, min(cx, w - 1))
                feet_y = max(0, min(feet_y, h - 1))
                if pitch_mask[feet_y, cx] > 0:
                    keep.append(i)

            if not keep:
                return detections
            return detections[keep]
        except Exception as e:
            logger.warning("Pitch filter failed: %s", e)
            return detections

    def _detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Run YOLO. Return numpy array shape (N, 6):
        [x1, y1, x2, y2, conf, class_id]
        """
        if self.yolo is None:
            return np.empty((0, 6))
        try:
            h, w = frame.shape[:2]
            scale = 1.0
            if w > 640:
                scale = 640.0 / w
                frame_r = cv2.resize(
                    frame, (640, int(h * scale))
                )
            else:
                frame_r = frame
            scale_back = 1.0 / scale

            results = self.yolo(
                frame_r,
                verbose=False,
                conf=YOLO_CONF,
                iou=YOLO_IOU,
                classes=[0],
            )

            dets = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    if int(box.cls) != 0:
                        continue
                    conf = float(box.conf)
                    x1, y1, x2, y2 = [
                        v * scale_back
                        for v in box.xyxy[0].tolist()
                    ]
                    x1, y1, x2, y2 = (int(x1), int(y1),
                                      int(x2), int(y2))
                    bw = x2 - x1
                    bh = y2 - y1
                    if bh < MIN_BOX_H or bw < MIN_BOX_W:
                        continue
                    if y1 < h * 0.12 and bh < 60:
                        continue
                    if bw / max(bh, 1) > MAX_ASPECT:
                        continue
                    dets.append([x1, y1, x2, y2, conf, 0])

            if not dets:
                return np.empty((0, 6))
            arr = np.array(dets, dtype=np.float32)
            arr = self._filter_to_pitch(frame, arr)
            return arr
        except Exception as e:
            logger.error("Detection failed: %s", e)
            return np.empty((0, 6))

    def _detect_team(self, frame: np.ndarray,
                      bbox: list) -> int:
        """
        HSV jersey colour with k-means 2-team clustering.
        Returns 0, 1, or -1 (unknown).
        """
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            fh, fw = frame.shape[:2]
            x1 = max(0, min(x1, fw - 1))
            x2 = max(x1 + 1, min(x2, fw))
            y1 = max(0, min(y1, fh - 1))
            y2 = max(y1 + 1, min(y2, fh))
            bh = y2 - y1
            bw = x2 - x1
            if bh < 8 or bw < 8:
                return -1

            trim = int(bw * 0.20)
            torso = frame[
                y1:y1 + int(bh * 0.50),
                x1 + trim:x2 - trim
            ]
            if torso.size == 0:
                return -1

            hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
            green = cv2.inRange(
                hsv,
                np.array([25, 20, 20]),
                np.array([95, 255, 255])
            )
            dark = hsv[:, :, 2] < 30
            jersey = ~((green > 0) | dark)
            if int(np.sum(jersey)) < 40:
                return -1

            hue = hsv[:, :, 0][jersey].astype(float)
            sat = hsv[:, :, 1][jersey].astype(float)
            val = hsv[:, :, 2][jersey].astype(float)

            sat_ratio = float(np.mean(sat > 60))
            brightness = float(np.mean(val)) / 255.0
            hi = sat > 60
            med_hue = float(
                np.median(hue[hi]) if hi.sum() >= 5
                else np.median(hue)
            )

            feat = np.array([
                med_hue / 180.0, sat_ratio, brightness
            ])
            self._colour_features.append(feat)

            if (len(self._colour_features) >= 6
                    and self._team_centroids is None):
                data = np.array(self._colour_features)
                w = np.array([2.0, 1.5, 0.5])
                dw = data * w
                c = np.array([dw[0], dw[len(dw) // 2]])
                for _ in range(20):
                    d = np.linalg.norm(
                        dw[:, None, :] - c[None, :, :], axis=2
                    )
                    labels = np.argmin(d, axis=1)
                    for k in range(2):
                        m = labels == k
                        if m.any():
                            c[k] = dw[m].mean(axis=0)
                self._team_centroids = c

            if self._team_centroids is not None:
                fw2 = feat * np.array([2.0, 1.5, 0.5])
                d0 = float(np.linalg.norm(
                    fw2 - self._team_centroids[0]
                ))
                d1 = float(np.linalg.norm(
                    fw2 - self._team_centroids[1]
                ))
                return 0 if d0 <= d1 else 1
            return -1
        except Exception:
            return -1

    def _extract_crop(self, frame: np.ndarray,
                       bbox: list) -> Optional[str]:
        """
        Top 55% torso crop as base64 JPEG.
        Only if sharpness (Laplacian variance) >= 15.
        """
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            fh, fw = frame.shape[:2]
            x1 = max(0, min(x1, fw - 1))
            x2 = max(x1 + 1, min(x2, fw))
            y1 = max(0, min(y1, fh - 1))
            y2 = max(y1 + 1, min(y2, fh))
            bh = y2 - y1
            if bh < 30:
                return None
            bot = y1 + int(bh * 0.55)
            crop = frame[y1:bot, x1:x2]
            if crop.size == 0:
                return None
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray, cv2.CV_64F).var() < 15:
                return None
            _, buf = cv2.imencode(
                ".jpg", crop,
                [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            return base64.b64encode(buf.tobytes()).decode()
        except Exception:
            return None

    @staticmethod
    def _iou(a, b):
        """Compute IoU between two bboxes [x1,y1,x2,y2]."""
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _center_dist(a, b):
        """Euclidean distance between bbox centers."""
        cx_a = (a[0] + a[2]) / 2.0
        cy_a = (a[1] + a[3]) / 2.0
        cx_b = (b[0] + b[2]) / 2.0
        cy_b = (b[1] + b[3]) / 2.0
        return ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5

    def update(self, frame: np.ndarray,
                frame_idx: int,
                timestamp: float,
                confirmed_ids: dict) -> dict:
        """
        Process one frame.

        Steps:
        1. Estimate camera motion
        2. Shift ALL track positions (motion compensation)
        3. Detect with YOLO
        4. Update ByteTrack
        5. Sync ByteTrack output with self._meta
           - IoU primary matching
           - Center distance fallback (100px)
        6. State transitions
        7. Mark lost/removed tracks
        8. Apply confirmed_ids labels
        9. Log per-frame metrics
        10. Return track_states and newly_confirmed

        Returns:
        {
            "track_states": list,      # confirmed + lost only
            "newly_confirmed": list,   # just hit threshold
        }
        """
        if self.yolo is None or self.bytetrack is None:
            return {"track_states": [], "newly_confirmed": []}

        try:
            # 1. Estimate camera motion
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dx, dy = self._estimate_camera_motion(gray)
            self._motion_dx = dx
            self._motion_dy = dy

            # 2. Shift ALL track positions BEFORE matching
            self._shift_all_tracks(dx, dy)

            # 3. Detect
            dets = self._detect(frame)

            # 4. ByteTrack update
            if len(dets) > 0:
                bt_out = self.bytetrack.update(dets, frame)
            else:
                bt_out = self.bytetrack.update(
                    np.empty((0, 6)), frame
                )

            active_ids = set()
            matched_count = 0
            new_count = 0

            # 5. Sync output with metadata
            for row in bt_out:
                x1 = int(row[0])
                y1 = int(row[1])
                x2 = int(row[2])
                y2 = int(row[3])
                tid = int(row[4])
                conf = float(row[5])
                active_ids.add(tid)

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                bbox = [x1, y1, x2, y2]

                if tid not in self._meta:
                    # Check if any existing non-removed track
                    # matches via center distance fallback
                    best_match_tid = None
                    best_match_dist = CENTER_DIST_GATE

                    for existing_tid, meta in self._meta.items():
                        if meta["state"] == "removed":
                            continue
                        existing_bbox = meta.get("bbox")
                        if existing_bbox is None:
                            continue

                        iou = self._iou(bbox, existing_bbox)
                        if iou >= 0.25:
                            # IoU match — use this track
                            best_match_tid = existing_tid
                            break

                        # Center distance fallback
                        dist = self._center_dist(bbox, existing_bbox)
                        if dist < best_match_dist:
                            best_match_dist = dist
                            best_match_tid = existing_tid

                    if best_match_tid is not None:
                        # Re-associate with existing track
                        meta = self._meta[best_match_tid]
                        meta["bbox"] = bbox
                        meta["frames_since_seen"] = 0
                        meta["last_frame"] = frame_idx
                        meta["positions"].append(
                            [cx, cy, frame_idx]
                        )
                        n = meta["detection_count"] + 1
                        meta["detection_count"] = n
                        meta["avg_confidence"] = (
                            meta["avg_confidence"] * (n - 1) + conf
                        ) / n
                        if meta["state"] == "lost":
                            meta["state"] = (
                                "confirmed"
                                if meta["detection_count"]
                                >= TENTATIVE_THRESHOLD
                                else "tentative"
                            )
                        if meta["team_id"] is None:
                            t = self._detect_team(frame, bbox)
                            if t >= 0:
                                meta["team_id"] = t
                        if conf > 0.6 and not meta["best_crop_b64"]:
                            meta["best_crop_b64"] = \
                                self._extract_crop(frame, bbox)
                        active_ids.add(best_match_tid)
                        matched_count += 1
                    else:
                        # Genuinely new track
                        team = self._detect_team(frame, bbox)
                        crop = self._extract_crop(frame, bbox)
                        self._meta[tid] = {
                            "track_id": tid,
                            "state": "tentative",
                            "detection_count": 1,
                            "frames_since_seen": 0,
                            "positions": [[cx, cy, frame_idx]],
                            "first_frame": frame_idx,
                            "last_frame": frame_idx,
                            "team_id": team if team >= 0 else None,
                            "coach_confirmed": False,
                            "coach_label": None,
                            "best_crop_b64": crop,
                            "surfaced_to_ui": False,
                            "bbox": bbox,
                            "avg_confidence": conf,
                        }
                        new_count += 1
                else:
                    meta = self._meta[tid]
                    meta["bbox"] = bbox
                    meta["frames_since_seen"] = 0
                    meta["last_frame"] = frame_idx
                    meta["positions"].append(
                        [cx, cy, frame_idx]
                    )
                    n = meta["detection_count"] + 1
                    meta["detection_count"] = n
                    meta["avg_confidence"] = (
                        meta["avg_confidence"] * (n - 1) + conf
                    ) / n

                    if meta["state"] == "lost":
                        meta["state"] = (
                            "confirmed"
                            if meta["detection_count"]
                            >= TENTATIVE_THRESHOLD
                            else "tentative"
                        )

                    if meta["team_id"] is None:
                        t = self._detect_team(frame, bbox)
                        if t >= 0:
                            meta["team_id"] = t
                    if (conf > 0.6
                            and not meta["best_crop_b64"]):
                        meta["best_crop_b64"] = \
                            self._extract_crop(frame, bbox)
                    matched_count += 1

            # 6+7. State transitions and lost/removed marking
            newly_confirmed = []
            lost_count = 0
            confirmed_count = 0

            for tid, meta in self._meta.items():
                if tid not in active_ids:
                    meta["frames_since_seen"] += 1
                    if (meta["frames_since_seen"] > 0
                            and meta["state"] != "removed"):
                        meta["state"] = "lost"
                        lost_count += 1
                    if meta["frames_since_seen"] > MAX_LOST_FRAMES:
                        meta["state"] = "removed"
                    continue

                if (meta["state"] in ("tentative", "lost")
                        and meta["detection_count"]
                            >= TENTATIVE_THRESHOLD):
                    meta["state"] = "confirmed"
                    if not meta["surfaced_to_ui"]:
                        meta["surfaced_to_ui"] = True
                        newly_confirmed.append(meta)

                if meta["state"] == "confirmed":
                    confirmed_count += 1

            # Count lost (for logging)
            lost_count = sum(
                1 for m in self._meta.values()
                if m["state"] == "lost"
            )

            # 8. Apply coach confirmed labels
            for tid, label in confirmed_ids.items():
                if tid in self._meta:
                    self._meta[tid]["coach_confirmed"] = True
                    self._meta[tid]["coach_label"] = label
                    self._meta[tid]["state"] = "confirmed"
                    self._meta[tid]["surfaced_to_ui"] = True

            # 9. Per-frame instrumentation logging
            logger.debug(
                "Frame %d: detections=%d active_tracks=%d "
                "matched=%d new=%d lost=%d confirmed=%d",
                frame_idx, len(dets),
                len([m for m in self._meta.values()
                     if m["state"] != "removed"]),
                matched_count, new_count,
                lost_count, confirmed_count
            )

            # 10. Build response — confirmed + lost only
            track_states = [
                {
                    "track_id": m["track_id"],
                    "state": m["state"],
                    "bbox": m["bbox"],
                    "team_id": m["team_id"],
                    "coach_confirmed": m["coach_confirmed"],
                    "coach_label": m["coach_label"],
                }
                for m in self._meta.values()
                if m["state"] in ("confirmed", "lost")
            ]

            return {
                "track_states": track_states,
                "newly_confirmed": [
                    {
                        "track_id": m["track_id"],
                        "bbox": m["bbox"],
                        "team_id": m["team_id"],
                        "crop_b64": m["best_crop_b64"],
                        "first_seen_frame": m["first_frame"],
                    }
                    for m in newly_confirmed
                ],
            }

        except Exception as e:
            logger.error("Tracker.update failed: %s", e)
            return {"track_states": [], "newly_confirmed": []}

    def process_batch(self,
                       frames: list,
                       frame_indices: list,
                       timestamps: list,
                       confirmed_ids: dict) -> dict:
        """
        Call self.update() for each frame sequentially.
        Accumulate newly_confirmed across all frames.
        Return final track_states from last frame.
        """
        all_newly_confirmed = []
        last_track_states = []

        for i, frame in enumerate(frames):
            fi = (frame_indices[i]
                  if i < len(frame_indices) else i)
            ts = (timestamps[i]
                  if i < len(timestamps) else 0.0)
            result = self.update(
                frame, fi, ts, confirmed_ids
            )
            all_newly_confirmed.extend(
                result.get("newly_confirmed", [])
            )
            last_track_states = result.get(
                "track_states", []
            )

        confirmed_count = sum(
            1 for m in self._meta.values()
            if m["state"] == "confirmed"
        )
        total = len([
            m for m in self._meta.values()
            if m["state"] != "removed"
        ])

        return {
            "track_states": last_track_states,
            "newly_confirmed": all_newly_confirmed,
            "tracks_total": total,
            "tracks_confirmed": confirmed_count,
        }

    def incremental_reid_patch(self,
                                confirmed_track_id: int,
                                confirmed_label: str) -> int:
        """
        After coach confirms a track, find fragmented versions
        nearby in space and time. Merge their positions.
        Returns count of merged tracks.
        """
        confirmed = self._meta.get(confirmed_track_id)
        if not confirmed or not confirmed["positions"]:
            return 0

        first_pos = confirmed["positions"][0]
        first_frame = confirmed["first_frame"]
        merged = 0
        to_merge = []

        for tid, meta in self._meta.items():
            if tid == confirmed_track_id:
                continue
            if meta["coach_confirmed"]:
                continue
            if meta["state"] == "removed":
                continue
            if not meta["positions"]:
                continue
            if abs(meta["last_frame"] - first_frame) > 10:
                continue
            last_pos = meta["positions"][-1]
            dist = (
                (last_pos[0] - first_pos[0]) ** 2 +
                (last_pos[1] - first_pos[1]) ** 2
            ) ** 0.5
            if dist < 50:
                to_merge.append(tid)

        for tid in to_merge:
            meta = self._meta[tid]
            confirmed["positions"] = (
                meta["positions"] + confirmed["positions"]
            )
            confirmed["positions"].sort(key=lambda p: p[2])
            confirmed["first_frame"] = min(
                confirmed["first_frame"],
                meta["first_frame"]
            )
            confirmed["detection_count"] += (
                meta["detection_count"]
            )
            self._meta[tid]["state"] = "removed"
            merged += 1

        return merged

    def get_confirmed_labels(self) -> dict:
        """Return {track_id: coach_label} for coach-confirmed."""
        return {
            tid: m["coach_label"]
            for tid, m in self._meta.items()
            if m["coach_confirmed"] and m["coach_label"]
        }

    def get_track_summary(self) -> dict:
        states = [m["state"] for m in self._meta.values()]
        return {
            "total": len(self._meta),
            "confirmed": states.count("confirmed"),
            "tentative": states.count("tentative"),
            "lost": states.count("lost"),
            "removed": states.count("removed"),
            "confirmed_by_coach": sum(
                1 for m in self._meta.values()
                if m["coach_confirmed"]
            ),
        }
