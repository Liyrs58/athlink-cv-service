"""
Stream tracker using BoxMOT BoT-SORT.
Per-session instances — no shared state.
Do NOT reimplement tracking logic.
Do NOT modify BoxMOT internals.
"""

import os
import logging
import base64
import cv2
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

TENTATIVE_THRESHOLD = 5
MAX_LOST_FRAMES = 30
YOLO_CONF = 0.25
YOLO_IOU = 0.45
MIN_BOX_H = 30
MIN_BOX_W = 10
MAX_ASPECT = 4.0
MOTION_CAP = 80
EMBEDDING_ALPHA = 0.7  # FootyVision EMA: keep 70% history, blend 30% new


class Tracker:
    """
    Per-session stateful tracker wrapping BoxMOT BoT-SORT.
    One instance per streaming session. No shared state.
    """

    def __init__(self, frame_w: int, frame_h: int, fps: float):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.fps = fps
        self.yolo = None
        self.botsort = None

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

        # Fallback motion compensation
        self._prev_gray = None

        # Team colour clustering
        self._colour_features = []
        self._team_centroids = None

    def load_model(self):
        """
        Load YOLO and BoT-SORT once. Never raise.
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
            from boxmot import BoTSORT
            self.botsort = BoTSORT(
                track_high_thresh=0.6,
                track_low_thresh=0.1,
                new_track_thresh=0.7,
                track_buffer=MAX_LOST_FRAMES,
                match_thresh=0.8,
                frame_rate=max(1, int(self.fps)),
            )
            logger.info("BoT-SORT initialised")

            # FootyVision paper optimal lambda weights (ICMIP 2024)
            # λfeat=0.3, λdist=0.3, λiou=0.3, λvel=0.1
            _lambda_map = {
                "lambda_feat": 0.3,
                "lambda_iou": 0.3,
                "lambda_dist": 0.3,
                "lambda_vel": 0.1,
            }
            applied = []
            for attr, val in _lambda_map.items():
                if hasattr(self.botsort, attr):
                    setattr(self.botsort, attr, val)
                    applied.append(attr)
            if applied:
                logger.info("FootyVision lambdas applied: %s",
                            applied)
            else:
                logger.warning(
                    "BoT-SORT does not expose lambda weight "
                    "attributes — using BoxMOT defaults"
                )
        except Exception as e:
            logger.error("BoT-SORT init failed: %s", e)

    def _estimate_motion_fallback(self,
                                   gray: np.ndarray) -> tuple:
        """
        Fallback camera motion compensation using
        Lucas-Kanade optical flow on border features.
        Used if BoT-SORT internal compensation is insufficient.
        Returns (dx, dy). Returns (0.0, 0.0) on any failure.
        """
        if self._prev_gray is None:
            self._prev_gray = gray
            return (0.0, 0.0)
        try:
            h, w = gray.shape[:2]
            mask = np.zeros(gray.shape, dtype=np.uint8)
            border = int(min(h, w) * 0.30)
            mask[:border, :] = 255
            mask[-border:, :] = 255
            mask[:, :border] = 255
            mask[:, -border:] = 255

            corners = cv2.goodFeaturesToTrack(
                self._prev_gray,
                maxCorners=150,
                qualityLevel=0.01,
                minDistance=8,
                mask=mask,
            )
            if corners is None or len(corners) < 8:
                self._prev_gray = gray
                return (0.0, 0.0)

            new_corners, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, corners, None
            )
            good_old = corners[status.ravel() == 1]
            good_new = new_corners[status.ravel() == 1]

            if len(good_old) < 6:
                self._prev_gray = gray
                return (0.0, 0.0)

            dx = float(np.median(
                good_new[:, 0] - good_old[:, 0]
            ))
            dy = float(np.median(
                good_new[:, 1] - good_old[:, 1]
            ))

            if abs(dx) > MOTION_CAP or abs(dy) > MOTION_CAP:
                logger.warning(
                    "Motion capped: dx=%.1f dy=%.1f", dx, dy
                )
                self._prev_gray = gray
                return (0.0, 0.0)

            self._prev_gray = gray
            return (dx, dy)
        except Exception as e:
            logger.warning("Motion fallback failed: %s", e)
            self._prev_gray = gray
            return (0.0, 0.0)

    def _filter_to_pitch(self, frame: np.ndarray,
                          detections: np.ndarray) -> np.ndarray:
        """
        Remove detections outside the pitch using green colour mask.
        FootyVision-style bystander removal via activation masks.
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
                # Clamp to frame bounds
                cx = max(0, min(cx, w - 1))
                feet_y = max(0, min(feet_y, h - 1))
                if pitch_mask[feet_y, cx] > 0:
                    keep.append(i)

            if not keep:
                # All filtered out — return original to avoid
                # losing all detections on a tricky frame
                return detections
            return detections[keep]
        except Exception as e:
            logger.warning("Pitch filter failed: %s", e)
            return detections

    def _detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Run YOLO. Return numpy array shape (N, 6):
        [x1, y1, x2, y2, conf, class_id]
        for use with BoT-SORT update().

        Filter:
        - class != 0 -> skip
        - bh < MIN_BOX_H -> skip
        - bw < MIN_BOX_W -> skip
        - y1 < frame_h*0.12 AND bh < 60 -> skip (stands)
        - bw/bh > MAX_ASPECT -> skip (ad boards)
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
                    x1,y1,x2,y2 = (int(x1), int(y1),
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
            # FootyVision: pitch boundary bystander removal
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
        Accumulates self._colour_features.
        Builds self._team_centroids when >= 6 samples.
        """
        try:
            x1,y1,x2,y2 = [int(v) for v in bbox]
            fh, fw = frame.shape[:2]
            x1=max(0,min(x1,fw-1)); x2=max(x1+1,min(x2,fw))
            y1=max(0,min(y1,fh-1)); y2=max(y1+1,min(y2,fh))
            bh = y2-y1; bw = x2-x1
            if bh < 8 or bw < 8:
                return -1

            trim = int(bw * 0.20)
            torso = frame[
                y1:y1+int(bh*0.50),
                x1+trim:x2-trim
            ]
            if torso.size == 0:
                return -1

            hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
            green = cv2.inRange(
                hsv,
                np.array([25, 20, 20]),
                np.array([95, 255, 255])
            )
            dark = hsv[:,:,2] < 30
            jersey = ~((green > 0) | dark)
            if int(np.sum(jersey)) < 40:
                return -1

            hue = hsv[:,:,0][jersey].astype(float)
            sat = hsv[:,:,1][jersey].astype(float)
            val = hsv[:,:,2][jersey].astype(float)

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
                c = np.array([dw[0], dw[len(dw)//2]])
                for _ in range(20):
                    d = np.linalg.norm(
                        dw[:,None,:] - c[None,:,:], axis=2
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
            x1,y1,x2,y2 = [int(v) for v in bbox]
            fh, fw = frame.shape[:2]
            x1=max(0,min(x1,fw-1)); x2=max(x1+1,min(x2,fw))
            y1=max(0,min(y1,fh-1)); y2=max(y1+1,min(y2,fh))
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

    def update(self, frame: np.ndarray,
                frame_idx: int,
                timestamp: float,
                confirmed_ids: dict) -> dict:
        """
        Process one frame.

        Steps:
        1. Detect with YOLO
        2. Update BoT-SORT — it handles Kalman, ReID,
           camera motion, two-stage association internally
        3. Sync BoT-SORT output with self._meta
        4. State transitions (tentative -> confirmed at 5)
        5. Mark lost tracks
        6. Apply confirmed_ids labels
        7. Return track_states and newly_confirmed

        Returns:
        {
            "track_states": list,      # confirmed + lost only
            "newly_confirmed": list,   # just hit threshold
        }
        """
        if self.yolo is None or self.botsort is None:
            return {"track_states": [], "newly_confirmed": []}

        try:
            # 1. Detect
            dets = self._detect(frame)

            # 2. BoT-SORT update
            # Input:  (N, 6) [x1,y1,x2,y2,conf,cls]
            # Output: (M, 8) [x1,y1,x2,y2,id,conf,cls,idx]
            if len(dets) > 0:
                bt_out = self.botsort.update(dets, frame)
            else:
                bt_out = self.botsort.update(
                    np.empty((0, 6)), frame
                )

            active_ids = set()

            # 3. Sync output with metadata
            for row in bt_out:
                x1 = int(row[0]); y1 = int(row[1])
                x2 = int(row[2]); y2 = int(row[3])
                tid = int(row[4])
                conf = float(row[5])
                active_ids.add(tid)

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                bbox = [x1, y1, x2, y2]

                if tid not in self._meta:
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
                        "embedding": None,
                    }
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
                        meta["avg_confidence"] * (n-1) + conf
                    ) / n

                    # FootyVision embedding EMA update
                    # BoT-SORT rows: [x1,y1,x2,y2,id,conf,cls,idx,...]
                    # Columns 8+ may contain embedding data
                    if len(row) > 8:
                        try:
                            new_emb = np.array(
                                row[8:], dtype=np.float32
                            )
                            if new_emb.size > 0:
                                if meta["embedding"] is None:
                                    meta["embedding"] = new_emb
                                else:
                                    meta["embedding"] = (
                                        EMBEDDING_ALPHA
                                        * meta["embedding"]
                                        + (1 - EMBEDDING_ALPHA)
                                        * new_emb
                                    )
                        except Exception:
                            pass  # embeddings not available

                    if meta["team_id"] is None:
                        t = self._detect_team(frame, bbox)
                        if t >= 0:
                            meta["team_id"] = t
                    if (conf > 0.6
                            and not meta["best_crop_b64"]):
                        meta["best_crop_b64"] = \
                            self._extract_crop(frame, bbox)

            # 4+5. State transitions and lost marking
            newly_confirmed = []
            for tid, meta in self._meta.items():
                if tid not in active_ids:
                    meta["frames_since_seen"] += 1
                    if meta["frames_since_seen"] > 0:
                        if meta["state"] not in (
                            "removed",
                        ):
                            meta["state"] = "lost"
                    if meta["frames_since_seen"] > MAX_LOST_FRAMES:
                        meta["state"] = "removed"
                    continue

                if (meta["state"] == "tentative"
                        and meta["detection_count"]
                            >= TENTATIVE_THRESHOLD):
                    meta["state"] = "confirmed"
                    if not meta["surfaced_to_ui"]:
                        meta["surfaced_to_ui"] = True
                        newly_confirmed.append(meta)

            # 6. Apply coach confirmed labels
            for tid, label in confirmed_ids.items():
                if tid in self._meta:
                    self._meta[tid]["coach_confirmed"] = True
                    self._meta[tid]["coach_label"] = label
                    self._meta[tid]["state"] = "confirmed"
                    self._meta[tid]["surfaced_to_ui"] = True

            # 7. Build response — confirmed + lost only
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
                (last_pos[0] - first_pos[0])**2 +
                (last_pos[1] - first_pos[1])**2
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
