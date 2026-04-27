import json
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
import torch
import sys
import os

# BotSort import with fallback
try:
    from boxmot import BotSort
except (ImportError, AttributeError):
    try:
        from boxmot.trackers.botsort.botsort import BotSort
    except ImportError:
        print("[ERROR] BotSort import failed")
        BotSort = None


class TrackerCore:
    def __init__(self, yolo_path, reid_path, device="cpu"):
        self.device = device

        # Resolve paths absolutely
        yolo_path = os.path.abspath(yolo_path)
        reid_path = os.path.abspath(reid_path)

        print(f"[TrackerCore] YOLO: {yolo_path} (exists: {os.path.exists(yolo_path)})")
        print(f"[TrackerCore] ReID: {reid_path} (exists: {os.path.exists(reid_path)})")

        # YOLO wants "cuda", boxmot wants "0" — normalise both
        yolo_device = "cuda" if device in ("0", "cuda") else "cpu"
        boxmot_device = "0" if device in ("0", "cuda") else "cpu"

        # Detection – YOLO only every N frames
        self.yolo = YOLO(yolo_path)
        self.yolo.to(yolo_device)

        # Tracking – runs EVERY frame
        if BotSort is None:
            raise RuntimeError("BotSort not available")

        self.tracker = BotSort(
            reid_weights=Path(reid_path),
            device=boxmot_device,
            half=True,
            track_buffer=60,           # match max_age
            match_thresh=0.25,         # tighter IoU for first association
            proximity_thresh=0.5,
            appearance_thresh=0.15,    # relaxed: allow more ReID matches
            track_high_thresh=0.6,
            track_low_thresh=0.1,
            new_track_thresh=0.65,     # slightly lower
            cmc_method="ecc",
            frame_rate=25,
            per_class=True,            # track teams separately
            asso_func="ciou",          # better for fast motion
            max_age=60,                # 2s occlusion survival
            det_thresh=0.25,           # catch distant players
        )

        self.frame_idx = 0
        self.results = []

    def detect(self, frame):
        """YOLO – returns np.array (N,6) [x1,y1,x2,y2,conf,cls]."""
        results = self.yolo.predict(frame, conf=0.05, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 6))

        dets = []
        for box in boxes:
            cls = int(box.cls.item())
            if cls in [1, 2, 3]:   # player classes (0 = ball)
                dets.append([
                    float(box.xyxy[0][0]), float(box.xyxy[0][1]),
                    float(box.xyxy[0][2]), float(box.xyxy[0][3]),
                    float(box.conf.item()), float(cls)
                ])
        return np.array(dets, dtype=float) if dets else np.empty((0, 6))

    def track(self, frame, dets):
        """BoT‑SORT update – always call, even with empty dets."""
        return self.tracker.update(dets, frame)

    def process_frame(self, frame, video_frame, dets, save=True):
        """Track and optionally save results."""
        tracks = self.track(frame, dets)

        if save:
            players = []
            for t in tracks:
                if len(t) < 8:
                    continue
                x1, y1, x2, y2, tid, conf, cls, _ = t
                players.append({
                    "trackId": int(tid),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(conf),
                    "class": int(cls)
                })
            self.results.append({
                "frameIndex": int(video_frame),
                "players": players,
                "detection_count": len(dets),
                "track_count": len(tracks)
            })

        return len(tracks) if len(tracks) > 0 else 0

    def save(self, job_id):
        out = {
            "jobId": job_id,
            "frames": self.results,
            "total_frames": len(self.results),
        }
        path = Path(f"temp/{job_id}/tracking/track_results.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        return path


def run_tracking(video_path, job_id, frame_stride=5, max_frames=None, device="cpu"):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Resolve model paths (Colab: /content/, local: ./models/)
    if os.path.exists("/content/roboflow_players.pt"):
        yolo_path = "/content/roboflow_players.pt"
        reid_path = "/content/athlink-cv-service/models/osnet_x1_0_msmt17.pt"
    else:
        # Local: use relative paths
        yolo_path = "models/roboflow_players.pt"
        reid_path = "models/osnet_x1_0_msmt17.pt"

    tracker = TrackerCore(
        yolo_path=yolo_path,
        reid_path=reid_path,
        device=device,
    )

    video_frame = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO only on stride frames
        if video_frame % frame_stride == 0:
            dets = tracker.detect(frame)
            save_this = True
        else:
            dets = np.empty((0, 6))
            save_this = False

        # Track EVERY frame (Kalman coasting keeps IDs during pans)
        n_tracks = tracker.process_frame(frame, video_frame, dets, save=save_this)

        if save_this:
            processed += 1
            if processed % 10 == 0:
                print(f"Video frame {video_frame:4d} | processed {processed:4d} | "
                      f"dets {len(dets):3d} | tracks {n_tracks:3d}")

        if max_frames and processed >= max_frames:
            break

        video_frame += 1

    cap.release()
    tracker.save(job_id)
    print(f"Done. Video frames: {video_frame}, processed: {processed}")
    return tracker.results


if __name__ == "__main__":
    import time
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    t0 = time.time()
    run_tracking(
        video_path="/Users/rudra/Downloads/1b16c594_villa_psg_40s_new.mp4",
        job_id="spawn_test",
        frame_stride=10,
        max_frames=None,
        device=device,
    )
    print(f"Total time: {time.time()-t0:.1f}s")
