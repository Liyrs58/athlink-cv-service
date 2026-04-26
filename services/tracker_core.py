import json
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
import boxmot
import torch

class TrackerCore:
    def __init__(self, yolo_path, reid_path, device="cpu"):
        self.device = device
        # Detection – YOLO only every N frames
        self.yolo = YOLO(yolo_path)
        self.yolo.to(device)

        # Tracking – runs EVERY frame
        self.tracker = boxmot.BotSort(
            reid_weights=Path(reid_path),
            device=device,
            half=False,
            track_buffer=150,          # survive 6s bench shots
            match_thresh=0.85,         # distance = 1-IoU, so IoU ≥ 0.15
            proximity_thresh=0.5,
            appearance_thresh=0.25,    # ReID bridges motion gaps
            track_high_thresh=0.03,    # ALL detections become high‑conf
            track_low_thresh=0.01,
            new_track_thresh=0.03,     # any detection can spawn a new track
            cmc_method="ecc",          # boxmot v12: gmc_method → cmc_method
            frame_rate=25,
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

    tracker = TrackerCore(
        yolo_path="models/roboflow_players.pt",
        reid_path="models/osnet_x0_25_msmt17.pt",   # fast CPU model
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
