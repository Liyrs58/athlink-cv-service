import json
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
import boxmot
import torch
import sys
import os

# Handle relative imports for both local and Colab environments
try:
    from services.vlm_state import VLMStateMachine, GameState
except (ImportError, ModuleNotFoundError):
    # Colab fallback: import from same directory
    sys.path.insert(0, os.path.dirname(__file__))
    from vlm_state import VLMStateMachine, GameState

class TrackerCore:
    def __init__(self, yolo_path, reid_path, device="cpu"):
        self.device = device
        # YOLO wants "cuda", boxmot wants "0" — normalise both
        yolo_device = "cuda" if device in ("0", "cuda") else "cpu"
        boxmot_device = "0" if device in ("0", "cuda") else "cpu"

        # Detection – YOLO only every N frames
        self.yolo = YOLO(yolo_path)
        self.yolo.to(yolo_device)

        # Tracking – runs EVERY frame
        self.tracker = boxmot.BotSort(
            reid_weights=Path(reid_path),
            device=boxmot_device,
            half=True,
            track_buffer=300,          # survive bench shots
            match_thresh=0.30,         # broadcast sports: loose IoU matching
            proximity_thresh=0.5,
            appearance_thresh=0.25,    # ReID bridges motion gaps
            track_high_thresh=0.6,     # confirmed detections only
            track_low_thresh=0.1,
            new_track_thresh=0.7,      # prevent phantom tracks
            cmc_method="sparseOptFlow", # camera motion compensation (broadcast pans)
            frame_rate=25,
        )

        # VLM state machine for game state detection
        self.vlm_sm = VLMStateMachine(device=device)
        self.id_override_map = {}  # Maps new_tid -> old_tid after resume

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
        """Track and optionally save results, with VLM state filtering."""
        # Analyze game state every 50 frames
        state = self.vlm_sm.analyze(frame, video_frame)

        # If bench shot or paused, freeze active tracks
        if state in (GameState.BENCH_SHOT, GameState.PAUSED, GameState.INJURY):
            if self.vlm_sm.prev_state == GameState.PLAY:
                # Transitioning to pause — freeze current tracks
                current_tracks = self.tracker.tracks
                self.vlm_sm.freeze_tracks(current_tracks, video_frame)
            # Skip tracker update during pause
            tracks = self.vlm_sm.frozen_tracks.values() if self.vlm_sm.frozen_tracks else []
        elif state == GameState.PLAY and self.vlm_sm.prev_state in (GameState.BENCH_SHOT, GameState.PAUSED, GameState.INJURY):
            # Resuming from pause — match visible players to frozen roster
            tracks = self.track(frame, dets)
            self.id_override_map = self.vlm_sm.resume_tracks(tracks, video_frame)
        else:
            # Normal tracking
            tracks = self.track(frame, dets)

        if save:
            players = []
            for t in tracks:
                if len(t) < 8:
                    continue
                x1, y1, x2, y2, tid, conf, cls, _ = t

                # Apply ID override if resuming from pause
                final_tid = int(tid)
                if final_tid in self.id_override_map:
                    final_tid = self.id_override_map[final_tid]

                players.append({
                    "trackId": final_tid,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(conf),
                    "class": int(cls),
                    "gameState": state.value
                })
            self.results.append({
                "frameIndex": int(video_frame),
                "players": players,
                "detection_count": len(dets),
                "track_count": len(tracks),
                "gameState": state.value
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
        yolo_path="models/yolov8n-pose.pt",
        reid_path="models/osnet_x1_0_msmt17.pt",   # better ReID model
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
