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

# VLM state machine import
try:
    from services.vlm_state import VLMStateMachine, GameState
except (ImportError, ModuleNotFoundError):
    sys.path.insert(0, os.path.dirname(__file__))
    from vlm_state import VLMStateMachine, GameState


class TrackerCore:
    def __init__(self, yolo_path, reid_path, device="cpu"):
        self.device = device

        yolo_path = os.path.abspath(yolo_path)
        reid_path = os.path.abspath(reid_path)

        print(f"[TrackerCore] YOLO: {yolo_path} (exists: {os.path.exists(yolo_path)})")
        print(f"[TrackerCore] ReID: {reid_path} (exists: {os.path.exists(reid_path)})")

        yolo_device = "cuda" if device in ("0", "cuda") else "cpu"
        boxmot_device = "0" if device in ("0", "cuda") else "cpu"

        self.yolo = YOLO(yolo_path)
        self.yolo.to(yolo_device)

        if BotSort is None:
            raise RuntimeError("BotSort not available")

        self.tracker = BotSort(
            reid_weights=Path(reid_path),
            device=boxmot_device,
            half=True,
            # --- Core association ---
            match_thresh=0.65,         # IoU ≥ 0.35 (relaxed from 0.7 for better persistence)
            asso_func="iou",
            per_class=False,
            # --- Track lifecycle ---
            track_high_thresh=0.45,    # slightly relaxed for more confirmed tracks
            track_low_thresh=0.1,      # BYTE low-conf recovery
            new_track_thresh=0.55,     # spawn slightly easier
            track_buffer=150,          # 6s @ 25fps — survive longer pans
            max_age=150,               # match track_buffer
            det_thresh=0.3,
            # --- ReID ---
            appearance_thresh=0.20,    # more permissive ReID matching
            proximity_thresh=0.5,
            # --- Camera motion ---
            cmc_method="ecc",
            frame_rate=25,
        )

        # VLM: game state + permanent player registry
        self.vlm = VLMStateMachine(device=device)
        self.id_remap = {}
        self._last_tracks = []

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
            if cls in [1, 2, 3]:
                dets.append([
                    float(box.xyxy[0][0]), float(box.xyxy[0][1]),
                    float(box.xyxy[0][2]), float(box.xyxy[0][3]),
                    float(box.conf.item()), float(cls)
                ])
        return np.array(dets, dtype=float) if dets else np.empty((0, 6))

    def track(self, frame, dets):
        return self.tracker.update(dets, frame)

    def process_frame(self, frame, video_frame, dets, save=True):
        """
        Track + permanent player ID assignment via PlayerRegistry.

        Key behavior:
        - PLAY: run BoT-SORT + registry remapping
        - BENCH_SHOT: skip tracker entirely (don't feed empty dets that kill tracks)
        - bench→play: registry flushes stale mappings, re-matches new IDs to slots
        """
        state = self.vlm.analyze(frame, video_frame)

        # Full freeze: field not visible — skip tracker entirely
        if state.is_freeze():
            if save:
                self.results.append({
                    "frameIndex": int(video_frame),
                    "players": [],
                    "detection_count": 0,
                    "track_count": 0,
                    "gameState": state.value,
                })
            return 0

        # Run BotSort (both PLAY and CELEBRATION)
        tracks = self.track(frame, dets)

        # Extract ReID embeddings from BoT-SORT track objects
        embed_map = {}
        if hasattr(self.tracker, 'active_tracks'):
            for strack in self.tracker.active_tracks:
                if strack.is_activated and hasattr(strack, 'smooth_feat') and strack.smooth_feat is not None:
                    embed_map[strack.id] = strack.smooth_feat.copy()

        # Identity assignment only during PLAY (not celebration — freeze assignment)
        if state.is_play() and len(tracks) > 0:
            self.id_remap = self.vlm.get_id_remap(frame, tracks, video_frame, embed_map)

        if save:
            players = []
            for t in tracks:
                if len(t) < 8:
                    continue
                x1, y1, x2, y2, tid, conf, cls, _ = t
                tid_int = int(tid)

                # Check referee registry first
                if tid_int in self.vlm.registry.botsort_to_ref:
                    rid = self.vlm.registry.botsort_to_ref[tid_int]
                    players.append({
                        "trackId": f"R{rid}",
                        "team": "REF",
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": float(conf),
                        "class": int(cls),
                        "gameState": state.value,
                    })
                    continue

                final_tid = self.id_remap.get(tid_int, None)
                if final_tid is None:
                    continue

                slot = self.vlm.registry.slots.get(final_tid)
                team = slot.team if slot else "UNK"

                players.append({
                    "trackId": final_tid,
                    "team": team,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(conf),
                    "class": int(cls),
                    "gameState": state.value,
                })
            self.results.append({
                "frameIndex": int(video_frame),
                "players": players,
                "detection_count": len(dets),
                "track_count": len(tracks),
                "gameState": state.value,
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


def run_tracking(video_path, job_id, frame_stride=1, max_frames=None, device="cpu"):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if os.path.exists("/content/roboflow_players.pt"):
        yolo_path = "/content/roboflow_players.pt"
        reid_path = "/content/athlink-cv-service/models/osnet_x1_0_msmt17.pt"
    else:
        yolo_path = "models/roboflow_players.pt"
        reid_path = "models/osnet_x1_0_msmt17.pt"

    tracker = TrackerCore(yolo_path=yolo_path, reid_path=reid_path, device=device)

    video_frame = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if video_frame % frame_stride == 0:
            dets = tracker.detect(frame)
            save_this = True
        else:
            dets = np.empty((0, 6))
            save_this = False

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
        frame_stride=1,
        device=device,
    )
    print(f"Total time: {time.time()-t0:.1f}s")
