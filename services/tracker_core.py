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

# Identity + VLM imports
try:
    from services.identity_core import IdentityCore, Track as IdentityTrack
    from services.vlm_state import VLMStateMachine, GameState
except (ImportError, ModuleNotFoundError):
    sys.path.insert(0, os.path.dirname(__file__))
    from identity_core import IdentityCore, Track as IdentityTrack
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

        # VLM: game state only (scene detection)
        self.vlm = VLMStateMachine(device=device)
        # Identity core: deterministic, GPU-optimized ID assignment
        self.identity = IdentityCore(max_players=22)
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
        Track + permanent player ID assignment via IdentityCore.

        Key behavior:
        - PLAY: run BoT-SORT + identity matching
        - CELEBRATION: run BotSort but freeze identity assignment
        - BENCH_SHOT: skip tracker entirely
        """
        state = self.vlm.analyze(frame, video_frame)

        # Full freeze: field not visible — skip tracker entirely
        if state.is_freeze():
            self.identity.freeze_on_bench(video_frame)
            if save:
                self.results.append({
                    "frameIndex": int(video_frame),
                    "players": [],
                    "detection_count": 0,
                    "track_count": 0,
                    "gameState": state.value,
                })
            return 0

        # Resume from freeze
        if state.is_play():
            self.identity.unfreeze_on_play(video_frame)

        # Run BotSort (both PLAY and CELEBRATION)
        tracks = self.track(frame, dets)

        # Extract ReID embeddings from BoT-SORT track objects (batched)
        embed_map = {}
        if hasattr(self.tracker, 'active_tracks'):
            for strack in self.tracker.active_tracks:
                if strack.is_activated and hasattr(strack, 'smooth_feat') and strack.smooth_feat is not None:
                    embed_map[strack.id] = strack.smooth_feat.copy()

        # Convert BotSort tracks to IdentityCore format
        identity_tracks = []
        for t in tracks:
            if len(t) < 7:
                continue
            tid = int(t[4])
            cls = int(t[6])
            bbox = np.array([float(t[0]), float(t[1]), float(t[2]), float(t[3])])
            conf = float(t[5])
            emb = embed_map.get(tid)
            identity_tracks.append(IdentityTrack(tid=tid, bbox=bbox, conf=conf, cls=cls, emb=emb))

        # Identity assignment only during PLAY (not celebration)
        if state.is_play() and len(identity_tracks) > 0:
            self.id_remap = self.identity.update(identity_tracks, video_frame)

        if save:
            players = []
            for t in tracks:
                if len(t) < 8:
                    continue
                x1, y1, x2, y2, tid, conf, cls, _ = t
                tid_int = int(tid)

                final_pid = self.id_remap.get(tid_int, None)
                if final_pid is None:
                    continue

                player = self.identity.players.get(final_pid)
                if player is None:
                    continue

                team_str = player.team.name if player.team.value >= 0 else "UNK"

                players.append({
                    "trackId": final_pid,
                    "team": team_str,
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
