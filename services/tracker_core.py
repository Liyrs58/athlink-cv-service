import json
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
import torch
import sys
import os

# Deep-EIoU tracker (replaces BoT-SORT)
try:
    from services.deep_eiou import DeepEIoUTracker, GTALink, build_tracklet_summaries
except (ImportError, ModuleNotFoundError):
    sys.path.insert(0, os.path.dirname(__file__))
    from deep_eiou import DeepEIoUTracker, GTALink, build_tracklet_summaries

# Identity + VLM imports
try:
    from services.identity_core import IdentityCore
    from services.vlm_state import VLMStateMachine, GameState
except (ImportError, ModuleNotFoundError):
    sys.path.insert(0, os.path.dirname(__file__))
    from identity_core import IdentityCore
    from vlm_state import VLMStateMachine, GameState


class TrackerCore:
    def __init__(self, yolo_path, device="cpu"):
        self.device = device

        yolo_path = os.path.abspath(yolo_path)
        print(f"[TrackerCore] YOLO: {yolo_path} (exists: {os.path.exists(yolo_path)})")

        yolo_device = "cuda" if device in ("0", "cuda") else "cpu"
        self.yolo = YOLO(yolo_path)
        self.yolo.to(yolo_device)

        # Deep-EIoU tracker (no external ReID model needed — uses YOLO crops via identity)
        self.tracker = DeepEIoUTracker(
            track_high_thresh=0.50,
            track_low_thresh=0.10,
            new_track_thresh=0.40,
            match_thresh=0.70,
            iou_only_thresh=0.80,
            max_age=150,
            min_hits=3,
            expand_r1=1.5,
            expand_r2=2.0,
        )
        print("[TrackerCore] Deep-EIoU tracker initialised")

        # VLM: game state only (scene detection)
        self.vlm = VLMStateMachine(device=device)
        # Identity core: deterministic ID assignment via ReID + position
        self.identity = IdentityCore()
        self.id_remap = {}  # de_tid -> int PID
        self._snapshot_taken = False
        self._needs_revival = False

        # Scene state tracking
        self._active_baseline = 0
        self._track_history = []
        self._transition_frame = -1

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
        """Run Deep-EIoU. dets is (N,6) [x1,y1,x2,y2,conf,cls]. Returns list of DETrack."""
        if len(dets) == 0:
            # Still call update so Kalman predicts and ages tracks
            return self.tracker.update(
                np.empty((0, 4)), np.empty((0,)), np.empty((0,)), None
            )
        bboxes  = dets[:, :4]
        scores  = dets[:, 4]
        classes = dets[:, 5]
        return self.tracker.update(bboxes, scores, classes, embeds=None)

    def _extract_embeds(self):
        """Extract mean embeddings from active DETrack objects."""
        embed_map = {}
        for tr in self.tracker.active_tracks:
            emb = tr.mean_embed
            if emb is not None:
                embed_map[tr.track_id] = emb
        return embed_map

    def _build_track_inputs(self, tracks, embed_map):
        """tracks is list of DETrack objects."""
        track_objs, positions, embeddings = [], {}, {}
        for tr in tracks:
            tid = tr.track_id
            bbox = tr.bbox  # [x1,y1,x2,y2]
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            positions[tid] = (cx, cy)
            if tid in embed_map:
                embeddings[tid] = embed_map[tid]
            track_objs.append(tr)  # DETrack has .track_id already
        return track_objs, positions, embeddings

    def _update_collapse_guard(self, n_tracks):
        """Priority B: detect pre-cutaway track collapse."""
        self._track_history.append(n_tracks)
        if len(self._track_history) > 20:
            self._track_history.pop(0)

        # Set baseline from first 30 frames
        if len(self._track_history) == 30 and self._baseline_tracks == 0:
            self._baseline_tracks = max(1, sum(self._track_history) / 30)

        if self._baseline_tracks > 0 and len(self._track_history) >= 10:
            recent_mean = sum(self._track_history[-10:]) / 10
            self._suspend_birth = recent_mean < 0.5 * self._baseline_tracks
        else:
            self._suspend_birth = False

    def process_frame(self, frame, video_frame, dets, save=True):
        """
        Step A: BoTSORT runtime params patched at init + log verified
        Step B: Early snapshot via collapse guard (before hard freeze)
        Step C: Log live params
        Step D+E: Snapshot on entry, revival-first on return
        """
        state = self.vlm.analyze(frame, video_frame)
        is_freeze = state.is_freeze()
        is_play = state.is_play()

        # Run BotSort (always except hard freeze)
        if not is_freeze:
            tracks = self.track(frame, dets)
            n_tracks = len(tracks)
        else:
            tracks = []
            n_tracks = 0

        # Step B: Collapse guard — detect pre-cutaway state BEFORE hard freeze
        if is_play:
            self._track_history.append(n_tracks)
            if len(self._track_history) > 20:
                self._track_history.pop(0)

            # Establish baseline (first 30 valid frames)
            if len(self._track_history) == 30 and self._active_baseline == 0:
                self._active_baseline = max(1, sum(self._track_history) / 30)

            # Detect collapse
            if self._active_baseline > 0 and len(self._track_history) >= 10:
                recent_mean = sum(self._track_history[-10:]) / 10
                active_drop = recent_mean / max(self._active_baseline, 1)

                # Step D: Early snapshot if collapse detected (not waiting for hard freeze)
                if active_drop <= 0.70 and not self._snapshot_taken:
                    saved = self.identity.snapshot_active(video_frame)
                    self._snapshot_taken = True
                    self._transition_frame = video_frame
                    print(f"[Guard] Frame {video_frame}: active_drop={active_drop:.2f} "
                          f"→ early snapshot ({saved} slots)")

        # Step A: hard freeze scene — only age, no association
        if is_freeze:
            self.identity.begin_frame(video_frame)
            self.identity.end_frame(video_frame)
            if save:
                self.results.append({
                    "frameIndex": int(video_frame),
                    "players": [],
                    "detection_count": 0,
                    "track_count": 0,
                    "gameState": state.value,
                    "analysis_valid": False,
                })
            return 0

        # Step E: Recovery on first valid play after freeze
        embed_map = self._extract_embeds()
        track_objs, positions, embeddings = self._build_track_inputs(tracks, embed_map)

        if is_play and len(track_objs) > 0:
            self.identity.begin_frame(video_frame)

            # First match: recovery-first if snapshot exists
            if self._snapshot_taken and not self._needs_revival:
                revived = self.identity.revive_cost_matrix(track_objs, embeddings, positions)
                self.id_remap = {tid: int(pid[1:]) for tid, pid in revived.items()}
                self._needs_revival = True  # only try once
                n_revived = len(revived)
                print(f"[Recovery] Frame {video_frame}: {n_revived} revived from snapshot")

            # Then normal assignment
            tid_to_pid = self.identity.assign_tracks(track_objs, embeddings, positions)
            self.identity.end_frame(video_frame)
            self.identity.maybe_log(detections_count=len(dets), tracks_count=n_tracks)

            # Merge: assign_tracks wins
            for tid, pid in tid_to_pid.items():
                self.id_remap[tid] = int(pid[1:])

        if save:
            players = []
            for tr in tracks:
                # DETrack object
                final_pid = self.id_remap.get(tr.track_id)
                if final_pid is None:
                    continue
                x1, y1, x2, y2 = tr.bbox
                players.append({
                    "trackId": final_pid,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(tr.score),
                    "class": int(tr.cls),
                    "gameState": state.value,
                    "analysis_valid": True,
                })
            self.results.append({
                "frameIndex": int(video_frame),
                "players": players,
                "detection_count": len(dets),
                "track_count": n_tracks,
                "gameState": state.value,
                "analysis_valid": True,
            })

        return n_tracks

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
    else:
        yolo_path = "models/roboflow_players.pt"

    tracker = TrackerCore(yolo_path=yolo_path, device=device)

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
