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
    from services.identity_core import IdentityCore
    from services.vlm_state import VLMStateMachine, GameState
except (ImportError, ModuleNotFoundError):
    sys.path.insert(0, os.path.dirname(__file__))
    from identity_core import IdentityCore
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
            asso_func="iou",
            per_class=False,
            max_age=150,
            det_thresh=0.3,
            proximity_thresh=0.5,
            cmc_method="ecc",
            # Target params (Step A: correct init values)
            match_thresh=0.30,
            appearance_thresh=0.55,
            new_track_thresh=0.40,
            track_high_thresh=0.50,
            track_low_thresh=0.10,
            track_buffer=150,
            frame_rate=25,
        )

        # Step A: patch runtime object for all candidate nested paths
        self._patch_botsort_params()

        # VLM: game state only (scene detection)
        self.vlm = VLMStateMachine(device=device)
        # Identity core: deterministic ID assignment via ReID + position
        self.identity = IdentityCore()
        self.id_remap = {}  # botsort_tid -> int PID
        self._snapshot_taken = False
        self._needs_revival = False

        # Step B+C: scene state tracking
        self._scene_state = "play"
        self._active_baseline = 0
        self._track_history = []
        self._transition_frame = -1

        self.frame_idx = 0
        self.results = []

        # Log initial params
        self._log_botsort_params()

    def _patch_botsort_params(self):
        """Step A: Patch actual runtime object used during update()."""
        params = {
            'match_thresh': 0.30,
            'appearance_thresh': 0.55,
            'new_track_thresh': 0.40,
            'track_high_thresh': 0.50,
            'track_low_thresh': 0.10,
            'track_buffer': 150,
            'frame_rate': 25,
        }
        for candidate in [self.tracker, getattr(self.tracker, 'tracker', None),
                         getattr(self.tracker, 'args', None)]:
            if candidate is None:
                continue
            for key, val in params.items():
                if hasattr(candidate, key):
                    setattr(candidate, key, val)

    def _log_botsort_params(self):
        """Step C: Log params from actual matcher object."""
        t = self.tracker
        params = {k: getattr(t, k, None) for k in [
            'match_thresh', 'appearance_thresh', 'new_track_thresh',
            'track_high_thresh', 'track_low_thresh', 'track_buffer', 'frame_rate'
        ]}
        print(f"[TrackerCore] Live BotSort params: {params}")

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

    def _extract_embeds(self):
        embed_map = {}
        try:
            src = getattr(self.tracker, 'active_tracks',
                          getattr(self.tracker, 'tracked_stracks', []))
            for strack in src:
                sid = getattr(strack, 'track_id', getattr(strack, 'id', None))
                feat = getattr(strack, 'smooth_feat', None)
                if sid is not None and feat is not None:
                    embed_map[int(sid)] = feat.copy()
        except Exception:
            pass
        return embed_map

    def _build_track_inputs(self, tracks, embed_map):
        track_objs, positions, embeddings = [], {}, {}
        for t in tracks:
            if len(t) < 5:
                continue
            tid = int(t[4])
            cx = (float(t[0]) + float(t[2])) / 2
            cy = (float(t[1]) + float(t[3])) / 2
            positions[tid] = (cx, cy)
            if tid in embed_map:
                embeddings[tid] = embed_map[tid]

            class _T:
                pass
            obj = _T()
            obj.track_id = tid
            track_objs.append(obj)
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
            for t in tracks:
                if len(t) < 8:
                    continue
                x1, y1, x2, y2, tid, conf, cls, _ = t
                tid_int = int(tid)
                final_pid = self.id_remap.get(tid_int)
                if final_pid is None:
                    continue
                players.append({
                    "trackId": final_pid,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(conf),
                    "class": int(cls),
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
