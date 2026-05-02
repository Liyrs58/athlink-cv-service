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

try:
    import torchvision
    import torchvision.transforms as T
    from torchvision.models import resnet50, ResNet50_Weights
except ImportError:
    torchvision = None

# Identity + VLM imports
try:
    from services.identity_core import IdentityCore
    from services.vlm_state import VLMStateMachine, GameState
    from services.track_suppressor import TrackSuppressor
    from services.role_filter import RoleFilter
    from services.crop_quality import CropQualityGate
except (ImportError, ModuleNotFoundError):
    sys.path.insert(0, os.path.dirname(__file__))
    from identity_core import IdentityCore
    from vlm_state import VLMStateMachine, GameState
    from track_suppressor import TrackSuppressor
    from role_filter import RoleFilter
    from crop_quality import CropQualityGate


class ReIDExtractor:
    def __init__(self, device="cpu"):
        self.device = "cuda" if "cuda" in str(device) else "cpu"
        if torch.backends.mps.is_available():
            self.device = "mps"
            
        self.mode = "HSV-fallback"
        self.model = None
        
        if torchvision is not None:
            try:
                print("[ReID] loading ResNet50 appearance model...")
                self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
                self.model.fc = torch.nn.Identity() 
                self.model.to(self.device)
                self.model.eval()
                self.transform = T.Compose([
                    T.ToPILImage(),
                    T.Resize((128, 128)), # Balanced for speed
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                self.mode = "ResNet50"
                print(f"[ReID] {self.mode} fallback is generic, identity confidence limited")
            except Exception as e:
                print(f"[ReID] ResNet50 fallback failed: {e}. Using HSV.")
        else:
            print("[ReID] torchvision not found. Using HSV fallback only - same-team identity unreliable.")

    def extract(self, crops: list) -> list:
        if self.model is None or not crops:
            return []
        
        features_list = []
        # Batching for performance
        batch_size = 16
        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i:i+batch_size]
            tensors = []
            for c in batch_crops:
                if c.size == 0:
                    tensors.append(torch.zeros((3, 128, 128)))
                else:
                    tensors.append(self.transform(c))
            
            batch_tensor = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                feat = self.model(batch_tensor)
                feat = torch.nn.functional.normalize(feat, p=2, dim=1)
                features_list.extend([f.cpu().numpy() for f in feat])
        return features_list


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
        self.reid = ReIDExtractor(device=device)
        self.suppressor = TrackSuppressor()
        self.role_filter = RoleFilter()
        self.crop_quality = CropQualityGate()
        self.id_remap = {}  # de_tid -> int PID
        self._snapshot_taken = False
        self._needs_revival = False

        # Scene state tracking
        self._active_baseline = 0
        self._track_history = []
        self._soft_collapse = False
        self._soft_recovery_frames = 0
        self._streak_collapse = 0
        self._streak_recover = 0
        self._transition_frame = -1
        self._prev_is_freeze = False

        # Officials tracking for output
        self._officials_this_frame = []

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

    def _extract_torso_hsv(self, frame: np.ndarray, bbox: list) -> np.ndarray:
        """Extract HSV hue/sat histogram from torso region as a fast ReID embedding."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        
        bw, bh = x2 - x1, y2 - y1
        if bw < 8 or bh < 15:
            return np.zeros(52, dtype=np.float32)  # 36 Hue + 16 Sat bins

        # Torso: top 15-55% of height, center 60% of width
        ty1, ty2 = y1 + int(bh * 0.15), y1 + int(bh * 0.55)
        tx1, tx2 = x1 + int(bw * 0.20), x1 + int(bw * 0.80)
        
        crop = frame[ty1:ty2, tx1:tx2]
        if crop.size == 0:
            return np.zeros(52, dtype=np.float32)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Avoid pitch green
        green_mask = cv2.inRange(hsv, np.array([25, 20, 20]), np.array([95, 255, 255]))
        non_green = cv2.bitwise_not(green_mask)
        
        h_hist = cv2.calcHist([hsv], [0], non_green, [36], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], non_green, [16], [0, 256]).flatten()
        
        hist = np.concatenate([h_hist, s_hist])
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist /= norm
        return hist.astype(np.float32)

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
        
        # Extract embeddings for all detections
        embeds = []
        for bbox in bboxes:
            embeds.append(self._extract_torso_hsv(frame, bbox))
        embeds = np.array(embeds)
        
        return self.tracker.update(bboxes, scores, classes, embeds=embeds)

    def _extract_embeds(self, frame, tracks):
        """Architecture Change 3: Extract real ResNet embeddings if available + HSV."""
        if not tracks:
            return {}
            
        tid_to_crop = {}
        h, w = frame.shape[:2]
        for tr in tracks:
            x1, y1, x2, y2 = [int(v) for v in tr.bbox]
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            tid_to_crop[tr.track_id] = crop

        tids = list(tid_to_crop.keys())
        crops = [tid_to_crop[tid] for tid in tids]
        
        # Load real embeddings if model exists
        if self.reid.mode == "ResNet50":
            feats = self.reid.extract(crops)
            return {tids[i]: feats[i] for i in range(len(tids))}
        
        # Fallback to HSV
        embed_map = {}
        for tr in tracks:
            emb = self._extract_torso_hsv(frame, tr.bbox)
            if emb is not None:
                embed_map[tr.track_id] = emb
        return embed_map

    def _detect_overlay(self, frame):
        """Architecture Change 5: Detect lower-third graphics."""
        h, w = frame.shape[:2]
        # Check bottom 25%
        roi = frame[int(h*0.75):, :]
        if roi.size == 0: return False
        # High homogeneity or high-contrast horizontal edges typically signify UI
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_intensity = np.mean(np.abs(edges))
        # If very uniform (solid bar) OR very high edge intensity (text on bar)
        return edge_intensity < 2.0 or edge_intensity > 40.0

    def process_frame(self, frame, video_frame, dets, save=True):
        """
        Step A: BoTSORT runtime params patched at init + log verified
        Step B: Early snapshot via collapse guard (before hard freeze)
        Step C: Log live params
        Step D+E: Snapshot on entry, revival-first on return
        Prompt 10: Referee filter (smart multi-signal)
        Prompt 11: Crop quality gating
        """
        state = self.vlm.analyze(frame, video_frame)
        is_freeze = state.is_freeze()
        is_play = state.is_play()

        # Bug #3: Reset tracker on freeze→play transition (purge ghost tracks)
        if self._prev_is_freeze and is_play:
            self.tracker.reset()
            self.id_remap.clear()
            self.identity.reset_for_scene()
            self.role_filter.reset()
            self._snapshot_taken = False
            self._needs_revival = False
            self._track_history.clear()
            self._active_baseline = 0
            print(f"[Reset] Frame {video_frame}: freeze→play, tracker + identity reset")
            
        # BUG FIX A: Snapshot on freeze entry
        if is_freeze and not self._prev_is_freeze:
            saved = self.identity.snapshot_active(video_frame)
            self._snapshot_taken = True
            print(f"[Snapshot] Frame {video_frame}: freeze entry saved {saved} slots")
            
        self._prev_is_freeze = is_freeze

        # Run tracker (always except hard freeze)
        if not is_freeze:
            tracks = self.track(frame, dets)
            n_tracks = len(tracks)
        else:
            tracks = []
            n_tracks = 0

        # Remove raw tracks collapse checking logic here

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

        # ── Prompt 9: Suppress noisy tracks before identity ──
        tracks, _sup_stats = self.suppressor.suppress(tracks, frame, video_frame)

        # BUG FIX D: Suppress stale predicted tracks (drift guard)
        visible_tracks = [t for t in tracks if t.time_since_update <= 1]
        stale_suppressed = len(tracks) - len(visible_tracks)
        if stale_suppressed > 0 and video_frame % 30 == 0:
            print(f"[TrackerCore] F{video_frame}: {stale_suppressed} stale tracks suppressed from render/assign")

        # ── Prompt 10: Filter out referees/officials (ENABLED — smart v2) ──
        filtered_tracks, self._officials_this_frame = self.role_filter.filter(
            visible_tracks, frame, video_frame
        )
        player_tracks = filtered_tracks
        # ── Prompt 11: Score crop quality for each track ──
        quality_scores = self.crop_quality.score_batch(player_tracks, frame, video_frame)
        self.crop_quality.maybe_log(video_frame)

        # ── Architecture Change 4: Hard Field Exclusions ──
        # Final hammer to keep refs/GK out of P1-P22 slots
        assignable_tracks = [
            t for t in player_tracks
            if int(getattr(t, 'cls', 0)) == 2 # Field players only
            and (quality_scores.get(t.track_id) is None or quality_scores[t.track_id].allow_assignment)
        ]
        
        # Architecture Change 5: Overlay Guard
        is_overlay = self._detect_overlay(frame)
        h, w = frame.shape[:2]
        overlay_blocked_tids = set()
        if is_overlay:
            for t in player_tracks:
                # If box bottom is in the lower third, block memory
                if t.bbox[3] > h * 0.70:
                    overlay_blocked_tids.add(t.track_id)
            if video_frame % 30 == 0:
                print(f"[OverlayGuard] Frame {video_frame}: blocked memory updates for {len(overlay_blocked_tids)} boxes")
                
        # BUG FIX 4: Wire Team Centroids to RoleFilter
        if is_play and not getattr(self, "_teams_initialized", False) and len(assignable_tracks) >= 15:
            valid_embeds = [tr.mean_embed for tr in assignable_tracks if tr.mean_embed is not None and tr.hits > 5]
            if len(valid_embeds) >= 10:
                try:
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(valid_embeds)
                    centers = kmeans.cluster_centers_
                    self.role_filter.set_team_centroids(centers[0], centers[1])
                    self._teams_initialized = True
                    print(f"[TrackerCore] Team centroids initialized via K-Means from {len(valid_embeds)} player samples")
                except Exception as e:
                    pass

        # ── SOFT-STATE TRACKING & COLLAPSE LOGIC ──
        if is_play and not is_overlay:
            current_count = len(assignable_tracks)
            
            # Maintain rolling baseline from strong play frames only
            if current_count >= 18 and not self._soft_collapse:
                self._track_history.append(current_count)
                if len(self._track_history) > 30:
                    self._track_history.pop(0)
                self._active_baseline = max(18.0, float(sum(self._track_history)) / len(self._track_history))
                
            if self._active_baseline > 0:
                # Trigger SoftCollapse if dropped below 65% for 3+ consecutive frames
                if current_count <= 0.65 * self._active_baseline:
                    self._streak_collapse += 1
                else:
                    self._streak_collapse = 0
                    
                # Trigger SoftRecovery if recovered above 80%
                if current_count >= 0.80 * self._active_baseline:
                    self._streak_recover += 1
                else:
                    self._streak_recover = 0
                    
                if self._streak_collapse >= 3 and not self._soft_collapse:
                    saved = self.identity.snapshot_active(video_frame)
                    self._soft_collapse = True
                    self._streak_collapse = 0
                    print(f"[SoftCollapse] Frame {video_frame}: current={current_count} baseline={self._active_baseline:.1f} saved={saved}")
                    
                elif self._streak_recover >= 1 and self._soft_collapse:
                    self._soft_collapse = False
                    self._soft_recovery_frames = 60
                    self._streak_recover = 0
                    print(f"[SoftRecovery] Frame {video_frame}: current={current_count} baseline={self._active_baseline:.1f} revived=candidates → entering recovery mode")

        # Periodic telemetry
        if is_play and video_frame % 30 == 0:
            mode = "collapse" if self._soft_collapse else ("recovery" if self._soft_recovery_frames > 0 else "normal")
            print(f"[SoftState] frame={video_frame} baseline={self._active_baseline:.1f} current={len(assignable_tracks)} mode={mode} low_streak={self._streak_collapse} recovery_left={self._soft_recovery_frames}")
            
        # Hard Collapse pre-cutaway fallback
        if is_play and self._active_baseline > 0 and len(assignable_tracks) <= 0.60 * self._active_baseline and not self._snapshot_taken:
            saved = self.identity.snapshot_active(video_frame)
            self._snapshot_taken = True
            print(f"[Guard] Frame {video_frame}: hard collapse detected → early snapshot ({saved} slots)")

        # Build set of track IDs that allow memory update
        memory_ok_tids = set()
        stale_memory_blocked = 0
        for t in player_tracks:
            tid = t.track_id
            q = quality_scores.get(tid)
            
            # Block if stale OR overlay OR low quality
            if t.time_since_update > 0:
                stale_memory_blocked += 1
                continue
            if tid in overlay_blocked_tids:
                continue
            if q is not None and not q.allow_memory_update:
                continue
            if self._soft_collapse: # Block memory during collapse
                continue
                
            memory_ok_tids.add(tid)
                
        if (stale_memory_blocked > 0 or self._soft_collapse) and video_frame % 30 == 0:
            msg = f"[TrackerCore] F{video_frame}: memory updates restricted"
            if stale_memory_blocked > 0: msg += f" (stale={stale_memory_blocked})"
            if self._soft_collapse: msg += " (SOFT_COLLAPSE active)"
            print(msg)

        # Step E: Recovery on first valid play after freeze
        embed_map = self._extract_embeds(frame, assignable_tracks)
        
        # Build inputs for identity
        track_objs, positions, embeddings = [], {}, {}
        for tr in assignable_tracks:
            tid = tr.track_id
            bx = tr.bbox
            positions[tid] = ((bx[0]+bx[2])/2, (bx[1]+bx[3])/2)
            if tid in embed_map:
                embeddings[tid] = embed_map[tid]
            track_objs.append(tr)

        if is_play and len(track_objs) > 0:
            self.identity.begin_frame(video_frame)

            # First match: recovery-first if snapshot exists (hard or soft)
            revived_tids = set()
            should_revive = (self._snapshot_taken and not self._needs_revival) or (self._soft_recovery_frames > 0)
            
            if should_revive:
                revived = self.identity.revive_cost_matrix(track_objs, embeddings, positions)
                self.id_remap.update({tid: int(pid[1:]) for tid, pid in revived.items()})
                revived_tids = set(revived.keys())
                
                if not self._needs_revival: # Hard revival used
                    self._needs_revival = True
                    print(f"[Recovery] Frame {video_frame}: {len(revived)} revived from scene snapshot")
                else: # Soft recovery
                    self._soft_recovery_frames -= 1
                    if len(revived) > 0:
                        print(f"[SoftRecovery] Frame {video_frame}: {len(revived)} revived from soft snapshot")

            # Then normal assignment — pass memory_ok set for gating, EXCLUDE revived
            normal_track_objs = [t for t in track_objs if t.track_id not in revived_tids]
            
            tid_to_pid = self.identity.assign_tracks(
                normal_track_objs, embeddings, positions,
                memory_ok_tids=memory_ok_tids,
            )
            self.identity.end_frame(video_frame)
            self.identity.maybe_log(detections_count=len(dets), tracks_count=n_tracks)

            # Merge: assign_tracks only updates un-revived mappings
            for tid, pid in tid_to_pid.items():
                self.id_remap[tid] = int(pid[1:])
                
            if should_revive:
                R = len(revived_tids)
                N = len(tid_to_pid)
                U = len(normal_track_objs) - N
                if R > 0 or N > 0:
                    print(f"[RecoveryAssign] F{video_frame} revived={R} normal={N} blocked={stale_memory_blocked+len(overlay_blocked_tids)} unassigned={U}")

        if save:
            players = []
            for tr in player_tracks:
                # DETrack object
                final_pid = self.id_remap.get(tr.track_id)
                if final_pid is None:
                    continue
                x1, y1, x2, y2 = tr.bbox
                q = quality_scores.get(tr.track_id)
                players.append({
                    "trackId": final_pid,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(tr.score),
                    "class": int(tr.cls),
                    "gameState": state.value,
                    "analysis_valid": True,
                    "crop_quality": float(q.score) if q else 1.0,
                })
            self.results.append({
                "frameIndex": int(video_frame),
                "players": players,
                "detection_count": len(dets),
                "track_count": n_tracks,
                "officials_filtered": len(self._officials_this_frame),
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
