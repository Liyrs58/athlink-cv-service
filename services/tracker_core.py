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
    """
    Priority order:
      1. OSNet (torchreid) — 512-D person-specific ReID, best for same-team
      2. ResNet50 (torchvision) — 2048-D generic ImageNet, limited same-team
      3. HSV histogram — 52-D color, team-level only

    OSNet weights: /content/osnet_x1_0_msmt17.pt (upload to Colab)
    """
    OSNET_PATH = "/content/osnet_x1_0_msmt17.pt"

    def __init__(self, device="cpu"):
        self.device = "cuda" if "cuda" in str(device) else "cpu"
        if torch.backends.mps.is_available():
            self.device = "mps"

        self.mode = "HSV-fallback"
        self.model = None
        self.transform = None
        self.feat_dim = 52  # HSV fallback dim

        # Try OSNet first
        self._try_load_osnet()

        # Fallback: ResNet50
        if self.model is None and torchvision is not None:
            self._try_load_resnet50()

        if self.model is None:
            print("[ReID] torchvision not found. Using HSV fallback only — same-team identity unreliable.")

    def _try_load_osnet(self):
        if not os.path.exists(self.OSNET_PATH):
            print(f"[ReID] OSNet weights not found at {self.OSNET_PATH} — skipping")
            return
        try:
            import torchreid

            # Load raw checkpoint and unwrap if needed
            ckpt = torch.load(self.OSNET_PATH, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

            # Strip "module." prefix (DataParallel checkpoints)
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            # Detect actual num_classes from checkpoint classifier head
            num_classes = 1000
            if "classifier.weight" in state_dict:
                num_classes = state_dict["classifier.weight"].shape[0]
            elif "classifier.0.weight" in state_dict:
                num_classes = state_dict["classifier.0.weight"].shape[0]
            print(f"[ReID] OSNet checkpoint num_classes={num_classes} (classifier head will be ignored for inference)")

            # Build model with matching num_classes so there's no shape error,
            # then we swap the head with Identity to extract backbone features only.
            self.model = torchreid.models.build_model(
                name="osnet_x1_0",
                num_classes=num_classes,
                pretrained=False,
            )
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[ReID] OSNet missing keys (non-classifier): {missing[:3]}")

            # Replace classifier with Identity so forward() returns 512-D features
            if hasattr(self.model, "classifier"):
                self.model.classifier = torch.nn.Identity()

            self.model.to(self.device)
            self.model.eval()
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.mode = "OSNet"
            self.feat_dim = 512
            print(f"[ReID] OSNet loaded — classifier head replaced with Identity, backbone active")
        except Exception as e:
            print(f"[ReID] OSNet load failed: {e}")
            self.model = None

    def _try_load_resnet50(self):
        try:
            print("[ReID] loading ResNet50 fallback...")
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.model.fc = torch.nn.Identity()
            self.model.to(self.device)
            self.model.eval()
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((128, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.mode = "ResNet50"
            self.feat_dim = 2048
            print("[ReID] ResNet50 fallback loaded — generic, identity confidence limited")
        except Exception as e:
            print(f"[ReID] ResNet50 failed: {e}")
            self.model = None

    def extract(self, crops: list) -> list:
        """Returns list of L2-normalised numpy arrays, one per crop."""
        if self.model is None or not crops:
            return []
        features_list = []
        batch_size = 16
        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i:i + batch_size]
            tensors = []
            for c in batch_crops:
                if c is None or c.size == 0:
                    tensors.append(torch.zeros(3, *self.transform.transforms[1].size[::-1]))
                else:
                    try:
                        tensors.append(self.transform(c))
                    except Exception:
                        tensors.append(torch.zeros(3, 128, 128))
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
        self._scene_snapshot_taken = False
        self._needs_scene_reset = False
        self._needs_scene_revival = False
        self._scene_revival_frames_left = 0   # keep trying scene revival for N frames
        self._scene_reset_at_frame = -1   # guard hard-collapse for 30 frames post-reset

        # Scene state tracking
        self._active_baseline = 0
        self._track_history = []
        self._soft_collapse = False
        self._soft_recovery_frames = 0
        self._streak_collapse = 0
        self._streak_recover = 0
        self._transition_frame = -1

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
        """Extract ReID embeddings. OSNet > ResNet50 > HSV fallback."""
        if not tracks:
            return {}

        h, w = frame.shape[:2]
        tids = []
        crops = []
        for tr in tracks:
            x1, y1, x2, y2 = [int(v) for v in tr.bbox]
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            tids.append(tr.track_id)
            crops.append(crop)

        if self.reid.mode in ("OSNet", "ResNet50"):
            feats = self.reid.extract(crops)
            if len(feats) == len(tids):
                return {tids[i]: feats[i] for i in range(len(tids))}

        # HSV fallback
        embed_map = {}
        for tr in tracks:
            emb = self._extract_torso_hsv(frame, tr.bbox)
            if emb is not None:
                embed_map[tr.track_id] = emb
        return embed_map

    def _pixel_to_pitch(self, frame_w: int, frame_h: int,
                        px: float, py: float) -> tuple:
        """Convert pixel foot-point to normalised pitch coords [0,1]x[0,1].
        Uses homography if available, otherwise proportional fallback."""
        H = getattr(self, '_homography', None)
        if H is not None:
            pt = np.array([[[px, py]]], dtype=np.float32)
            mapped = cv2.perspectiveTransform(pt, H)
            mx, my = float(mapped[0, 0, 0]), float(mapped[0, 0, 1])
            # Clamp to pitch bounds (105m x 68m)
            return (np.clip(mx / 105.0, 0.0, 1.0),
                    np.clip(my / 68.0, 0.0, 1.0))
        # Proportional fallback
        return (np.clip(px / frame_w, 0.0, 1.0),
                np.clip(py / frame_h, 0.0, 1.0))

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

        if is_freeze:
            self._needs_scene_reset = True

        # Bug #3: Reset tracker on freeze→play transition (purge ghost tracks)
        if self._needs_scene_reset and is_play:
            self.tracker.reset()
            self.id_remap.clear()
            self.identity.reset_for_scene(frame_id=video_frame)
            self.role_filter.reset()
            self._scene_snapshot_taken = False
            self._needs_scene_revival = True
            self._scene_revival_frames_left = 90
            self._track_history.clear()
            self._scene_reset_at_frame = video_frame  # guard hard-collapse for 30 frames

            # Reset soft state
            self._soft_collapse = False
            self._soft_recovery_frames = 0
            self._streak_collapse = 0
            self._streak_recover = 0
            if self._active_baseline == 0.0:
                self._active_baseline = 18.0
            print(f"[SoftStateReset] Frame {video_frame}: mode=normal baseline={self._active_baseline:.1f} low_streak=0 recovery_left=0")
            print(f"[Reset] Frame {video_frame}: freeze→play, tracker + identity reset")
            self._needs_scene_reset = False
            
        # BUG FIX A: Snapshot on freeze entry
        if is_freeze and not self._scene_snapshot_taken:
            saved = self.identity.snapshot_scene(video_frame)
            self._scene_snapshot_taken = True
            print(f"[Snapshot] Frame {video_frame}: freeze entry saved {saved} slots")

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
                    saved = self.identity.snapshot_soft(video_frame)
                    self._soft_collapse = True
                    self.identity.in_soft_collapse = True
                    self.identity.locks.in_collapse = True   # audit flag
                    self.identity.in_soft_recovery = False
                    self._streak_collapse = 0
                    print(f"[SoftCollapse] Frame {video_frame}: current={current_count} baseline={self._active_baseline:.1f} saved={saved}")

                elif self._streak_recover >= 1 and self._soft_collapse:
                    self._soft_collapse = False
                    self.identity.in_soft_collapse = False
                    self.identity.locks.in_collapse = False
                    self._soft_recovery_frames = 60
                    self.identity.in_soft_recovery = True
                    self._streak_recover = 0
                    snap_len = len(self.identity._soft_snapshot) if hasattr(self.identity, '_soft_snapshot') and self.identity._soft_snapshot else 0
                    print(f"[SoftRecovery] Frame {video_frame}: current={current_count} baseline={self._active_baseline:.1f} snapshot_slots={snap_len} revived=candidates → entering recovery mode")

        # Periodic telemetry
        if is_play and video_frame % 30 == 0:
            mode = "collapse" if self._soft_collapse else ("recovery" if self._soft_recovery_frames > 0 else "normal")
            print(f"[SoftState] frame={video_frame} baseline={self._active_baseline:.1f} current={len(assignable_tracks)} mode={mode} low_streak={self._streak_collapse} recovery_left={self._soft_recovery_frames}")
            
        # Hard Collapse pre-cutaway fallback
        # Disabled for 30 frames after scene reset (snapshot would be empty/stale)
        frames_since_reset = video_frame - self._scene_reset_at_frame
        hard_collapse_ok = (
            is_play
            and self._active_baseline > 0
            and len(assignable_tracks) <= 0.60 * self._active_baseline
            and not self._scene_snapshot_taken
            and frames_since_reset > 30
        )
        if hard_collapse_ok:
            saved = self.identity.snapshot_scene(video_frame)
            self._scene_snapshot_taken = True
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
            if self._soft_collapse or self._soft_recovery_frames > 0: # Block memory during collapse and recovery
                continue
                
            memory_ok_tids.add(tid)
                
        if (stale_memory_blocked > 0 or self._soft_collapse or self._soft_recovery_frames > 0) and video_frame % 30 == 0:
            msg = f"[TrackerCore] F{video_frame}: memory updates restricted"
            if stale_memory_blocked > 0: msg += f" (stale={stale_memory_blocked})"
            if self._soft_collapse: msg += " (SOFT_COLLAPSE active)"
            if self._soft_recovery_frames > 0: msg += " (SOFT_RECOVERY active)"
            print(msg)

        # Step E: Recovery on first valid play after freeze
        embed_map = self._extract_embeds(frame, assignable_tracks)
        h_frame, w_frame = frame.shape[:2]

        # Build inputs for identity
        track_objs, positions, embeddings, pitch_positions, team_labels = [], {}, {}, {}, {}
        for tr in assignable_tracks:
            tid = tr.track_id
            bx = tr.bbox
            cx = (bx[0] + bx[2]) / 2
            # Use foot-point (bottom-centre) for pitch projection
            foot_y = float(bx[3])
            positions[tid] = (cx, foot_y)
            pitch_positions[tid] = self._pixel_to_pitch(w_frame, h_frame, cx, foot_y)
            if tid in embed_map:
                embeddings[tid] = embed_map[tid]
            # Team label from id_remap history or track attribute
            team_labels[tid] = getattr(tr, 'team', None)
            track_objs.append(tr)

        # Pass pitch coords and team labels to identity core
        self.identity.pitch_positions = pitch_positions
        self.identity.team_labels = team_labels
        self.identity.reid_mode = self.reid.mode

        # Clear scene recovery protection after 60 frames
        if (self.identity.in_scene_recovery
                and self._scene_reset_at_frame >= 0
                and video_frame - self._scene_reset_at_frame > 90):
            self.identity.in_scene_recovery = False

        # Track-level meta for this frame (pid, source, identity_valid, confidence)
        meta_by_tid = {}

        if is_play and len(track_objs) > 0:
            present_tids = {int(tr.track_id) for tr in track_objs}
            self.identity.begin_frame(video_frame, present_tids=present_tids)

            # First match: recovery-first if snapshot exists (hard or soft)
            revived_tids = set()
            should_revive_scene = self._needs_scene_revival
            should_revive_soft = (self._soft_recovery_frames > 0)

            if should_revive_scene:
                self._scene_revival_frames_left -= 1
                scene_window_expired = self._scene_revival_frames_left <= 0
                # Only attempt revival for tracks not yet locked
                unlocked_for_revival = [
                    t for t in track_objs
                    if not self.identity.locks.is_tid_locked(int(t.track_id))
                ]
                if unlocked_for_revival:
                    revived, scene_meta = self.identity.revive_cost_matrix(
                        unlocked_for_revival, embeddings, positions
                    )
                    self.id_remap.update({tid: int(pid[1:]) for tid, pid in revived.items()})
                    meta_by_tid.update(scene_meta)
                    revived_tids.update(revived.keys())
                    if revived:
                        print(f"[SceneRevival] frame={video_frame} revived={len(revived)} remaining_window={self._scene_revival_frames_left}")
                elif not scene_window_expired:
                    # All tracks are locked — revival complete
                    self._needs_scene_revival = False
                    print(f"[SceneRevivalDone] frame={video_frame} all tracks locked")
                if scene_window_expired:
                    still_unlocked = [
                        t for t in track_objs
                        if int(t.track_id) not in revived_tids
                        and not self.identity.locks.is_tid_locked(int(t.track_id))
                    ]
                    forced, force_meta = self.identity.force_commit_remaining_scene_slots(
                        still_unlocked, embeddings, positions
                    )
                    self.id_remap.update({tid: int(pid[1:]) for tid, pid in forced.items()})
                    meta_by_tid.update(force_meta)
                    revived_tids.update(forced.keys())
                    self._needs_scene_revival = False
                    self.identity.in_scene_recovery = False
                    print(f"[SceneRevivalExpired] frame={video_frame} forced={len(forced)} commits before exit")

            if should_revive_soft:
                is_first_frame = (self._soft_recovery_frames == 60)
                self._soft_recovery_frames -= 1
                if self._soft_recovery_frames <= 0:
                    self.identity.in_soft_recovery = False
                cand = len(track_objs)

                revived_soft, soft_meta = self.identity.revive_from_soft_snapshot(
                    track_objs, embeddings, positions,
                    is_first_recovery_frame=is_first_frame,
                )
                self.id_remap.update({tid: int(pid[1:]) for tid, pid in revived_soft.items()})
                meta_by_tid.update(soft_meta)
                revived_tids.update(revived_soft.keys())

                if is_first_frame:
                    print(f"[SoftRecoveryEntry] candidates={cand} revived={len(revived_soft)}")
                    if len(revived_soft) == 0:
                        snap_len = len(self.identity._soft_snapshot) if hasattr(self.identity, "_soft_snapshot") and self.identity._soft_snapshot else 0
                        if snap_len == 0:
                            print(f"[SoftRecoveryCause] F{video_frame} no snapshot or consumed early")
                        elif len(embeddings) == 0:
                            print(f"[SoftRecoveryCause] F{video_frame} no embeddings available")
                        elif cand == 0:
                            print(f"[SoftRecoveryCause] F{video_frame} no candidate tracks")
                        else:
                            print(f"[SoftRecoveryCause] F{video_frame} costs too high")

            # Normal assignment — identity_restricted is the single source of truth.
            # assign_tracks also enforces this internally as a second safety net.
            normal_track_objs = [t for t in track_objs if t.track_id not in revived_tids]
            restricted = self.identity._identity_restricted
            restricted_reason = self.identity._identity_restricted_reason()

            tid_to_pid, normal_meta = self.identity.assign_tracks(
                normal_track_objs, embeddings, positions,
                memory_ok_tids=memory_ok_tids,
                allow_new_assignments=not restricted,
            )
            meta_by_tid.update(normal_meta)
            self.identity.end_frame(video_frame)
            self.identity.maybe_log(detections_count=len(dets), tracks_count=n_tracks)

            # Merge: assign_tracks only updates un-revived mappings
            for tid, pid in tid_to_pid.items():
                self.id_remap[tid] = int(pid[1:])

            # Drop id_remap entries for tracks that became unassigned this frame
            for tid, m in meta_by_tid.items():
                if m.pid is None and tid in self.id_remap:
                    del self.id_remap[tid]

            # Lock-aware accounting — count provisional as normal (they should be 0 in restricted)
            locked_kept = sum(1 for m in meta_by_tid.values() if m.source == "locked")
            revived_n = sum(1 for m in meta_by_tid.values() if m.source == "revived")
            provisional_n = sum(1 for m in meta_by_tid.values() if m.source == "provisional")
            unknown_n = sum(1 for m in meta_by_tid.values() if m.source == "unknown")
            unassigned_n = sum(1 for m in meta_by_tid.values() if m.source == "unassigned")

            # Invariant check: if restricted, provisional must be 0
            if restricted and provisional_n > 0:
                print(
                    f"[IdentityInvariantFAIL] F{video_frame} provisional={provisional_n} "
                    f"during restricted mode reason={restricted_reason}"
                )

            if should_revive_scene or should_revive_soft or video_frame % 30 == 0:
                print(
                    f"[RecoveryAssign] F{video_frame} locked={locked_kept} "
                    f"revived={revived_n} normal={provisional_n} unknown={unknown_n} "
                    f"blocked={stale_memory_blocked+len(overlay_blocked_tids)} "
                    f"unassigned={unassigned_n} restricted={restricted} reason={restricted_reason}"
                )
        elif is_play:
            # No tracks this frame — still tick the lock TTLs so stale ones expire
            self.identity.begin_frame(video_frame, present_tids=set())
            self.identity.end_frame(video_frame)

        if save:
            players = []
            assignment_pending = bool(self.identity._identity_restricted)
            for tr in player_tracks:
                tid = tr.track_id
                meta = meta_by_tid.get(tid)
                x1, y1, x2, y2 = tr.bbox
                q = quality_scores.get(tid)

                if meta is not None and meta.pid is not None and meta.identity_valid:
                    out_pid = int(meta.pid[1:])
                    source = meta.source
                    identity_valid = True
                    identity_confidence = float(meta.confidence)
                    player_id = meta.pid
                    display_id = meta.pid
                else:
                    # Uncertain — render as raw track (T<id>), no P-id
                    out_pid = None
                    source = "unassigned"
                    identity_valid = False
                    identity_confidence = float(meta.confidence) if meta else 0.0
                    player_id = None
                    display_id = "?" if assignment_pending else None

                players.append({
                    "trackId": out_pid if out_pid is not None else int(tid),
                    "rawTrackId": int(tid),
                    "playerId": player_id,
                    "displayId": display_id,
                    "assignment_pending": assignment_pending and not identity_valid,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(tr.score),
                    "class": int(tr.cls),
                    "gameState": state.value,
                    "analysis_valid": True,
                    "crop_quality": float(q.score) if q else 1.0,
                    "identity_valid": identity_valid,
                    "assignment_source": source,
                    "identity_confidence": identity_confidence,
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

    # Final identity-lock summary (the real success metric)
    id_summary = tracker.identity.end_run_summary()
    summary = id_summary  # end_run_summary merges lock summary
    valid = sum(1 for f in tracker.results for p in f.get("players", []) if p.get("identity_valid"))
    invalid = sum(1 for f in tracker.results for p in f.get("players", []) if not p.get("identity_valid"))
    total_boxes = valid + invalid
    valid_id_coverage = round(valid / max(total_boxes, 1), 3)
    print("=" * 60)
    print("[Identity Summary]")
    print(f"  identity_switches            = {summary['identity_switches']}")
    print(f"  id_rebind_count              = {summary['id_rebind_count']}")
    print(f"  pid_takeover_count           = {summary['pid_takeover_count']}")
    print(f"  switches_blocked             = {summary['switches_blocked']}")
    print(f"  soft_recovery_rebinds_blocked= {summary['soft_recovery_rebinds_blocked']}")
    print(f"  collapse_lock_attempts       = {summary['collapse_lock_attempts']}  (blocked, not written)")
    print(f"  collapse_lock_creations      = {summary['collapse_lock_creations']}  (must be 0)")
    print(f"  locks_created                = {summary['locks_created']}")
    print(f"  locks_expired                = {summary['locks_expired']}")
    print(f"  locks_live_at_end            = {summary['locks_live']}")
    print(f"  lock_retention_rate          = {summary['lock_retention_rate']}")
    print(f"  excessive_lock_churn         = {summary['excessive_lock_churn']}")
    print(f"  valid_identity_frames        = {valid}")
    print(f"  unknown_boxes                = {invalid}")
    print(f"  valid_id_coverage            = {valid_id_coverage}")
    if summary['excessive_lock_churn']:
        print(f"[IdentityWarning] excessive lock churn: {summary['locks_created']} locks created (target ≤40)")
    print("=" * 60)

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
