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
    from services.camera_motion_service import CameraMotionDetector, log_camera_motion
except (ImportError, ModuleNotFoundError):
    sys.path.insert(0, os.path.dirname(__file__))
    from identity_core import IdentityCore
    from vlm_state import VLMStateMachine, GameState
    from track_suppressor import TrackSuppressor
    from role_filter import RoleFilter
    from crop_quality import CropQualityGate
    from camera_motion_service import CameraMotionDetector, log_camera_motion


class ReIDExtractor:
    """
    Priority order:
      1. OSNet (torchreid) — 512-D person-specific ReID, best for same-team
      2. ResNet50 (torchvision) — 2048-D generic ImageNet, limited same-team
      3. HSV histogram — 52-D color, team-level only

    OSNet weights: /content/osnet_x1_0_msmt17.pt (upload to Colab)
    """
    HF_OSNET_REPO = "kaiyangzhou/osnet"
    HF_OSNET_FILE = (
        "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_"
        "stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
    )

    def __init__(self, device="cpu"):
        self.device = "cuda" if "cuda" in str(device) else "cpu"
        if torch.backends.mps.is_available():
            self.device = "mps"

        self.mode = "HSV-fallback"
        self.model = None
        self.transform = None
        self.feat_dim = 52  # HSV fallback dim
        self.target_size = (256, 128)  # default, overridden when model loads

        # Try OSNet first
        self._try_load_osnet()

        # Fallback: ResNet50
        if self.model is None and torchvision is not None:
            self._try_load_resnet50()

        if self.model is None:
            print("[ReID] torchvision not found. Using HSV fallback only — same-team identity unreliable.")

    def _candidate_osnet_paths(self) -> list[str]:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return [
            os.environ.get("OSNET_WEIGHTS", ""),
            "/content/osnet_x1_0_msmt17.pt",
            "/content/athlink-cv-service/models/osnet_x1_0_msmt17.pt",
            os.path.join(repo_root, "models", "osnet_x1_0_msmt17.pt"),
            os.path.join(repo_root, "osnet_x1_0_msmt17.pt"),
        ]

    def _find_osnet_weights(self) -> str | None:
        """Priority order for OSNet weights:
        1. Football-specific fine-tuned (Liyrs58/football-osnet-reid) — local then HF Hub
        2. Sports-tuned (CondadosAI/osnet-trackers, SportsMOT) — local then HF Hub
        3. MSMT17 pedestrian surveillance — local only (last resort)
        """
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Priority 1a: Football-specific weights (local)
        football_local = [
            os.environ.get("OSNET_FOOTBALL_WEIGHTS", ""),
            "/content/football_osnet_x1_0.pth.tar",
            "/content/athlink-cv-service/models/football_osnet_x1_0.pth.tar",
            os.path.join(repo_root, "models", "football_osnet_x1_0.pth.tar"),
        ]
        for path in football_local:
            if path and os.path.exists(path) and os.path.getsize(path) > 1_000_000:
                print(f"[ReID] Using football-specific OSNet weights: {path}")
                return path

        # Priority 1b: Football-specific weights (HF Hub download)
        try:
            from huggingface_hub import hf_hub_download
            dest = hf_hub_download(
                repo_id="Liyrs58/football-osnet-reid",
                filename="football_osnet_x1_0.pth.tar",
                local_dir=os.path.join(repo_root, "models"),
                local_dir_use_symlinks=False,
            )
            print(f"[ReID] Downloaded football OSNet from HF Hub: {dest}")
            return dest
        except Exception:
            pass  # not yet available — fall through

        # Priority 2a: SportsMOT sports-tuned weights (local)
        sports_local = [
            os.environ.get("OSNET_SPORTS_WEIGHTS", ""),
            "/content/sports_model.pth.tar-60",
            "/content/athlink-cv-service/models/sports_model.pth.tar-60",
            os.path.join(repo_root, "models", "sports_model.pth.tar-60"),
        ]
        for path in sports_local:
            if path and os.path.exists(path) and os.path.getsize(path) > 1_000_000:
                print(f"[ReID] Using sports-tuned OSNet weights: {path}")
                return path

        # Priority 2b: SportsMOT weights (HF Hub download)
        try:
            from huggingface_hub import hf_hub_download
            dest = hf_hub_download(
                repo_id="CondadosAI/osnet-trackers",
                filename="sports_model.pth.tar-60",
                local_dir=os.path.join(repo_root, "models"),
                local_dir_use_symlinks=False,
            )
            print(f"[ReID] Downloaded sports OSNet from HF Hub: {dest}")
            return dest
        except Exception as e:
            print(f"[ReID] HF Hub sports download failed: {e}")

        # Priority 3: MSMT17 pedestrian weights (local only — last resort)
        for path in self._candidate_osnet_paths():
            if path and os.path.exists(path) and os.path.getsize(path) > 1_000_000:
                print(f"[ReID] Using MSMT17 OSNet weights (last resort): {path}")
                return path

        return None

    def _download_osnet_weights(self, target_path: str) -> bool:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        try:
            from huggingface_hub import hf_hub_download
            import shutil

            print("[ReID] Downloading OSNet via huggingface_hub...")
            downloaded = hf_hub_download(
                repo_id=self.HF_OSNET_REPO,
                filename=self.HF_OSNET_FILE,
                local_dir=os.path.dirname(target_path),
                local_dir_use_symlinks=False,
            )

            if downloaded != target_path:
                shutil.copy2(downloaded, target_path)

            print(f"[ReID] OSNet downloaded to {target_path}")
            return True

        except Exception as e:
            print(f"[ReID] huggingface_hub OSNet download failed: {e}")
            return False

    def _try_load_osnet(self):
        osnet_path = self._find_osnet_weights()

        if osnet_path is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            osnet_path = os.path.join(repo_root, "models", "osnet_x1_0_msmt17.pt")

            if not self._download_osnet_weights(osnet_path):
                print("[ReID] OSNet unavailable; falling back.")
                return

        print(f"[ReID] Loading OSNet from {osnet_path}")
        try:
            import torchreid

            # Load raw checkpoint and unwrap if needed
            ckpt = torch.load(osnet_path, map_location="cpu")
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
            if "cuda" in str(self.device):
                self.model.half()
            self.model.eval()
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.target_size = (256, 128)
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
            if "cuda" in str(self.device):
                self.model.half()
            self.model.eval()
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.target_size = (128, 128)
            self.mode = "ResNet50"
            self.feat_dim = 2048
            print("[ReID] ResNet50 fallback loaded — generic, identity confidence limited")
        except Exception as e:
            print(f"[ReID] ResNet50 failed: {e}")
            self.model = None

    def _zero_tensor(self) -> torch.Tensor:
        """Zero crop matching target size."""
        h, w = self.target_size
        return torch.zeros(3, h, w)

    def extract(self, crops: list) -> list:
        """Returns list of L2-normalised numpy arrays, one per crop."""
        if self.model is None or not crops:
            return []
        features_list = []
        batch_size = 32
        
        # GPU Accelerated Preprocessing
        h_target, w_target = self.target_size
        
        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i:i + batch_size]
            batch_tensors = []
            
            for c in batch_crops:
                if c is None or c.size == 0:
                    batch_tensors.append(self._zero_tensor().to(self.device))
                else:
                    try:
                        # Move to GPU as-is (H, W, C) -> (C, H, W)
                        t = torch.from_numpy(c.transpose(2, 0, 1)).to(self.device).float() / 255.0
                        # Resize on GPU
                        t = torch.nn.functional.interpolate(t.unsqueeze(0), size=(h_target, w_target), mode='bilinear', align_corners=False).squeeze(0)
                        # Normalize on GPU
                        t = T.functional.normalize(t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        batch_tensors.append(t)
                    except Exception:
                        batch_tensors.append(self._zero_tensor().to(self.device))
            
            batch_tensor = torch.stack(batch_tensors)
            if "cuda" in str(self.device):
                batch_tensor = batch_tensor.half()
            
            with torch.no_grad():
                feat = self.model(batch_tensor)
                feat = torch.nn.functional.normalize(feat, p=2, dim=1)
                features_list.extend([f.cpu().numpy() for f in feat])
        return features_list


class _NoOpIdentity:
    """Drop-in stub when use_identity=False — all calls are no-ops."""
    def __getattr__(self, name):
        return lambda *a, **kw: None
    def assign_tracks(self, *a, **kw):
        return {}
    def end_run_summary(self):
        pass


class TrackerCore:
    def __init__(self, yolo_path, device="cpu", use_identity=True):
        self.device = device

        yolo_path = os.path.abspath(yolo_path)
        print(f"[TrackerCore] YOLO: {yolo_path} (exists: {os.path.exists(yolo_path)})")

        yolo_device = "cuda" if ("cuda" in str(device) or str(device) == "0") else "cpu"
        self.yolo = YOLO(yolo_path)
        self.yolo.to(yolo_device)
        # We don't call .half() here, as it can cause dtype mismatch errors during fusion.
        # Instead, we pass half=True to the predict() call below.

        # Derive class-id sets from model.names so we work with either:
        #   roboflow_players.pt  → {0:'ball', 1:'goalkeeper', 2:'player', 3:'referee'}
        #   yolov8m.pt (COCO)    → {0:'person', 32:'sports ball', ...}
        names_lower = {int(k): str(v).lower() for k, v in self.yolo.names.items()}
        ball_keys = {k for k, v in names_lower.items() if v in ("ball", "sports ball")}
        player_keys = {k for k, v in names_lower.items()
                       if v in ("player", "goalkeeper", "referee", "person")}
        if not ball_keys:
            ball_keys = {0}        # roboflow default
        if not player_keys:
            player_keys = {1, 2, 3}  # roboflow default
        self.ball_class_ids = ball_keys
        self.player_class_ids = player_keys
        self._is_coco_model = "person" in names_lower.values()
        print(f"[TrackerCore] class ids — ball={sorted(ball_keys)} "
              f"player={sorted(player_keys)} coco={self._is_coco_model}")

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
        # Re-enabled Phase D: proactive snapshot now fires before collapse (line 569)
        self.identity = IdentityCore()
        self.reid = ReIDExtractor(device=device) if use_identity else None
        self._use_identity = use_identity
        self.suppressor = TrackSuppressor()
        self.role_filter = RoleFilter()
        self.crop_quality = CropQualityGate()
        self.id_remap = {}  # de_tid -> int PID
        self._switched_pids: set = set()  # PIDs that changed lock this session
        self._scene_snapshot_taken = False
        self._needs_scene_reset = False
        self._needs_scene_revival = False
        self._scene_revival_frames_left = 0   # keep trying scene revival for N frames
        self._scene_reset_at_frame = -1   # guard hard-collapse for 30 frames post-reset
        self._in_freeze_segment = False

        # Scene state tracking
        self._active_baseline = 18.0  # sensible default; updated from real data after frame 1
        self._track_history = []
        self._soft_collapse = False
        self._soft_recovery_frames = 0
        self._streak_collapse = 0
        self._streak_recover = 0
        self._transition_frame = -1
        self._frames_in_collapse = 0  # phase 2 telemetry: feeds match_report.quality.softCollapseFraction
        self._last_snapshot_frame = -100  # proactive snapshot: throttle to every 15 frames min

        # Officials tracking for output
        self._officials_this_frame = []
        self._suspected_officials_this_frame = []  # held-back tracks (ref-leaning, unconfirmed)

        self.frame_idx = 0
        self.results = []

    def detect(self, frame):
        """YOLO – returns np.array (N,6) [x1,y1,x2,y2,conf,cls].

        Side effect: stashes the highest-confidence ball detection of
        the current frame on `self._last_ball_det` so the per-frame writer can
        persist it. Class ids derived from model.names at __init__ time:
        roboflow → {0:ball, 1:goalkeeper, 2:player, 3:referee};
        coco yolov8 → {0:person, 32:sports ball}. Output cls is normalised
        to roboflow numbering so downstream consumers stay unchanged."""
        self._last_ball_det = None
        # Disable FP16 (half=False) for YOLO to prevent dtype mismatch during fusion
        results = self.yolo.predict(frame, conf=0.05, verbose=False, half=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 6))

        dets = []
        best_ball = None
        for box in boxes:
            cls = int(box.cls.item())
            x1, y1 = float(box.xyxy[0][0]), float(box.xyxy[0][1])
            x2, y2 = float(box.xyxy[0][2]), float(box.xyxy[0][3])
            conf = float(box.conf.item())
            if cls in self.ball_class_ids:
                if best_ball is None or conf > best_ball[4]:
                    best_ball = [x1, y1, x2, y2, conf, 0.0]
            elif cls in self.player_class_ids:
                # Normalise to class 2 (player) when running the COCO fallback,
                # since downstream code (e.g. trajectory filters) keys on cls==2.
                out_cls = 2.0 if self._is_coco_model else float(cls)
                dets.append([x1, y1, x2, y2, conf, out_cls])
        if best_ball is not None:
            self._last_ball_det = {
                "bbox": [best_ball[0], best_ball[1], best_ball[2], best_ball[3]],
                "confidence": best_ball[4],
                "source": "yolo",
            }
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
                np.empty((0, 4)), np.empty((0,)), np.empty((0,)), None, frame=frame
            )
        bboxes  = dets[:, :4]
        scores  = dets[:, 4]
        classes = dets[:, 5]

        # Extract embeddings for all detections
        embeds = []
        for bbox in bboxes:
            embeds.append(self._extract_torso_hsv(frame, bbox))
        embeds = np.array(embeds)

        return self.tracker.update(bboxes, scores, classes, embeds=embeds, frame=frame)

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

        embed_map = {}
        hsv_map = {}
        for tr in tracks:
            hsv_map[tr.track_id] = self._extract_torso_hsv(frame, tr.bbox)

        if self.reid is not None and self.reid.mode in ("OSNet", "ResNet50"):
            feats = self.reid.extract(crops)
            if len(feats) == len(tids):
                for i, tid in enumerate(tids):
                    embed_map[tid] = {
                        "emb": feats[i],
                        "hsv": hsv_map.get(tid)
                    }
                return embed_map

        # Fallback (or if ReID failed)
        for tid, hsv in hsv_map.items():
            embed_map[tid] = {"emb": hsv, "hsv": hsv}
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

    def process_frame(self, frame, video_frame, dets, save=True, camera_motion=None):
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
        entered_freeze = is_freeze and not self._in_freeze_segment

        if is_freeze:
            self._in_freeze_segment = True
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
            print(f"[Reset] Frame {video_frame}: freeze→play, tracker reset + identity recovery")
            self._needs_scene_reset = False
            self._in_freeze_segment = False
            
        # Freeze-entry snapshot must refresh even if a prior hard-collapse snapshot exists.
        if entered_freeze:
            saved = self.identity.snapshot_scene(video_frame, merge_existing=True)
            self._scene_snapshot_taken = True
            print(f"[Snapshot] Frame {video_frame}: freeze entry refreshed {saved} slots")

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
        # `suspected` are tracks with ref-leaning votes but not yet confirmed; we
        # drop them entirely from this frame to avoid transient P-id leakage.
        filtered_tracks, self._officials_this_frame, suspected_officials = self.role_filter.filter(
            visible_tracks, frame, video_frame
        )
        player_tracks = filtered_tracks
        self._suspected_officials_this_frame = suspected_officials
        
        # Collect all official/suspected-official TIDs to block in identity
        _official_tids_this_frame = set()
        for ofc in (self._officials_this_frame or []):
            if hasattr(ofc, 'track_id'):
                _official_tids_this_frame.add(int(ofc.track_id))
        for sofc in (suspected_officials or []):
            if hasattr(sofc, 'track_id'):
                _official_tids_this_frame.add(int(sofc.track_id))
        self._official_tids_this_frame = _official_tids_this_frame
        
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
        # Camera-aware baseline: rolling median of player-only counts (excludes officials).
        # Adapts both up (wide shots) and down (broadcast zoom-in to box).
        if is_play and not is_overlay:
            n_officials = len(self._officials_this_frame) if self._officials_this_frame else 0
            current_count = max(0, len(assignable_tracks) - 0)  # assignable already excludes officials
            _ = n_officials  # kept for future expected-count subtraction if assignable changes

            # Track recovery even while collapsed so recovery trigger can fire
            if self._soft_collapse:
                if current_count >= 0.60 * self._active_baseline:
                    self._streak_recover += 1
                else:
                    self._streak_recover = 0

            # Always feed the rolling window (no gate on >=18 — that's what trapped the baseline high).
            if not self._soft_collapse:
                self._track_history.append(current_count)
                if len(self._track_history) > 30:
                    self._track_history.pop(0)
                # Median = robust to brief detection drops, adapts to broadcast zoom.
                sorted_hist = sorted(self._track_history)
                mid = len(sorted_hist) // 2
                if len(sorted_hist) % 2 == 0 and len(sorted_hist) >= 2:
                    median_count = (sorted_hist[mid - 1] + sorted_hist[mid]) / 2.0
                else:
                    median_count = float(sorted_hist[mid])
                # Floor at 8 (small-side coverage); no upper cap.
                self._active_baseline = max(8.0, median_count)


                # Trigger SoftCollapse if dropped below 65% for 3+ consecutive frames
                if current_count <= 0.65 * self._active_baseline:
                    self._streak_collapse += 1
                else:
                    self._streak_collapse = 0

                # Trigger SoftRecovery if recovered above 60%
                if current_count >= 0.60 * self._active_baseline:
                    self._streak_recover += 1
                else:
                    self._streak_recover = 0

                if self._streak_collapse >= 3 and not self._soft_collapse:
                    # Check if identity bootstrap is complete before allowing soft collapse
                    bootstrap_complete = (
                        self.identity.locks.collapse_lock_creations == 0 and  # no collapsed-blocks yet
                        (self.identity.locks.locks_created >= 5 or self.identity.locks.count_live_locks() >= 5)
                    )
                    if bootstrap_complete:
                        saved = self.identity.snapshot_soft(video_frame)
                        self._soft_collapse = True
                        self.identity.in_soft_collapse = True
                        self.identity.locks.in_collapse = True   # audit flag
                        self.identity.in_soft_recovery = False
                        self._streak_collapse = 0
                        print(f"[SoftCollapse] Frame {video_frame}: current={current_count} baseline={self._active_baseline:.1f} saved={saved}")
                    else:
                        # Bootstrap not complete — don't activate soft collapse yet
                        self._streak_collapse = 0
                        if video_frame % 30 == 0:
                            print(f"[Bootstrap] Frame {video_frame}: soft_collapse blocked (not ready) — live_locks={self.identity.locks.count_live_locks()} created={self.identity.locks.locks_created}")

            # Recovery transition — checked OUTSIDE the if-not-collapse gate so it can fire when collapsed
            if self._streak_recover >= 1 and self._soft_collapse and is_play and not is_overlay:
                self._soft_collapse = False
                self.identity.in_soft_collapse = False
                self.identity.locks.in_collapse = False
                snap_len = len(self.identity._soft_snapshot) if hasattr(self.identity, '_soft_snapshot') and self.identity._soft_snapshot else 0
                if snap_len == 0:
                    self._soft_recovery_frames = 0
                    self.identity.in_soft_recovery = False
                    print(f"[SoftRecovery] Frame {video_frame}: snapshot empty → skip recovery, normal mode")
                else:
                    self._soft_recovery_frames = 60
                    self.identity.in_soft_recovery = True
                    current_count = len(assignable_tracks)  # Re-compute for log
                    print(f"[SoftRecovery] Frame {video_frame}: current={current_count} baseline={self._active_baseline:.1f} snapshot_slots={snap_len} revived=candidates → entering recovery mode")
                self._streak_recover = 0

        # Track frames-in-collapse for match_report.json telemetry (phase 2)
        if self._soft_collapse:
            self._frames_in_collapse = getattr(self, "_frames_in_collapse", 0) + 1

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
        track_objs, positions, embeddings, pitch_positions, team_labels, tid_bboxes = [], {}, {}, {}, {}, {}
        for tr in assignable_tracks:
            tid = tr.track_id
            bx = tr.bbox
            cx = (bx[0] + bx[2]) / 2
            # Use foot-point (bottom-centre) for pitch projection
            foot_y = float(bx[3])
            positions[tid] = (cx, foot_y)
            tid_bboxes[tid] = [float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3])]
            pitch_positions[tid] = self._pixel_to_pitch(w_frame, h_frame, cx, foot_y)
            if tid in embed_map:
                embeddings[tid] = embed_map[tid]
            # Team label from id_remap history or track attribute
            team_labels[tid] = getattr(tr, 'team', None)
            track_objs.append(tr)

        # PROACTIVE SNAPSHOT: Capture healthy track state before collapse
        # Runs AFTER embeddings are built so embed_map is available
        if is_play and self._active_baseline > 0:
            current_count = len(assignable_tracks)
            is_healthy = (current_count >= 0.35 * max(self._active_baseline, 1.0))
            should_snapshot = (
                is_healthy and
                (not hasattr(self, '_last_snapshot_frame') or video_frame - self._last_snapshot_frame >= 15)
            )
            if should_snapshot:
                # Snapshot from identity slots first (locked/revived tracks)
                saved = self.identity.snapshot_soft(video_frame)
                # If no slots yet, seed empty slots with raw track embeddings
                if saved == 0 and len(embed_map) > 0:
                    seeded = self.identity.seed_provisional_from_tracks(
                        embed_map, positions, pitch_positions, team_labels, video_frame
                    )
                    saved = seeded
                self._last_snapshot_frame = video_frame
                if saved > 0:
                    print(f"[ProactiveSnapshot] Frame {video_frame}: {saved} slots captured (baseline={self._active_baseline:.1f})")

        # Pass pitch coords and team labels to identity core
        self.identity.pitch_positions = pitch_positions
        self.identity.team_labels = team_labels
        if self.reid is not None:
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
            self.identity.begin_frame(
                video_frame,
                present_tids=present_tids,
                frame_width=w_frame,
                frame_height=h_frame,
            )
            cluster_tids = self.identity.congestion_detector.detect(
                list(tid_bboxes.items())
            ) if tid_bboxes else set()

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
                        unlocked_for_revival, embeddings, positions,
                        cluster_tids=cluster_tids,
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
                        still_unlocked, embeddings, positions,
                        cluster_tids=cluster_tids,
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
                    cluster_tids=cluster_tids,
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
                camera_motion=camera_motion,
                official_tids=getattr(self, '_official_tids_this_frame', None),
                tid_bboxes=tid_bboxes,
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

            # Accumulate switched PIDs from lock rebinds/takeovers this frame
            for _, pid, _, _, _ in self.identity.locks._switch_log:
                self._switched_pids.add(pid)

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

            # ── VLM Gating: Check embedding drift before re-analyzing ──
            high_drift_pids = []
            should_reanalyze_vlm = False
            vlm_skip_reason = None

            # Check if any tracked player (with valid P-id) has high drift (similarity < threshold)
            if meta_by_tid:
                for tid, meta in meta_by_tid.items():
                    if meta.pid and meta.pid != "UNK":
                        if hasattr(self.identity, 'drift_tracker'):
                            tracker = self.identity.drift_tracker
                            # Check if this P-id has drift data and if latest similarity is below threshold
                            if meta.pid in tracker.pid_history:
                                history = tracker.pid_history[meta.pid]
                                if history and history[-1] < tracker.drift_threshold:
                                    high_drift_pids.append(meta.pid)
                                    should_reanalyze_vlm = True

                # Log VLM gating decision
                if should_reanalyze_vlm:
                    print(f"[VLMGate] Frame {video_frame}: REANALYZE high_drift_pids={high_drift_pids}")
                elif video_frame % 30 == 0:
                    print(f"[VLMGate] Frame {video_frame}: SKIPPED no high drift detected ({len(meta_by_tid)} active)")

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
                    display_id = ("!" + meta.pid) if meta.pid in self._switched_pids else meta.pid
                    
                    slot = self.identity.get_slot(meta.pid)
                    team_id = slot.team_id if slot else None
                    team_confidence = 0.91 if team_id is not None else 0.0
                    consensus = "CONFIRMED" if source == "locked" else "AMBIGUOUS"
                else:
                    # Uncertain — render as raw track (T<id>), no P-id
                    out_pid = None
                    source = "unassigned"
                    identity_valid = False
                    identity_confidence = 0.0
                    player_id = None
                    display_id = "?" if assignment_pending else None
                    team_id = None
                    team_confidence = 0.0
                    consensus = "NEEDS_REVIEW"

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
                    "team_id": team_id,
                    "team_confidence": team_confidence,
                    "role": "player" if player_id else "unknown",
                    "is_official": False,
                    "consensus": consensus,
                })

            officials_list = []
            for ofc in self._officials_this_frame:
                ox1, oy1, ox2, oy2 = ofc.bbox
                ofc_obj = {
                    "trackId": int(ofc.track_id),
                    "rawTrackId": int(ofc.track_id),
                    "playerId": None,
                    "displayId": "REF",
                    "assignment_pending": False,
                    "bbox": [float(ox1), float(oy1), float(ox2), float(oy2)],
                    "confidence": float(ofc.score),
                    "class": int(ofc.cls),
                    "gameState": state.value,
                    "analysis_valid": True,
                    "crop_quality": 1.0,
                    "identity_valid": False,
                    "assignment_source": "official",
                    "identity_confidence": 0.0,
                    "team_id": None,
                    "team_confidence": 0.0,
                    "role": "official",
                    "is_official": True,
                    "consensus": "CONFIRMED",
                }
                officials_list.append(ofc_obj)
                # Officials are NEVER added to players list to prevent identity leakage

            ball_list = []
            last_ball = getattr(self, "_last_ball_det", None)
            if last_ball is not None:
                ball_list.append(last_ball)

            # Enforce 1:1 Identity Invariants
            final_players = []
            seen_raw = {}
            seen_pid = {}
            
            for p in players:
                raw = p.get("rawTrackId")
                pid = p.get("playerId")
                
                # If this raw track ID is already seen...
                if raw in seen_raw:
                    existing = seen_raw[raw]
                    # Keep the one with higher identity confidence
                    if p.get("identity_confidence", 0) > existing.get("identity_confidence", 0):
                        existing["playerId"] = None
                        existing["identity_valid"] = False
                        existing["identity_confidence"] = 0.0
                        existing["consensus"] = "NEEDS_REVIEW"
                        existing["assignment_source"] = "unassigned"
                    else:
                        p["playerId"] = None
                        p["identity_valid"] = False
                        p["identity_confidence"] = 0.0
                        p["consensus"] = "NEEDS_REVIEW"
                        p["assignment_source"] = "unassigned"
                        pid = None

                seen_raw[raw] = p
                
                # If this PID is already seen...
                if pid is not None:
                    if pid in seen_pid:
                        existing = seen_pid[pid]
                        # Keep the one with higher confidence
                        if p.get("identity_confidence", 0) > existing.get("identity_confidence", 0):
                            existing["playerId"] = None
                            existing["identity_valid"] = False
                            existing["identity_confidence"] = 0.0
                            existing["consensus"] = "NEEDS_REVIEW"
                            existing["assignment_source"] = "unassigned"
                        else:
                            p["playerId"] = None
                            p["identity_valid"] = False
                            p["identity_confidence"] = 0.0
                            p["consensus"] = "NEEDS_REVIEW"
                            p["assignment_source"] = "unassigned"
                    else:
                        seen_pid[pid] = p

            for p in players:
                final_players.append(p)

            self.results.append({
                "frameIndex": int(video_frame),
                "players": final_players,
                "officials": officials_list,
                "ball": ball_list,
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


def _resolve_model_path(primary: str, fallback: str) -> str:
    """Pick the first existing path, env-overridable.

    YOLO_MODEL_PATH overrides the primary; YOLO_FALLBACK_PATH overrides
    the fallback. Used by run_tracking for the fallback probe."""
    primary = os.environ.get("YOLO_MODEL_PATH", primary)
    fallback = os.environ.get("YOLO_FALLBACK_PATH", fallback)
    return primary if os.path.exists(primary) else fallback


def _probe_first_frames_for_players(tracker: "TrackerCore", cap, n_frames: int = 3) -> int:
    """Sample n_frames across the video and count player-class detections.

    Used to decide whether to fall back from roboflow_players.pt (which can
    return only ball boxes on out-of-distribution clips) to a COCO yolov8m.
    Restores cap position when done."""
    saved_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    sample_idxs = [0, max(1, total // 4), max(2, total // 2)][:n_frames] if total > 0 else list(range(n_frames))
    n_players = 0
    for fi in sample_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        dets = tracker.detect(frame)
        n_players += len(dets)
    cap.set(cv2.CAP_PROP_POS_FRAMES, saved_pos)
    return n_players


def run_tracking(video_path, job_id, frame_stride=1, max_frames=None, device="cpu", use_identity=True):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {video_path}")

    primary = "/content/roboflow_players.pt" if os.path.exists("/content/roboflow_players.pt") \
              else "models/roboflow_players.pt"
    fallback = "yolov8m.pt" if os.path.exists("yolov8m.pt") else "models/yolov8m.pt"
    yolo_path = _resolve_model_path(primary, fallback)

    tracker = TrackerCore(yolo_path=yolo_path, device=device, use_identity=use_identity)

    # Phase 2: Initialize camera motion detector
    camera_motion_detector = CameraMotionDetector()
    camera_motions = []  # Store for each processed frame

    # Fallback probe: if the primary model returns zero player detections on
    # the first 3 sampled frames, swap to yolov8m (COCO 'person' class) and
    # rebuild the tracker. Only meaningful when primary is a roboflow model.
    if not tracker._is_coco_model and os.path.exists(fallback):
        n_players = _probe_first_frames_for_players(tracker, cap, n_frames=3)
        print(f"[run_tracking] primary-model probe: {n_players} player dets across 3 frames")
        if n_players == 0:
            print(f"[run_tracking] FALLBACK → {fallback} (primary returned 0 player dets)")
            tracker = TrackerCore(yolo_path=fallback, device=device)

    video_frame = 0
    processed = 0
    diagnostics = []  # per-sampled-frame tracker health for Phase A acceptance

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if video_frame % frame_stride == 0:
            # Phase 2: Detect camera motion for this processed frame
            motion = camera_motion_detector.estimate(frame, video_frame)
            camera_motions.append(motion)
            if motion["motion_class"] != "stable":
                print(log_camera_motion(video_frame, motion))

            # Raw YOLO inference for diagnostics — share with tracker.detect via
            # a single-pass that records the histogram, then filters.
            # Disable FP16 (half=False) for YOLO to prevent dtype mismatch during fusion
            yolo_results = tracker.yolo.predict(frame, conf=0.05, verbose=False, half=False)
            yolo_boxes = yolo_results[0].boxes
            if yolo_boxes is None or len(yolo_boxes) == 0:
                raw_classes = []
            else:
                raw_classes = yolo_boxes.cls.cpu().numpy().astype(int).tolist()
            class_histogram = {}
            for c in raw_classes:
                class_histogram[c] = class_histogram.get(c, 0) + 1
            # Now run the standard detect path (re-runs YOLO; cheap on GPU and
            # keeps the existing _last_ball_det side-effect clean).
            dets = tracker.detect(frame)
            save_this = True
            tracker_input_count = len(dets)
            player_candidate_count = sum(class_histogram.get(c, 0)
                                         for c in tracker.player_class_ids)
            diagnostics.append({
                "frameIndex": video_frame,
                "raw_box_count": len(raw_classes),
                "class_histogram": class_histogram,
                "player_candidate_count": player_candidate_count,
                "tracker_input_count": tracker_input_count,
                "active_tracks": -1,  # filled in after process_frame below
                "camera_motion": motion["motion_class"],
            })
        else:
            dets = np.empty((0, 6))
            save_this = False
            motion = None

        n_tracks = tracker.process_frame(frame, video_frame, dets, save=save_this, camera_motion=motion)

        if save_this:
            diagnostics[-1]["active_tracks"] = int(n_tracks)
            processed += 1
            if processed % 10 == 0:
                print(f"Video frame {video_frame:4d} | processed {processed:4d} | "
                      f"dets {len(dets):3d} | tracks {n_tracks:3d}")

        if max_frames and processed >= max_frames:
            break

        video_frame += 1

    cap.release()
    tracker.save(job_id)

    # Phase 2: Save camera motion data
    camera_motion_path = Path(f"temp/{job_id}/tracking/camera_motion.json")
    camera_motion_path.parent.mkdir(parents=True, exist_ok=True)
    with open(camera_motion_path, "w") as f:
        json.dump({
            "total_frames": video_frame,
            "processed_frames": processed,
            "motion_statistics": {
                "stable": sum(1 for m in camera_motions if m["motion_class"] == "stable"),
                "pan": sum(1 for m in camera_motions if m["motion_class"] == "pan"),
                "fast_pan": sum(1 for m in camera_motions if m["motion_class"] == "fast_pan"),
                "cut": sum(1 for m in camera_motions if m["motion_class"] == "cut"),
                "unknown": sum(1 for m in camera_motions if m["motion_class"] == "unknown"),
            },
            "motions": camera_motions,
        }, f, indent=2)
    print(f"\n✓ Camera motion saved to {camera_motion_path}")

    # Write per-frame diagnostics for Phase A acceptance gate
    diag_path = Path(f"temp/{job_id}/tracking/tracker_diagnostics.json")
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    with open(diag_path, "w") as f:
        json.dump({"frames": diagnostics, "count": len(diagnostics)}, f, indent=2)
    # Console summary
    if diagnostics:
        import statistics
        track_counts = [d["active_tracks"] for d in diagnostics if d["active_tracks"] >= 0]
        nonzero_input = sum(1 for d in diagnostics if d["tracker_input_count"] > 0)
        pct_nonzero = nonzero_input / max(len(diagnostics), 1) * 100
        print(f"[diagnostics] {len(diagnostics)} sampled frames | "
              f"tracker_input>0 on {pct_nonzero:.1f}% | "
              f"median active tracks={statistics.median(track_counts) if track_counts else 0:.1f} | "
              f"p10={(sorted(track_counts)[len(track_counts)//10] if len(track_counts)>=10 else min(track_counts) if track_counts else 0):.1f}")

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

    # Persist identity metrics next to track_results.json so phase 2 match_report
    # can read them without re-parsing logs.
    try:
        import json as _json
        metrics_path = Path(f"temp/{job_id}/tracking/identity_metrics.json")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_payload = {
            **{k: v for k, v in summary.items()},
            "valid_identity_frames": valid,
            "unknown_boxes": invalid,
            "valid_id_coverage": valid_id_coverage,
            "frames_in_collapse": getattr(tracker, "_frames_in_collapse", 0),
            "frames_processed": processed,
        }
        with open(metrics_path, "w") as fh:
            _json.dump(metrics_payload, fh, indent=2, default=str)
        print(f"[IdentityMetrics] persisted to {metrics_path}")
    except Exception as e:
        print(f"[IdentityMetrics] persist failed: {e}")

    # Export drift report if tracking completed
    if hasattr(tracker.identity, 'drift_tracker'):
        try:
            drift_report = tracker.identity.drift_tracker.export_report()
            drift_report_path = Path(f"temp/{job_id}/tracking/drift_report.json")
            drift_report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(drift_report_path, "w") as f:
                json.dump(drift_report, f, indent=2, default=str)
            print(f"[DriftReport] exported to {drift_report_path}")
            print(f"[DriftReport] players tracked: {len(drift_report.get('players', {}))}")
        except Exception as e:
            print(f"[DriftReport] ERROR exporting: {e}")

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
