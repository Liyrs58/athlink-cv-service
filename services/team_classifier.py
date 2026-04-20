"""
SigLIP-based Team Classifier for Football Broadcast Tracking.

Replaces naive RGB/HSV k-means with:
  1. Torso-only crops (upper 15-50% of bbox, middle 60% width)
  2. SigLIP vision encoder → 768-dim embeddings
  3. UMAP dimensionality reduction → 3 dims
  4. K-Means with K=2 (two outfield teams only)
  5. Temporal smoothing: majority-vote per track_id, lock after N observations
  6. Referee/GK detected as outliers (far from both centroids)

Usage:
    classifier = TeamClassifier()
    classifier.fit(fit_frames, fit_detections)  # ~30-100 frames
    team = classifier.classify(frame, bbox, track_id)  # per detection
"""

import os
import logging
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Lazy-load heavy deps to avoid startup penalty
_torch = None
_siglip_model = None
_siglip_processor = None

SIGLIP_MODEL_ID = os.getenv("SIGLIP_MODEL", "google/siglip-base-patch16-224")
DEVICE = None  # auto-detect


def _get_device():
    global DEVICE
    if DEVICE is not None:
        return DEVICE
    try:
        import torch
        if torch.cuda.is_available():
            DEVICE = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            DEVICE = "mps"
        else:
            DEVICE = "cpu"
    except ImportError:
        DEVICE = "cpu"
    return DEVICE


def _get_siglip():
    """Lazy-load SigLIP model + processor."""
    global _siglip_model, _siglip_processor, _torch
    if _siglip_model is not None:
        return _siglip_model, _siglip_processor

    import torch as torch_mod
    _torch = torch_mod

    try:
        from transformers import AutoProcessor, SiglipVisionModel
    except ImportError:
        raise ImportError(
            "transformers package required. Install: pip install transformers"
        )

    device = _get_device()
    logger.info("Loading SigLIP model %s on %s...", SIGLIP_MODEL_ID, device)
    _siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_ID)
    _siglip_model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_ID).to(device).eval()
    logger.info("SigLIP model loaded successfully.")
    return _siglip_model, _siglip_processor


def _torso_crop(frame: np.ndarray, bbox) -> Optional[np.ndarray]:
    """Extract torso-only crop: upper 15%-50% vertically, middle 20%-80% horizontally.

    This isolates the jersey from shorts/legs (which dominate naive full-bbox crops)
    and removes the head/sky above.
    """
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    h, w = y2 - y1, x2 - x1
    if h < 20 or w < 10:
        return None

    # Clamp to frame bounds
    fh, fw = frame.shape[:2]
    x1, x2 = max(0, x1), min(fw, x2)
    y1, y2 = max(0, y1), min(fh, y2)
    h, w = y2 - y1, x2 - x1
    if h < 20 or w < 10:
        return None

    ty1 = y1 + int(h * 0.15)
    ty2 = y1 + int(h * 0.50)
    tx1 = x1 + int(w * 0.20)
    tx2 = x1 + int(w * 0.80)

    if ty2 <= ty1 or tx2 <= tx1:
        return None

    crop = frame[ty1:ty2, tx1:tx2]
    return crop if crop.size > 0 else None


def _embed_batch(crops: List[np.ndarray], batch_size: int = 32) -> np.ndarray:
    """Embed a list of BGR crops through SigLIP → (N, 768) float32 array."""
    model, processor = _get_siglip()
    device = _get_device()

    # SigLIP expects RGB PIL-like images
    rgb_crops = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops]

    all_embeddings = []
    for i in range(0, len(rgb_crops), batch_size):
        batch = rgb_crops[i : i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with _torch.no_grad():
            outputs = model(**inputs)
        # pooler_output: (batch, 768)
        embs = outputs.pooler_output.cpu().numpy()
        all_embeddings.append(embs)

    return np.vstack(all_embeddings)


class TeamClassifier:
    """SigLIP + UMAP + KMeans team classifier with temporal smoothing.

    Call fit() once on ~30-100 frames, then classify() per detection.
    """

    def __init__(
        self,
        smoothing_window: int = 10,
        outlier_threshold: float = 2.5,
        n_clusters: int = 2,
    ):
        self.smoothing_window = smoothing_window
        self.outlier_threshold = outlier_threshold
        self.n_clusters = n_clusters

        self._reducer = None  # UMAP, lazy init
        self._kmeans = None  # sklearn KMeans, lazy init
        self._fitted = False

        # Temporal state per track
        self.track_history: Dict[int, List[int]] = defaultdict(list)
        self.track_locked: Dict[int, int] = {}

        # Cache embeddings per track to avoid re-computing
        self._track_embedding_cache: Dict[int, np.ndarray] = {}

    def _ensure_sklearn_umap(self):
        """Lazy-import UMAP and sklearn."""
        if self._reducer is not None:
            return

        try:
            import umap
        except ImportError:
            raise ImportError("umap-learn required. Install: pip install umap-learn")
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            raise ImportError("scikit-learn required. Install: pip install scikit-learn")

        self._reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
        self._kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)

    def fit(
        self,
        frames: List[np.ndarray],
        detections: List[List[dict]],
    ) -> None:
        """Fit the team classifier on a set of frames + detections.

        Args:
            frames: List of BGR frames.
            detections: detections[i] = [{"bbox": [x1,y1,x2,y2], "track_id": int}, ...]
                        One list per frame.

        Requires at least 20 valid torso crops to fit.
        """
        self._ensure_sklearn_umap()

        crops = []
        for frame, dets in zip(frames, detections):
            for d in dets:
                c = _torso_crop(frame, d["bbox"])
                if c is not None:
                    crops.append(c)

        if len(crops) < 20:
            logger.warning(
                "Only %d torso crops collected (need 20+). "
                "Team classification may be unreliable.",
                len(crops),
            )
            if len(crops) < 4:
                raise ValueError(
                    f"Not enough torso crops ({len(crops)}). Need at least 4 for k=2."
                )

        logger.info("Fitting SigLIP team classifier on %d torso crops...", len(crops))

        # Embed through SigLIP
        embeddings = _embed_batch(crops)
        logger.info("SigLIP embeddings: shape %s", embeddings.shape)

        # UMAP reduction
        reduced = self._reducer.fit_transform(embeddings)
        logger.info("UMAP reduced: shape %s", reduced.shape)

        # K-Means clustering
        self._kmeans.fit(reduced)
        self._fitted = True

        # Log cluster sizes
        labels = self._kmeans.labels_
        for c in range(self.n_clusters):
            count = int((labels == c).sum())
            logger.info("  Cluster %d: %d samples", c, count)

        logger.info("TeamClassifier fitted successfully.")

    def classify(self, frame: np.ndarray, bbox, track_id: int) -> int:
        """Classify a single detection.

        Returns:
            0 = team A, 1 = team B, -1 = referee/goalkeeper/unknown
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before classify().")

        # Lock-in: if track has enough confident history, return cached
        if track_id in self.track_locked:
            return self.track_locked[track_id]

        # Check embedding cache — only embed once per track
        if track_id in self._track_embedding_cache:
            reduced = self._reducer.transform(
                self._track_embedding_cache[track_id].reshape(1, -1)
            )
        else:
            crop = _torso_crop(frame, bbox)
            if crop is None:
                return -1

            emb = _embed_batch([crop])
            self._track_embedding_cache[track_id] = emb[0]
            reduced = self._reducer.transform(emb)

        # Distance to both centroids
        dists = np.linalg.norm(
            reduced - self._kmeans.cluster_centers_, axis=1
        )
        sorted_dists = np.sort(dists)
        min_dist = sorted_dists[0]
        second_dist = sorted_dists[1] if len(sorted_dists) > 1 else float("inf")

        # Outlier rejection: referee/GK if too far from nearest centroid
        # or if the two distances are too similar (ambiguous)
        if min_dist > self.outlier_threshold or (second_dist - min_dist) < 0.3:
            team = -1
        else:
            team = int(self._kmeans.predict(reduced)[0])

        # Accumulate history
        self.track_history[track_id].append(team)

        # Lock in after smoothing_window observations
        if len(self.track_history[track_id]) >= self.smoothing_window:
            majority = Counter(self.track_history[track_id]).most_common(1)[0][0]
            self.track_locked[track_id] = majority
            return majority

        # Early frames: return running majority
        return Counter(self.track_history[track_id]).most_common(1)[0][0]

    def classify_batch_tracks(
        self,
        tracks: List[Dict],
        video_path: str,
        sample_frames: int = 50,
    ) -> None:
        """Batch classify all tracks from a completed tracking run.

        This is the drop-in replacement for _cluster_teams_per_track().
        It:
          1. Samples N frames evenly from the video
          2. Fits the classifier on all visible players
          3. Classifies each track and writes teamId in-place

        Args:
            tracks: List of track dicts with 'trajectory' entries containing 'bbox' and 'frameIndex'.
            video_path: Path to the source video.
            sample_frames: Number of frames to sample for fitting.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Cannot open video for team classification: %s", video_path)
            for t in tracks:
                t.setdefault("teamId", -1)
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            for t in tracks:
                t.setdefault("teamId", -1)
            return

        # Build a frame→track lookup for quick access
        frame_to_dets: Dict[int, List[dict]] = defaultdict(list)
        for t in tracks:
            for entry in (t.get("trajectory") or []):
                fi = entry.get("frameIndex")
                bbox = entry.get("bbox")
                if fi is not None and bbox is not None:
                    frame_to_dets[fi].append({
                        "bbox": bbox,
                        "track_id": t.get("trackId", -1),
                    })

        available_frames = sorted(frame_to_dets.keys())
        if not available_frames:
            cap.release()
            for t in tracks:
                t.setdefault("teamId", -1)
            return

        # Sample frames evenly for fitting
        step = max(1, len(available_frames) // sample_frames)
        fit_frame_indices = available_frames[::step][:sample_frames]

        # Read the actual frames
        fit_frames = []
        fit_dets = []
        for fi in fit_frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                continue
            fit_frames.append(frame)
            fit_dets.append(frame_to_dets[fi])

        if not fit_frames:
            cap.release()
            for t in tracks:
                t.setdefault("teamId", -1)
            return

        # Step 1: Fit
        try:
            self.fit(fit_frames, fit_dets)
        except (ValueError, ImportError) as e:
            logger.warning("SigLIP team classifier fit failed: %s. Falling back to HSV.", e)
            cap.release()
            # Fall back to old method
            from services.tracking_service import _cluster_teams_per_track
            _cluster_teams_per_track(tracks, k=2)
            return

        # Step 2: Classify each track using a representative frame
        for t in tracks:
            traj = t.get("trajectory") or []
            if not traj:
                t["teamId"] = -1
                continue

            # Pick the middle trajectory entry as representative
            mid = traj[len(traj) // 2]
            fi = mid.get("frameIndex")
            bbox = mid.get("bbox")
            track_id = t.get("trackId", -1)

            if fi is None or bbox is None:
                t["teamId"] = -1
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                t["teamId"] = -1
                continue

            team = self.classify(frame, bbox, track_id)
            if team >= 0:
                t["teamId"] = team
            else:
                # SigLIP returned -1 (no confident assignment).
                # Use last locked team for this track, else default to team 0.
                prev = self.track_locked.get(track_id)
                t["teamId"] = prev if prev is not None and prev >= 0 else 0
                t["teamOutlier"] = True

        cap.release()

        # Log results
        team_counts = Counter(t.get("teamId", -1) for t in tracks)
        logger.info("SigLIP team classification results: %s", dict(team_counts))


def classify_teams_siglip(tracks: List[Dict], video_path: str) -> None:
    """Drop-in replacement for _cluster_teams_per_track().

    Call this instead of _cluster_teams_per_track(filtered, k=3) at line 1624
    of tracking_service.py.
    """
    classifier = TeamClassifier()
    classifier.classify_batch_tracks(tracks, video_path)
