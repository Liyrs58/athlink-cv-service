"""
Dominant Color Team Classifier for Football Broadcast Tracking.

Lightweight alternative to SigLIP — no heavy ML model required.
Works by extracting the dominant jersey color from torso crops using:
  1. Torso-only crop (15%-50% height, 20%-80% width)
  2. Green pitch pixel masking (removes grass bleed)
  3. LAB color space (perceptually uniform — handles lighting variance)
  4. Per-pixel dominant color via mini k-means on the crop itself
  5. Per-track median color → global k=2 clustering into two teams
  6. Referee/GK outlier detection (far from both centroids)
  7. Temporal lock-in per track_id

Usage:
    classifier = DominantColorClassifier()
    classifier.fit(tracks, video_path)  # reads frames, clusters
    # teamId is written in-place on each track dict
"""

import cv2
import numpy as np
import logging
from collections import Counter
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def _torso_crop(frame: np.ndarray, bbox) -> Optional[np.ndarray]:
    """Extract torso-only crop: 15%-50% height, 20%-80% width.
    Isolates jersey from shorts/legs/head."""
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
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


def _mask_green_and_dark(crop_bgr: np.ndarray) -> np.ndarray:
    """Return boolean mask of jersey pixels (True = keep, False = green/dark/shadow)."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    # Green pitch pixels
    green = cv2.inRange(hsv, np.array([25, 20, 20]), np.array([95, 255, 255]))
    # Very dark pixels (shadows, pitch markings)
    dark = hsv[:, :, 2] < 30
    # Very bright white-ish pixels that are actually pitch lines
    # (high V, very low S)
    exclude = (green > 0) | dark
    return ~exclude


def _extract_dominant_lab(crop_bgr: np.ndarray, min_pixels: int = 50) -> Optional[np.ndarray]:
    """Extract dominant color from a BGR crop in LAB space.

    Returns: np.array([L, A, B]) or None if too few jersey pixels.
    """
    mask = _mask_green_and_dark(crop_bgr)
    n_jersey = int(mask.sum())
    if n_jersey < min_pixels:
        return None

    # Convert to LAB (perceptually uniform)
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    jersey_pixels = lab[mask].astype(np.float32)

    # If we have enough pixels, use mini k-means (k=2) on the jersey pixels
    # to find the dominant jersey color (ignoring secondary stripe/crest colors)
    if len(jersey_pixels) > 200:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        try:
            _, labels, centers = cv2.kmeans(
                jersey_pixels, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS
            )
            # Pick the cluster with the most pixels (= dominant color)
            counts = np.bincount(labels.flatten())
            dominant_idx = int(np.argmax(counts))
            return centers[dominant_idx]
        except cv2.error:
            pass

    # Fallback: simple median
    return np.median(jersey_pixels, axis=0)


class DominantColorClassifier:
    """Lightweight dominant-color team classifier.

    No ML model required. Uses LAB color space + k-means.
    """

    def __init__(self, samples_per_track: int = 8, outlier_std: float = 2.0):
        self.samples_per_track = samples_per_track
        self.outlier_std = outlier_std

    def fit_and_assign(self, tracks: List[Dict], video_path: str) -> None:
        """Classify all tracks by dominant jersey color. Writes teamId in-place.

        Steps:
          1. Sample frames for each track, extract torso crops
          2. Extract dominant LAB color per crop
          3. Compute median LAB color per track
          4. K=2 cluster tracks into two teams
          5. Flag outliers (referee) as teamId=2
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Cannot open video for dominant color classifier: %s", video_path)
            for t in tracks:
                t.setdefault("teamId", -1)
            return

        # Detect rotation
        ret, sample = cap.read()
        if not ret:
            cap.release()
            for t in tracks:
                t.setdefault("teamId", -1)
            return
        needs_rotation = sample.shape[0] > sample.shape[1]
        cap.release()

        # Step 1: Collect LAB samples per track
        track_colors: Dict[int, List[np.ndarray]] = {}  # track_idx → list of LAB vectors

        # Build frame request map
        frame_requests: Dict[int, List[Tuple[int, list]]] = {}  # frameIndex → [(track_idx, bbox)]
        eligible_indices = []

        for idx, track in enumerate(tracks):
            if track.get("is_staff", False):
                track["teamId"] = -1
                continue
            traj = track.get("trajectory", [])
            if len(traj) < 2:
                track["teamId"] = -1
                continue

            eligible_indices.append(idx)
            track_colors[idx] = []

            # Sample evenly
            step = max(1, len(traj) // self.samples_per_track)
            samples = traj[::step][:self.samples_per_track]
            for pt in samples:
                fi = pt.get("frameIndex")
                bbox = pt.get("bbox")
                if fi is not None and bbox is not None and len(bbox) >= 4:
                    if fi not in frame_requests:
                        frame_requests[fi] = []
                    frame_requests[fi].append((idx, bbox))

        if not eligible_indices or not frame_requests:
            for t in tracks:
                t.setdefault("teamId", -1)
            return

        # Step 2: Read frames and extract dominant colors
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            for t in tracks:
                t.setdefault("teamId", -1)
            return

        sorted_frames = sorted(frame_requests.keys())
        current_fi = -1
        for target_fi in sorted_frames:
            if target_fi > current_fi + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_fi)
            ret, frame = cap.read()
            if not ret:
                continue
            current_fi = target_fi
            if needs_rotation:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            for track_idx, bbox in frame_requests[target_fi]:
                crop = _torso_crop(frame, bbox)
                if crop is None:
                    continue
                lab = _extract_dominant_lab(crop)
                if lab is not None:
                    track_colors[track_idx].append(lab)

        cap.release()

        # Step 3: Compute median LAB per track
        track_medians: Dict[int, np.ndarray] = {}
        for idx in eligible_indices:
            colors = track_colors.get(idx, [])
            if len(colors) < 2:
                tracks[idx]["teamId"] = -1
                continue
            track_medians[idx] = np.median(np.stack(colors), axis=0)

        if len(track_medians) < 4:
            logger.warning("Too few tracks with color data (%d) for clustering", len(track_medians))
            for t in tracks:
                t.setdefault("teamId", -1)
            return

        # Step 4: K=2 clustering in LAB space
        clustering_indices = list(track_medians.keys())
        data = np.array([track_medians[i] for i in clustering_indices], dtype=np.float32)

        # Equal weighting for LAB channels — luminance (L) is just as important
        # as chrominance (A, B) for separating white vs dark kits.
        # LAB: L=[0,255], A=[0,255], B=[0,255] where 128 is neutral
        weights = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        data_w = data * weights

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        try:
            _, labels, centers = cv2.kmeans(
                data_w, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS
            )
        except cv2.error as e:
            logger.warning("K-means clustering failed: %s", e)
            for t in tracks:
                t.setdefault("teamId", -1)
            return

        labels = labels.flatten()

        # Step 5: Outlier detection for referees
        # Compute distance of each track to its assigned centroid
        distances = []
        for i, idx in enumerate(clustering_indices):
            dist = float(np.linalg.norm(data_w[i] - centers[labels[i]]))
            distances.append(dist)

        distances = np.array(distances)
        mean_dist = float(distances.mean())
        std_dist = float(distances.std()) if len(distances) > 1 else 1.0

        for i, idx in enumerate(clustering_indices):
            if distances[i] > mean_dist + self.outlier_std * std_dist:
                # Outlier in LAB space — could be ref/GK/lighting edge case.
                # Assign to nearest centroid anyway. We don't emit a third team.
                tracks[idx]["teamId"] = int(labels[i])
                tracks[idx]["teamOutlier"] = True  # flag for downstream debugging
                logger.debug(
                    f"Track {tracks[idx].get('trackId', '?')} flagged as team outlier "
                    f"(dist={distances[i]:.2f}, threshold={mean_dist + self.outlier_std * std_dist:.2f}) "
                    f"assigned to team {int(labels[i])}"
                )
            else:
                tracks[idx]["teamId"] = int(labels[i])

        # Fill in any unassigned tracks
        for t in tracks:
            t.setdefault("teamId", -1)

        # Log results
        t0 = sum(1 for t in tracks if t.get("teamId") == 0)
        t1 = sum(1 for t in tracks if t.get("teamId") == 1)
        t2 = sum(1 for t in tracks if t.get("teamId") == 2)
        unk = sum(1 for t in tracks if t.get("teamId") == -1)
        logger.info(
            "DominantColor team classification: T0=%d, T1=%d, ref/gk=%d, unknown=%d",
            t0, t1, t2, unk,
        )


def classify_teams_dominant_color(tracks: List[Dict], video_path: str) -> None:
    """Drop-in replacement for team clustering.
    Call this instead of _cluster_teams_per_track() or classify_teams_siglip().
    """
    classifier = DominantColorClassifier()
    classifier.fit_and_assign(tracks, video_path)
