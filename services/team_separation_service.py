"""
Team Separation Service — assigns players to teams using jersey colour clustering.

Reads frames directly from the video file (no pre-extracted frames needed).
Uses the same HSV colour feature approach as team_service.py but designed
for the /analyse pipeline where only a video path is available.

No sklearn — uses numpy k-means from team_service.py.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

UPPER_BODY_RATIO = 0.50   # top 50% of bbox — jersey only
HORIZ_TRIM = 0.20         # trim 20% each side — tight crop to torso centre
MIN_JERSEY_PIXELS = 80    # minimum non-green pixels to trust the sample
SAMPLES_PER_TRACK = 8     # frames to sample per track


# ---------------------------------------------------------------------------
# HSV colour name mapping
# ---------------------------------------------------------------------------

def hsv_to_colour_name(h: float, s: float, v: float) -> str:
    """
    Convert HSV values (OpenCV scale: H 0-180, S 0-255, V 0-255) to a
    human-readable colour name for coaching reports.
    """
    if v < 50:
        return "black"
    if s < 40:
        return "white" if v > 180 else "grey"

    # Hue wheel (OpenCV: 0-180)
    if h < 12 or h >= 165:
        return "red"
    if h < 20:
        return "dark red"
    if h < 28:
        return "orange"
    if h < 35:
        return "yellow"
    if h < 50:
        return "lime green"
    if h < 85:
        return "green"
    if h < 100:
        return "teal"
    if h < 130:
        return "blue"
    if h < 145:
        return "dark blue"
    if h < 165:
        return "purple"
    return "unknown"


def _detailed_colour_name(h: float, s: float, v: float) -> str:
    """More descriptive name combining brightness and hue."""
    base = hsv_to_colour_name(h, s, v)

    if base in ("black", "white", "grey"):
        return base

    if v < 100:
        return f"dark {base}"
    if s < 80 and v > 180:
        return f"light {base}"
    return base


# ---------------------------------------------------------------------------
# Colour extraction from a single frame crop
# ---------------------------------------------------------------------------

def extract_player_colour(frame: np.ndarray, bbox: list) -> Optional[np.ndarray]:
    """
    Extract HSV colour vector from the jersey region of a player.

    Args:
        frame: BGR image (full video frame)
        bbox: [x1, y1, x2, y2] pixel coordinates

    Returns:
        np.array([norm_hue, sat_ratio, brightness]) or None if extraction fails.
        norm_hue   ∈ [0,1] — median hue of high-saturation pixels / 180
        sat_ratio  ∈ [0,1] — fraction of jersey pixels with saturation > 60
        brightness ∈ [0,1] — mean Value channel / 255
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h_img, w_img = frame.shape[:2]

    # Clamp to frame bounds
    x1, x2 = max(0, x1), min(w_img, x2)
    y1, y2 = max(0, y1), min(h_img, y2)

    box_h = y2 - y1
    box_w = x2 - x1
    if box_h < 8 or box_w < 8:
        return None

    # Crop: upper 55% height (jersey), trim 15% each side (avoid arms)
    trim = int(box_w * HORIZ_TRIM)
    crop = frame[y1: y1 + int(box_h * UPPER_BODY_RATIO),
                 x1 + trim: x2 - trim]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Mask out pitch green (aggressive — any pixel with green-ish hue)
    green_mask = cv2.inRange(hsv,
                             np.array([25, 20, 20]),
                             np.array([95, 255, 255]))
    # Also mask very dark pixels (shadows, pitch markings)
    dark_mask = hsv[:, :, 2] < 30
    exclude = (green_mask > 0) | dark_mask
    jersey_px = ~exclude

    total_px = int(np.sum(jersey_px))
    if total_px < MIN_JERSEY_PIXELS:
        return None

    sat_vals = hsv[:, :, 1][jersey_px].astype(float)
    val_vals = hsv[:, :, 2][jersey_px].astype(float)
    hue_vals = hsv[:, :, 0][jersey_px].astype(float)

    sat_ratio = float(np.mean(sat_vals > 60))
    brightness = float(np.mean(val_vals)) / 255.0

    # Median hue from high-saturation pixels only (stable for coloured kits)
    hi_sat = sat_vals > 60
    if hi_sat.sum() >= 10:
        median_hue = float(np.median(hue_vals[hi_sat])) / 180.0
    else:
        median_hue = 0.0  # white/grey kit — no dominant hue

    return np.array([median_hue, sat_ratio, brightness])


# ---------------------------------------------------------------------------
# Raw HSV centroid extraction (for reporting team colours)
# ---------------------------------------------------------------------------

def _extract_raw_hsv(frame: np.ndarray, bbox: list) -> Optional[np.ndarray]:
    """
    Extract raw median HSV values (not normalised) from jersey region.
    Used to report team colours in human-readable form.
    Returns np.array([H, S, V]) in OpenCV scale (H:0-180, S:0-255, V:0-255).
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h_img, w_img = frame.shape[:2]
    x1, x2 = max(0, x1), min(w_img, x2)
    y1, y2 = max(0, y1), min(h_img, y2)

    box_h = y2 - y1
    box_w = x2 - x1
    if box_h < 8 or box_w < 8:
        return None

    trim = int(box_w * HORIZ_TRIM)
    crop = frame[y1: y1 + int(box_h * UPPER_BODY_RATIO),
                 x1 + trim: x2 - trim]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    green_mask = cv2.inRange(hsv,
                             np.array([25, 20, 20]),
                             np.array([95, 255, 255]))
    dark_mask = hsv[:, :, 2] < 30
    exclude = (green_mask > 0) | dark_mask
    jersey_px = ~exclude
    if int(np.sum(jersey_px)) < MIN_JERSEY_PIXELS:
        return None

    h = float(np.median(hsv[:, :, 0][jersey_px]))
    s = float(np.median(hsv[:, :, 1][jersey_px]))
    v = float(np.median(hsv[:, :, 2][jersey_px]))
    return np.array([h, s, v])


# ---------------------------------------------------------------------------
# K-means (numpy only, k-means++ init)
# ---------------------------------------------------------------------------

def _kmeans(data: np.ndarray, k: int = 2, max_iters: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """K-means clustering with k-means++ initialisation. No sklearn."""
    rng = np.random.RandomState(42)
    n = data.shape[0]

    # k-means++ seeding
    centroids = [data[rng.randint(n)]]
    for _ in range(k - 1):
        dists = np.array([
            min(np.linalg.norm(x - c) ** 2 for c in centroids)
            for x in data
        ])
        probs = dists / dists.sum()
        centroids.append(data[rng.choice(n, p=probs)])
    centroids = np.array(centroids)

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iters):
        dists = np.linalg.norm(
            data[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2
        )
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for i in range(k):
            mask = labels == i
            if mask.any():
                centroids[i] = data[mask].mean(axis=0)

    return labels, centroids


# ---------------------------------------------------------------------------
# Main pipeline: cluster_teams
# ---------------------------------------------------------------------------

def cluster_teams(tracks: List[dict], video_path: str) -> Dict:
    """
    Run the full team separation pipeline.

    Strategy:
      1. If tracks already have teamId from tracking_service (SigLIP or HSV),
         skip re-clustering and only compute colour names for reporting.
      2. Otherwise, try SigLIP classifier first, then fall back to HSV k-means.

    Args:
        tracks: list of track dicts from run_tracking() — modified in place
        video_path: path to the video file

    Returns:
        {
            "status": "ok" | "failed",
            "team_0_players": int,
            "team_1_players": int,
            "team_0_colour_hsv": [h, s, v],
            "team_1_colour_hsv": [h, s, v],
            "team_0_colour_name": str,
            "team_1_colour_name": str,
        }
    """
    # Check if tracks already have valid team assignments from tracking_service
    t0_pre = sum(1 for t in tracks if t.get("teamId") == 0)
    t1_pre = sum(1 for t in tracks if t.get("teamId") == 1)
    already_assigned = (t0_pre >= 3 and t1_pre >= 3)

    if not already_assigned:
        # No valid pre-assignment — cascade: Dominant Color → SigLIP → HSV
        logger.info("No pre-existing team assignment found. Running dominant color classifier...")
        assigned = False
        try:
            from services.dominant_color_classifier import classify_teams_dominant_color
            classify_teams_dominant_color(tracks, video_path)
            t0 = sum(1 for t in tracks if t.get("teamId") == 0)
            t1 = sum(1 for t in tracks if t.get("teamId") == 1)
            if t0 >= 3 and t1 >= 3:
                assigned = True
                logger.info("Dominant color team classification complete (T0=%d, T1=%d).", t0, t1)
        except Exception as e:
            logger.warning("Dominant color classifier failed (%s)", e)

        if not assigned:
            try:
                from services.team_classifier import classify_teams_siglip
                classify_teams_siglip(tracks, video_path)
                logger.info("SigLIP team classification complete.")
            except (ImportError, Exception) as e:
                logger.warning("SigLIP classifier failed (%s), falling back to HSV k-means", e)
                _hsv_cluster_teams(tracks, video_path)
    else:
        logger.info("Tracks already have team assignments (T0=%d, T1=%d) — skipping re-clustering", t0_pre, t1_pre)

    # Validate final assignment
    if not validate_separation(tracks):
        logger.warning("Team separation validation failed — fewer than 3 players in one team")
        # Don't wipe existing assignments if they're partially useful
        t0_count = sum(1 for t in tracks if t.get("teamId") == 0)
        t1_count = sum(1 for t in tracks if t.get("teamId") == 1)
        if t0_count == 0 and t1_count == 0:
            _mark_all_unknown(tracks)
            return _failed_result()

    # ── Compute team colours for reporting (always runs) ──────────────
    return _compute_team_colours(tracks, video_path)


def _hsv_cluster_teams(tracks: List[dict], video_path: str) -> None:
    """HSV-based k-means team clustering (fallback). Modifies tracks in-place."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Could not open video for team separation: %s", video_path)
        _mark_all_unknown(tracks)
        return

    # Detect portrait orientation
    ret, sample = cap.read()
    if not ret or sample is None:
        cap.release()
        _mark_all_unknown(tracks)
        return
    needs_rotation = sample.shape[0] > sample.shape[1]
    cap.release()

    # Pre-compute which frames we need to read
    frame_requests = {}  # frame_index → list of (track_idx, bbox)
    confirmed_indices = []

    for track_idx, track in enumerate(tracks):
        if track.get("is_staff", False):
            track["teamId"] = -1
            continue
        confirmed = track.get("confirmed_detections", 0)
        if not isinstance(confirmed, (int, float)) or confirmed < 5:
            track["teamId"] = -1
            continue
        traj = track.get("trajectory", [])
        if len(traj) < 2:
            track["teamId"] = -1
            continue
        confirmed_indices.append(track_idx)
        step = max(1, len(traj) // SAMPLES_PER_TRACK)
        samples = traj[::step][:SAMPLES_PER_TRACK]
        for pt in samples:
            fi = pt["frameIndex"]
            if fi not in frame_requests:
                frame_requests[fi] = []
            frame_requests[fi].append((track_idx, pt["bbox"]))

    if not confirmed_indices or not frame_requests:
        _mark_all_unknown(tracks)
        return

    sorted_frames = sorted(frame_requests.keys())
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        _mark_all_unknown(tracks)
        return

    feature_samples = {i: [] for i in confirmed_indices}
    current_frame_idx = -1
    for target_fi in sorted_frames:
        if target_fi > current_frame_idx + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_fi)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        current_frame_idx = target_fi
        if needs_rotation:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        for track_idx, bbox in frame_requests[target_fi]:
            feat = extract_player_colour(frame, bbox)
            if feat is not None:
                feature_samples[track_idx].append(feat)
    cap.release()

    feature_vectors = {}
    for track_idx in confirmed_indices:
        samples = feature_samples[track_idx]
        if not samples:
            tracks[track_idx]["teamId"] = -1
            continue
        feature_vectors[track_idx] = np.median(samples, axis=0)

    if len(feature_vectors) < 4:
        _mark_all_unknown(tracks)
        return

    clustering_indices = list(feature_vectors.keys())
    data = np.array([feature_vectors[i] for i in clustering_indices])
    weights = np.array([2.0, 1.5, 0.5])
    data_w = data * weights

    labels, centroids_w = _kmeans(data_w, k=2)
    for i, track_idx in enumerate(clustering_indices):
        tracks[track_idx]["teamId"] = int(labels[i])

    for track in tracks:
        track.setdefault("teamId", -1)


def _compute_team_colours(tracks: List[dict], video_path: str) -> Dict:
    """Extract team colour names from frames using HSV. Does NOT modify teamId."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return _colour_result_from_counts(tracks)

    ret, sample = cap.read()
    if not ret:
        cap.release()
        return _colour_result_from_counts(tracks)
    needs_rotation = sample.shape[0] > sample.shape[1]
    cap.release()

    # Collect raw HSV samples per team
    team_0_raw = []
    team_1_raw = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return _colour_result_from_counts(tracks)

    # Sample middle trajectory point from each track
    for track in tracks:
        tid = track.get("teamId", -1)
        if tid not in (0, 1):
            continue
        traj = track.get("trajectory", [])
        if not traj:
            continue
        mid = traj[len(traj) // 2]
        fi = mid.get("frameIndex")
        bbox = mid.get("bbox")
        if fi is None or bbox is None:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        if needs_rotation:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        raw = _extract_raw_hsv(frame, bbox)
        if raw is not None:
            if tid == 0:
                team_0_raw.append(raw)
            else:
                team_1_raw.append(raw)

    cap.release()

    t0_hsv = np.median(team_0_raw, axis=0).tolist() if team_0_raw else [0, 0, 0]
    t1_hsv = np.median(team_1_raw, axis=0).tolist() if team_1_raw else [0, 0, 0]

    t0_count = sum(1 for t in tracks if t.get("teamId") == 0)
    t1_count = sum(1 for t in tracks if t.get("teamId") == 1)

    t0_base = hsv_to_colour_name(*t0_hsv)
    t1_base = hsv_to_colour_name(*t1_hsv)

    if t0_base == t1_base:
        if t0_hsv[2] < t1_hsv[2]:
            t0_name = f"dark {t0_base}"
            t1_name = f"light {t1_base}"
        else:
            t0_name = f"light {t0_base}"
            t1_name = f"dark {t1_base}"
    else:
        t0_name = _detailed_colour_name(*t0_hsv)
        t1_name = _detailed_colour_name(*t1_hsv)

    logger.info(
        "Team separation: team_0=%d (%s, HSV=[%.0f,%.0f,%.0f]), "
        "team_1=%d (%s, HSV=[%.0f,%.0f,%.0f])",
        t0_count, t0_name, *t0_hsv,
        t1_count, t1_name, *t1_hsv,
    )

    return {
        "status": "ok",
        "team_0_players": t0_count,
        "team_1_players": t1_count,
        "team_0_colour_hsv": [round(v, 1) for v in t0_hsv],
        "team_1_colour_hsv": [round(v, 1) for v in t1_hsv],
        "team_0_colour_name": t0_name,
        "team_1_colour_name": t1_name,
    }


def _colour_result_from_counts(tracks: List[dict]) -> Dict:
    """Minimal result when video is unavailable for colour extraction."""
    t0 = sum(1 for t in tracks if t.get("teamId") == 0)
    t1 = sum(1 for t in tracks if t.get("teamId") == 1)
    return {
        "status": "ok" if (t0 >= 3 and t1 >= 3) else "partial",
        "team_0_players": t0,
        "team_1_players": t1,
        "team_0_colour_hsv": [0, 0, 0],
        "team_1_colour_hsv": [0, 0, 0],
        "team_0_colour_name": "unknown",
        "team_1_colour_name": "unknown",
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_separation(tracks: List[dict]) -> bool:
    """Returns True if both teams have at least 3 players."""
    t0 = sum(1 for t in tracks if t.get("teamId") == 0)
    t1 = sum(1 for t in tracks if t.get("teamId") == 1)
    return t0 >= 3 and t1 >= 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mark_all_unknown(tracks: List[dict]):
    """Mark all tracks as team unknown."""
    for t in tracks:
        t["teamId"] = -1


def _failed_result() -> Dict:
    return {
        "status": "failed",
        "team_0_players": 0,
        "team_1_players": 0,
        "team_0_colour_hsv": [0, 0, 0],
        "team_1_colour_hsv": [0, 0, 0],
        "team_0_colour_name": "unknown",
        "team_1_colour_name": "unknown",
    }
