"""Team color assignment via HSV histogram clustering.
"""

import cv2
import numpy as np
import logging
import os
import json
from pathlib import Path
from typing import List

NUM_TEAMS = 3
UPPER_BODY_RATIO = 0.55  # top 55% — jersey focus, avoid shorts/pitch
logger = logging.getLogger(__name__)


def assign_teams(
    tracks: List[dict],
    frames_dir: str,
    job_id: str,
    output_dir: str,
) -> List[dict]:
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Extract a 3-element feature vector per track ──────────────────────
    # [norm_hue, saturation_ratio, brightness]
    # norm_hue     : median hue of high-saturation pixels / 180  → [0,1]
    # sat_ratio    : fraction of jersey pixels with saturation > 60 → [0,1]
    #                  ~ 0 for white/grey kits, ~ 1 for colored kits
    # brightness   : mean Value channel of jersey pixels / 255   → [0,1]
    #
    # Using all three dimensions lets k-means separate:
    #   sky-blue  : high sat, hue ~100/180
    #   red/orange: high sat, hue ~0–10/180
    #   white     : low sat,  hue undefined
    #   yellow GK : high sat, hue ~30/180

    feature_vectors: dict = {}   # track_idx → np.array([norm_hue, sat_ratio, brightness])
    is_short_track: dict = {}    # track_idx → bool

    for track_idx, track in enumerate(tracks):
        traj = track.get("trajectory", [])
        if not traj:
            track["teamId"] = -1
            continue

        hits = track.get("hits", 0)
        is_short_track[track_idx] = hits < 15

        # Sample up to 8 evenly-spaced frames
        step = max(1, len(traj) // 8)
        sample_pts = traj[::step][:8]

        frame_features = []
        for pt in sample_pts:
            fi = pt["frameIndex"]
            frame_path = Path(frames_dir) / f"frame_{fi:06d}.jpg"
            if not frame_path.exists():
                continue

            img = cv2.imread(str(frame_path))
            if img is None:
                continue

            x1, y1, x2, y2 = [int(v) for v in pt["bbox"]]
            h_img, w_img = img.shape[:2]
            x1, x2 = max(0, x1), min(w_img, x2)
            y1, y2 = max(0, y1), min(h_img, y2)

            if x2 - x1 < 4 or y2 - y1 < 4:
                continue

            # Crop: upper 55% height, trim 15% each side horizontally
            box_h = y2 - y1
            box_w = x2 - x1
            trim = int(box_w * 0.15)
            crop = img[y1: y1 + int(box_h * UPPER_BODY_RATIO),
                       x1 + trim: x2 - trim]
            if crop.size == 0:
                continue

            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

            # Exclude grass green
            green_mask = cv2.inRange(hsv,
                                     np.array([35, 40, 40]),
                                     np.array([85, 255, 255]))
            jersey_mask = cv2.bitwise_not(green_mask)

            jersey_px = jersey_mask > 0
            total_px = int(np.sum(jersey_px))
            if total_px < 80:
                continue

            sat = hsv[:, :, 1].astype(float)
            val = hsv[:, :, 2].astype(float)
            hue = hsv[:, :, 0].astype(float)

            sat_vals = sat[jersey_px]
            val_vals = val[jersey_px]
            hue_vals = hue[jersey_px]

            sat_ratio = float(np.mean(sat_vals > 60))
            brightness = float(np.mean(val_vals)) / 255.0

            # Median hue from HIGH-saturation pixels only (avoid noise from white)
            hi_sat_mask = (sat_vals > 60)
            if hi_sat_mask.sum() >= 10:
                median_hue = float(np.median(hue_vals[hi_sat_mask])) / 180.0
            else:
                median_hue = 0.0  # no dominant color → white/grey kit

            frame_features.append([median_hue, sat_ratio, brightness])

        if not frame_features:
            track["teamId"] = -1
            continue

        avg_feat = np.mean(frame_features, axis=0)
        feature_vectors[track_idx] = avg_feat

        logger.debug(
            "Track %s | hits=%d | hue=%.2f sat_ratio=%.2f brightness=%.2f",
            track.get("trackId"), hits,
            avg_feat[0], avg_feat[1], avg_feat[2]
        )

    if not feature_vectors:
        for track in tracks:
            track.setdefault("teamId", -1)
        return tracks

    # ── 2. K-means (k=2) on long tracks ──────────────────────────────────────
    # FIX 2: Exclude potential officials from clustering
    clustering_indices = [
        idx for idx in feature_vectors
        if not is_short_track.get(idx, True) and not tracks[idx].get("is_official", False)
    ]

    if len(clustering_indices) < 2:
        # fall back: use all tracks (but still exclude officials)
        clustering_indices = [
            idx for idx in feature_vectors if not tracks[idx].get("is_official", False)
        ]

    if len(clustering_indices) < 2:
        # last resort: use all tracks including officials
        clustering_indices = list(feature_vectors.keys())

    data = np.array([feature_vectors[i] for i in clustering_indices])

    # Weight the dimensions: hue matters most for colored kits,
    # sat_ratio distinguishes white from colored.
    weights = np.array([2.0, 1.5, 0.5])
    data_w = data * weights

    labels, centroids_w = _kmeans(data_w, k=2)

    for i, track_idx in enumerate(clustering_indices):
        tracks[track_idx]["teamId"] = int(labels[i])

    # ── 3. Assign short / unclustered tracks to nearest centroid ─────────────
    # FIX 2: Skip officials, assign them teamId -2
    clustered_set = set(clustering_indices)
    for track_idx, feat in feature_vectors.items():
        if track_idx in clustered_set:
            continue
        if tracks[track_idx].get("is_official", False):
            tracks[track_idx]["teamId"] = -2
            continue
        feat_w = feat * weights
        d0 = float(np.linalg.norm(feat_w - centroids_w[0]))
        d1 = float(np.linalg.norm(feat_w - centroids_w[1]))
        tracks[track_idx]["teamId"] = 0 if d0 <= d1 else 1

    # ── 4. Goalkeeper — track most unlike its team's hue centroid ────────────
    # FIX 2: Skip officials when finding goalkeeper
    team_hues: dict = {0: [], 1: []}
    for track_idx, feat in feature_vectors.items():
        tid = tracks[track_idx].get("teamId")
        if tid == -2:  # Skip officials
            continue
        if tid in team_hues:
            team_hues[tid].append(feat[0])

    team_median_hue = {
        tid: float(np.median(v)) if v else 0.0
        for tid, v in team_hues.items()
    }

    gk_idx = None
    gk_dist = 0.0
    for track_idx, feat in feature_vectors.items():
        tid = tracks[track_idx].get("teamId")
        if tid == -2:  # Skip officials
            continue
        if tid not in team_median_hue:
            continue
        dist = abs(feat[0] - team_median_hue[tid])
        if dist > gk_dist:
            gk_dist = dist
            gk_idx = track_idx

    if gk_idx is not None and gk_dist > (15.0 / 180.0):  # >15 hue units
        tracks[gk_idx]["teamId"] = 2
        logger.info(
            "Goalkeeper: track %s (hue dist %.3f)",
            tracks[gk_idx].get("trackId"), gk_dist
        )
    else:
        logger.info("No goalkeeper detected (max hue dist %.3f)", gk_dist)

    # ── 5. Role field ─────────────────────────────────────────────────────────
    # FIX 2: Mark officials with special role
    for track in tracks:
        track.setdefault("teamId", -1)
        if track.get("teamId") == -2:
            track["role"] = "official"
        elif track.get("teamId") == 2:
            track["role"] = "goalkeeper"
        else:
            track["role"] = "player"

    # ── 6. Save ───────────────────────────────────────────────────────────────
    t0 = sum(1 for t in tracks if t.get("teamId") == 0)
    t1 = sum(1 for t in tracks if t.get("teamId") == 1)
    t2 = sum(1 for t in tracks if t.get("teamId") == 2)
    t_official = sum(1 for t in tracks if t.get("teamId") == -2)
    logger.info("Teams: team0=%d  team1=%d  goalkeeper=%d  officials=%d", t0, t1, t2, t_official)

    results = [
        {
            "trackId": t.get("trackId"),
            "teamId": t.get("teamId", -1),
            "role": t.get("role", "player"),
            "hits": t.get("hits", 0),
            "firstSeen": t.get("firstSeen", 0),
            "lastSeen": t.get("lastSeen", 0),
        }
        for t in tracks
    ]
    out_path = Path(output_dir) / "team_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    return tracks


def _kmeans(data: np.ndarray, k: int = 2, max_iters: int = 30) -> tuple:
    """K-means with k-means++ initialization for stability."""
    np.random.seed(42)
    n = data.shape[0]

    # k-means++ seed
    centroids = [data[np.random.randint(n)]]
    for _ in range(k - 1):
        dists = np.array([
            min(np.linalg.norm(x - c) ** 2 for c in centroids)
            for x in data
        ])
        probs = dists / dists.sum()
        centroids.append(data[np.random.choice(n, p=probs)])
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
