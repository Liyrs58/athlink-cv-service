"""
Re-identification service for merging fragmented player tracks.

Uses HSV colour histograms and spatial continuity to identify
track fragments that belong to the same player.
"""
import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# ReID parameters
MAX_GAP_SECONDS = 5.0       # Max time gap between fragments (camera pans can be long)
MAX_GAP_PIXELS = 400.0      # Max spatial distance (camera pans shift players a lot)
APPEARANCE_THRESHOLD = 0.50  # Bhattacharyya distance threshold (more lenient)
HIST_BINS = 36               # Hue bins for histogram
FPS_DEFAULT = 25.0


def extract_track_appearance(track: Dict, video_path: str, fps: float = FPS_DEFAULT) -> Optional[np.ndarray]:
    """
    Extract HSV colour histogram from torso region at 3 frames
    (start, middle, end of track lifespan).
    Returns normalised histogram or None if extraction fails.
    """
    traj = track.get("trajectory", [])
    if len(traj) < 2:
        return None

    # Pick 3 sample points: start, middle, end
    indices = [0, len(traj) // 2, len(traj) - 1]
    # Deduplicate
    indices = sorted(set(indices))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    needs_rotation = raw_h > raw_w

    histograms = []
    for idx in indices:
        entry = traj[idx]
        frame_idx = entry["frameIndex"]
        bbox = entry["bbox"]

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        if needs_rotation:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        hist = _extract_torso_histogram(frame, bbox)
        if hist is not None:
            histograms.append(hist)

    cap.release()

    if not histograms:
        return None

    # Average histograms
    combined = histograms[0].copy()
    for h in histograms[1:]:
        combined += h
    combined /= len(histograms)
    cv2.normalize(combined, combined, 0, 1, cv2.NORM_MINMAX)
    return combined


def _extract_torso_histogram(frame: np.ndarray, bbox: list) -> Optional[np.ndarray]:
    """Extract HSV hue histogram from upper body (torso) region."""
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    h_frame, w_frame = frame.shape[:2]
    x1 = max(0, min(x1, w_frame - 1))
    x2 = max(x1 + 1, min(x2, w_frame))
    y1 = max(0, min(y1, h_frame - 1))
    y2 = max(y1 + 1, min(y2, h_frame))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 5:
        return None

    # Torso: upper 50% height, centre 60% width
    ch, cw = crop.shape[:2]
    torso_top = 0
    torso_bottom = int(ch * 0.5)
    torso_left = int(cw * 0.2)
    torso_right = int(cw * 0.8)
    torso = crop[torso_top:torso_bottom, torso_left:torso_right]
    if torso.size == 0:
        return None

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

    # Mask out green (pitch) pixels
    green_mask = cv2.inRange(hsv, np.array([25, 20, 20]), np.array([95, 255, 255]))
    non_green = cv2.bitwise_not(green_mask)

    hist = cv2.calcHist([hsv], [0], non_green, [HIST_BINS], [0, 180])
    if hist.sum() < 10:
        return None
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def compute_appearance_similarity(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """
    Bhattacharyya distance between two histograms.
    0 = identical, 1 = completely different.
    """
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)


def compute_spatial_continuity(track1: Dict, track2: Dict,
                               max_gap_frames: int = 50,
                               max_distance_px: float = MAX_GAP_PIXELS) -> bool:
    """
    Check if track1's end and track2's start are close enough
    to plausibly be the same player.

    track1 must end before track2 starts (or small overlap allowed).
    """
    traj1 = track1.get("trajectory", [])
    traj2 = track2.get("trajectory", [])
    if not traj1 or not traj2:
        return False

    last1 = traj1[-1]
    first2 = traj2[0]

    # Frame gap check
    frame_gap = first2["frameIndex"] - last1["frameIndex"]
    if frame_gap < -5 or frame_gap > max_gap_frames:
        return False

    # Spatial distance check
    bbox1 = last1["bbox"]
    bbox2 = first2["bbox"]
    cx1 = (bbox1[0] + bbox1[2]) / 2.0
    cy1 = (bbox1[1] + bbox1[3]) / 2.0
    cx2 = (bbox2[0] + bbox2[2]) / 2.0
    cy2 = (bbox2[1] + bbox2[3]) / 2.0
    dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5

    # Size similarity check (height ratio within 60% — camera zoom changes sizes)
    h1 = bbox1[3] - bbox1[1]
    h2 = bbox2[3] - bbox2[1]
    if h1 > 0 and h2 > 0:
        ratio = h2 / h1
        if ratio < 0.4 or ratio > 2.5:
            return False

    return dist <= max_distance_px


def _find_connected_components(n: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    """Union-Find to get connected components."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in edges:
        union(a, b)

    components = {}
    for i in range(n):
        root = find(i)
        if root not in components:
            components[root] = []
        components[root].append(i)

    return list(components.values())


def merge_fragmented_tracks(tracks: List[Dict], video_path: str,
                            fps: float = FPS_DEFAULT) -> List[Dict]:
    """
    Merge fragmented tracks that represent the same player.

    Uses appearance similarity (HSV histogram) + spatial continuity.
    Returns merged track list.
    """
    if len(tracks) < 2:
        return tracks

    max_gap_frames = int(MAX_GAP_SECONDS * fps)
    n = len(tracks)

    # Sort by firstSeen for temporal ordering
    tracks_sorted = sorted(enumerate(tracks), key=lambda x: x[1].get("firstSeen", 0))
    idx_map = [orig_idx for orig_idx, _ in tracks_sorted]
    sorted_tracks = [t for _, t in tracks_sorted]

    # Extract appearance histograms (batch: open video once)
    logger.info(f"ReID: extracting appearance for {n} tracks...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("ReID: cannot open video, skipping merge")
        return tracks

    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    needs_rotation = raw_h > raw_w

    histograms = [None] * n
    # Collect all frame indices we need
    frame_requests = []  # (sorted_idx, traj_entry_idx, frame_idx)
    for si, track in enumerate(sorted_tracks):
        traj = track.get("trajectory", [])
        if len(traj) < 2:
            continue
        sample_indices = [0, len(traj) // 2, len(traj) - 1]
        sample_indices = sorted(set(sample_indices))
        for ti in sample_indices:
            frame_requests.append((si, ti, traj[ti]["frameIndex"], traj[ti]["bbox"]))

    # Sort by frame index for sequential reads
    frame_requests.sort(key=lambda x: x[2])

    # Process frames sequentially
    track_hists = {}  # si -> list of histograms
    current_frame_idx = -1
    current_frame = None

    for si, ti, frame_idx, bbox in frame_requests:
        if frame_idx != current_frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            if needs_rotation:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            current_frame = frame
            current_frame_idx = frame_idx

        if current_frame is None:
            continue

        hist = _extract_torso_histogram(current_frame, bbox)
        if hist is not None:
            if si not in track_hists:
                track_hists[si] = []
            track_hists[si].append(hist)

    cap.release()

    # Average histograms per track
    for si, hists in track_hists.items():
        combined = hists[0].copy()
        for h in hists[1:]:
            combined += h
        combined /= len(hists)
        cv2.normalize(combined, combined, 0, 1, cv2.NORM_MINMAX)
        histograms[si] = combined

    # Build edges: check all pairs where spatial continuity holds
    edges = []
    for i in range(n):
        if histograms[i] is None:
            continue
        for j in range(i + 1, n):
            if histograms[j] is None:
                continue

            # Check spatial continuity (either direction)
            spatially_close = (
                compute_spatial_continuity(sorted_tracks[i], sorted_tracks[j],
                                          max_gap_frames=max_gap_frames) or
                compute_spatial_continuity(sorted_tracks[j], sorted_tracks[i],
                                          max_gap_frames=max_gap_frames)
            )
            if not spatially_close:
                continue

            # Check appearance similarity
            dist = compute_appearance_similarity(histograms[i], histograms[j])
            if dist < APPEARANCE_THRESHOLD:
                edges.append((i, j))
                logger.debug(
                    f"ReID edge: track {sorted_tracks[i].get('trackId')} <-> "
                    f"{sorted_tracks[j].get('trackId')} (bhatt={dist:.3f})"
                )

    if not edges:
        logger.info("ReID: no merge candidates found")
        return tracks

    # Find connected components
    components = _find_connected_components(n, edges)

    # Merge each component
    merged = []
    merged_count = 0
    for component in components:
        if len(component) == 1:
            merged.append(sorted_tracks[component[0]])
            continue

        # Merge all tracks in this component
        primary = sorted_tracks[component[0]]
        for ci in component[1:]:
            secondary = sorted_tracks[ci]
            primary["trajectory"].extend(secondary.get("trajectory", []))
            primary["hits"] = primary.get("hits", 0) + secondary.get("hits", 0)
            primary["lastSeen"] = max(
                primary.get("lastSeen", 0),
                secondary.get("lastSeen", 0)
            )
            primary["_confirmed_detections"] = (
                primary.get("_confirmed_detections", primary.get("confirmed_detections", 0)) +
                secondary.get("_confirmed_detections", secondary.get("confirmed_detections", 0))
            )
            merged_count += 1

        # Sort trajectory by frame index and deduplicate
        primary["trajectory"].sort(key=lambda e: e["frameIndex"])
        # Remove duplicate frame entries (keep highest confidence)
        seen_frames = {}
        deduped = []
        for entry in primary["trajectory"]:
            fi = entry["frameIndex"]
            if fi not in seen_frames:
                seen_frames[fi] = len(deduped)
                deduped.append(entry)
            else:
                # Keep the one with higher confidence
                existing = deduped[seen_frames[fi]]
                if entry.get("confidence", 0) > existing.get("confidence", 0):
                    deduped[seen_frames[fi]] = entry
        primary["trajectory"] = deduped
        # Use lowest track ID in the component
        all_ids = [sorted_tracks[ci].get("trackId", 9999) for ci in component]
        primary["trackId"] = min(all_ids)

        logger.info(
            f"ReID merged {len(component)} fragments into track {primary['trackId']}: "
            f"IDs={[sorted_tracks[ci].get('trackId') for ci in component]}"
        )
        merged.append(primary)

    logger.info(f"ReID: {n} tracks -> {len(merged)} tracks ({merged_count} fragments merged)")
    return merged
