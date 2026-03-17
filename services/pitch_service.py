import cv2
import numpy as np
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

PITCH_WIDTH = 105.0   # metres (standard football pitch length)
PITCH_HEIGHT = 68.0   # metres (standard football pitch width)


# ---------------------------------------------------------------------------
# Field / homography detection helpers
# ---------------------------------------------------------------------------

def _green_field_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask


def _find_field_corners(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect 4 pitch corners using green-field contour approximation.
    Returns (4, 2) float32 in TL, TR, BR, BL order, or None on failure.
    """
    mask = _green_field_mask(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    frame_area = frame.shape[0] * frame.shape[1]
    if cv2.contourArea(largest) < frame_area * 0.08:
        return None

    hull = cv2.convexHull(largest)
    epsilon = 0.03 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True).reshape(-1, 2).astype(np.float32)

    if len(approx) < 4:
        rect = cv2.boundingRect(largest)
        x, y, w, h = rect
        return np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)

    # Pick 4 extremal corners via quadrant-score method (TL, TR, BR, BL)
    center = approx.mean(axis=0)
    corners = []
    for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        scores = dx * (approx[:, 0] - center[0]) + dy * (approx[:, 1] - center[1])
        corners.append(approx[int(np.argmax(scores))])
    return np.array(corners, dtype=np.float32)


def validate_homography(H: np.ndarray, frame_w: int, frame_h: int) -> bool:
    """
    Test H by projecting known pixel locations and checking world coords
    are within pitch bounds and sensibly located.
    Returns True if valid, False if H is garbage.
    """
    test_pixels = [
        [frame_w / 2, frame_h / 2],      # centre
        [frame_w * 0.1, frame_h * 0.5],  # left edge
        [frame_w * 0.9, frame_h * 0.5],  # right edge
    ]
    pts = np.array(test_pixels, dtype=np.float32).reshape(-1, 1, 2)
    try:
        world = cv2.perspectiveTransform(pts, H)
    except Exception:
        return False
    for pt in world:
        x, y = float(pt[0][0]), float(pt[0][1])
        if not (-10 < x < 115 and -10 < y < 78):
            return False
    centre_x, centre_y = float(world[0][0][0]), float(world[0][0][1])
    if not (30 < centre_x < 75 and 15 < centre_y < 53):
        return False
    return True


def _estimate_homography(frame: np.ndarray) -> Optional[np.ndarray]:
    """Estimate homography mapping pixel coords → pitch metres."""
    corners = _find_field_corners(frame)
    if corners is None:
        return None

    dst_pts = np.array([
        [0,           0],
        [PITCH_WIDTH, 0],
        [PITCH_WIDTH, PITCH_HEIGHT],
        [0,           PITCH_HEIGHT],
    ], dtype=np.float32)

    H, _ = cv2.findHomography(corners, dst_pts, cv2.RANSAC)
    return H


def _transform_point(H: np.ndarray, px: float, py: float) -> Tuple[float, float]:
    pt = np.array([[[px, py]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


# ---------------------------------------------------------------------------
# Job ID helpers
# ---------------------------------------------------------------------------

def _base_job_id(job_id: str) -> str:
    """Strip known suffixes to get the tracking job's base ID."""
    for suffix in ("_final_pitch", "_final_tactics", "_final", "_pitch", "_tactics"):
        if job_id.endswith(suffix):
            return job_id[: -len(suffix)]
    return job_id


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_tracking(job_id: str) -> Optional[Dict]:
    """
    Load tracking data containing player trajectories.
    Tries the given job_id first, then the base job_id (strips known suffixes).
    Within each folder, tries track_results.json then team_results.json.
    """
    candidates = [job_id]
    base = _base_job_id(job_id)
    if base != job_id:
        candidates.append(base)

    for jid in candidates:
        for filename in ("track_results.json", "team_results.json"):
            path = Path(f"temp/{jid}/tracking/{filename}")
            print(f"[pitch_service] looking for tracking data at: {path.resolve()}")
            if path.exists():
                print(f"[pitch_service] found: {path.resolve()}")
                with open(path) as f:
                    data = json.load(f)
                # Only accept files that have trajectory data
                tracks = data.get("tracks", data if isinstance(data, list) else [])
                if tracks and isinstance(tracks[0], dict) and "trajectory" in tracks[0]:
                    return data
                print(f"[pitch_service] skipping {filename} — no trajectory data")

    print(f"[pitch_service] no tracking data found for job '{job_id}'")
    return None


def _load_team_map(job_id: str) -> Dict[int, int]:
    """Returns {trackId: teamId}. Tries base job_id if suffix variant not found."""
    candidates = [job_id, _base_job_id(job_id)]
    for jid in candidates:
        path = Path(f"temp/{jid}/tracking/team_results.json")
        if path.exists():
            with open(path) as f:
                raw = json.load(f)
            tracks = raw["tracks"] if isinstance(raw, dict) else raw
            return {t["trackId"]: t.get("teamId", -1) for t in tracks}
    return {}


# ---------------------------------------------------------------------------
# Interpolation helper
# ---------------------------------------------------------------------------

def _interpolate_trajectory(trajectory_2d: List[Dict]) -> List[Dict]:
    """
    Linearly interpolate x/y for any gaps up to 45 frames between consecutive trajectory points.
    Points are assumed sorted by frameIndex.
    Add velocity-based smoothing and reliability flags.
    """
    if len(trajectory_2d) < 2:
        return trajectory_2d

    filled = []
    for i in range(len(trajectory_2d) - 1):
        a = trajectory_2d[i]
        b = trajectory_2d[i + 1]
        filled.append(a)
        gap = b["frameIndex"] - a["frameIndex"]
        if 1 < gap <= 45:  # Allow interpolation up to 45 frames
            for s in range(1, gap):
                t = s / gap
                filled.append({
                    "frameIndex": a["frameIndex"] + s,
                    "x": round(a["x"] + t * (b["x"] - a["x"]), 2),
                    "y": round(a["y"] + t * (b["y"] - a["y"]), 2),
                    "reliable": False,
                })
        elif gap > 45:
            # Gap too large, do not interpolate
            pass
    filled.append(trajectory_2d[-1])

    # Physics-based velocity smoothing: flag unreliable positions
    for i in range(1, len(filled) - 1):
        # Compute velocity from surrounding 5 frames using central difference
        if i >= 2 and i + 2 < len(filled):
            prev_pt = filled[i - 2]
            curr_pt = filled[i]
            next_pt = filled[i + 2]

            frame_gap = next_pt["frameIndex"] - prev_pt["frameIndex"]
            if frame_gap > 0:
                dx = next_pt["x"] - prev_pt["x"]
                dy = next_pt["y"] - prev_pt["y"]
                dist_m = (dx**2 + dy**2)**0.5
                velocity_ms = dist_m / (frame_gap * (1.0 / 25.0))  # assume 25fps

                # If velocity > 35 m/s (impossible even for shots), mark unreliable
                if velocity_ms > 35.0:
                    filled[i]["reliable"] = False

    return filled


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def map_pitch(
    video_path: str,
    job_id: str,
    frame_stride: int = 5,
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Detect pitch homography from video frames and map every tracked player's
    trajectory to 2D pitch coordinates (metres, 0-105 × 0-68).

    Requires track_results.json produced by run_tracking().
    """
    tracking = _load_tracking(job_id)
    if tracking is None:
        raise ValueError(
            f"No tracking data found for job '{job_id}'. "
            "Run /api/v1/track/players-with-teams first."
        )

    team_map = _load_team_map(job_id)

    # ------------------------------------------------------------------
    # Read frames sequentially, build per-frame homography table.
    # Carry forward the last good H when detection fails.
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1
    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1

    needs_rotation = raw_h > raw_w
    if needs_rotation:
        frame_h, frame_w = raw_w, raw_h
    else:
        frame_h, frame_w = raw_h, raw_w

    # frame_index -> H matrix
    frame_homographies: Dict[int, np.ndarray] = {}
    last_good_H: Optional[np.ndarray] = None
    frames_checked = 0
    raw_frame_idx = 0
    processed_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if raw_frame_idx % frame_stride != 0:
            raw_frame_idx += 1
            continue

        if max_frames and processed_count >= max_frames:
            break

        current_frame_idx = raw_frame_idx

        if needs_rotation:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        H = _estimate_homography(frame)
        if H is not None:
            if validate_homography(H, frame_w, frame_h):
                last_good_H = H
            else:
                H = None  # reject bad homography
        if H is None and last_good_H is not None:
            # Carry forward previous validated H
            H = last_good_H

        if H is not None:
            frame_homographies[current_frame_idx] = H

        frames_checked += 1
        if frames_checked % 50 == 0:
            logger.info("[pitch_service] frames checked: %d (frame_idx=%d)", frames_checked, current_frame_idx)
            print(f"[pitch_service] frames checked: {frames_checked} (frame_idx={current_frame_idx})")

        raw_frame_idx += 1
        processed_count += 1

    cap.release()

    homography_found = last_good_H is not None
    calibration_valid = homography_found  # already validated per-frame above

    if not homography_found:
        logger.warning("No homography found for job %s — world coords unavailable", job_id)
    else:
        logger.info("Homography validated for job %s", job_id)

    # ------------------------------------------------------------------
    # Transform player trajectories
    # ------------------------------------------------------------------
    players = []
    for track in tracking["tracks"]:
        track_id = track["trackId"]
        team_id = team_map.get(track_id, -1)
        trajectory_2d = []

        for point in track["trajectory"]:
            fi = point["frameIndex"]
            bbox = point["bbox"]
            # Use foot position: bottom-centre of bbox
            px = (float(bbox[0]) + float(bbox[2])) / 2.0
            py = float(bbox[3])

            if calibration_valid:
                # Find the closest available homography for this frame
                H = frame_homographies.get(fi)
                if H is None and frame_homographies:
                    nearest = min(frame_homographies.keys(), key=lambda k: abs(k - fi))
                    H = frame_homographies[nearest]

                if H is not None:
                    x, y = _transform_point(H, px, py)
                    x = max(0.0, min(PITCH_WIDTH, x))
                    y = max(0.0, min(PITCH_HEIGHT, y))
                else:
                    # No H available — skip this point
                    continue
            else:
                # calibration failed — do NOT write proportional fallback world coords
                # downstream services must check calibration_valid before trusting coords
                x = px / frame_w * PITCH_WIDTH
                y = py / frame_h * PITCH_HEIGHT

            trajectory_2d.append({
                "frameIndex": fi,
                "x": round(x, 2),
                "y": round(y, 2),
            })

        # Sort by frameIndex then interpolate gaps
        trajectory_2d.sort(key=lambda p: p["frameIndex"])
        trajectory_2d = _interpolate_trajectory(trajectory_2d)

        players.append({
            "trackId": track_id,
            "teamId": team_id,
            "trajectory2d": trajectory_2d,
        })

    # ------------------------------------------------------------------
    # Add ball world coordinates only if homography is valid
    # ------------------------------------------------------------------
    if calibration_valid and tracking.get("ball_trajectory"):
        ball_trajectory_2d = []

        for ball_det in tracking["ball_trajectory"]:
            fi = ball_det["frameIndex"]
            px = float(ball_det["x"])
            py = float(ball_det["y"])
            source = ball_det.get("source", "unknown")
            confidence = ball_det.get("confidence", 0.5)

            # Find the closest available homography for this frame
            H = frame_homographies.get(fi)
            if H is None and frame_homographies:
                # Use nearest frame's H
                nearest = min(frame_homographies.keys(), key=lambda k: abs(k - fi))
                H = frame_homographies[nearest]

            if H is not None:
                # Transform pixel coordinates to world coordinates
                pixel_pt = np.array([[px, py]], dtype=np.float32).reshape(-1, 1, 2)
                world_pt = cv2.perspectiveTransform(pixel_pt, H)
                world_x = float(world_pt[0][0][0])
                world_y = float(world_pt[0][0][1])

                # Clamp to pitch boundaries
                world_x = max(0.0, min(PITCH_WIDTH, world_x))
                world_y = max(0.0, min(PITCH_HEIGHT, world_y))
            else:
                # Proportional fallback
                world_x = px / frame_w * PITCH_WIDTH
                world_y = py / frame_h * PITCH_HEIGHT

            # Reliability flag based on source
            reliable = (source == "yolo")

            ball_trajectory_2d.append({
                "frameIndex": fi,
                "x": round(world_x, 2),
                "y": round(world_y, 2),
                "source": source,
                "confidence": confidence,
                "reliable": reliable,
            })

        # Sort by frameIndex
        ball_trajectory_2d.sort(key=lambda p: p["frameIndex"])

        # Add ball entry to players list
        players.append({
            "trackId": -1,
            "teamId": -1,
            "is_ball": True,
            "trajectory2d": ball_trajectory_2d,
        })

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    output_dir = Path(f"temp/{job_id}/pitch")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / "pitch_map.json")

    result = {
        "jobId": job_id,
        "framesProcessed": frames_checked,
        "homographyFound": homography_found,
        "calibration_valid": calibration_valid,
        "calibration_failed": not calibration_valid,
        "players": players,
        "outputPath": output_path,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Pitch mapping done: %d players, homography=%s", len(players), homography_found)
    return result
