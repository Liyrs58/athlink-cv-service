"""
routes/whatif.py

Phase 1: What-If Mode — 3D pose extraction from broadcast football footage.
Test endpoint that runs RTMPose (via YOLO11-Pose) + MotionBERT on an
existing completed job's tracking data.
"""

import logging
import os
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException

from services.job_queue_service import get_job, list_jobs
from services.pose_extractor import extract_poses_from_video
from services.motion_lifter import lift_to_3d

router = APIRouter(prefix="/api/v1/whatif", tags=["whatif"])
logger = logging.getLogger(__name__)


def _load_job_result(job_id: str) -> Optional[Dict]:
    """Load a completed job result from the job queue store."""
    job = get_job(job_id)
    if job is None:
        return None
    if job.get("status") != "completed":
        return None
    return job.get("result")


def _find_video_path(job_id: str, result: Dict) -> Optional[str]:
    """
    Find the video file for a job.
    During processing the video is at /tmp/{job_id}_{filename},
    but it's cleaned up after completion. Check annotated video
    or temp directory as fallbacks.
    """
    # Check annotated video (always kept)
    annotated = result.get("annotated_video_path")
    if annotated and os.path.exists(annotated):
        return annotated

    # Check common temp patterns
    import glob
    patterns = [
        f"/tmp/{job_id}_*",
        f"/tmp/{job_id}.*",
    ]
    for pat in patterns:
        matches = glob.glob(pat)
        for m in matches:
            if m.endswith((".mp4", ".mov", ".avi", ".MP4", ".MOV")):
                return m

    # Check temp/{job_id} directory for original
    temp_dir = os.path.join("temp", job_id)
    if os.path.isdir(temp_dir):
        for f in os.listdir(temp_dir):
            if f.endswith((".mp4", ".mov", ".avi", ".MP4", ".MOV")):
                return os.path.join(temp_dir, f)

    return None


def _build_frame_data(
    result: Dict, start_frame: int, end_frame: int
) -> List[Dict]:
    """
    Convert tracking output to the frame_data format needed by pose_extractor.

    The job result stores tracks as a list, each with a 'trajectory' list of
    {frameIndex, bbox, confidence, timestampSeconds}. We pivot this into
    per-frame format: [{frame_index, players: [{track_id, bbox, team_id, ...}]}]
    """
    # Get raw tracks from the job result
    # Tracks are embedded in the result at various places;
    # the full track data is stored on disk
    import json

    # Try loading from disk first (has full trajectory)
    track_file = os.path.join("temp", str(result.get("job_id", "")), "tracking", "track_results.json")
    tracks = None

    if os.path.exists(track_file):
        try:
            with open(track_file) as f:
                data = json.load(f)
            tracks = data if isinstance(data, list) else data.get("tracks", [])
            logger.info("Loaded %d tracks from disk: %s", len(tracks), track_file)
        except Exception as e:
            logger.warning("Failed to load tracks from %s: %s", track_file, e)

    if not tracks:
        logger.warning("No track data found on disk, cannot build frame data")
        return []

    # Pivot: frame_index -> list of players
    frames_map: Dict[int, List[Dict]] = {}

    for track in tracks:
        tid = track.get("trackId")
        team_id = track.get("teamId", -1)
        trajectory = track.get("trajectory", [])

        for point in trajectory:
            fi = point.get("frameIndex")
            if fi is None or fi < start_frame or fi > end_frame:
                continue

            bbox = point.get("bbox")
            if not bbox or len(bbox) < 4:
                continue

            frames_map.setdefault(fi, []).append({
                "track_id": tid,
                "bbox": bbox,
                "confidence": point.get("confidence", 0),
                "team_id": team_id,
            })

    # Sort by frame index
    frame_data = []
    for fi in sorted(frames_map.keys()):
        frame_data.append({
            "frame_index": fi,
            "players": frames_map[fi],
        })

    logger.info(
        "Built frame data: %d frames (range %d-%d), avg %.1f players/frame",
        len(frame_data), start_frame, end_frame,
        sum(len(f["players"]) for f in frame_data) / max(len(frame_data), 1),
    )
    return frame_data


def _run_sanity_checks(poses_3d: List[Dict]) -> Dict:
    """Verify the 3D poses make anatomical sense."""
    checks = {
        "frames_with_players": 0,
        "avg_players_per_frame": 0.0,
        "players_with_3d": 0,
        "head_above_hips": True,
        "sample_player": None,
    }

    total_players = 0
    frames_with_players = 0
    players_with_3d = 0

    for frame in poses_3d:
        players = frame.get("players", [])
        if players:
            frames_with_players += 1
            total_players += len(players)

            for player in players:
                kp3d = player.get("keypoints_3d", [])
                if not kp3d or len(kp3d) < 17:
                    continue

                players_with_3d += 1

                # Sanity: nose (0) Y should be above hips (11, 12) Y
                # In MotionBERT root-relative space, Y-up convention
                nose_y = kp3d[0][1]
                hip_y = (kp3d[11][1] + kp3d[12][1]) / 2.0
                if nose_y < hip_y - 0.3:  # tolerance
                    checks["head_above_hips"] = False

                if not checks["sample_player"] and kp3d:
                    checks["sample_player"] = {
                        "track_id": player.get("track_id"),
                        "team_id": player.get("team_id"),
                        "keypoints_3d_sample": [
                            [round(c, 4) for c in kp] for kp in kp3d[:5]
                        ],
                    }

    checks["frames_with_players"] = frames_with_players
    checks["players_with_3d"] = players_with_3d
    if frames_with_players > 0:
        checks["avg_players_per_frame"] = round(
            total_players / frames_with_players, 1
        )

    return checks


@router.get("/test/{job_id}")
async def test_pose_extraction(
    job_id: str,
    start_frame: int = 250,
    end_frame: int = 375,
):
    """
    Phase 1 test endpoint.
    Loads an existing completed job, runs YOLO11-Pose + MotionBERT
    on a window of frames, returns 3D poses with sanity checks.
    """
    result = _load_job_result(job_id)
    if not result:
        raise HTTPException(404, f"Job {job_id} not found or not completed")

    video_path = _find_video_path(job_id, result)
    if not video_path:
        raise HTTPException(
            400,
            f"Video file not found for job {job_id}. "
            "The original video is cleaned up after processing. "
            "Re-submit the video or provide a video_path query param.",
        )

    logger.info(
        "What-If Phase 1: job=%s video=%s frames=%d-%d",
        job_id, video_path, start_frame, end_frame,
    )

    frame_data = _build_frame_data(result, start_frame, end_frame)
    if not frame_data:
        raise HTTPException(400, "No tracking data found for the given frame range")

    # Step 1: 2D pose
    logger.info("Step 1: Extracting 2D poses...")
    poses_2d = extract_poses_from_video(video_path, frame_data)

    # Get video dimensions for normalisation
    import cv2
    cap = cv2.VideoCapture(video_path)
    vw = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920
    vh = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080
    cap.release()

    # Step 2: Lift to 3D
    logger.info("Step 2: Lifting to 3D with MotionBERT...")
    poses_3d = lift_to_3d(poses_2d, video_width=vw, video_height=vh)

    # Sanity
    sanity = _run_sanity_checks(poses_3d)

    logger.info(
        "Phase 1 complete: %d frames, %d players with 3D, head_ok=%s",
        sanity["frames_with_players"],
        sanity["players_with_3d"],
        sanity["head_above_hips"],
    )

    return {
        "job_id": job_id,
        "video_path": video_path,
        "frames_processed": len(poses_3d),
        "sanity_checks": sanity,
        "poses": poses_3d,
    }


@router.get("/test/{job_id}/summary")
async def test_pose_summary(job_id: str):
    """Quick check: does the job exist and have tracking data?"""
    result = _load_job_result(job_id)
    if not result:
        raise HTTPException(404, f"Job {job_id} not found or not completed")

    video_path = _find_video_path(job_id, result)
    tracking = result.get("tracking", {})

    return {
        "job_id": job_id,
        "status": "ready" if video_path else "no_video",
        "video_path": video_path,
        "total_tracks": tracking.get("total_tracks", 0),
        "frames_processed": tracking.get("frames_processed", 0),
    }
