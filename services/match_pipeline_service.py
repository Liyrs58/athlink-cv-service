import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import cv2

logger = logging.getLogger(__name__)

CHECKPOINT_INTERVAL_SECONDS = 300  # 5 min of video time


def get_checkpoint_path(job_id):
    # type: (str) -> str
    return str(Path("temp") / job_id / "checkpoints" / "checkpoint.json")


def load_checkpoint(job_id):
    # type: (str) -> Optional[Dict[str, Any]]
    """Read checkpoint.json if it exists. Returns None on any error."""
    path = Path(get_checkpoint_path(job_id))
    try:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    except Exception as e:
        logger.warning("Failed to load checkpoint for %s: %s", job_id, e)
    return None


def save_checkpoint(job_id, state):
    # type: (str, Dict[str, Any]) -> None
    """Write checkpoint.json atomically via temp file + rename."""
    ckpt_dir = Path("temp") / job_id / "checkpoints"
    try:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        tmp_path = ckpt_dir / "checkpoint.tmp"
        final_path = ckpt_dir / "checkpoint.json"
        with open(tmp_path, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(str(tmp_path), str(final_path))
    except Exception as e:
        logger.error("Failed to save checkpoint for %s: %s", job_id, e)


def run_full_match(
    job_id,          # type: str
    video_path,      # type: str
    frame_stride=2,  # type: int
    force_restart=False,  # type: bool
):
    # type: (...) -> Dict[str, Any]
    """
    Main pipeline function for processing a full match video.

    Processes in 5-minute chunks with checkpointing for resumability.
    """
    from services.tracking_service import run_tracking
    from services.team_service import assign_teams
    from services.job_queue_service import get_job

    # STEP 1 — Video metadata
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: {}".format(video_path))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0:
        fps = 25.0
    total_seconds = total_frames / fps

    # STEP 2 — Check for existing checkpoint
    checkpoint = load_checkpoint(job_id)
    resumed_from_frame = 0
    accumulated_tracks = []   # type: list
    accumulated_ball = []     # type: list
    max_track_id = 0
    chunks_already_done = 0

    if checkpoint and not force_restart:
        resumed_from_frame = checkpoint.get("last_completed_frame", 0) + 1
        accumulated_tracks = checkpoint.get("track_data_so_far", {}).get("tracks", [])
        accumulated_ball = checkpoint.get("track_data_so_far", {}).get("ball_trajectory", [])
        max_track_id = checkpoint.get("track_data_so_far", {}).get("max_track_id", 0)
        chunks_already_done = checkpoint.get("chunks_completed", 0)
        logger.info("[match_pipeline] resuming from frame %d", resumed_from_frame)
    else:
        logger.info("[match_pipeline] starting fresh for job %s", job_id)

    # STEP 3 — Process in chunks
    chunk_size_frames = int(CHECKPOINT_INTERVAL_SECONDS * fps)
    chunk_start = 0
    chunk_index = 0
    total_chunks = max(1, (total_frames + chunk_size_frames - 1) // chunk_size_frames)

    while chunk_start < total_frames:
        chunk_end = min(chunk_start + chunk_size_frames, total_frames)

        # Skip chunks entirely if already processed
        if chunk_end <= resumed_from_frame:
            chunk_start = chunk_end
            chunk_index += 1
            continue

        # Compute max_frames for this chunk
        chunk_frame_count = chunk_end - chunk_start
        max_frames_for_chunk = chunk_frame_count // frame_stride

        if max_frames_for_chunk < 1:
            chunk_start = chunk_end
            chunk_index += 1
            continue

        # Create a chunk-specific job ID for tracking service
        chunk_job_id = "{}_chunk_{}".format(job_id, chunk_index)
        chunk_output_dir = Path("temp") / chunk_job_id / "tracking"
        chunk_output_dir.mkdir(parents=True, exist_ok=True)

        # Run tracking on this chunk
        try:
            chunk_result = run_tracking(
                video_path=video_path,
                job_id=chunk_job_id,
                frame_stride=frame_stride,
                max_frames=max_frames_for_chunk,
                max_track_age=50,
            )
        except Exception as e:
            logger.error("[match_pipeline] chunk %d tracking failed: %s", chunk_index, e)
            chunk_start = chunk_end
            chunk_index += 1
            continue

        # Run team assignment on chunk tracks
        chunk_tracks = chunk_result.get("tracks", [])
        frames_dir = "temp/{}/frames".format(chunk_job_id)
        try:
            assign_teams(
                tracks=chunk_tracks,
                frames_dir=frames_dir,
                job_id=chunk_job_id,
                output_dir=str(chunk_output_dir),
            )
        except Exception as e:
            logger.warning("[match_pipeline] chunk %d team assignment failed: %s", chunk_index, e)

        # Merge chunk tracks with track ID offset
        track_id_offset = max_track_id + 1 if max_track_id > 0 else 0
        for track in chunk_tracks:
            original_id = track.get("trackId", 0)
            new_id = original_id + track_id_offset
            track["trackId"] = new_id
            # Offset frame indices to be absolute
            for pt in track.get("trajectory", []):
                pt["frameIndex"] = pt.get("frameIndex", 0) + chunk_start
            track["firstSeen"] = track.get("firstSeen", 0) + chunk_start
            track["lastSeen"] = track.get("lastSeen", 0) + chunk_start
            accumulated_tracks.append(track)
            max_track_id = max(max_track_id, new_id)

        # Merge ball trajectory with frame offset
        for ball in chunk_result.get("ball_trajectory", []):
            ball["frameIndex"] = ball.get("frameIndex", 0) + chunk_start
            accumulated_ball.append(ball)

        chunks_already_done += 1

        # Save checkpoint
        state = {
            "job_id": job_id,
            "last_completed_frame": chunk_end - 1,
            "last_completed_second": round((chunk_end - 1) / fps, 2),
            "total_frames_processed": len(accumulated_tracks),
            "chunks_completed": chunks_already_done,
            "track_data_so_far": {
                "tracks": accumulated_tracks,
                "ball_trajectory": accumulated_ball,
                "max_track_id": max_track_id,
            },
            "created_at": checkpoint.get("created_at", datetime.now(timezone.utc).isoformat()) if checkpoint else datetime.now(timezone.utc).isoformat(),
        }
        save_checkpoint(job_id, state)

        # Update job progress
        job = get_job(job_id)
        if job is not None:
            progress = chunk_end / total_frames
            job["progress"] = round(progress, 4)
            job["current_chunk"] = chunk_index

        logger.info(
            "[match_pipeline] chunk %d/%d complete (frames %d-%d)",
            chunk_index + 1, total_chunks, chunk_start, chunk_end - 1,
        )

        chunk_start = chunk_end
        chunk_index += 1

    # STEP 4 — Write final merged results
    output_dir = Path("temp") / job_id / "tracking"
    output_dir.mkdir(parents=True, exist_ok=True)

    final_track_results = {
        "jobId": job_id,
        "videoPath": video_path,
        "frameStride": frame_stride,
        "framesProcessed": total_frames,
        "trackCount": len(accumulated_tracks),
        "tracks": accumulated_tracks,
        "ballDetections": len(accumulated_ball),
        "ball_trajectory": accumulated_ball,
        "metadata": {
            "fps": fps,
            "width": width,
            "height": height,
            "totalFrames": total_frames,
        },
    }

    track_results_path = output_dir / "track_results.json"
    with open(track_results_path, "w") as f:
        json.dump(final_track_results, f, indent=2)

    # Write team_results from accumulated tracks
    team_results = [
        {
            "trackId": t.get("trackId"),
            "teamId": t.get("teamId", -1),
            "role": t.get("role", "player"),
            "hits": t.get("hits", 0),
            "firstSeen": t.get("firstSeen", 0),
            "lastSeen": t.get("lastSeen", 0),
        }
        for t in accumulated_tracks
    ]
    team_results_path = output_dir / "team_results.json"
    with open(team_results_path, "w") as f:
        json.dump(team_results, f, indent=2)

    logger.info(
        "[match_pipeline] tracking complete: %d tracks, %d ball detections",
        len(accumulated_tracks), len(accumulated_ball),
    )

    # Upload tracking results to Supabase
    try:
        from services.storage_service import upload_file_from_path, BUCKET_RESULTS
        upload_file_from_path(
            BUCKET_RESULTS,
            "{}/track_results.json".format(job_id),
            str(track_results_path),
        )
        upload_file_from_path(
            BUCKET_RESULTS,
            "{}/team_results.json".format(job_id),
            str(team_results_path),
        )
    except Exception:
        pass

    # STEP 5 — Run pitch mapping
    try:
        from services.pitch_service import map_pitch
        map_pitch(video_path=video_path, job_id=job_id)
        logger.info("[match_pipeline] pitch mapping complete")
    except Exception as e:
        logger.error("[match_pipeline] pitch mapping failed: %s", e)

    # STEP 6 — Run analytics
    analytics_available = False
    try:
        from services.analytics_service import build_analytics_report
        report = build_analytics_report(job_id)
        analytics_dir = Path("temp") / job_id / "analytics"
        analytics_dir.mkdir(parents=True, exist_ok=True)
        with open(analytics_dir / "analytics_report.json", "w") as f:
            json.dump(report, f, indent=2)
        analytics_available = True
        logger.info("[match_pipeline] analytics report saved")
    except Exception as e:
        logger.error("[match_pipeline] analytics failed: %s", e)

    # STEP 7 — Upload to Supabase
    upload_result = None
    try:
        from services.storage_service import upload_job_results
        upload_result = upload_job_results(job_id)
    except Exception:
        pass

    return {
        "job_id": job_id,
        "total_frames": total_frames,
        "frames_processed": total_frames,
        "chunks_completed": chunks_already_done,
        "resumed_from_frame": resumed_from_frame,
        "duration_seconds": round(total_seconds, 2),
        "analytics_available": analytics_available,
        "upload_result": upload_result,
    }


def get_match_progress(job_id):
    # type: (str) -> Dict[str, Any]
    """Read checkpoint and job status to report progress."""
    from services.job_queue_service import get_job

    checkpoint = load_checkpoint(job_id)
    job = get_job(job_id)

    status = "unknown"
    if job is not None:
        status = job.get("status", "unknown")

    progress_pct = 0.0
    last_completed_second = 0.0
    total_seconds_estimated = 0.0
    chunks_done = 0
    resumed = False

    if checkpoint is not None:
        last_completed_second = checkpoint.get("last_completed_second", 0.0)
        chunks_done = checkpoint.get("chunks_completed", 0)
        resumed = checkpoint.get("last_completed_frame", 0) > 0

    if job is not None:
        progress_pct = round(job.get("progress", 0.0) * 100, 2)

    # Estimate total seconds from video if possible
    track_path = Path("temp") / job_id / "tracking" / "track_results.json"
    if track_path.exists():
        try:
            with open(track_path) as f:
                td = json.load(f)
            meta = td.get("metadata", {})
            total_frames = meta.get("totalFrames", 0)
            fps = meta.get("fps", 25)
            if fps > 0 and total_frames > 0:
                total_seconds_estimated = round(total_frames / fps, 2)
        except Exception:
            pass

    return {
        "job_id": job_id,
        "status": status,
        "progress_pct": progress_pct,
        "last_completed_second": last_completed_second,
        "total_seconds_estimated": total_seconds_estimated,
        "chunks_done": chunks_done,
        "resumed": resumed,
    }
