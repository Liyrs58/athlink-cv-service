"""
Live streaming analysis — server-side BoT-SORT with progress polling.

Architecture:
  POST /stream/start        → upload video, start background BoT-SORT pipeline
  GET  /stream/{id}/progress → poll current tracking progress (reads progress.json)
  POST /stream/{id}/confirm  → coach confirms a player identity
  POST /stream/{id}/finalise → wait for pipeline, return job_id for full report
"""

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
import json
import uuid
import time
import os
import logging

from services.job_queue_service import create_job, submit_job

logger = logging.getLogger(__name__)
router = APIRouter()

SESSIONS = {}
SESSION_TTL = 30 * 60


def _clean_stale_sessions():
    now = time.time()
    stale = [
        sid for sid, s in SESSIONS.items()
        if now - s["created_at"] > SESSION_TTL
    ]
    for sid in stale:
        del SESSIONS[sid]


@router.post("/stream/start")
async def stream_start(
    video: UploadFile = File(...),
    frame_width: int = Form(0),
    frame_height: int = Form(0),
    clip_type: str = Form(default="match"),
    fps: float = Form(default=25.0),
):
    """
    Start a live streaming session.
    Saves video, starts background BoT-SORT pipeline, returns session_id immediately.
    """
    try:
        _clean_stale_sessions()
        session_id = str(uuid.uuid4())[:8]

        # Save video file
        video_path = f"/tmp/{session_id}_video.mp4"
        content = await video.read()
        with open(video_path, "wb") as f:
            f.write(content)

        # Progress file — tracking_service writes to this every 10 frames
        progress_path = f"/tmp/{session_id}_progress.json"

        # Write initial progress
        with open(progress_path, "w") as f:
            json.dump({
                "frames_processed": 0,
                "total_frames": 0,
                "tracks": [],
                "ball_bbox": None,
                "status": "starting",
            }, f)

        # Create job for the full pipeline
        job_id = str(uuid.uuid4())[:8]
        create_job(job_id)

        SESSIONS[session_id] = {
            "session_id": session_id,
            "created_at": time.time(),
            "video_path": video_path,
            "progress_path": progress_path,
            "job_id": job_id,
            "clip_type": clip_type,
            "confirmed_ids": {},
            "status": "processing",
        }

        # Start the REAL analysis pipeline in background
        from routes.analyse import _run_analysis_pipeline
        submit_job(
            job_id,
            _run_analysis_pipeline,
            job_id,
            video_path,
            progress_path=progress_path,
        )

        return JSONResponse({
            "session_id": session_id,
            "job_id": job_id,
            "status": "processing",
            "clip_type": clip_type,
        })
    except Exception as e:
        logger.error("stream_start failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/stream/{session_id}/progress")
async def stream_progress(session_id: str):
    """
    Poll current tracking progress.
    Reads progress.json written by tracking_service every 10 frames.
    """
    try:
        session = SESSIONS.get(session_id)
        if not session:
            return JSONResponse({"error": "session not found"}, status_code=404)

        progress_path = session["progress_path"]
        if not os.path.exists(progress_path):
            return JSONResponse({
                "frames_processed": 0,
                "total_frames": 0,
                "tracks": [],
                "ball_bbox": None,
                "status": "starting",
            })

        try:
            with open(progress_path, "r") as f:
                progress = json.load(f)
        except (json.JSONDecodeError, IOError):
            # File is being written to — return last known state
            return JSONResponse({
                "frames_processed": 0,
                "total_frames": 0,
                "tracks": [],
                "ball_bbox": None,
                "status": "processing",
            })

        return JSONResponse(progress)
    except Exception as e:
        logger.error("stream_progress failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/stream/{session_id}/confirm")
async def stream_confirm(
    session_id: str,
    track_id: int = Form(...),
    jersey_number: str = Form(default=""),
    player_name: str = Form(default=""),
    team_colour: str = Form(default=""),
):
    """Coach confirms a player identity."""
    try:
        session = SESSIONS.get(session_id)
        if not session:
            return JSONResponse({"error": "session not found"}, status_code=404)

        if jersey_number:
            label = f"#{jersey_number}"
        elif player_name:
            label = player_name
        elif team_colour:
            label = team_colour
        else:
            label = f"Player {track_id}"

        session["confirmed_ids"][track_id] = label

        return JSONResponse({
            "track_id": track_id,
            "confirmed_label": label,
            "status": "confirmed",
        })
    except Exception as e:
        logger.error("stream_confirm failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/stream/{session_id}/finalise")
async def stream_finalise(session_id: str):
    """
    Return the job_id for the background pipeline.
    Frontend polls /api/v1/jobs/status/{job_id} as normal.
    """
    try:
        session = SESSIONS.get(session_id)
        if not session:
            return JSONResponse({"error": "session not found"}, status_code=404)

        job_id = session["job_id"]

        return JSONResponse({
            "job_id": job_id,
            "session_id": session_id,
            "poll_url": f"/api/v1/jobs/status/{job_id}",
            "message": "Full analysis running. Poll for results.",
        })
    except Exception as e:
        logger.error("stream_finalise failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/stream/{session_id}/status")
async def stream_status(session_id: str):
    """Current session status."""
    try:
        session = SESSIONS.get(session_id)
        if not session:
            return JSONResponse({"error": "session not found"}, status_code=404)
        return JSONResponse({
            "session_id": session_id,
            "job_id": session["job_id"],
            "clip_type": session["clip_type"],
            "status": session["status"],
            "uptime_seconds": int(time.time() - session["created_at"]),
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
