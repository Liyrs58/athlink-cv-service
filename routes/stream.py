"""
Frame Shuttle API — live streaming analysis.
Per-session Tracker instances. Real pipeline on finalise.
"""

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List
import numpy as np
import cv2
import json
import uuid
import time
import logging

from services.stream_tracker_service import Tracker
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
    frame_width: int = Form(...),
    frame_height: int = Form(...),
    clip_type: str = Form(default="match"),
    fps: float = Form(default=25.0),
):
    """
    Start a new streaming session.
    Saves video to disk. Creates per-session Tracker.
    """
    try:
        _clean_stale_sessions()
        session_id = str(uuid.uuid4())[:8]

        # Save video file
        video_path = f"/tmp/{session_id}_video.mp4"
        content = await video.read()
        with open(video_path, "wb") as f:
            f.write(content)

        # Create per-session tracker
        tracker = Tracker(frame_width, frame_height, fps)
        tracker.load_model()

        SESSIONS[session_id] = {
            "session_id": session_id,
            "created_at": time.time(),
            "video_path": video_path,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "clip_type": clip_type,
            "fps": fps,
            "frames_processed": 0,
            "confirmed_ids": {},
            "tracker": tracker,
        }

        return JSONResponse({
            "session_id": session_id,
            "status": "ready",
            "clip_type": clip_type,
        })
    except Exception as e:
        logger.error("stream_start failed: %s", e)
        return JSONResponse({"error": str(e)})


@router.post("/stream/{session_id}/frames")
async def stream_frames(
    session_id: str,
    frames: List[UploadFile] = File(...),
    frame_indices: str = Form(...),
    timestamps: str = Form(...),
):
    """Process a batch of frames via BoT-SORT tracker."""
    try:
        _clean_stale_sessions()
        session = SESSIONS.get(session_id)
        if not session:
            return JSONResponse({"error": "session not found"})

        t_start = time.time()

        try:
            indices = json.loads(frame_indices)
        except Exception:
            indices = list(range(len(frames)))
        try:
            ts_list = json.loads(timestamps)
        except Exception:
            ts_list = [0.0] * len(frames)

        # Decode frames
        np_frames = []
        for f in frames:
            data = await f.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                np_frames.append(img)
            else:
                np_frames.append(
                    np.zeros((100, 100, 3), dtype=np.uint8)
                )

        tracker = session["tracker"]
        result = tracker.process_batch(
            np_frames,
            indices,
            ts_list,
            session["confirmed_ids"],
        )
        session["frames_processed"] += len(np_frames)

        return JSONResponse({
            "session_id": session_id,
            "frames_processed": session["frames_processed"],
            "track_states": result["track_states"],
            "newly_confirmed": result["newly_confirmed"],
            "tracks_total": result["tracks_total"],
            "tracks_confirmed": result["tracks_confirmed"],
            "processing_ms": int(
                (time.time() - t_start) * 1000
            ),
        })
    except Exception as e:
        logger.error("stream_frames failed: %s", e)
        return JSONResponse({"error": str(e)})


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
            return JSONResponse({"error": "session not found"})

        if jersey_number:
            label = f"#{jersey_number}"
        elif player_name:
            label = player_name
        elif team_colour:
            label = team_colour
        else:
            label = f"Player {track_id}"

        session["confirmed_ids"][track_id] = label
        tracker = session["tracker"]
        if track_id in tracker._meta:
            tracker._meta[track_id]["coach_confirmed"] = True
            tracker._meta[track_id]["coach_label"] = label
            tracker._meta[track_id]["state"] = "confirmed"
            tracker._meta[track_id]["surfaced_to_ui"] = True

        merged = tracker.incremental_reid_patch(
            track_id, label
        )

        return JSONResponse({
            "track_id": track_id,
            "confirmed_label": label,
            "tracks_merged": merged,
            "status": "confirmed",
        })
    except Exception as e:
        logger.error("stream_confirm failed: %s", e)
        return JSONResponse({"error": str(e)})


@router.post("/stream/{session_id}/finalise")
async def stream_finalise(session_id: str):
    """
    Trigger REAL full analysis pipeline.
    No fake output. No pixel math. No manual job status setting.
    Real pipeline. Real report.
    """
    try:
        session = SESSIONS.get(session_id)
        if not session:
            return JSONResponse({"error": "session not found"})

        video_path = session["video_path"]
        tracker = session["tracker"]
        confirmed_labels = tracker.get_confirmed_labels()
        summary = tracker.get_track_summary()

        job_id = str(uuid.uuid4())[:8]
        create_job(job_id)

        from routes.analyse import _run_analysis_pipeline
        submit_job(
            job_id,
            _run_analysis_pipeline,
            job_id,
            video_path,
            pre_confirmed_labels=confirmed_labels,
        )

        return JSONResponse({
            "job_id": job_id,
            "session_id": session_id,
            "tracks_confirmed": summary["confirmed"],
            "tracks_total": summary["total"],
            "poll_url": f"/api/v1/jobs/status/{job_id}",
            "message": (
                "Full analysis running. Poll for results."
            ),
        })
    except Exception as e:
        logger.error("stream_finalise failed: %s", e)
        return JSONResponse({"error": str(e)})


@router.get("/stream/{session_id}/status")
async def stream_status(session_id: str):
    """Current session status."""
    try:
        session = SESSIONS.get(session_id)
        if not session:
            return JSONResponse({"error": "session not found"})
        summary = session["tracker"].get_track_summary()
        return JSONResponse({
            "session_id": session_id,
            "frames_processed": session["frames_processed"],
            "clip_type": session["clip_type"],
            "uptime_seconds": int(
                time.time() - session["created_at"]
            ),
            **summary,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)})
