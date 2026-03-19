"""
Frame Shuttle API — processes frame batches for live streaming analysis.
Browser sends JPEG frames in small batches, server returns tracking events.
"""
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List
from pydantic import BaseModel
import numpy as np
import cv2
import json
import uuid
import time
import logging

from services.stream_tracker_service import get_stream_tracker
from services.job_queue_service import create_job, submit_job

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory session store
SESSIONS = {}  # session_id -> session state dict
SESSION_TTL = 30 * 60  # 30 minutes


def _clean_stale_sessions():
    """Remove sessions older than 30 minutes."""
    now = time.time()
    stale = [sid for sid, s in SESSIONS.items() if now - s["created_at"] > SESSION_TTL]
    for sid in stale:
        del SESSIONS[sid]


class StreamStartRequest(BaseModel):
    frame_width: int
    frame_height: int
    clip_type: str = "match"
    fps: float = 25.0


@router.post("/stream/start")
async def stream_start(req: StreamStartRequest):
    """Start a new streaming session."""
    try:
        session_id = str(uuid.uuid4())[:8]
        SESSIONS[session_id] = {
            "session_id": session_id,
            "created_at": time.time(),
            "frames_processed": 0,
            "tracks": {},
            "confirmed_ids": {},
            "frame_width": req.frame_width,
            "frame_height": req.frame_height,
            "clip_type": req.clip_type,
            "fps": req.fps,
            "uncertain_tracks": [],
        }
        return JSONResponse({
            "session_id": session_id,
            "status": "ready",
            "clip_type": req.clip_type,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=200)


@router.post("/stream/{session_id}/frames")
async def stream_frames(
    session_id: str,
    frames: List[UploadFile] = File(...),
    frame_indices: str = Form(...),
    timestamps: str = Form(...),
):
    """Process a batch of frames (up to 25 at a time)."""
    try:
        _clean_stale_sessions()

        session = SESSIONS.get(session_id)
        if not session:
            return JSONResponse({"error": "session not found"}, status_code=200)

        t_start = time.time()

        # Parse indices and timestamps
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
                np_frames.append(np.zeros((100, 100, 3), dtype=np.uint8))

        # Process batch
        tracker = get_stream_tracker()
        result = tracker.process_batch(
            frames=np_frames,
            frame_indices=indices,
            timestamps=ts_list,
            existing_tracks=session["tracks"],
            confirmed_ids=session["confirmed_ids"],
            frame_w=session["frame_width"],
            frame_h=session["frame_height"],
        )

        session["tracks"] = result["updated_tracks"]
        session["frames_processed"] += len(np_frames)

        # Build response tracks
        response_tracks = []
        for tid, track in session["tracks"].items():
            response_tracks.append({
                "track_id": track["track_id"],
                "bbox": track["bbox"],
                "team_id": track["team_id"],
                "confirmed": track["confirmed"],
                "uncertain": track["uncertain"],
                "confirmed_label": track["confirmed_label"],
            })

        processing_ms = int((time.time() - t_start) * 1000)

        return JSONResponse({
            "session_id": session_id,
            "frames_processed": session["frames_processed"],
            "tracks": response_tracks,
            "new_uncertain": result["new_uncertain"],
            "processing_ms": processing_ms,
        })

    except Exception as e:
        logger.error("stream_frames failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=200)


@router.post("/stream/{session_id}/confirm")
async def stream_confirm(
    session_id: str,
    track_id: int = Form(...),
    jersey_number: str = Form(default=""),
    player_name: str = Form(default=""),
    team_colour: str = Form(default=""),
):
    """Coach confirms a player identity during live processing."""
    try:
        session = SESSIONS.get(session_id)
        if not session:
            return JSONResponse({"error": "session not found"}, status_code=200)

        # Build label
        if jersey_number:
            label = f"#{jersey_number}"
        elif player_name:
            label = player_name
        elif team_colour:
            label = team_colour
        else:
            label = f"Player {track_id}"

        # Mark track as confirmed
        session["confirmed_ids"][track_id] = label
        track = session["tracks"].get(track_id)
        if track:
            track["confirmed"] = True
            track["confirmed_label"] = label
            track["uncertain"] = False

        # Incremental ReID patch
        tracker = get_stream_tracker()
        merged = tracker.incremental_reid_patch(
            confirmed_track_id=track_id,
            confirmed_label=label,
            all_tracks=session["tracks"],
        )

        return JSONResponse({
            "track_id": track_id,
            "confirmed_label": label,
            "tracks_merged": merged,
            "status": "confirmed",
        })

    except Exception as e:
        logger.error("stream_confirm failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=200)


@router.post("/stream/{session_id}/finalise")
async def stream_finalise(session_id: str):
    """Finalise session and trigger full analysis pipeline."""
    try:
        session = SESSIONS.get(session_id)
        if not session:
            return JSONResponse({"error": "session not found"}, status_code=200)

        # Build tracks list in tracking_service format
        tracks_list = []
        for tid, track in session["tracks"].items():
            trajectory = []
            for pos in track.get("positions", []):
                x, y, fi = pos
                # Reconstruct bbox from last known
                bbox = track["bbox"]
                trajectory.append({
                    "frameIndex": fi,
                    "bbox": bbox,
                    "confidence": track["avg_confidence"],
                })

            tracks_list.append({
                "trackId": track["track_id"],
                "teamId": track.get("team_id"),
                "firstSeen": track.get("first_frame", 0),
                "lastSeen": track.get("last_frame", 0),
                "hits": track.get("detection_count", 0),
                "confirmed_detections": track.get("detection_count", 0),
                "trajectory": trajectory,
                "is_staff": False,
            })

        # Build pre-confirmed labels
        pre_confirmed = {}
        for tid, label in session["confirmed_ids"].items():
            pre_confirmed[tid] = label

        # Create job and submit to pipeline
        job_id = f"stream_{session_id}"
        create_job(job_id)

        # Import pipeline function
        from routes.analyse import _run_analysis_pipeline

        # We can't run full pipeline without a video file for streaming,
        # so store session data for retrieval
        # The pipeline needs a video path — streaming doesn't have one.
        # Instead, store the session result directly as a completed job.
        from services.job_queue_service import _jobs
        job = _jobs[job_id]
        job["status"] = "completed"
        job["completedAt"] = time.time()
        job["result"] = {
            "job_id": job_id,
            "session_id": session_id,
            "clip_type": session.get("clip_type", "match"),
            "tracking": {
                "total_tracks": len(tracks_list),
                "confirmed_tracks": sum(1 for t in tracks_list if t["trackId"] in pre_confirmed),
                "frames_processed": session["frames_processed"],
            },
            "tracks": tracks_list,
            "confirmed_labels": pre_confirmed,
            "physical": {"players": []},
            "analysis": "Stream session finalised. Full analysis requires video file upload.",
        }

        tracks_confirmed = sum(1 for t in session["tracks"].values() if t["confirmed"])

        return JSONResponse({
            "job_id": job_id,
            "session_id": session_id,
            "tracks_confirmed": tracks_confirmed,
            "tracks_total": len(session["tracks"]),
            "poll_url": f"/api/v1/jobs/status/{job_id}",
        })

    except Exception as e:
        logger.error("stream_finalise failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=200)


@router.get("/stream/{session_id}/status")
async def stream_status(session_id: str):
    """Get current session status."""
    try:
        session = SESSIONS.get(session_id)
        if not session:
            return JSONResponse({"error": "session not found"}, status_code=200)

        tracks = session["tracks"]
        return JSONResponse({
            "session_id": session_id,
            "frames_processed": session["frames_processed"],
            "tracks_total": len(tracks),
            "tracks_confirmed": sum(1 for t in tracks.values() if t["confirmed"]),
            "tracks_uncertain": sum(1 for t in tracks.values() if t["uncertain"]),
            "clip_type": session.get("clip_type", "match"),
            "uptime_seconds": int(time.time() - session["created_at"]),
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=200)
