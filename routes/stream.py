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
        _clean_stale_sessions()
        # Reset tracker state for new session
        tracker = get_stream_tracker()
        tracker.reset_session()

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

        # Build per-player physical stats from stream positions
        fps_val = session.get("fps", 5.0)
        frame_w = session.get("frame_width", 1920)
        frame_h = session.get("frame_height", 1080)
        # Rough conversion: assume 105m pitch spans full width
        px_to_m = 105.0 / max(frame_w, 1)

        players_physical = []
        for track_data in tracks_list:
            tid = track_data["trackId"]
            traj = track_data.get("trajectory", [])
            if len(traj) < 2:
                continue

            # Compute distance from trajectory positions
            positions = session["tracks"].get(tid, {}).get("positions", [])
            total_dist_m = 0.0
            max_speed_ms = 0.0
            sprint_count = 0
            for j in range(1, len(positions)):
                dx = (positions[j][0] - positions[j-1][0]) * px_to_m
                dy = (positions[j][1] - positions[j-1][1]) * px_to_m
                d = (dx*dx + dy*dy) ** 0.5
                total_dist_m += d
                # Speed in m/s (frames are at fps_val)
                dt = 1.0 / fps_val if fps_val > 0 else 0.2
                speed = d / dt if dt > 0 else 0
                if speed > max_speed_ms:
                    max_speed_ms = speed
                if speed > 5.5:  # ~20 km/h sprint threshold
                    sprint_count += 1

            team_id = track_data.get("teamId")
            label = pre_confirmed.get(tid, f"Player {tid}")

            players_physical.append({
                "track_id": tid,
                "label": label,
                "team_id": team_id,
                "distance_m": round(total_dist_m, 1),
                "max_speed_kmh": round(max_speed_ms * 3.6, 1),
                "sprint_count": sprint_count,
                "frames_visible": track_data.get("hits", 0),
                "confidence": "approximate",
            })

        # Build analysis text
        total_players = len(tracks_list)
        confirmed = sum(1 for t in tracks_list if t["trackId"] in pre_confirmed)
        team_counts = {}
        for t in tracks_list:
            tid_team = t.get("teamId")
            if tid_team is not None:
                team_counts[tid_team] = team_counts.get(tid_team, 0) + 1

        analysis_lines = [
            f"Live stream analysis completed — {session['frames_processed']} frames processed at {fps_val} fps.",
            f"Detected {total_players} players ({confirmed} confirmed by coach).",
        ]
        if team_counts:
            for t_id, cnt in sorted(team_counts.items()):
                analysis_lines.append(f"Team {t_id}: {cnt} players detected.")

        avg_dist = sum(p["distance_m"] for p in players_physical) / max(len(players_physical), 1)
        if avg_dist > 0:
            analysis_lines.append(f"Average distance covered: {avg_dist:.0f}m (approximate — based on pixel displacement).")

        top_sprint = sorted(players_physical, key=lambda p: p["max_speed_kmh"], reverse=True)[:3]
        if top_sprint:
            fastest = top_sprint[0]
            analysis_lines.append(f"Fastest player: {fastest['label']} at {fastest['max_speed_kmh']} km/h.")

        analysis_text = "\n".join(analysis_lines)

        # Create job with real results
        job_id = f"stream_{session_id}"
        create_job(job_id)

        from services.job_queue_service import _jobs
        job = _jobs[job_id]
        job["status"] = "completed"
        job["completedAt"] = time.time()
        job["result"] = {
            "job_id": job_id,
            "session_id": session_id,
            "clip_type": session.get("clip_type", "match"),
            "tracking": {
                "total_tracks": total_players,
                "confirmed_tracks": confirmed,
                "frames_processed": session["frames_processed"],
            },
            "tracks": tracks_list,
            "confirmed_labels": pre_confirmed,
            "physical": {"players": players_physical},
            "analysis": analysis_text,
            "velocities": {
                "players": [
                    {
                        "track_id": p["track_id"],
                        "label": p["label"],
                        "team_id": p["team_id"],
                        "distance_metres": p["distance_m"],
                        "max_speed_kmh": p["max_speed_kmh"],
                        "sprint_count": p["sprint_count"],
                    }
                    for p in players_physical
                ],
            },
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
