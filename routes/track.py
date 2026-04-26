from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import logging
import os

from services.tracking_service import run_tracking
from services.team_service import assign_teams
from services.job_queue_service import create_job, get_job, submit_job
from services.validation_service import validate_football_content
from models.analysis import QueuedJobResponse

router = APIRouter()
logger = logging.getLogger(__name__)


class TrackPlayersRequest(BaseModel):
    jobId: str = Field(..., description="Unique job identifier")
    videoPath: str = Field(..., description="Absolute path to video file")
    frameStride: int = Field(5, ge=1, le=60, description="Process every Nth frame")
    maxFrames: Optional[int] = Field(None, ge=1, description="Max frames to process")
    maxTrackAge: int = Field(10, ge=1, description="Frames before a lost track is dropped")


class TrackPlayersResponse(BaseModel):
    jobId: str
    videoPath: str
    frameStride: int
    framesProcessed: int
    trackCount: int
    outputPath: str


def _run_teams_job(req: TrackPlayersRequest) -> dict:
    result = run_tracking(
        video_path=req.videoPath,
        job_id=req.jobId,
        frame_stride=req.frameStride,
        max_frames=req.maxFrames,
        max_track_age=req.maxTrackAge,
    )

    tracks_with_teams = assign_teams(
        tracks=result["tracks"],
        frames_dir=f"temp/{req.jobId}/frames",
        job_id=req.jobId,
        output_dir=f"temp/{req.jobId}/tracking",
    )

    team0_count = sum(1 for t in tracks_with_teams if t.get("teamId") == 0)
    team1_count = sum(1 for t in tracks_with_teams if t.get("teamId") == 1)
    team2_count = sum(1 for t in tracks_with_teams if t.get("teamId") == 2)

    # Upload to Supabase if configured
    try:
        from services.storage_service import upload_file_from_path, BUCKET_RESULTS
        upload_file_from_path(
            BUCKET_RESULTS,
            f"{req.jobId}/track_results.json",
            f"temp/{req.jobId}/tracking/track_results.json"
        )
        upload_file_from_path(
            BUCKET_RESULTS,
            f"{req.jobId}/team_results.json",
            f"temp/{req.jobId}/tracking/team_results.json"
        )
    except Exception:
        pass

    return {
        "jobId": result["jobId"],
        "videoPath": result["videoPath"],
        "frameStride": result["frameStride"],
        "framesProcessed": result["framesProcessed"],
        "trackCount": result["trackCount"],
        "team0Count": team0_count,
        "team1Count": team1_count,
        "team2Count": team2_count,
        "outputPath": f"temp/{req.jobId}/tracking/team_results.json",
    }


@router.post("/players-with-teams", response_model=QueuedJobResponse,
              summary="Track players and assign teams (async)")
async def track_players_with_teams(req: TrackPlayersRequest):
    """Run BoT-SORT tracking + HSV team color assignment. Returns immediately;
    poll job status until complete."""
    if not os.path.exists(req.videoPath):
        raise HTTPException(status_code=400, detail=f"Video not found: {req.videoPath}")

    # Brick 22: auto-reject non-football videos
    print("DEBUG: Starting validation...")
    validation = validate_football_content(req.videoPath)
    print("DEBUG: Validation completed.")
    if not validation["valid"]:
        raise HTTPException(status_code=422, detail=validation["reason"])

    if get_job(req.jobId) is not None:
        raise HTTPException(status_code=409, detail=f"Job already exists: {req.jobId}")

    print("DEBUG: Creating job...")
    create_job(req.jobId)
    print("DEBUG: Submitting job...")
    submit_job(req.jobId, _run_teams_job, req)
    print("DEBUG: Job submitted.")

    return QueuedJobResponse(
        jobId=req.jobId,
        status="queued",
        message=f"Job queued. Poll GET /api/v1/jobs/status/{req.jobId}",
    )


@router.post("/players", response_model=TrackPlayersResponse,
              summary="Track players without team assignment (sync)")
async def track_players(req: TrackPlayersRequest):
    """Run BoT-SORT tracking only (no team colors). Synchronous — waits for
    completion before responding."""
    if not os.path.exists(req.videoPath):
        raise HTTPException(status_code=400, detail=f"Video not found: {req.videoPath}")

    # Brick 22: auto-reject non-football videos
    validation = validate_football_content(req.videoPath)
    if not validation["valid"]:
        raise HTTPException(status_code=422, detail=validation["reason"])

    try:
        result = run_tracking(
            video_path=req.videoPath,
            job_id=req.jobId,
            frame_stride=req.frameStride,
            max_frames=req.maxFrames,
            max_track_age=req.maxTrackAge,
        )
    except Exception as e:
        logger.exception("Tracking failed for job %s", req.jobId)
        raise HTTPException(status_code=500, detail=str(e))

    return TrackPlayersResponse(
        jobId=result["jobId"],
        videoPath=result["videoPath"],
        frameStride=result["frameStride"],
        framesProcessed=result["framesProcessed"],
        trackCount=result["trackCount"],
        outputPath=f"temp/{req.jobId}/tracking/track_results.json",
    )
