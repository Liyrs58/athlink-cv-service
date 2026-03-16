from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
import logging
import os

from services.match_pipeline_service import run_full_match, get_match_progress, load_checkpoint
from services.job_queue_service import create_job, get_job, submit_job
from models.analysis import QueuedJobResponse

router = APIRouter()
logger = logging.getLogger(__name__)


class MatchRunRequest(BaseModel):
    jobId: str = Field(..., description="Unique job identifier")
    videoPath: str = Field(..., description="Absolute path to video file")
    frameStride: int = Field(2, ge=1, le=60, description="Process every Nth frame")
    forceRestart: bool = Field(False, description="Ignore existing checkpoint and restart")


def _run_match_job(req: MatchRunRequest) -> dict:
    return run_full_match(
        job_id=req.jobId,
        video_path=req.videoPath,
        frame_stride=req.frameStride,
        force_restart=req.forceRestart,
    )


@router.post("/run", response_model=QueuedJobResponse,
             summary="Run full match pipeline (async)")
async def match_run(req: MatchRunRequest):
    """Process a full match video with chunked tracking, team assignment,
    pitch mapping, and analytics. Resumes from checkpoint if available."""
    if not os.path.exists(req.videoPath):
        raise HTTPException(status_code=400, detail="Video not found: {}".format(req.videoPath))

    existing = get_job(req.jobId)
    if existing is not None:
        raise HTTPException(status_code=409, detail="Job already exists: {}".format(req.jobId))

    create_job(req.jobId)
    submit_job(req.jobId, _run_match_job, req)

    return QueuedJobResponse(
        jobId=req.jobId,
        status="queued",
        message="Match pipeline queued. Poll GET /api/v1/match/progress/{}".format(req.jobId),
    )


@router.get("/progress/{jobId}",
            summary="Get match pipeline progress")
async def match_progress(jobId: str):
    """Return progress of a running or completed match pipeline job."""
    checkpoint = load_checkpoint(jobId)
    job = get_job(jobId)

    if checkpoint is None and job is None:
        raise HTTPException(status_code=404, detail="No match pipeline found for job '{}'".format(jobId))

    return get_match_progress(jobId)
