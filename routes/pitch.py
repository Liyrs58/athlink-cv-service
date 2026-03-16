from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import logging
import os

from services.pitch_service import map_pitch
from services.job_queue_service import create_job, get_job, submit_job
from models.analysis import QueuedJobResponse

router = APIRouter()
logger = logging.getLogger(__name__)


class PitchMapRequest(BaseModel):
    jobId: str = Field(..., description="Base job identifier (tracking must exist)")
    videoPath: str = Field(..., description="Absolute path to video file")
    frameStride: int = Field(5, ge=1, le=60, description="Process every Nth frame")
    maxFrames: Optional[int] = Field(None, ge=1, description="Max frames to process")


def _run_pitch_job(req: PitchMapRequest) -> dict:
    return map_pitch(
        video_path=req.videoPath,
        job_id=req.jobId,
        frame_stride=req.frameStride,
        max_frames=req.maxFrames,
    )


@router.post("/map", response_model=QueuedJobResponse,
              summary="Map player positions to pitch coordinates (async)")
async def pitch_map(req: PitchMapRequest):
    """Compute homography from detected pitch lines and project player
    bounding-box centers onto a 105x68 m coordinate system."""
    if not os.path.exists(req.videoPath):
        raise HTTPException(status_code=400, detail=f"Video not found: {req.videoPath}")

    pitch_job_id = f"{req.jobId}_pitch"

    if get_job(pitch_job_id) is not None:
        raise HTTPException(status_code=409, detail=f"Job already exists: {pitch_job_id}")

    create_job(pitch_job_id)
    submit_job(pitch_job_id, _run_pitch_job, req)

    return QueuedJobResponse(
        jobId=pitch_job_id,
        status="queued",
        message=f"Job queued. Poll GET /api/v1/jobs/status/{pitch_job_id}",
    )
