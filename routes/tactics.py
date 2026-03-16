from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import logging

from services.tactics_service import analyze_tactics
from services.job_queue_service import create_job, get_job, submit_job
from models.analysis import QueuedJobResponse

router = APIRouter()
logger = logging.getLogger(__name__)


class TacticsRequest(BaseModel):
    jobId: str = Field(..., description="Base job identifier (tracking + pitch must exist)")


def _run_tactics_job(req: TacticsRequest) -> dict:
    return analyze_tactics(job_id=req.jobId)


@router.post("/analyze", response_model=QueuedJobResponse,
              summary="Run tactical analysis (async)")
async def tactics_analyze(req: TacticsRequest):
    """Compute formations, heatmaps, passing lanes, space occupation,
    and event detection (passes, shots, tackles)."""
    tactics_job_id = f"{req.jobId}_tactics"

    if get_job(tactics_job_id) is not None:
        raise HTTPException(status_code=409, detail=f"Job already exists: {tactics_job_id}")

    create_job(tactics_job_id)
    submit_job(tactics_job_id, _run_tactics_job, req)

    return QueuedJobResponse(
        jobId=tactics_job_id,
        status="queued",
        message=f"Job queued. Poll GET /api/v1/jobs/status/{tactics_job_id}",
    )
