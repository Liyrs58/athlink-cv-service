"""
routes/highlight.py

Brick 16 — POST /api/v1/highlight/detect
"""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from models.analysis import QueuedJobResponse
from services.job_queue_service import create_job, get_job, submit_job
from services.highlight_service import detect_highlights

router = APIRouter()


class HighlightDetectRequest(BaseModel):
    jobId: str = Field(..., description="Base job identifier")
    minRunDistance: float = Field(15.0, description="Minimum distance (metres) for a long-run highlight")
    minAcceleration: float = Field(8.0, description="Minimum speed (m/s) for an acceleration burst")


@router.post("/detect", response_model=QueuedJobResponse,
             summary="Detect highlights from tracked player movements (async)")
def highlight_detect(req: HighlightDetectRequest):
    """Scan all pitch-mapped tracks for long runs, acceleration bursts,
    and penetrating runs into the final third. Async — poll for completion."""
    pitch_path = Path("temp") / req.jobId / "pitch" / "pitch_map.json"
    if not pitch_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No pitch map for job '{req.jobId}'. Run pitch/map first.",
        )

    highlight_job_id = f"{req.jobId}_highlights"

    existing = get_job(highlight_job_id)
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Highlight job '{highlight_job_id}' already exists (status: {existing['status']}).",
        )

    create_job(highlight_job_id)
    submit_job(
        highlight_job_id,
        detect_highlights,
        req.jobId, req.minRunDistance, req.minAcceleration,
    )

    return {
        "jobId": highlight_job_id,
        "status": "queued",
        "message": f"Highlight detection queued. Poll GET /api/v1/jobs/status/{highlight_job_id}",
    }
