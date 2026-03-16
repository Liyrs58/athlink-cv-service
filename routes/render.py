"""
routes/render.py

Brick 13 — POST /api/v1/render/{jobId}
Async: submits render job, returns immediately.
Poll GET /api/v1/jobs/status/{jobId}_render
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.job_queue_service import create_job, get_job, submit_job
from services.render_service import run_render
from models.analysis import QueuedJobResponse

router = APIRouter()


class RenderRequest(BaseModel):
    jobId: str = Field(..., description="Base job identifier (tracking must exist)")
    includeMinimap: bool = Field(False, description="Include pitch minimap overlay")


def _run_render_job(req: RenderRequest) -> dict:
    return run_render(req.jobId, req.includeMinimap)


@router.post("/{job_id}", response_model=QueuedJobResponse,
              summary="Render annotated video with overlays (async)")
def start_render(job_id: str, req: RenderRequest):
    """Produce an MP4 with team-colored bounding boxes, ball trail, and
    optional pitch minimap overlay."""
    track_path = Path("temp") / job_id / "tracking" / "track_results.json"
    if not track_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No tracking results for job '{job_id}'. Run players-with-teams first."
        )

    render_job_id = f"{job_id}_render"

    existing = get_job(render_job_id)
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Render job '{render_job_id}' already exists (status: {existing['status']})."
        )

    create_job(render_job_id)
    submit_job(render_job_id, _run_render_job, RenderRequest(jobId=job_id, includeMinimap=req.includeMinimap))

    return {
        "jobId":   render_job_id,
        "status":  "queued",
        "message": f"Render queued. Poll GET /api/v1/jobs/status/{render_job_id}",
    }
