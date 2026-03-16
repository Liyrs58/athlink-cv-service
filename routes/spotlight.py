"""
routes/spotlight.py

Brick 14 — POST /api/v1/spotlight/select
Brick 15 — POST /api/v1/spotlight/render
Brick 17 — POST /api/v1/spotlight/export-clip
"""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from models.analysis import QueuedJobResponse
from services.job_queue_service import create_job, get_job, submit_job
from services.spotlight_service import (
    select_player,
    render_spotlight,
    export_clip,
)

router = APIRouter()


# ── Request models ───────────────────────────────────────────────────────────

class SpotlightSelectRequest(BaseModel):
    jobId: str = Field(..., description="Base job identifier")
    trackId: int = Field(..., description="Track ID of the player to select")
    startSecond: float = Field(..., description="Start of the time window (seconds)")
    endSecond: float = Field(..., description="End of the time window (seconds)")


class SpotlightRenderRequest(BaseModel):
    jobId: str = Field(..., description="Base job identifier")
    trackId: int = Field(..., description="Track ID of the player to spotlight")
    startSecond: float = Field(..., description="Start of the clip (seconds)")
    endSecond: float = Field(..., description="End of the clip (seconds)")
    effectStyle: str = Field("glow", description="Effect style: circle, glow, or arrow")


class ClipExportRequest(BaseModel):
    jobId: str = Field(..., description="Base job identifier")
    trackId: int = Field(..., description="Track ID of the player to spotlight")
    startSecond: float = Field(..., description="Start of the clip (seconds)")
    endSecond: float = Field(..., description="End of the clip (seconds)")
    effectStyle: str = Field("glow", description="Effect style: circle, glow, or arrow")
    includeSlowmo: bool = Field(False, description="Add slow-motion to middle section")
    slowmoSection: float = Field(0.5, description="Fraction of clip to slow-mo (0.0-1.0)")


# ── Brick 14: Select ────────────────────────────────────────────────────────

@router.post("/select",
             summary="Get trajectory frames for a single player")
def spotlight_select(req: SpotlightSelectRequest):
    """Return bounding boxes and pitch coordinates for one player
    within a time window. Synchronous — returns immediately."""
    track_path = Path("temp") / req.jobId / "tracking" / "track_results.json"
    if not track_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No tracking results for job '{req.jobId}'.",
        )

    try:
        return select_player(req.jobId, req.trackId, req.startSecond, req.endSecond)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── Brick 15: Render ────────────────────────────────────────────────────────

@router.post("/render", response_model=QueuedJobResponse,
             summary="Render spotlight video on one player (async)")
def spotlight_render(req: SpotlightRenderRequest):
    """Render a clip highlighting one player with the chosen visual effect.
    Other players are dimmed. Async — poll for completion."""
    track_path = Path("temp") / req.jobId / "tracking" / "track_results.json"
    if not track_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No tracking results for job '{req.jobId}'.",
        )

    if req.effectStyle not in ("circle", "glow", "arrow"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid effectStyle '{req.effectStyle}'. Use circle, glow, or arrow.",
        )

    render_job_id = f"{req.jobId}_spot_{req.trackId}"

    existing = get_job(render_job_id)
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Spotlight job '{render_job_id}' already exists (status: {existing['status']}).",
        )

    create_job(render_job_id)
    submit_job(
        render_job_id,
        render_spotlight,
        req.jobId, req.trackId, req.startSecond, req.endSecond, req.effectStyle,
    )

    return {
        "jobId": render_job_id,
        "status": "queued",
        "message": f"Spotlight render queued. Poll GET /api/v1/jobs/status/{render_job_id}",
    }


# ── Brick 17: Export clip ────────────────────────────────────────────────────

@router.post("/export-clip", response_model=QueuedJobResponse,
             summary="Export spotlight clip with optional slow-mo (async)")
def spotlight_export_clip(req: ClipExportRequest):
    """Render a clip with spotlight effect, optional slow-motion in the middle
    section, and a burned-in timestamp. Async — poll for completion."""
    track_path = Path("temp") / req.jobId / "tracking" / "track_results.json"
    if not track_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No tracking results for job '{req.jobId}'.",
        )

    if req.effectStyle not in ("circle", "glow", "arrow"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid effectStyle '{req.effectStyle}'. Use circle, glow, or arrow.",
        )

    clip_job_id = f"{req.jobId}_clip_{req.trackId}"

    existing = get_job(clip_job_id)
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Clip job '{clip_job_id}' already exists (status: {existing['status']}).",
        )

    create_job(clip_job_id)
    submit_job(
        clip_job_id,
        export_clip,
        req.jobId, req.trackId, req.startSecond, req.endSecond,
        req.effectStyle, req.includeSlowmo, req.slowmoSection,
    )

    return {
        "jobId": clip_job_id,
        "status": "queued",
        "message": f"Clip export queued. Poll GET /api/v1/jobs/status/{clip_job_id}",
    }
