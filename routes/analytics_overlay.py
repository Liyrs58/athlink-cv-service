"""
routes/analytics_overlay.py

Analytics overlay routes - POST for rendering, GET for download.
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.job_queue_service import create_job, get_job, submit_job
from services.analytics_overlay_service import render_analytics_highlight
from models.analysis import QueuedJobResponse

router = APIRouter()


class AnalyticsOverlayRequest(BaseModel):
    overlays: list = Field(
        default=["pass_network", "xg", "sprint_trails", "formation", "event_labels"],
        description="List of overlays to apply"
    )
    highlightOnly: bool = Field(True, description="Render highlights only or full match")
    paddingSeconds: float = Field(2.0, description="Seconds of padding around highlights")


def _run_analytics_overlay_job(job_id: str, req: AnalyticsOverlayRequest) -> dict:
    """Run the analytics overlay rendering job."""
    
    # Determine output path
    if req.highlightOnly:
        output_path = f"temp/{job_id}/highlights/analytics_highlight.mp4"
    else:
        output_path = f"temp/{job_id}/render/analytics_full.mp4"
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Render the video
    if req.highlightOnly:
        return render_analytics_highlight(
            job_id=job_id,
            output_path=output_path,
            overlays=req.overlays,
            padding_seconds=req.paddingSeconds
        )
    else:
        from services.analytics_overlay_service import render_full_match_analytics
        return render_full_match_analytics(
            job_id=job_id,
            overlays=req.overlays
        )


@router.post("/{job_id}", response_model=QueuedJobResponse,
              summary="Render analytics overlay video (async)")
def start_analytics_overlay(job_id: str, req: AnalyticsOverlayRequest):
    """Render video with analytics overlays (xG, passes, formation, etc.)."""
    
    # Check if tracking data exists
    track_path = Path("temp") / job_id / "tracking" / "track_results.json"
    if not track_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No tracking results for job '{job_id}'. Run players-with-teams first."
        )
    
    overlay_job_id = f"{job_id}_analytics_overlay"
    
    # Check if job already exists
    existing = get_job(overlay_job_id)
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Analytics overlay job '{overlay_job_id}' already exists (status: {existing['status']})."
        )
    
    # Create and submit job
    job_payload = {
        "job_id": job_id,
        "overlays": req.overlays,
        "highlight_only": req.highlightOnly,
        "padding_seconds": req.paddingSeconds
    }
    
    job = create_job(overlay_job_id, "analytics_overlay", job_payload)
    submit_job(overlay_job_id, _run_analytics_overlay_job, job_id, req)
    
    return QueuedJobResponse(
        jobId=overlay_job_id,
        status="queued",
        message=f"Analytics overlay rendering queued. Poll GET /api/v1/jobs/status/{overlay_job_id}"
    )


@router.get("/{job_id}/download",
             summary="Download rendered analytics overlay video")
def download_analytics_overlay(job_id: str):
    """Download the rendered analytics overlay video."""
    
    overlay_job_id = f"{job_id}_analytics_overlay"
    
    # Check job status
    job = get_job(overlay_job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Analytics overlay job '{overlay_job_id}' not found."
        )
    
    if job.get("status") != "completed":
        raise HTTPException(
            status_code=404,
            detail=f"Analytics overlay video not yet rendered for job '{job_id}'. Current status: {job.get('status')}"
        )
    
    # Determine output file path
    highlight_path = Path("temp") / job_id / "highlights" / "analytics_highlight.mp4"
    full_path = Path("temp") / job_id / "render" / "analytics_full.mp4"
    
    if highlight_path.exists():
        video_path = highlight_path
    elif full_path.exists():
        video_path = full_path
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Analytics overlay video not found for job '{job_id}'."
        )
    
    # Return video file as streaming response
    from fastapi.responses import FileResponse
    
    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=f"analytics_highlight_{job_id}.mp4"
    )
