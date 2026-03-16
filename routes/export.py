"""
routes/export.py

Brick 12 — GET /api/v1/export/{jobId}
"""

from fastapi import APIRouter, HTTPException

from services.export_service import build_export

router = APIRouter()


@router.get("/{job_id}", summary="Export aggregated analysis results")
def export_job(job_id: str):
    """Return all tracking, team, pitch, tactics, and event data for a job
    as a single JSON payload suitable for mobile client consumption."""
    try:
        return build_export(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Export failed: {exc}")
