from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import logging

from services.analytics_service import build_analytics_report, get_available_services

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/analytics/{jobId}",
            summary="Get unified EPL analytics report")
async def get_analytics(jobId: str):
    """Run all available analytics services and return a single structured report."""
    track_path = Path("temp") / jobId / "tracking" / "track_results.json"
    if not track_path.exists():
        raise HTTPException(
            status_code=404,
            detail="track_results.json not found for job '{}'".format(jobId)
        )
    try:
        result = build_analytics_report(jobId)
        available = get_available_services(jobId)
        dq = result.get("data_quality", {})
        return JSONResponse(
            content=result,
            headers={
                "X-Available-Services": ",".join(available),
                "X-Data-Quality": str(dq.get("overall", "unknown")),
                "X-Analyst-Ready": str(result.get("analyst_ready", False)).lower(),
                "X-Frames-Analysed": str(dq.get("frames_analysed", 0)),
            },
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Analytics report failed for job %s", jobId)
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.get("/analytics/{jobId}/available",
            summary="Check which analytics services are ready")
async def get_available(jobId: str):
    """Fast check — returns which services have their required input files."""
    available = get_available_services(jobId)
    return {"available": available, "job_id": jobId}
