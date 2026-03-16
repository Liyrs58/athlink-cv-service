from fastapi import APIRouter, HTTPException
import logging

from services.xg_service import compute_xg

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/xg/{jobId}",
             summary="Get xG (expected goals) analysis")
async def get_xg(jobId: str):
    """Compute and return expected goals model for a completed job."""
    try:
        result = compute_xg(jobId)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("xG computation failed for job %s", jobId)
        raise HTTPException(status_code=500, detail={"error": str(e)})
