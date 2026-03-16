from fastapi import APIRouter, HTTPException
import logging

from services.formation_service import compute_formations

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/formation/{jobId}",
             summary="Get tactical formation analysis")
async def get_formation(jobId: str):
    """Detect formations and shape shifts over time for a completed job."""
    try:
        result = compute_formations(jobId)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Formation computation failed for job %s", jobId)
        raise HTTPException(status_code=500, detail={"error": str(e)})
