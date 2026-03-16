from fastapi import APIRouter, HTTPException
import logging

from services.pressing_service import compute_pressing

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/pressing/{jobId}",
             summary="Get pressing intensity and PPDA analysis")
async def get_pressing(jobId: str):
    """Compute pressing metrics, PPDA, and recovery time for a completed job."""
    try:
        result = compute_pressing(jobId)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Pressing computation failed for job %s", jobId)
        raise HTTPException(status_code=500, detail={"error": str(e)})
