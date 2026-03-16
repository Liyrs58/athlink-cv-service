from fastapi import APIRouter, HTTPException
import logging

from services.pass_network_service import compute_pass_network

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/pass-network/{jobId}",
             summary="Get pass network analysis")
async def get_pass_network(jobId: str):
    """Compute and return the pass network graph for a completed job."""
    try:
        result = compute_pass_network(jobId)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Pass network computation failed for job %s", jobId)
        raise HTTPException(status_code=500, detail={"error": str(e)})
