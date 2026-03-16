from fastapi import APIRouter, HTTPException, Query
import logging
from typing import Optional

from services.heatmap_service import compute_heatmaps

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/heatmap/{jobId}",
             summary="Get player heatmaps and sprint analysis")
async def get_heatmap(jobId: str, trackId: Optional[int] = Query(None)):
    """Compute and return per-player distance stats and heatmaps."""
    try:
        result = compute_heatmaps(jobId)
        if trackId is not None:
            key = str(trackId)
            if key not in result["players"]:
                raise HTTPException(
                    status_code=404,
                    detail="trackId {} not found".format(trackId),
                )
            return {
                "fps": result["fps"],
                "players": {key: result["players"][key]},
                "team_summary": result["team_summary"],
            }
        return result
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Heatmap computation failed for job %s", jobId)
        raise HTTPException(status_code=500, detail={"error": str(e)})
