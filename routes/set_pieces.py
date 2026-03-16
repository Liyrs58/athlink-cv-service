from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import logging

from services.set_piece_service import detect_set_pieces

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/{jobId}")
async def get_set_pieces(
    jobId: str,
    type: Optional[str] = Query(None, description="Filter by set piece type: corner, free_kick, throw_in, penalty"),
    team: Optional[int] = Query(None, description="Filter by team: 0 or 1")
):
    """Get set piece detection analysis."""
    try:
        result = detect_set_pieces(jobId)
        
        # Filter by team if specified
        if team is not None:
            if team not in [0, 1]:
                raise HTTPException(status_code=400, detail="Team must be 0 or 1")
        
        # Filter by type if specified
        if type:
            valid_types = ["corner", "free_kick", "throw_in", "penalty", "goal_kick", "kick_off"]
            if type not in valid_types:
                raise HTTPException(status_code=400, detail=f"Type must be one of: {', '.join(valid_types)}")
            
            filtered_set_pieces = [sp for sp in result["set_pieces"] if sp["type"] == type]
            result["set_pieces"] = filtered_set_pieces
        
        # Filter by team if specified
        if team is not None:
            filtered_set_pieces = [sp for sp in result["set_pieces"] if sp["team"] == team]
            result["set_pieces"] = filtered_set_pieces
        
        return result
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting set pieces for job {jobId}: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})
