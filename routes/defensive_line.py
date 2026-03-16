from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import logging

from services.defensive_line_service import compute_defensive_lines

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/{jobId}")
async def get_defensive_lines(
    jobId: str,
    team: Optional[int] = Query(None, description="Filter by team: 0 or 1"),
    summary: Optional[bool] = Query(False, description="Return only average statistics without timeline")
):
    """Get defensive line height and team compactness analysis."""
    try:
        result = compute_defensive_lines(jobId)
        
        # Filter by team if specified
        if team is not None:
            if team not in [0, 1]:
                raise HTTPException(status_code=400, detail="Team must be 0 or 1")
            
            team_key = f"team_{team}"
            if team_key not in result:
                raise HTTPException(status_code=404, detail=f"No data found for team {team}")
            
            team_data = result[team_key]
            
            # Return summary only if requested
            if summary:
                return {
                    "avg_defensive_line_depth_m": team_data.get("avg_defensive_line_depth_m"),
                    "avg_team_width_m": team_data.get("avg_team_width_m"),
                    "avg_team_length_m": team_data.get("avg_team_length_m"),
                    "avg_shape_area_m2": team_data.get("avg_shape_area_m2"),
                    "high_line_pct": team_data.get("high_line_pct"),
                    "low_block_pct": team_data.get("low_block_pct"),
                    "out_of_shape_events": team_data.get("out_of_shape_events", [])
                }
            else:
                return {
                    "team_0": team_data if team == 0 else {},
                    "team_1": team_data if team == 1 else {},
                    "frame_count": result["frame_count"]
                }
        
        # Return summary only if requested
        if summary:
            summary_result = {}
            for team_key in ["team_0", "team_1"]:
                if team_key in result:
                    team_data = result[team_key]
                    summary_result[team_key] = {
                        "avg_defensive_line_depth_m": team_data.get("avg_defensive_line_depth_m"),
                        "avg_team_width_m": team_data.get("avg_team_width_m"),
                        "avg_team_length_m": team_data.get("avg_team_length_m"),
                        "avg_shape_area_m2": team_data.get("avg_shape_area_m2"),
                        "high_line_pct": team_data.get("high_line_pct"),
                        "low_block_pct": team_data.get("low_block_pct"),
                        "out_of_shape_events": team_data.get("out_of_shape_events", [])
                    }
            return summary_result
        
        # Return full result
        return result
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting defensive lines for job {jobId}: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})
