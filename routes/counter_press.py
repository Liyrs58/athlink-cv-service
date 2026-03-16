from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import logging

from services.counter_press_service import compute_counter_press

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/{jobId}")
async def get_counter_press(
    jobId: str,
    team: Optional[int] = Query(None, description="Filter by team: 0 or 1"),
    intensity: Optional[str] = Query(None, description="Filter by intensity: high, medium, low, no_press"),
    outcome: Optional[str] = Query(None, description="Filter by outcome: success_regain, fail_possession_lost, fail_shot_conceded")
):
    """Get counter-pressing analysis."""
    try:
        result = compute_counter_press(jobId)
        
        # Filter by team if specified
        if team is not None:
            if team not in [0, 1]:
                raise HTTPException(status_code=400, detail="Team must be 0 or 1")
            
            team_key = f"team_{team}"
            if team_key not in result:
                raise HTTPException(status_code=404, detail=f"No data found for team {team}")
            
            team_data = result[team_key]
            
            # Filter windows by intensity
            if intensity:
                valid_intensities = ["high_intensity", "medium_intensity", "low_intensity", "no_press"]
                if intensity not in valid_intensities:
                    raise HTTPException(status_code=400, detail=f"Intensity must be one of: {', '.join(valid_intensities)}")
                team_data["windows"] = [w for w in team_data["windows"] if w["intensity"] == intensity]
            
            # Filter windows by outcome
            if outcome:
                valid_outcomes = ["success_regain", "fail_possession_lost", "fail_shot_conceded", "success_foul_won"]
                if outcome not in valid_outcomes:
                    raise HTTPException(status_code=400, detail=f"Outcome must be one of: {', '.join(valid_outcomes)}")
                team_data["windows"] = [w for w in team_data["windows"] if w["outcome"] == outcome]
            
            return {
                "team_0": team_data if team == 0 else {},
                "team_1": team_data if team == 1 else {},
                "total_turnovers_analysed": result["total_turnovers_analysed"]
            }
        
        # Apply filters to both teams if specified
        if intensity or outcome:
            for team_key in ["team_0", "team_1"]:
                if team_key in result:
                    team_data = result[team_key]
                    
                    if intensity:
                        valid_intensities = ["high_intensity", "medium_intensity", "low_intensity", "no_press"]
                        if intensity in valid_intensities:
                            team_data["windows"] = [w for w in team_data["windows"] if w["intensity"] == intensity]
                    
                    if outcome:
                        valid_outcomes = ["success_regain", "fail_possession_lost", "fail_shot_conceded", "success_foul_won"]
                        if outcome in valid_outcomes:
                            team_data["windows"] = [w for w in team_data["windows"] if w["outcome"] == outcome]
        
        return result
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting counter-press analysis for job {jobId}: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})
