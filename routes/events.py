from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import logging

from services.event_service import detect_events

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/{jobId}")
async def get_events(
    jobId: str,
    type: Optional[str] = Query(None, description="Filter by event type: pass, shot, dribble, turnover, carry"),
    team: Optional[int] = Query(None, description="Filter by team: 0 or 1"),
    from_time: Optional[float] = Query(None, description="Filter from time in seconds"),
    to_time: Optional[float] = Query(None, description="Filter to time in seconds")
):
    """Get event timeline for a job with optional filtering."""
    try:
        # Detect events
        result = detect_events(jobId)
        
        # Apply filters to events
        filtered_events = result["events"]
        
        # Filter by type
        if type:
            filtered_events = [e for e in filtered_events if e["type"] == type]
        
        # Filter by team
        if team is not None:
            filtered_events = [e for e in filtered_events if e["team"] == team]
        
        # Filter by time window
        if from_time is not None:
            filtered_events = [e for e in filtered_events if e["time_seconds"] >= from_time]
        
        if to_time is not None:
            filtered_events = [e for e in filtered_events if e["time_seconds"] <= to_time]
        
        return {
            "events": filtered_events,
            "summary": result["summary"],
            "ball_track_frames": result["ball_track_frames"],
            "interpolated_frames": result["interpolated_frames"],
            "possession_sequences": result["possession_sequences"]
        }
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting events for job {jobId}: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@router.get("/{jobId}/summary")
async def get_events_summary(jobId: str):
    """Get only the summary statistics for events."""
    try:
        result = detect_events(jobId)
        return result["summary"]
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting events summary for job {jobId}: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})
