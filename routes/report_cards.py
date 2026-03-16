from fastapi import APIRouter, HTTPException, Response
from typing import Dict, Any
import logging

from services.report_card_service import generate_player_report, generate_team_report
from services.storage_service import _base_job_id, _load_json
from pathlib import Path

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/{jobId}/player/{trackId}")
async def get_player_report(jobId: str, trackId: int):
    """Generate PDF report for a specific player."""
    try:
        # Check if player exists in tracking data
        track_data = _load_json(Path(f"temp/{jobId}/tracking/track_results.json"))
        if not track_data:
            raise HTTPException(status_code=404, detail="Tracking data not found")
        
        # Verify track_id exists
        player_found = False
        if "players" in track_data:
            for player in track_data["players"]:
                if player["trackId"] == trackId:
                    player_found = True
                    break
        
        if not player_found:
            raise HTTPException(status_code=404, detail=f"Player {trackId} not found")
        
        # Generate PDF
        pdf_bytes = generate_player_report(jobId, trackId)
        
        # Return PDF with appropriate headers
        headers = {
            "Content-Type": "application/pdf",
            "Content-Disposition": f"attachment; filename=\"player_{trackId}_report.pdf\""
        }
        
        return Response(content=pdf_bytes, headers=headers)
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating player report for job {jobId}, player {trackId}: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@router.get("/{jobId}/team/{team}")
async def get_team_report(jobId: str, team: int):
    """Generate PDF report for an entire team."""
    try:
        if team not in [0, 1]:
            raise HTTPException(status_code=400, detail="Team must be 0 or 1")
        
        # Check if team data exists
        team_data = _load_json(Path(f"temp/{jobId}/tracking/team_results.json"))
        if not team_data:
            raise HTTPException(status_code=404, detail="Team data not found")
        
        # Verify team has players
        team_map = {}
        tracks_list = team_data.get("tracks", team_data) if isinstance(team_data, dict) else team_data
        for t in tracks_list:
            team_map[t["trackId"]] = t.get("teamId", -1)
        
        team_players = [tid for tid, t in team_map.items() if t == team]
        if not team_players:
            raise HTTPException(status_code=404, detail=f"No players found for team {team}")
        
        # Generate PDF
        pdf_bytes = generate_team_report(jobId, team)
        
        # Return PDF with appropriate headers
        headers = {
            "Content-Type": "application/pdf",
            "Content-Disposition": f"attachment; filename=\"team_{team}_report.pdf\""
        }
        
        return Response(content=pdf_bytes, headers=headers)
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating team report for job {jobId}, team {team}: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@router.get("/{jobId}/available")
async def get_available_reports(jobId: str):
    """Get list of available players and teams for report generation."""
    try:
        # Load tracking data
        track_data = _load_json(Path(f"temp/{jobId}/tracking/track_results.json"))
        team_data = _load_json(Path(f"temp/{jobId}/tracking/team_results.json"))
        
        if not track_data or not team_data:
            raise HTTPException(status_code=404, detail="Tracking or team data not found")
        
        # Get all player track IDs
        players = []
        if "players" in track_data:
            players = [player["trackId"] for player in track_data["players"]]
        
        # Get teams that have players
        team_map = {}
        tracks_list = team_data.get("tracks", team_data) if isinstance(team_data, dict) else team_data
        for t in tracks_list:
            team_map[t["trackId"]] = t.get("teamId", -1)
        
        teams = sorted(set(team_map.values()))
        teams = [t for t in teams if t in [0, 1]]  # Filter valid teams
        
        return {
            "players": players,
            "teams": teams,
            "job_id": jobId
        }
        
    except Exception as e:
        logger.error(f"Error getting available reports for job {jobId}: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})
