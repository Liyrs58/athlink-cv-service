"""Player performance report cards with aggregated stats.
"""

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import Color, white, black, grey, lightgrey, red, green, orange
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

logger = logging.getLogger(__name__)

# Constants
PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 40
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN
CONTENT_HEIGHT = PAGE_HEIGHT - 2 * MARGIN

# Pitch dimensions for drawing
PITCH_DRAW_WIDTH = 300
PITCH_DRAW_HEIGHT = 180
SPRINT_PITCH_WIDTH = 200
SPRINT_PITCH_HEIGHT = 120

# Import helper functions
def _base_job_id(job_id):
    # type: (str) -> str
    for suffix in ("_final_tactics", "_final_pitch", "_final", "_tactics", "_pitch"):
        if job_id.endswith(suffix):
            return job_id[:-len(suffix)]
    return job_id

def _load_json(path):
    # type: (Path) -> Optional[Dict]
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

def _get_team_color(team_id):
    """Get team colour for PDF elements."""
    return green if team_id == 0 else orange

def _interpolate_color(value):
    """Interpolate color from white (0) to red (1)."""
    # Clamp value between 0 and 1
    value = max(0, min(1, value))
    
    # Interpolate from white to red
    r = 1.0
    g = 1.0 - value
    b = 1.0 - value
    
    return Color(r, g, b)

def _draw_pitch(c, x, y, width, height, opacity=0.3):
    """Draw a football pitch with markings."""
    # Pitch outline
    c.setStrokeColor(white)
    c.setFillColor(green)
    c.rect(x, y, width, height, fill=1, stroke=1)
    
    # Set opacity for markings
    c.setFillColorOpacity(opacity)
    c.setStrokeColorOpacity(opacity)
    
    # Centre line
    c.line(x + width/2, y, x + width/2, y + height)
    
    # Centre circle
    centre_x = x + width/2
    centre_y = y + height/2
    radius = min(width, height) * 0.15
    c.circle(centre_x, centre_y, radius, fill=0, stroke=1)
    
    # Penalty areas (simplified)
    pen_width = width * 0.16
    pen_height = height * 0.44
    pen_depth = width * 0.12
    
    # Left penalty area
    c.rect(x, y + (height - pen_height)/2, pen_depth, pen_height, fill=0, stroke=1)
    
    # Right penalty area
    c.rect(x + width - pen_depth, y + (height - pen_height)/2, pen_depth, pen_height, fill=0, stroke=1)
    
    # Reset opacity
    c.setFillColorOpacity(1.0)
    c.setStrokeColorOpacity(1.0)

def _draw_heatmap(c, x, y, width, height, heatmap_data, pitch_bg=True):
    """Draw a heatmap on a pitch."""
    if pitch_bg:
        _draw_pitch(c, x, y, width, height)
    
    # Draw heatmap cells
    if not heatmap_data:
        return
    
    grid_w = 21
    grid_h = 14
    cell_width = width / grid_w
    cell_height = height / grid_h
    
    for i, value in enumerate(heatmap_data):
        if value <= 0:
            continue
            
        grid_x = i % grid_w
        grid_y = i // grid_w
        
        cell_x = x + grid_x * cell_width
        cell_y = y + (grid_h - grid_y - 1) * cell_height  # Flip Y axis
        
        c.setFillColor(_interpolate_color(value))
        c.rect(cell_x, cell_y, cell_width, cell_height, fill=1, stroke=0)

def _draw_bar_chart(c, x, y, width, height, data, labels):
    """Draw a simple bar chart."""
    if not data:
        return
    
    max_value = max(data) if data else 1
    bar_width = width / len(data)
    
    for i, (value, label) in enumerate(zip(data, labels)):
        if max_value > 0:
            bar_height = (value / max_value) * height
        else:
            bar_height = 0
        
        bar_x = x + i * bar_width
        bar_y = y
        
        # Draw bar
        c.setFillColor(lightgrey)
        c.rect(bar_x, bar_y, bar_width - 2, bar_height, fill=1, stroke=0)
        
        # Draw label
        c.setFillColor(black)
        c.setFont("Helvetica", 8)
        text_width = c.stringWidth(str(value), "Helvetica", 8)
        text_x = bar_x + (bar_width - text_width) / 2
        text_y = bar_y - 12
        c.drawString(text_x, text_y, str(value))

def _get_player_data(job_id: str, track_id: int) -> Dict[str, Any]:
    """Get comprehensive player data from analytics cache or compute."""
    # Try to load from analytics cache first
    analytics_path = Path(f"temp/{job_id}/analytics/analytics_report.json")
    if analytics_path.exists():
        analytics_data = _load_json(analytics_path)
        if analytics_data and "players" in analytics_data:
            player_key = str(track_id)
            if player_key in analytics_data["players"]:
                return analytics_data["players"][player_key]
    
    # If not cached, load and compute from individual services
    player_data = {}
    
    # Load basic tracking data
    track_data = _load_json(Path(f"temp/{job_id}/tracking/track_results.json"))
    team_data = _load_json(Path(f"temp/{job_id}/tracking/team_results.json"))
    
    if track_data and team_data:
        # Get team info
        team_map = {}
        tracks_list = team_data.get("tracks", team_data) if isinstance(team_data, dict) else team_data
        for t in tracks_list:
            team_map[t["trackId"]] = t.get("teamId", -1)
        
        player_data["team"] = team_map.get(track_id, -1)
        
        # Extract trajectory data
        trajectory = []
        if "players" in track_data:
            for player in track_data["players"]:
                if player["trackId"] == track_id:
                    trajectory = player.get("trajectory2d", [])
                    break
        
        # Calculate basic metrics
        total_distance = 0.0
        sprint_distance = 0.0
        top_speed = 0.0
        sprint_count = 0
        
        if len(trajectory) > 1:
            prev_x, prev_y = trajectory[0]["x"], trajectory[0]["y"]
            for i in range(1, len(trajectory)):
                curr_x, curr_y = trajectory[i]["x"], trajectory[i]["y"]
                dist = math.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                total_distance += dist
                
                # Simple speed calculation (would need frame times for accuracy)
                speed = dist * 25  # Assuming 25 FPS
                if speed > top_speed:
                    top_speed = speed
                
                # Sprint detection (simplified)
                if speed > 7.0:  # 7 m/s threshold
                    sprint_distance += dist
                    sprint_count += 1
                
                prev_x, prev_y = curr_x, curr_y
        
        player_data["total_distance_km"] = round(total_distance / 1000, 1)
        player_data["sprint_distance_km"] = round(sprint_distance / 1000, 1)
        player_data["top_speed_kmh"] = round(top_speed * 3.6, 1)
        player_data["sprint_count"] = sprint_count
    
    # Load heatmap data if available
    heatmap_data = _load_json(Path(f"temp/{job_id}/heatmaps/heatmap.json"))
    if heatmap_data and "players" in heatmap_data:
        player_key = str(track_id)
        if player_key in heatmap_data["players"]:
            player_data["heatmap"] = heatmap_data["players"][player_key].get("heatmap", [])
            player_data["sprint_heatmap"] = heatmap_data["players"][player_key].get("sprint_heatmap", [])
    
    # Load event data if available
    events_data = _load_json(Path(f"temp/{job_id}/events/event_timeline.json"))
    if events_data and "events" in events_data:
        player_events = [e for e in events_data["events"] if e.get("player_track_id") == track_id]
        
        # Count event types
        event_counts = {
            "passes": len([e for e in player_events if e["type"] == "pass"]),
            "shots": len([e for e in player_events if e["type"] == "shot"]),
            "dribbles": len([e for e in player_events if e["type"] == "dribble"]),
            "turnovers": len([e for e in player_events if e["type"] == "turnover"]),
            "carries": len([e for e in player_events if e["type"] == "carry"])
        }
        
        player_data["event_counts"] = event_counts
        
        # Passing stats
        passes = [e for e in player_events if e["type"] == "pass"]
        passes_received = len([e for e in events_data["events"] if e.get("to_player_track_id") == track_id])
        
        if passes:
            avg_pass_distance = sum(e.get("distance_m", 0) for e in passes) / len(passes)
        else:
            avg_pass_distance = 0
        
        player_data["passes_made"] = len(passes)
        player_data["passes_received"] = passes_received
        player_data["avg_pass_distance"] = round(avg_pass_distance, 1)
    
    return player_data

def generate_player_report(job_id: str, track_id: int) -> bytes:
    """Generate PDF report for a single player."""
    # Get player data
    player_data = _get_player_data(job_id, track_id)
    
    if not player_data:
        raise ValueError(f"No data found for player {track_id}")
    
    # Create PDF buffer
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    
    # Set up fonts
    c.setFont("Helvetica", 12)
    
    # HEADER
    # Left side
    c.setFont("Helvetica-Bold", 20)
    c.drawString(MARGIN, PAGE_HEIGHT - MARGIN - 20, "AthLink")
    c.setFont("Helvetica", 12)
    c.drawString(MARGIN, PAGE_HEIGHT - MARGIN - 40, "Match Analysis")
    
    # Right side
    c.setFont("Helvetica-Bold", 20)
    player_text = f"Player #{track_id}"
    c.drawString(PAGE_WIDTH - MARGIN - c.stringWidth(player_text, "Helvetica-Bold", 20), 
                 PAGE_HEIGHT - MARGIN - 20, player_text)
    
    # Team badge
    team = player_data.get("team", -1)
    if team in [0, 1]:
        c.setFillColor(_get_team_color(team))
        c.rect(PAGE_WIDTH - MARGIN - 60, PAGE_HEIGHT - MARGIN - 35, 50, 20, fill=1, stroke=0)
    
    # Separator line
    c.setStrokeColor(black)
    c.setLineWidth(1)
    c.line(MARGIN, PAGE_HEIGHT - MARGIN - 60, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - MARGIN - 60)
    
    # SECTION 1 - Physical Stats
    y_pos = PAGE_HEIGHT - MARGIN - 100
    section_height = 80
    
    # Draw 4 metric boxes
    metrics = [
        ("Total Distance (km)", player_data.get("total_distance_km", 0)),
        ("Sprint Distance (km)", player_data.get("sprint_distance_km", 0)),
        ("Top Speed (km/h)", player_data.get("top_speed_kmh", 0)),
        ("Sprint Count", player_data.get("sprint_count", 0))
    ]
    
    box_width = CONTENT_WIDTH / 4 - 10
    for i, (label, value) in enumerate(metrics):
        x = MARGIN + i * (box_width + 10)
        
        # Background
        c.setFillColor(lightgrey)
        c.rect(x, y_pos, box_width, section_height, fill=1, stroke=0)
        
        # Label
        c.setFillColor(grey)
        c.setFont("Helvetica", 9)
        c.drawString(x + 5, y_pos + section_height - 15, label)
        
        # Value
        c.setFillColor(black)
        c.setFont("Helvetica-Bold", 18)
        value_text = str(value)
        c.drawString(x + 5, y_pos + section_height - 45, value_text)
    
    # SECTION 2 - Position Heatmap
    y_pos -= section_height + 40
    c.setFont("Helvetica-Bold", 14)
    c.drawString(MARGIN, y_pos, "Position Heatmap")
    y_pos -= 20
    
    heatmap_x = MARGIN
    heatmap_y = y_pos
    _draw_heatmap(c, heatmap_x, heatmap_y, PITCH_DRAW_WIDTH, PITCH_DRAW_HEIGHT, 
                  player_data.get("heatmap", []))
    
    # SECTION 3 - Passing Stats (if available)
    if "passes_made" in player_data:
        y_pos -= PITCH_DRAW_HEIGHT + 40
        c.setFont("Helvetica-Bold", 14)
        c.drawString(MARGIN, y_pos, "Passing Statistics")
        y_pos -= 25
        
        c.setFont("Helvetica", 11)
        stats_text = [
            f"Passes Made: {player_data.get('passes_made', 0)}",
            f"Passes Received: {player_data.get('passes_received', 0)}",
            f"Average Distance: {player_data.get('avg_pass_distance', 0)}m"
        ]
        
        for i, text in enumerate(stats_text):
            c.drawString(MARGIN, y_pos - i * 15, text)
        
        # Badges
        if player_data.get("passes_made", 0) > 10:
            c.setFillColor(green)
            c.setFont("Helvetica-Bold", 12)
            c.drawString(MARGIN + 200, y_pos, "Key Passer")
    
    # SECTION 4 - Event Breakdown (if available)
    if "event_counts" in player_data:
        y_pos -= 120
        c.setFont("Helvetica-Bold", 14)
        c.drawString(MARGIN, y_pos, "Event Breakdown")
        y_pos -= 25
        
        event_data = player_data["event_counts"]
        event_labels = ["Passes", "Shots", "Dribbles", "Turnovers", "Carries"]
        event_values = [event_data.get("passes", 0), event_data.get("shots", 0),
                       event_data.get("dribbles", 0), event_data.get("turnovers", 0),
                       event_data.get("carries", 0)]
        
        _draw_bar_chart(c, MARGIN, y_pos - 60, CONTENT_WIDTH, 60, event_values, event_labels)
    
    # SECTION 5 - Sprint Heatmap (if available)
    if player_data.get("sprint_heatmap"):
        # Place beside passing stats if space allows
        sprint_x = MARGIN + CONTENT_WIDTH - SPRINT_PITCH_WIDTH - 50
        sprint_y = PAGE_HEIGHT - MARGIN - 300
        
        c.setFont("Helvetica-Bold", 12)
        c.drawString(sprint_x, sprint_y + 20, "Sprint Heatmap")
        
        _draw_heatmap(c, sprint_x, sprint_y, SPRINT_PITCH_WIDTH, SPRINT_PITCH_HEIGHT,
                      player_data.get("sprint_heatmap", []))
    
    # FOOTER
    c.setFont("Helvetica", 8)
    c.setFillColor(grey)
    footer_text = f"Generated by AthLink CV Service - Job ID: {job_id} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    c.drawString(MARGIN, MARGIN - 20, footer_text)
    
    # Save PDF
    c.save()
    buffer.seek(0)
    
    # Also save to file
    output_dir = Path(f"temp/{job_id}/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / f"player_{track_id}.pdf", "wb") as f:
        f.write(buffer.getvalue())
    
    return buffer.getvalue()

def generate_team_report(job_id: str, team: int) -> bytes:
    """Generate PDF report for an entire team."""
    # Load team data
    team_data = _load_json(Path(f"temp/{job_id}/tracking/team_results.json"))
    if not team_data:
        raise ValueError("No team data found")
    
    # Get players for this team
    team_map = {}
    tracks_list = team_data.get("tracks", team_data) if isinstance(team_data, dict) else team_data
    for t in tracks_list:
        team_map[t["trackId"]] = t.get("teamId", -1)
    
    team_players = [tid for tid, t in team_map.items() if t == team]
    
    if not team_players:
        raise ValueError(f"No players found for team {team}")
    
    # Create PDF buffer
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    
    # Page 1: Team Summary
    c.setFont("Helvetica-Bold", 24)
    team_text = f"Team {team} Report"
    c.drawString(MARGIN, PAGE_HEIGHT - MARGIN - 30, team_text)
    
    c.setFont("Helvetica", 12)
    c.drawString(MARGIN, PAGE_HEIGHT - MARGIN - 60, f"Job ID: {job_id}")
    c.drawString(MARGIN, PAGE_HEIGHT - MARGIN - 80, f"Players: {len(team_players)}")
    
    # Load analytics data for team summary
    analytics_data = _load_json(Path(f"temp/{job_id}/analytics/analytics_report.json"))
    if analytics_data:
        y_pos = PAGE_HEIGHT - MARGIN - 120
        
        # Add key metrics
        summary = analytics_data.get("match_summary", {})
        
        metrics = [
            ("Total Passes", summary.get("total_passes", "N/A")),
            ("Possession %", f"{summary.get('possession_pct', {}).get(f'team_{team}', 'N/A')}%"),
            ("xG", summary.get(f"xg_team_{team}", "N/A")),
            ("Shots", summary.get(f"shots_team_{team}", "N/A"))
        ]
        
        c.setFont("Helvetica-Bold", 14)
        c.drawString(MARGIN, y_pos, "Team Statistics")
        y_pos -= 25
        
        c.setFont("Helvetica", 12)
        for label, value in metrics:
            c.drawString(MARGIN, y_pos, f"{label}: {value}")
            y_pos -= 20
    
    # Subsequent pages: Individual player reports
    for i, track_id in enumerate(team_players):
        c.showPage()
        
        try:
            # Generate player data for this page (simplified version)
            player_data = _get_player_data(job_id, track_id)
            
            # Player header
            c.setFont("Helvetica-Bold", 18)
            c.drawString(MARGIN, PAGE_HEIGHT - MARGIN - 30, f"Player #{track_id}")
            
            # Team badge
            c.setFillColor(_get_team_color(team))
            c.rect(PAGE_WIDTH - MARGIN - 60, PAGE_HEIGHT - MARGIN - 35, 50, 20, fill=1, stroke=0)
            
            # Key metrics
            y_pos = PAGE_HEIGHT - MARGIN - 80
            metrics = [
                ("Distance", f"{player_data.get('total_distance_km', 0)} km"),
                ("Top Speed", f"{player_data.get('top_speed_kmh', 0)} km/h"),
                ("Sprints", player_data.get("sprint_count", 0)),
                ("Passes", player_data.get("passes_made", 0))
            ]
            
            c.setFont("Helvetica", 12)
            for label, value in metrics:
                c.drawString(MARGIN, y_pos, f"{label}: {value}")
                y_pos -= 20
            
            # Mini heatmap
            if player_data.get("heatmap"):
                y_pos -= 30
                c.setFont("Helvetica-Bold", 12)
                c.drawString(MARGIN, y_pos, "Position Heatmap")
                y_pos -= 20
                
                # Draw smaller heatmap
                mini_width = 200
                mini_height = 120
                _draw_heatmap(c, MARGIN, y_pos - mini_height, mini_width, mini_height,
                              player_data.get("heatmap", []))
        
        except Exception as e:
            logger.warning(f"Failed to generate player {track_id} page: {e}")
            # Continue with next player
            continue
    
    # Save PDF
    c.save()
    buffer.seek(0)
    
    # Also save to file
    output_dir = Path(f"temp/{job_id}/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / f"team_{team}.pdf", "wb") as f:
        f.write(buffer.getvalue())

    return buffer.getvalue()


def build_match_pdf(job_id: str, output_path: str) -> str:
    """Render a single-page PDF summary from match_report.json.

    Lightweight, mood-board-aligned: header, quality metrics row, two-team metrics
    row, top players list, events count. No charts (heatmap/per-player pages are
    generated separately via generate_player_report).
    """
    report_path = Path(f"temp/{job_id}/match_report.json")
    if not report_path.exists():
        from services.match_report_service import build_match_report as _build
        _build(job_id)
    with open(report_path) as f:
        report = json.load(f)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(output_path, pagesize=A4)

    # Header
    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(MARGIN, PAGE_HEIGHT - MARGIN - 8, "AthLink CV — Match Report")
    c.setFont("Helvetica", 10)
    c.setFillColor(grey)
    c.drawString(MARGIN, PAGE_HEIGHT - MARGIN - 24, f"Job: {report.get('jobId', job_id)}")

    # Quality strip
    y = PAGE_HEIGHT - MARGIN - 60
    quality = report.get("quality", {}) or {}
    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(MARGIN, y, "Tracking quality")
    y -= 14
    c.setFont("Helvetica", 9)
    quality_lines = [
        ("Stable IDs", quality.get("stableIds")),
        ("Track resurrection", quality.get("trackResurrection")),
        ("ID switches /min", quality.get("lowIdSwitchesPerMin")),
        ("Pitch detection", quality.get("pitchDetectionCoverage")),
        ("Match confidence", quality.get("matchConfidence")),
        ("Valid ID coverage", quality.get("validIdCoverage")),
        ("Unique IDs", quality.get("uniqueIds")),
        ("Unknown boxes", quality.get("unknownBoxes")),
    ]
    for label, val in quality_lines:
        if val is None:
            display = "—"
        elif isinstance(val, float):
            display = f"{val:.3f}"
        else:
            display = str(val)
        c.drawString(MARGIN + 10, y, f"{label}: {display}")
        y -= 12

    # Teams + metrics block
    y -= 10
    teams = report.get("teams", {}) or {}
    metrics = report.get("metrics", {}) or {}
    c.setFont("Helvetica-Bold", 11)
    c.drawString(MARGIN, y, "Teams")
    y -= 14
    c.setFont("Helvetica", 9)
    home, away = teams.get("home", {}), teams.get("away", {})
    c.drawString(
        MARGIN + 10, y,
        f"Home — players={home.get('playerCount', 0)} formation={home.get('formation') or '—'} "
        f"conf={home.get('formationConfidence') or '—'}"
    )
    y -= 12
    c.drawString(
        MARGIN + 10, y,
        f"Away — players={away.get('playerCount', 0)} formation={away.get('formation') or '—'} "
        f"conf={away.get('formationConfidence') or '—'}"
    )
    y -= 18
    c.setFont("Helvetica-Bold", 11)
    c.drawString(MARGIN, y, "Metrics (Home / Away)")
    y -= 14
    c.setFont("Helvetica", 9)
    rows = [
        ("Possession %", metrics.get("possessionPct")),
        ("Pass accuracy", metrics.get("passAccuracy")),
        ("Pressures", metrics.get("pressures")),
        ("Distance covered (km)", metrics.get("distanceCoveredKm")),
        ("Sprints", metrics.get("sprints")),
        ("xG", metrics.get("expectedGoals")),
        ("Big chances", metrics.get("bigChances")),
        ("Shot accuracy", metrics.get("shotAccuracy")),
        ("Turnovers won", metrics.get("turnoversWon")),
    ]
    for label, vals in rows:
        if not vals or len(vals) != 2:
            continue
        c.drawString(MARGIN + 10, y, f"{label}: {vals[0]} / {vals[1]}")
        y -= 12

    # Players (top 8)
    y -= 8
    c.setFont("Helvetica-Bold", 11)
    c.drawString(MARGIN, y, "Players (top 8 by distance)")
    y -= 14
    c.setFont("Helvetica", 9)
    players = report.get("players", []) or []
    sorted_players = sorted(players, key=lambda p: -(p.get("distanceM") or 0))[:8]
    for p in sorted_players:
        c.drawString(
            MARGIN + 10, y,
            f"{p.get('playerId', '?'):<6} {p.get('team') or '—':<6}  "
            f"distance={p.get('distanceM', 0):.1f}m  sprints={p.get('sprints', 0)}  "
            f"top={p.get('topSpeedKmh', 0):.1f} km/h"
        )
        y -= 12

    # Events count
    y -= 8
    n_events = len(report.get("events", []) or [])
    c.drawString(MARGIN, y, f"Events captured: {n_events}")

    c.save()
    return output_path
