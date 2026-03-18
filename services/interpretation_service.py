import os
import json
import math
import urllib.request

def build_rich_context(events, tracks, vel_summary, shape_summary, velocities, job_id):
    """Build honest, coach-friendly context for Claude AI report."""

    # FIX 1d: Only include non-staff tracks in analysis
    on_pitch_players = [t for t in tracks if not t.get('is_staff', False)]
    if len(on_pitch_players) > 30:
        on_pitch_players = sorted(
            on_pitch_players,
            key=lambda t: t.get('confirmed_detections', 0),
            reverse=True
        )[:30]

    confirmed = [t for t in on_pitch_players if t.get("confirmed_detections", 0) >= 5]
    total_duration = events[-1]["end_time"] if events else 0
    on_pitch_count = len(on_pitch_players)

    # PART A: Data quality checks
    warnings = []
    max_speed_kmh = vel_summary.get('max_speed_kmh', 0)
    total_sprints = vel_summary.get('total_sprints', 0)
    avg_width_m = shape_summary.get('avg_width_metres', 0)

    # Handle None and string values from shape data
    shape_data_available = (
        avg_width_m is not None and
        avg_width_m != "data unavailable" and
        isinstance(avg_width_m, (int, float))
    )

    if on_pitch_count > 25:
        warnings.append("Note: player count seems high — some duplicates may exist. Treat individual player stats as approximate.")

    if max_speed_kmh > 35:
        # Will exclude from prompt entirely
        pass

    if total_sprints > 50:
        warnings.append("Sprint count may be inflated by tracking noise. Focus on relative comparison between players, not absolute numbers.")

    if shape_data_available and avg_width_m > 70:
        # Will replace with unavailable message
        pass

    # Build situation timeline for team rhythm
    situation_timeline = []
    for e in events:
        situation_timeline.append(f"{e['start_time']:.0f}s-{e['end_time']:.0f}s: {e['situation']}")

    # Build per-player stats (only realistic values)
    per_player_stats = []
    players_with_5_detections = [v for v in velocities if v.get('confirmed_detections', 0) >= 5]
    for v in players_with_5_detections:
        track_id = v['track_id']
        sprint_count = v.get('sprint_count', 0)
        max_speed_ms = v.get('max_speed_ms', 0)
        max_speed_player_kmh = round(max_speed_ms * 3.6, 1)
        distance = v.get('distance_metres', 0)

        # Skip if speed is unrealistic
        if max_speed_player_kmh > 35:
            per_player_stats.append(f"  PLAYER #{track_id}: data unreliable for this player")
        else:
            per_player_stats.append(f"  PLAYER #{track_id}: {sprint_count} sprints | {max_speed_player_kmh} km/h | {distance}m")

    # Start building prompt with honest structure
    lines = [
        "You are a football coach assistant.",
        "Write a coaching report from this match data ONLY.",
        "Use plain English only — no jargon, no analytics terms.",
        "Be specific with player numbers and stats.",
        "Be honest when data seems unreliable.",
        "",
        "CRITICAL: Do not make any claim about historical trends unless historical_context is explicitly provided and non-empty.",
        "Only draw conclusions from the data explicitly passed in this current request.",
        "If historical_context is provided, reference it as 'across X previous matches' where X is the number of matches.",
        "Do not extrapolate or infer patterns beyond the current match data.",
        "",
        "CONFIRMED DATA (reliable):",
        f"- Clip duration: {total_duration:.0f}s",
        f"- Match phases: {', '.join(situation_timeline)}",
        f"- Players tracked: {on_pitch_count} on-pitch players",
        "",
    ]

    # Add warnings if any
    if warnings:
        lines.append("DATA QUALITY NOTES:")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    lines.append("PHYSICAL DATA (use with care — broadcast tracking):")
    if per_player_stats:
        for stat in per_player_stats:
            lines.append(stat)
    else:
        lines.append("  (No players with 5+ confirmed detections)")
    lines.append("")

    lines.append("FOR EACH PLAYER with 5+ confirmed detections:")
    lines.append("Write exactly this format:")
    lines.append("")
    lines.append("PLAYER #{track_id}")
    lines.append("Sprints: {count} | Top speed: {speed} km/h | Distance: {distance}m")
    lines.append("")
    lines.append("What they did: [1 sentence observation]")
    lines.append("What to work on: [1 sentence]")
    lines.append("Training focus: [1 specific drill or instruction]")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("TEAM RHYTHM (most reliable section):")
    for timeline_entry in situation_timeline:
        lines.append(f"  {timeline_entry}")
    lines.append("")

    # Add shape data if available
    if shape_data_available:
        lines.append("TEAM SHAPE (formation width/depth):")
        lines.append(f"- Average width: {shape_summary.get('avg_width_metres', 'N/A')}m")
        lines.append(f"- Average depth: {shape_summary.get('avg_depth_metres', 'N/A')}m")
        lines.append("")
    else:
        lines.append("TEAM SHAPE: Data unavailable for this clip (camera angle or detection issues)")
        lines.append("")

    lines.append("THE ONE THING TO ADDRESS BEFORE SATURDAY:")
    lines.append("[Based only on the most reliable data point — usually the situation timeline")
    lines.append("and relative player comparison, not absolute speed/distance]")
    lines.append("")

    lines.append("IMPORTANT: If any stat seems physically impossible")
    lines.append("(speed over 35 km/h, distance over 800m in 40s),")
    lines.append("say 'data unreliable for this player' rather than")
    lines.append("reporting the number.")

    return "\n".join(lines)


def interpret_events(events, tracks, job_id, velocity_summary=None, shape_summary=None, velocities=None, memory=None):
    if not events:
        return [{"job_id": job_id, "analysis": "No events to analyse.", "events": []}]

    prompt = build_rich_context(
        events, tracks,
        velocity_summary or {},
        shape_summary or {},
        velocities or [],
        job_id
    )

    if memory:
        prompt += f"\n\n=== HISTORICAL CONTEXT ===\n{memory}\n\nIMPORTANT: This is historical data from previous matches. Reference it as 'across previous matches' and do not mix it with current match conclusions unless explicitly comparing."

    data = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 800,
        "messages": [{"role": "user", "content": prompt}]
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
            "anthropic-version": "2023-06-01"
        }
    )

    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
            text = result["content"][0]["text"]
            return [{"job_id": job_id, "analysis": text, "events": events}]
    except Exception as e:
        return [{"job_id": job_id, "analysis": f"API error: {e}", "events": events}]
