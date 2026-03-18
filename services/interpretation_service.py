import os
import json
import math
import urllib.request

from services.confidence_service import (
    score_track_confidence,
    score_physical_metric,
    build_data_confidence_summary,
)


def safe_float(val):
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def build_rich_context(events, tracks, vel_summary, shape_summary, velocities, job_id, team_separation=None, data_confidence=None):
    """Build honest, coach-friendly context for Claude AI report."""
    team_sep = team_separation or {}
    conf = data_confidence or {}

    on_pitch_players = [t for t in tracks if not t.get('is_staff', False)]
    if len(on_pitch_players) > 30:
        on_pitch_players = sorted(
            on_pitch_players,
            key=lambda t: t.get('confirmed_detections', 0),
            reverse=True
        )[:30]

    confirmed = [t for t in on_pitch_players if t.get("confirmed_detections", 0) >= 5]
    total_duration = safe_float(events[-1]["end_time"]) if events else 0
    on_pitch_count = len(on_pitch_players)

    max_speed_kmh = safe_float(vel_summary.get('max_speed_kmh', 0))
    total_sprints = safe_float(vel_summary.get('total_sprints', 0))
    avg_width_m = safe_float(shape_summary.get('avg_width_metres'))
    avg_depth_m = safe_float(shape_summary.get('avg_depth_metres'))
    shape_data_available = avg_width_m is not None

    # Situation timeline
    situation_timeline = []
    for e in events:
        start = safe_float(e.get('start_time', 0)) or 0
        end = safe_float(e.get('end_time', 0)) or 0
        situation_timeline.append(f"{start:.0f}s-{end:.0f}s: {e['situation']}")

    # Track team map
    track_team_map = {}
    for t in tracks:
        tid = t.get("trackId")
        if tid is not None:
            track_team_map[tid] = t.get("teamId", -1)

    t0_name = team_sep.get("team_0_colour_name", "A")
    t1_name = team_sep.get("team_1_colour_name", "B")
    has_teams = team_sep.get("status") == "ok"

    # Build per-player stats with confidence-aware language
    per_player_lines = []
    for v in velocities:
        track_id = v['track_id']
        # Find the track to get confidence
        trk = next((t for t in tracks if t.get("trackId") == track_id), None)
        if not trk or trk.get("confirmed_detections", 0) < 5:
            continue

        tc = score_track_confidence(trk)
        conf_level = tc["level"]

        sprint_count = v.get('sprint_count', 0)
        max_speed_ms = safe_float(v.get('max_speed_ms', 0))
        distance = safe_float(v.get('distance_metres', 0)) or 0
        max_speed_player_kmh = round(max_speed_ms * 3.6, 1) if max_speed_ms else 0

        team_label = ""
        if has_teams:
            ptid = track_team_map.get(track_id, -1)
            if ptid == 0:
                team_label = f" [{t0_name}]"
            elif ptid == 1:
                team_label = f" [{t1_name}]"

        if max_speed_player_kmh > 35:
            per_player_lines.append(f"  PLAYER #{track_id}{team_label} [confidence: {conf_level}]: data unreliable for this player")
        elif conf_level == "low":
            per_player_lines.append(f"  PLAYER #{track_id}{team_label} [confidence: LOW]: data insufficient for reliable conclusions")
        elif conf_level == "medium":
            sprint_lo = max(0, sprint_count - 2)
            sprint_hi = sprint_count + 2
            dist_lo = round(distance * 0.75, 0)
            dist_hi = round(distance * 1.25, 0)
            per_player_lines.append(
                f"  PLAYER #{track_id}{team_label} [confidence: MEDIUM]: "
                f"approximately {sprint_lo}-{sprint_hi} sprints | "
                f"~{max_speed_player_kmh} km/h ±20% | "
                f"{dist_lo}-{dist_hi}m"
            )
        else:  # high
            per_player_lines.append(
                f"  PLAYER #{track_id}{team_label} [confidence: HIGH]: "
                f"{sprint_count} sprints | "
                f"{max_speed_player_kmh} km/h ±10% | "
                f"{distance}m ±12%"
            )

    duration_str = f"{total_duration:.0f}" if isinstance(total_duration, (int, float)) else "unknown"

    lines = [
        "You are a football coach assistant.",
        "Write a coaching report from this match data ONLY.",
        "Use plain English only — no jargon, no analytics terms.",
        "Be specific with player numbers and stats.",
        "Be honest when data seems unreliable.",
        "",
        "CRITICAL HONESTY RULES:",
        "- For HIGH confidence data: state facts directly. E.g. 'Player 17 made 5 sprints'",
        "- For MEDIUM confidence data: use hedged language. E.g. 'Player 17 made approximately 4-6 sprints'",
        "- For LOW confidence data: say 'Player X's data is insufficient for reliable conclusions'",
        "- NEVER report a number without noting its confidence context",
        "- The 'one thing to address before Saturday' must come ONLY from HIGH or MEDIUM confidence data",
        "",
        "CRITICAL: Do not make any claim about historical trends unless historical_context is explicitly provided and non-empty.",
        "",
    ]

    # Add confidence summary
    if conf:
        lines.append(f"DATA QUALITY GRADE: {conf.get('overall_grade', '?')}")
        lines.append(f"EXPLANATION: {conf.get('grade_explanation', '')}")
        lines.append(f"High confidence players: {conf.get('high_confidence_players', 0)}")
        lines.append(f"Medium confidence players: {conf.get('medium_confidence_players', 0)}")
        lines.append(f"Low confidence players: {conf.get('low_confidence_players', 0)}")
        lines.append("")

    lines.append("CONFIRMED DATA (reliable):")
    lines.append(f"- Clip duration: {duration_str}s")
    lines.append(f"- Match phases: {', '.join(situation_timeline)}")
    lines.append(f"- Players tracked: {on_pitch_count} on-pitch players")
    lines.append("")

    if team_sep.get("status") == "ok":
        lines.append("TEAM IDENTIFICATION:")
        lines.append(f"- Team A ({t0_name} jersey): {team_sep.get('team_0_players', 0)} players detected")
        lines.append(f"- Team B ({t1_name} jersey): {team_sep.get('team_1_players', 0)} players detected")
        lines.append("")

    lines.append("STRUCTURE YOUR REPORT IN THREE SECTIONS:")
    lines.append("")
    lines.append("## RELIABLE INSIGHTS (team-level, high confidence)")
    lines.append("[Team rhythm, situation timeline, team shape — things we can trust]")
    lines.append("")
    lines.append("## APPROXIMATE ESTIMATES (player-level, medium confidence)")
    lines.append("[Individual player stats with ranges and margins of error]")
    lines.append("")
    lines.append("## DATA GAPS")
    lines.append("[What could not be measured and why — ball possession, pass accuracy, dead ball positions]")
    lines.append("")

    lines.append("PHYSICAL DATA BY PLAYER:")
    if per_player_lines:
        for line in per_player_lines:
            lines.append(line)
    else:
        lines.append("  (No players with 5+ confirmed detections)")
    lines.append("")

    lines.append("TEAM RHYTHM:")
    for entry in situation_timeline:
        lines.append(f"  {entry}")
    lines.append("")

    if shape_data_available:
        lines.append("TEAM SHAPE:")
        t0_shape = shape_summary.get("team_0", {})
        t1_shape = shape_summary.get("team_1", {})
        if has_teams and t0_shape.get("avg_width_metres") is not None:
            lines.append(f"- Team A ({t0_name}): width {t0_shape['avg_width_metres']}m, depth {t0_shape.get('avg_depth_metres', '?')}m")
        if has_teams and t1_shape.get("avg_width_metres") is not None:
            lines.append(f"- Team B ({t1_name}): width {t1_shape['avg_width_metres']}m, depth {t1_shape.get('avg_depth_metres', '?')}m")
        lines.append("")
    else:
        lines.append("TEAM SHAPE: Data unavailable for this clip")
        lines.append("")

    lines.append("THE ONE THING TO ADDRESS BEFORE SATURDAY:")
    lines.append("[Based ONLY on HIGH or MEDIUM confidence data — never from LOW confidence tracks]")
    lines.append("")

    lines.append("IMPORTANT: If any stat seems physically impossible (speed over 35 km/h, distance over 800m in 40s), say 'data unreliable' rather than reporting the number.")

    return "\n".join(lines)


def interpret_events(events, tracks, job_id, velocity_summary=None, shape_summary=None, velocities=None, memory=None, team_separation=None, data_confidence=None, brain_summary=None):
    if not events:
        return [{"job_id": job_id, "analysis": "No events to analyse.", "events": []}]

    prompt = build_rich_context(
        events, tracks,
        velocity_summary or {},
        shape_summary or {},
        velocities or [],
        job_id,
        team_separation=team_separation,
        data_confidence=data_confidence,
    )

    # Prefix with Observer Brain analysis when available
    if brain_summary:
        health = brain_summary.get("tracking_health", {})
        brain_prefix = "\n".join([
            "=== BRAIN ANALYSIS ===",
            f"Tracking health: {health.get('data_reliability', 'unknown')}",
            f"Brain verdict: {brain_summary.get('brain_verdict', '')}",
            f"Metrics to trust: {brain_summary.get('metrics_to_trust', [])}",
            f"Metrics to question: {brain_summary.get('metrics_to_question', [])}",
            f"Anomalies detected: {brain_summary.get('anomalies_summary', 'none')}",
            f"Match phases with confidence: {brain_summary.get('match_phases', [])}",
            "=== END BRAIN ANALYSIS ===",
            "",
            "INSTRUCTION: Your coaching report must reflect the brain's confidence levels. "
            "Only make strong claims about metrics the brain says to trust. "
            "For metrics to question, use hedged language ('approximately', 'suggests', 'may indicate').",
            "",
        ])
        prompt = brain_prefix + prompt

    if memory:
        prompt += f"\n\n=== HISTORICAL CONTEXT ===\n{memory}\n\nIMPORTANT: This is historical data from previous matches. Reference it as 'across previous matches' and do not mix it with current match conclusions unless explicitly comparing."

    data = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1000,
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
