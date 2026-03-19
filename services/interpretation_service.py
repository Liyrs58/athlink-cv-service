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
    """Build plain-English coaching context. No jargon, no raw technical values."""
    team_sep = team_separation or {}
    conf = data_confidence or {}

    on_pitch_players = [t for t in tracks if not t.get('is_staff', False)]
    if len(on_pitch_players) > 30:
        on_pitch_players = sorted(
            on_pitch_players,
            key=lambda t: t.get('confirmed_detections', 0),
            reverse=True
        )[:30]

    total_duration = safe_float(events[-1]["end_time"]) if events else 0

    t0_name = team_sep.get("team_0_colour_name", "Team A")
    t1_name = team_sep.get("team_1_colour_name", "Team B")
    has_teams = team_sep.get("status") == "ok"

    # Situation timeline
    situation_timeline = []
    for e in events:
        start = safe_float(e.get('start_time', 0)) or 0
        end = safe_float(e.get('end_time', 0)) or 0
        label = "open play" if e['situation'] == 'OPEN_PLAY' else "stoppage"
        situation_timeline.append(f"{start:.0f}s–{end:.0f}s: {label}")

    # Separate open play and dead ball windows
    open_play_windows = [e for e in events if e.get('situation') == 'OPEN_PLAY']
    dead_ball_windows = [e for e in events if e.get('situation') != 'OPEN_PLAY']

    # Build team shape per situation phase
    def _shape_for_phase(phase_name, width, depth):
        if width is None:
            return None
        return f"roughly {width:.0f}m wide, {depth:.0f}m deep" if depth else f"roughly {width:.0f}m wide"

    sh_t0 = shape_summary.get("team_0", {})
    sh_t1 = shape_summary.get("team_1", {})
    t0_shape_str = _shape_for_phase("open play", sh_t0.get("avg_width_metres"), sh_t0.get("avg_depth_metres"))
    t1_shape_str = _shape_for_phase("open play", sh_t1.get("avg_width_metres"), sh_t1.get("avg_depth_metres"))

    # Build per-player numbers, skip low confidence and inactive players
    # Number players within each team by distance (Player 1 = most distance)
    team_player_velocities = {"team_0": [], "team_1": [], "unassigned": []}
    track_team_map = {}
    for t in tracks:
        tid = t.get("trackId")
        if tid is not None:
            track_team_map[tid] = t.get("teamId", -1)

    for v in velocities:
        track_id = v['track_id']
        trk = next((t for t in tracks if t.get("trackId") == track_id), None)
        if not trk or trk.get("confirmed_detections", 0) < 5:
            continue
        tc = score_track_confidence(trk)
        conf_level = tc["level"]
        if conf_level == "low":
            continue  # skip low confidence players

        team_id = track_team_map.get(track_id, -1)
        team_key = "team_0" if team_id == 0 else ("team_1" if team_id == 1 else "unassigned")

        distance = safe_float(v.get('distance_metres', 0)) or 0
        max_speed_ms = safe_float(v.get('max_speed_ms', 0))
        sprint_count = v.get('sprint_count', 0)
        max_speed_kmh = round(max_speed_ms * 3.6, 1) if max_speed_ms else 0

        pct = {"high": 0.12, "medium": 0.25}.get(conf_level, 0.25)
        dist_lo = round(distance * (1 - pct), 0)
        dist_hi = round(distance * (1 + pct), 0)

        team_player_velocities[team_key].append({
            "track_id": track_id,
            "conf_level": conf_level,
            "distance": distance,
            "dist_range": f"{dist_lo:.0f}–{dist_hi:.0f}m",
            "max_speed_kmh": max_speed_kmh,
            "sprint_count": sprint_count,
        })

    # Sort by distance descending, assign sequential player numbers per team
    player_lines = []
    for team_key, team_name in [("team_0", t0_name), ("team_1", t1_name), ("unassigned", "Unassigned")]:
        sorted_players = sorted(team_player_velocities[team_key], key=lambda p: p["distance"], reverse=True)
        for idx, p in enumerate(sorted_players, 1):
            player_label = f"{team_name} Player {idx}"
            conf = p["conf_level"]
            if conf == "high":
                player_lines.append(
                    f"  {player_label} [HIGH confidence]: "
                    f"distance {p['dist_range']} | "
                    f"{p['max_speed_kmh']} km/h peak (±10%) | "
                    f"{p['sprint_count']} sprint(s)"
                )
            else:
                speed_lo = round(p['max_speed_kmh'] * 0.8, 1)
                speed_hi = round(p['max_speed_kmh'] * 1.2, 1)
                player_lines.append(
                    f"  {player_label} [MEDIUM confidence]: "
                    f"roughly {p['dist_range']} | "
                    f"approximately {speed_lo}–{speed_hi} km/h peak | "
                    f"approximately {max(0, p['sprint_count'] - 2)}–{p['sprint_count'] + 2} sprint(s)"
                )

    duration_str = f"{total_duration:.0f}" if isinstance(total_duration, (int, float)) else "unknown"

    lines = [
        "You are a football analyst writing a post-match report for a grassroots or semi-professional coach.",
        "",
        "The coach reads this on their phone. They played or coached today and want to know what the data actually showed.",
        "",
        "STRICT RULES — break these and the report is useless:",
        "",
        "1. NEVER use these words:",
        "   entropy, Voronoi, Shannon, homography, spectral, belief state, heuristic, algorithm,",
        "   pixel, calibration, neuro-symbolic, anomaly, interpolated, biomechanical.",
        "   Replace them with plain English always.",
        "",
        "2. NEVER invent tactical labels.",
        "   Do not write 'high press', 'gegenpressing', 'low block', 'progressive carries'",
        "   unless the data explicitly shows it. If you cannot prove it from the numbers, do not say it.",
        "",
        "3. ONLY make claims about metrics the brain trusts.",
        "   The brain analysis section tells you what to trust and what to question.",
        "   If a metric is in 'metrics_to_question' do not mention it in the report at all.",
        "   If confidence is low on a metric, skip it entirely.",
        "",
        "4. Every tactical claim needs a number behind it.",
        "   BAD: 'The team pressed well in the first half.'",
        "   GOOD: 'During open play your team spread to roughly 36m wide.'",
        "   BAD: 'Player 7 showed good work rate.'",
        "   GOOD: 'Blue Player 1 covered the most ground — roughly 140–180m.'",
        "",
        "5. When you describe shape, describe it at a specific moment.",
        "   BAD: 'The team averaged 36m width.'",
        "   GOOD: 'During open play your team spread to roughly 36m wide.'",
        "",
        "6. Approximate estimates must say they are approximate.",
        "   Use: 'roughly', 'around', 'approximately', 'suggests' for any MEDIUM confidence metric.",
        "",
        "7. Short sentences. Maximum 20 words per sentence.",
        "   Write like you are texting a coach, not writing a thesis.",
        "",
        "8. No praise for the sake of it.",
        "   Do not write 'great work rate' or 'good intensity.' Only write what the data shows.",
        "",
        "REPORT STRUCTURE — use exactly these sections:",
        "",
        "## WHAT HAPPENED",
        "3-4 sentences maximum.",
        "Describe the match flow using only the situation timeline. Reference actual timestamps.",
        "Do not invent what caused stoppages.",
        "",
        "## TEAM SHAPE",
        "2-3 sentences. Use actual width and depth numbers.",
        "Do not label formations (4-4-2 etc) — you cannot confirm these from the data.",
        "If one team was significantly wider or narrower, say so and give the number.",
        "",
        "## WHO COVERED THE MOST GROUND",
        "List the top 3-4 players by distance covered. Use the distance range, not a single number.",
        "One sentence per player maximum. Only include HIGH or MEDIUM confidence players.",
        "Skip LOW confidence players entirely.",
        "",
        "## SPRINT EFFORTS",
        "ONLY include this section if 'sprint_counts' is in the trusted metrics list above.",
        "If 'sprint_counts' is in the questioned metrics list, skip this section entirely and do not mention sprints anywhere.",
        "If included: list players who sprinted, give count, note peak speed with margin.",
        "",
        "## SPACE CONTROL",
        "Only include if space control data is provided below.",
        "Do not use the word Voronoi.",
        "Say it plainly: 'Blue had the ball in more areas of the pitch for most of this clip — controlling roughly X% of the available space.'",
        "One or two sentences only.",
        "",
        "## ONE THING TO WORK ON",
        "This is the most important section.",
        "Pick ONE specific, actionable thing the data suggests needs improvement.",
        "It must be based directly on the data — not generic advice.",
        "BAD: 'Work on pressing higher up the pitch.'",
        "GOOD: 'During stoppages your team's shape tightened to around 20m wide. Work on maintaining width when play stops.'",
        "Maximum 4 sentences.",
        "",
    ]

    lines.append(f"CLIP DURATION: {duration_str} seconds")
    lines.append("")

    lines.append("SITUATION TIMELINE:")
    for entry in situation_timeline:
        lines.append(f"  {entry}")
    lines.append("")

    if has_teams:
        lines.append("TEAM IDENTIFICATION:")
        lines.append(f"  Team A name: {t0_name} ({team_sep.get('team_0_players', 0)} players detected)")
        lines.append(f"  Team B name: {t1_name} ({team_sep.get('team_1_players', 0)} players detected)")
        lines.append("")

    lines.append("TEAM SHAPE (use these numbers directly):")
    if t0_shape_str:
        lines.append(f"  {t0_name}: {t0_shape_str} (averaged across open play phases)")
    else:
        lines.append(f"  {t0_name}: shape data unavailable")
    if t1_shape_str:
        lines.append(f"  {t1_name}: {t1_shape_str} (averaged across open play phases)")
    else:
        lines.append(f"  {t1_name}: shape data unavailable")
    lines.append("")

    lines.append("PLAYER DISTANCES (use these for 'who covered the most ground'):")
    lines.append("Players are numbered 1, 2, 3... within each team, by distance covered (Player 1 = most distance).")
    if player_lines:
        for pl in player_lines:
            lines.append(pl)
    else:
        lines.append("  (No players with sufficient tracking data)")
    lines.append("")

    return "\n".join(lines)


def interpret_events(events, tracks, job_id, velocity_summary=None, shape_summary=None, velocities=None, memory=None, team_separation=None, data_confidence=None, brain_summary=None, voronoi=None, entropy=None):
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
            "=== DATA QUALITY SUMMARY ===",
            f"Overall data reliability: {health.get('data_reliability', 'unknown')}",
            f"Summary: {brain_summary.get('brain_verdict', '')}",
            f"Metrics you can trust: {brain_summary.get('metrics_to_trust', [])}",
            f"Metrics to question (skip these in the report): {brain_summary.get('metrics_to_question', [])}",
            f"Issues detected: {brain_summary.get('anomalies_summary', 'none')}",
            "=== END DATA QUALITY SUMMARY ===",
            "",
            "INSTRUCTION: Your coaching report must reflect the data quality above. "
            "Only make strong claims about metrics listed as trusted. "
            "For metrics listed as 'to question', do not mention them at all in your report.",
            "",
        ])
        prompt = brain_prefix + prompt

    # Space control context (plain English, no Voronoi label)
    if voronoi and voronoi.get("status") == "ok":
        ts = team_separation or {}
        t0_name = ts.get("team_0_colour_name", "Team A")
        t1_name = ts.get("team_1_colour_name", "Team B")
        dom_team = t0_name if voronoi.get("dominant_team") == 0 else t1_name
        prompt += "\n".join([
            "",
            "SPACE CONTROL DATA:",
            f"  {t0_name} controlled {voronoi.get('team_0_control_pct', 0):.1f}% of available pitch space",
            f"  {t1_name} controlled {voronoi.get('team_1_control_pct', 0):.1f}% of available pitch space",
            f"  {dom_team} had the advantage by {voronoi.get('control_margin', 0):.1f}%",
            f"  (Based on {voronoi.get('frames_analysed', 0)} frames)",
            "",
        ])

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
