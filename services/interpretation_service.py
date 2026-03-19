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


def _compute_phase_widths(events, tracks, calibration):
    """
    For each situation event, compute average lateral spread (width) per team
    using trajectory entries that fall within that time window.

    Returns a list of dicts:
      { "label": "open play", "start": 0, "end": 22, "team_0_width": 36.2, "team_1_width": 21.4 }
    """
    ppm = 1.0
    if calibration and calibration.get("pixels_per_metre"):
        ppm = max(calibration["pixels_per_metre"], 0.1)

    # fps assumed 25, frame_stride 2 — convert seconds to approximate frame range
    fps = 25.0
    stride = 2

    phase_shapes = []
    for e in events:
        start_s = safe_float(e.get("start_time", 0)) or 0
        end_s = safe_float(e.get("end_time", 0)) or 0
        label = "open play" if e.get("situation") == "OPEN_PLAY" else "stoppage"

        start_frame = int(start_s * fps / stride) * stride
        end_frame = int(end_s * fps / stride) * stride

        team_xs = {"team_0": [], "team_1": []}
        for t in tracks:
            team_id = t.get("teamId", -1)
            if team_id not in (0, 1):
                continue
            key = f"team_{team_id}"
            for entry in t.get("trajectory", []):
                fi = entry.get("frameIndex", 0)
                if start_frame <= fi <= end_frame:
                    bbox = entry.get("bbox")
                    if bbox and len(bbox) >= 4:
                        cx = (bbox[0] + bbox[2]) / 2.0 / ppm
                        team_xs[key].append(cx)

        row = {"label": label, "start": start_s, "end": end_s,
               "team_0_width": None, "team_1_width": None}
        for key in ("team_0", "team_1"):
            xs = team_xs[key]
            if len(xs) >= 3:
                row[f"{key}_width"] = round(max(xs) - min(xs), 1)
        phase_shapes.append(row)

    return phase_shapes


def build_rich_context(events, tracks, vel_summary, shape_summary, velocities, job_id,
                       team_separation=None, data_confidence=None, calibration=None,
                       brain_summary=None):
    """Build plain-English coaching context for the Claude prompt."""
    team_sep = team_separation or {}

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
        situation_timeline.append(f"{start:.0f}s\u2013{end:.0f}s: {label}")

    # Per-phase shape data
    phase_shapes = _compute_phase_widths(events, tracks, calibration)

    # Check whether brain trusts sprint data
    sprints_trusted = True
    if brain_summary:
        questioned = brain_summary.get("metrics_to_question", [])
        if "sprint_counts" in questioned:
            sprints_trusted = False

    # Build per-player numbers, skip low confidence and inactive players
    # Number players within each team by distance (Player 1 = most distance)
    team_player_velocities = {"team_0": [], "team_1": [], "unassigned": []}
    track_team_map = {}
    track_position_map = {}
    for t in tracks:
        tid = t.get("trackId")
        if tid is not None:
            track_team_map[tid] = t.get("teamId", -1)
            # Infer position from trajectory
            traj = t.get("trajectory", [])
            positions_list = []
            vis_frac = calibration.get("visible_fraction", 0.55) if calibration else 0.55
            for entry in traj:
                bbox = entry.get("bbox", [])
                if len(bbox) >= 4:
                    px = (bbox[0] + bbox[2]) / 2.0
                    positions_list.append({"pixel_x": px, "visible_fraction": vis_frac})
            # Simplified position inference
            if positions_list:
                avg_x = sum((p["pixel_x"] / 1920.0) * (105.0 * p["visible_fraction"]) for p in positions_list) / len(positions_list)
                if avg_x < 35:
                    track_position_map[tid] = "Defender"
                elif avg_x < 70:
                    track_position_map[tid] = "Midfielder"
                else:
                    track_position_map[tid] = "Forward"
            else:
                track_position_map[tid] = "Unknown"

    for v in velocities:
        track_id = v['track_id']
        trk = next((t for t in tracks if t.get("trackId") == track_id), None)
        if not trk or trk.get("confirmed_detections", 0) < 5:
            continue
        tc = score_track_confidence(trk)
        conf_level = tc["level"]
        if conf_level == "low":
            continue

        team_id = track_team_map.get(track_id, -1)
        team_key = "team_0" if team_id == 0 else ("team_1" if team_id == 1 else "unassigned")

        distance = safe_float(v.get('distance_metres', 0)) or 0
        max_speed_ms = safe_float(v.get('max_speed_ms', 0))
        sprint_count = v.get('sprint_count', 0)
        max_speed_kmh = round(max_speed_ms * 3.6, 1) if max_speed_ms else 0

        pct = {"high": 0.12, "medium": 0.25}.get(conf_level, 0.25)
        dist_lo = round(distance * (1 - pct), 0)
        dist_hi = round(distance * (1 + pct), 0)

        position = track_position_map.get(track_id, "Unknown")

        team_player_velocities[team_key].append({
            "track_id": track_id,
            "conf_level": conf_level,
            "distance": distance,
            "dist_range": f"{dist_lo:.0f}\u2013{dist_hi:.0f}m",
            "max_speed_kmh": max_speed_kmh,
            "sprint_count": sprint_count,
            "position": position,
        })

    # Sort by distance descending, assign sequential player numbers per team
    player_lines = []
    for team_key, team_name in [("team_0", t0_name), ("team_1", t1_name), ("unassigned", "Unassigned")]:
        sorted_players = sorted(team_player_velocities[team_key], key=lambda p: p["distance"], reverse=True)
        for idx, p in enumerate(sorted_players, 1):
            position = p.get("position", "Unknown")
            player_label = f"{team_name} {position}"
            conf = p["conf_level"]
            if conf == "high":
                if sprints_trusted:
                    player_lines.append(
                        f"  {player_label} [HIGH confidence]: "
                        f"distance {p['dist_range']} | "
                        f"{p['max_speed_kmh']} km/h peak (\u00b110%) | "
                        f"{p['sprint_count']} sprint(s)"
                    )
                else:
                    player_lines.append(
                        f"  {player_label} [HIGH confidence]: "
                        f"distance {p['dist_range']} | "
                        f"{p['max_speed_kmh']} km/h peak (\u00b110%)"
                    )
            else:
                speed_lo = round(p['max_speed_kmh'] * 0.8, 1)
                speed_hi = round(p['max_speed_kmh'] * 1.2, 1)
                if sprints_trusted:
                    player_lines.append(
                        f"  {player_label} [MEDIUM confidence]: "
                        f"roughly {p['dist_range']} | "
                        f"approximately {speed_lo}\u2013{speed_hi} km/h peak | "
                        f"approximately {max(0, p['sprint_count'] - 2)}\u2013{p['sprint_count'] + 2} sprint(s)"
                    )
                else:
                    player_lines.append(
                        f"  {player_label} [MEDIUM confidence]: "
                        f"roughly {p['dist_range']} | "
                        f"approximately {speed_lo}\u2013{speed_hi} km/h peak"
                    )

    duration_str = f"{total_duration:.0f}" if isinstance(total_duration, (int, float)) else "unknown"

    lines = [
        "You are writing a post-match report for a grassroots or semi-professional football coach.",
        "",
        "The coach reads this on their phone after training or a match.",
        "They are not academics. They want to know what happened, what the numbers show, and what to fix.",
        "",
        "BANNED WORDS — never use these:",
        "entropy, Voronoi, Shannon, homography, spectral, biomechanical,",
        "calibration, pixel, heuristic, algorithm, interpolated,",
        "belief state, neuro-symbolic, anomaly, parametric.",
        "Replace every one with plain English.",
        "",
        "BANNED TACTICS — never write these unless the data proves it:",
        "'high press', 'low block', 'gegenpressing', 'progressive carries',",
        "'half-space', 'pressing triggers', 'counterpressing'.",
        "If you cannot point to a specific number that proves it, do not write it.",
        "",
        "RULES:",
        "",
        "Rule 1 \u2014 Every tactical claim needs a number.",
        "BAD: 'The team pressed well.'",
        "GOOD: 'During open play Blue spread to roughly 36m wide.'",
        "",
        "Rule 2 \u2014 Shape must be described per phase, not as an average.",
        "BAD: 'Blue averaged 36m width.'",
        "GOOD: 'During open play (0\u201322s) Blue spread to roughly 36m wide.",
        "When play stopped (22\u201333s) the shape dropped to around 20m.'",
        "",
        "Rule 3 \u2014 Never label a formation.",
        "Do not write 4-4-2, 4-3-3, or any formation number.",
        "Describe what you see: 'Blue had more players in defence than attack'",
        "not 'Blue played a 4-5-1.'",
        "",
        "Rule 4 \u2014 Mark approximations clearly.",
        "For any MEDIUM confidence metric use: 'roughly', 'around', 'approximately', 'suggests'.",
        "For any metric the brain questions \u2014 skip it entirely. Do not mention it at all.",
        "",
        "Rule 5 \u2014 Short sentences. Maximum 20 words each.",
        "Write like you are texting a coach, not writing a thesis.",
        "",
        "Rule 6 \u2014 No generic praise.",
        "Do not write 'good work rate', 'great intensity', 'solid defensive shape'",
        "unless a specific number supports it.",
        "",
        "REPORT STRUCTURE \u2014 use exactly these sections:",
        "",
        "## WHAT HAPPENED",
        "3\u20134 sentences maximum.",
        "Describe match flow using only the situation timeline.",
        "Reference actual timestamps. Do not invent what caused stoppages.",
        "",
        "## TEAM SHAPE",
        "2\u20133 sentences.",
        "Describe shape during open play AND during dead ball separately \u2014 use the per-phase data provided.",
        "Give actual width numbers. Do not label formations.",
        "If one team was significantly wider or narrower say so and give the number.",
        "",
        "## WHO COVERED THE MOST GROUND",
        "List top 3\u20134 players by distance only.",
        "Use player numbers (Blue Player 1, Red Player 1 etc).",
        "Use distance range not a single number.",
        "One sentence per player maximum.",
        "Only include HIGH or MEDIUM confidence players. Skip LOW confidence entirely.",
        "",
        "## SPRINT EFFORTS",
        "ONLY include this section if sprints are listed as trusted above.",
        "If sprints are questioned \u2014 delete this section entirely. Do not mention sprints anywhere.",
        "If included: which players sprinted, how many times, peak speed with margin.",
        "",
        "## PITCH CONTROL",
        "Only include if space control data is provided below.",
        "Do not use the word Voronoi.",
        "Plain English only. One or two sentences only.",
        "",
        "## ONE THING TO WORK ON",
        "Most important section. Coaches remember this one.",
        "Pick ONE specific thing the data suggests needs work.",
        "Must be based directly on a number in the data.",
        "Must be actionable \u2014 something to do in training.",
        "BAD: 'Work on pressing higher up the pitch.'",
        "GOOD: 'When play stopped (22\u201333 seconds), Blue's shape dropped to roughly 20m wide.",
        "Players were very bunched together.",
        "Work on maintaining width during dead balls so you are harder to press at restarts.'",
        "Maximum 4 sentences.",
        "Never write generic football advice.",
        "If the data does not clearly suggest something specific, write:",
        "'The clip is too short to identify a clear priority \u2014 analyse a longer sequence.'",
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
        lines.append(f"  {t0_name} ({team_sep.get('team_0_players', 0)} players detected)")
        lines.append(f"  {t1_name} ({team_sep.get('team_1_players', 0)} players detected)")
        lines.append("")

    lines.append("TEAM SHAPE PER PHASE (use these numbers directly, describe each phase separately):")
    if phase_shapes:
        for ps in phase_shapes:
            t0w = f"{ps['team_0_width']:.0f}m wide" if ps["team_0_width"] is not None else "no data"
            t1w = f"{ps['team_1_width']:.0f}m wide" if ps["team_1_width"] is not None else "no data"
            lines.append(
                f"  {ps['label'].capitalize()} ({ps['start']:.0f}s\u2013{ps['end']:.0f}s): "
                f"{t0_name} {t0w}, {t1_name} {t1w}"
            )
    else:
        lines.append("  (Shape data unavailable)")
    lines.append("")

    lines.append("PLAYER DISTANCES (use these for 'who covered the most ground'):")
    lines.append("Players listed by position and distance covered (top = most distance).")
    if not sprints_trusted:
        lines.append("NOTE: Sprint data is unreliable for this clip. Do NOT mention sprints in the report.")
    if player_lines:
        for pl in player_lines:
            lines.append(pl)
    else:
        lines.append("  (No players with sufficient tracking data)")
    lines.append("")

    return "\n".join(lines)


def interpret_events(events, tracks, job_id, velocity_summary=None, shape_summary=None,
                     velocities=None, memory=None, team_separation=None, data_confidence=None,
                     brain_summary=None, voronoi=None, entropy=None, calibration=None):
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
        calibration=calibration,
        brain_summary=brain_summary,
    )

    # Prefix with Observer Brain data quality summary
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
            f"  {t0_name} controlled roughly {voronoi.get('team_0_control_pct', 0):.0f}% of available pitch space",
            f"  {t1_name} controlled roughly {voronoi.get('team_1_control_pct', 0):.0f}% of available pitch space",
            f"  {dom_team} had the advantage",
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
