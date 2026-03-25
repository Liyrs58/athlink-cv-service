"""
Three-layer AI brain for coaching report generation.

Layer 1 — EYES:       Gemini 1.5 Pro watches the video clip
Layer 2 — AUDITOR:    Claude Vision cross-checks tracking data against frames
Layer 3 — SYNTHESISER: Claude writes the final coaching report
"""

import os
import json
import base64
import logging
import urllib.request
import cv2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call_claude(messages: list, max_tokens: int = 1024, model: str = "claude-sonnet-4-6") -> dict:
    """Call Claude API and return the parsed JSON response body."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    data = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def _extract_key_frames(video_path: str, count: int = 8) -> list:
    """Extract `count` evenly-spaced frames from the video, return as base64 JPEG strings."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = [int(i * total / count) for i in range(count)]
    frames_b64 = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        # Resize to keep token cost manageable (max 800px wide)
        h, w = frame.shape[:2]
        if w > 800:
            scale = 800 / w
            frame = cv2.resize(frame, (800, int(h * scale)))
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frames_b64.append(base64.b64encode(buf).decode('utf-8'))

    cap.release()
    return frames_b64


def _parse_json_response(text: str) -> dict:
    """Try to parse JSON from a model response, stripping markdown fences if needed."""
    text = text.strip()
    if text.startswith("```"):
        # Strip ```json ... ``` fences
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw_text": text}


# ---------------------------------------------------------------------------
# LAYER 1 — EYES (Gemini 1.5 Pro watches the video)
# ---------------------------------------------------------------------------

def gemini_watch_clip(video_path: str, team_0_name: str, team_1_name: str) -> dict:
    """Upload video to Gemini and get tactical observations."""
    try:
        import google.generativeai as genai

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")

        genai.configure(api_key=api_key)

        # Upload video file
        logger.info("Uploading video to Gemini File API...")
        video_file = genai.upload_file(path=video_path)
        logger.info("Gemini upload complete: %s", video_file.name)

        # Wait for file to be processed
        import time
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise RuntimeError(f"Gemini file processing failed: {video_file.state.name}")

        # FIX 4: gemini-2.0-flash deprecated on RunPod — use 1.5 [2026-03-25]
        model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
        model = genai.GenerativeModel(model_name)

        prompt = f"""You are an elite football analyst with 20 years of experience coaching from grassroots Sunday league to professional academies. You understand every formation, every pressing system, every tactical concept in modern football.

A standard football pitch is 90-110 metres long and 45-90 metres wide.

You are watching a clip featuring {team_0_name} (first team) vs {team_1_name} (second team).

Analyse EVERYTHING you can observe. Be specific. Use exact timestamps if visible on screen. Name positions by role (centre-back, holding midfielder, right winger etc). Reference specific moments.

Return a JSON object with ALL of these fields populated:

{{
  "team_0": {{
    "name": "{team_0_name}",
    "formation_observed": "e.g. 4-3-3, 4-4-2, 3-5-2 — state what you actually saw, not a guess",
    "formation_notes": "how the shape shifted between phases of play",
    "defensive_shape": "how they organised without the ball — high block, mid block, low block, man-marking, zonal",
    "defensive_line_height": "deep, medium, or high — with specific observation",
    "defensive_compactness": "how tight horizontally and vertically when defending",
    "backline_coordination": "do the defenders move together or individually",
    "pressing_triggers": "exactly what triggers their press — back pass, goalkeeper ball, specific zones",
    "press_intensity": "high, medium, low — with specific observation",
    "counter_press": "do they counter-press immediately on losing the ball or drop off",
    "transition_attack_to_defence": "how fast and organised when losing the ball",
    "transition_defence_to_attack": "how fast and direct when winning the ball",
    "width_in_possession": "how wide do they stretch when attacking",
    "width_out_of_possession": "how compact horizontally when defending",
    "build_up_style": "short passing from back, direct long balls, mixed",
    "build_up_from_goalkeeper": "short distribution, long kicks, mixed — under pressure or composed",
    "progression_method": "how do they move the ball forward — through the lines, wide and cross, direct",
    "attacking_patterns": "recurring patterns you see — overlaps, combinations, set plays",
    "overloads_created": "where on the pitch are they creating numerical advantages",
    "ball_loss_zones": "where exactly are balls being lost and why — technical error, pressed, poor decision",
    "second_ball_tendency": "do they win or lose the second ball consistently",
    "set_piece_shape_attacking": "how they set up for corners and free kicks going forward",
    "set_piece_shape_defending": "how they defend corners and free kicks — zonal, man, mixed",
    "pressing_coordination": "is pressing coordinated between lines or individual",
    "channel_discipline": "do wide players track back into their defensive channels",
    "ball_near_vs_ball_far": "what are ball-far players doing — compacting, standing, making runs",
    "tactical_vulnerability": "the single most exploitable weakness you saw",
    "tactical_strength": "the single most effective thing they do",
    "one_thing_to_fix": "most important coaching point with a specific drill suggestion"
  }},
  "team_1": {{
    "name": "{team_1_name}",
    "formation_observed": "e.g. 4-3-3, 4-4-2, 3-5-2",
    "formation_notes": "how the shape shifted between phases",
    "defensive_shape": "high block, mid block, low block, man-marking, zonal",
    "defensive_line_height": "deep, medium, or high",
    "defensive_compactness": "tight or stretched horizontally and vertically",
    "backline_coordination": "move together or individually",
    "pressing_triggers": "what triggers their press",
    "press_intensity": "high, medium, low",
    "counter_press": "immediate or drop off",
    "transition_attack_to_defence": "speed and organisation losing the ball",
    "transition_defence_to_attack": "speed and directness winning the ball",
    "width_in_possession": "how wide when attacking",
    "width_out_of_possession": "how compact when defending",
    "build_up_style": "short, direct, mixed",
    "build_up_from_goalkeeper": "distribution pattern",
    "progression_method": "how they move the ball forward",
    "attacking_patterns": "recurring patterns",
    "overloads_created": "where they create advantages",
    "ball_loss_zones": "where and why they lose the ball",
    "second_ball_tendency": "win or lose second balls",
    "set_piece_shape_attacking": "corner and free kick setup",
    "set_piece_shape_defending": "how they defend set pieces",
    "pressing_coordination": "coordinated or individual pressing",
    "channel_discipline": "wide players tracking back",
    "ball_near_vs_ball_far": "ball-far player activity",
    "tactical_vulnerability": "most exploitable weakness",
    "tactical_strength": "most effective strength",
    "one_thing_to_fix": "most important coaching point with drill"
  }},
  "match_context": {{
    "dominant_team": "which team controlled the clip and why",
    "momentum_shifts": "did momentum shift and when",
    "physical_intensity": "high, medium, low intensity match",
    "referee_context": "any notable decisions or stoppages observed",
    "weather_pitch_conditions": "any observable conditions affecting play"
  }},
  "individual_observations": [
    {{
      "team": "team_0 or team_1",
      "position": "e.g. centre-back, striker, left midfielder",
      "shirt_colour": "colour description",
      "observation": "specific tactical or technical observation",
      "positive_or_negative": "positive, negative, or mixed",
      "coaching_point": "specific actionable feedback for this player"
    }}
  ],
  "key_moments": [
    {{
      "timestamp": "approximate time if visible",
      "description": "what happened",
      "significance": "why this moment matters tactically",
      "team_affected": "team_0, team_1, or both"
    }}
  ]
}}

Populate individual_observations with observations for as many players as you can identify — aim for at least 6-8 individual observations across both teams.
Populate key_moments with up to 5 significant moments.
Return ONLY valid JSON. No markdown. No explanation. No apologies if something is hard to see — just describe what you can observe."""

        response = model.generate_content([video_file, prompt])
        result = _parse_json_response(response.text)
        logger.info("Gemini analysis complete")
        return result

    except Exception as e:
        logger.warning("Gemini layer failed: %s", e)
        return {"error": f"Gemini unavailable: {e}", "one_thing_to_fix": "Analysis unavailable"}


# ---------------------------------------------------------------------------
# LAYER 1b — EYES (Gemini event counting and psychology)
# ---------------------------------------------------------------------------

def gemini_count_events(video_path: str, team_0_name: str, team_1_name: str) -> dict:
    """Second Gemini pass — count specific events and identify patterns."""
    try:
        import google.generativeai as genai
        import time

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")

        genai.configure(api_key=api_key)
        video_file = genai.upload_file(path=video_path)

        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise RuntimeError("Gemini file processing failed")

        # FIX 4: gemini-2.0-flash deprecated on RunPod — use 1.5 [2026-03-25]
        model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
        model = genai.GenerativeModel(model_name)

        prompt = f"""You are a professional football data analyst. Watch this clip and COUNT specific events as precisely as possible.

Teams: {team_0_name} vs {team_1_name}

Count and observe the following for EACH team separately:

EVENTS TO COUNT:
- Passes attempted and completed (estimate)
- Shots on target vs off target
- Corners won
- Free kicks won
- Throw-ins
- Headers contested
- Tackles attempted
- Times ball went out of play
- Times goalkeeper had the ball
- Number of times each team entered the opposition half

PLAYER BEHAVIOUR:
- Which players are organising and communicating (pointing, directing teammates)
- Which players show positive body language after mistakes vs negative
- Which players are pressing with intensity vs jogging
- Any player who makes the same mistake more than once
- Any player who shows leadership by position or communication
- Any player who looks fatigued or is moving asymmetrically (possible injury risk)

TACTICAL EVENTS:
- How many times did the defensive line break (opponent got in behind)
- How many times was the press beaten with one pass
- How many successful counter-attacks
- How many times did wide players fail to track back
- Any visible shirt numbers on players

PHYSICAL CONDITIONS:
- Pitch quality (good, average, poor)
- Any visible weather effects
- Time of day (day/floodlit/dusk)

Return JSON:
{{
  "team_0_events": {{
    "passes_estimated": number,
    "shots_on_target": number,
    "shots_off_target": number,
    "corners": number,
    "free_kicks": number,
    "headers_won": number,
    "tackles_attempted": number,
    "defensive_line_breaks_conceded": number,
    "press_beaten_by_single_pass": number,
    "successful_counter_attacks": number,
    "wide_players_failed_to_track_back": number,
    "times_entered_opposition_half": number
  }},
  "team_1_events": {{
    "passes_estimated": number,
    "shots_on_target": number,
    "shots_off_target": number,
    "corners": number,
    "free_kicks": number,
    "headers_won": number,
    "tackles_attempted": number,
    "defensive_line_breaks_conceded": number,
    "press_beaten_by_single_pass": number,
    "successful_counter_attacks": number,
    "wide_players_failed_to_track_back": number,
    "times_entered_opposition_half": number
  }},
  "player_psychology": [
    {{
      "team": "team_0 or team_1",
      "description": "shirt colour and position",
      "shirt_number": "if visible",
      "behaviour": "specific observed behaviour",
      "interpretation": "what this suggests about their mental state or role",
      "coaching_note": "specific coaching point"
    }}
  ],
  "leadership_identified": [
    {{
      "team": "team_0 or team_1",
      "description": "shirt colour and position",
      "evidence": "specific behaviour that indicates leadership"
    }}
  ],
  "injury_risk_flags": [
    {{
      "team": "team_0 or team_1",
      "description": "shirt colour and position",
      "observation": "specific asymmetric movement or fatigue sign"
    }}
  ],
  "shirt_numbers_visible": ["list any visible shirt numbers with team and position"],
  "physical_conditions": {{
    "pitch_quality": "good/average/poor",
    "weather_visible": "description or none visible",
    "lighting": "daylight/floodlit/dusk"
  }},
  "repeated_mistakes": [
    {{
      "team": "team_0 or team_1",
      "player_description": "position and colour",
      "mistake": "specific mistake observed more than once",
      "count": number,
      "coaching_point": "how to fix this"
    }}
  ]
}}

Return ONLY valid JSON. No markdown. No explanation."""

        response = model.generate_content([video_file, prompt])
        result = _parse_json_response(response.text)
        logger.info("Gemini event counting complete")
        return result

    except Exception as e:
        logger.warning("Gemini event counting failed: %s", e)
        return {}


# ---------------------------------------------------------------------------
# LAYER 2 — AUDITOR (Claude Vision cross-checks data against frames)
# ---------------------------------------------------------------------------

def claude_audit_tracking(frames: list, tracking_summary: dict) -> dict:
    """Send key frames + tracking summary to Claude Vision for cross-checking."""
    try:
        if not frames:
            raise ValueError("No frames provided for audit")

        # Build content blocks: frames as images + tracking data as text
        content = []
        for i, b64_frame in enumerate(frames[:8]):
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64_frame,
                },
            })

        summary_text = json.dumps(tracking_summary, indent=2, default=str)

        content.append({
            "type": "text",
            "text": f"""You are a senior football data analyst auditing AI tracking data against actual match footage. Your job is to verify what the computer tracked matches what a human expert can see.

Tracking data claims:
{summary_text}

Standard football pitch dimensions: 90-110m long, 45-90m wide.
A standard 11v11 match has 22 outfield players plus 2 goalkeepers.
A grassroots or semi-pro match may have fewer players visible depending on camera angle.

Examine each frame carefully and answer:

1. PLAYER COUNT: How many players can you actually count in each frame? Does this match the tracking claim?
2. TEAM IDENTIFICATION: Can you identify two distinct teams by kit colour? Do the team colour labels in tracking data match what you see?
3. BALL VISIBILITY: Is the ball visible in these frames? Does it appear to be tracked correctly?
4. DISTANCE PLAUSIBILITY: A player cannot cover more than ~150m in 40 seconds at match intensity. Flag any distance claims that are physically impossible.
5. SPEED PLAUSIBILITY: Maximum human sprint speed is ~36 km/h for elite athletes, ~28 km/h for amateur. Flag speeds above 32 km/h for grassroots footage.
6. TRACKING GAPS: Are there players on the pitch who appear to have no tracking box? Estimate how many are untracked.
7. PITCH COVERAGE: Does the camera show the full pitch, half the pitch, or a close-up? This affects how reliable the metrics are.

Return JSON:
{{
  "player_count_valid": true or false,
  "players_tracked": number you can count across frames,
  "players_estimated_total": your best estimate of total players on pitch,
  "tracking_coverage_percent": estimated percentage of players being tracked,
  "team_colours_valid": true or false,
  "team_0_colour_confirmed": "colour you actually see",
  "team_1_colour_confirmed": "colour you actually see",
  "ball_tracking_valid": true or false,
  "ball_visible_in_frames": true or false,
  "camera_coverage": "full_pitch, half_pitch, or close_up",
  "unreliable_metrics": ["list each metric that should not be trusted with reason"],
  "physically_impossible_values": ["list any values that defy physics with explanation"],
  "data_quality_issues": ["list any other concerns about the tracking data"],
  "confidence": "high, medium, or low",
  "confidence_reason": "one sentence explaining the confidence level"
}}

Return ONLY valid JSON. No markdown.""",
        })

        result = _call_claude(
            messages=[{"role": "user", "content": content}],
            max_tokens=1024,
            model="claude-sonnet-4-6",
        )
        text = result["content"][0]["text"]
        audit = _parse_json_response(text)
        logger.info("Claude audit complete — confidence: %s", audit.get("confidence", "unknown"))
        return audit

    except Exception as e:
        logger.warning("Claude audit layer failed: %s", e)
        return {
            "confidence": "low",
            "unreliable_metrics": [],
            "player_count_valid": False,
        }


# ---------------------------------------------------------------------------
# LAYER 3 — SYNTHESISER (Claude writes the coaching report)
# ---------------------------------------------------------------------------

def synthesise_report(
    gemini_observations: dict,
    audit_result: dict,
    tracking_data: dict,
    team_0_name: str,
    team_1_name: str,
    gemini_events: dict = None,
) -> str:
    """Combine all layers into a final coaching report."""
    try:
        # Filter tracking data — remove metrics flagged as unreliable
        unreliable = set(audit_result.get("unreliable_metrics", []))
        filtered_tracking_data = {
            k: v for k, v in tracking_data.items()
            if k not in unreliable
        }

        # Build event stats block from gemini_events
        if gemini_events and not gemini_events.get("raw_text"):
            t0e = gemini_events.get("team_0_events", {})
            t1e = gemini_events.get("team_1_events", {})
            event_stats_block = f"""{team_0_name}:
- Passes estimated: {t0e.get('passes_estimated', '?')}
- Shots on target: {t0e.get('shots_on_target', '?')} | Off target: {t0e.get('shots_off_target', '?')}
- Corners: {t0e.get('corners', '?')} | Free kicks: {t0e.get('free_kicks', '?')}
- Headers won: {t0e.get('headers_won', '?')} | Tackles: {t0e.get('tackles_attempted', '?')}
- Defensive line breaks conceded: {t0e.get('defensive_line_breaks_conceded', '?')}
- Press beaten by single pass: {t0e.get('press_beaten_by_single_pass', '?')}
- Counter-attacks: {t0e.get('successful_counter_attacks', '?')}
- Wide players failed to track back: {t0e.get('wide_players_failed_to_track_back', '?')}
- Times entered opposition half: {t0e.get('times_entered_opposition_half', '?')}

{team_1_name}:
- Passes estimated: {t1e.get('passes_estimated', '?')}
- Shots on target: {t1e.get('shots_on_target', '?')} | Off target: {t1e.get('shots_off_target', '?')}
- Corners: {t1e.get('corners', '?')} | Free kicks: {t1e.get('free_kicks', '?')}
- Headers won: {t1e.get('headers_won', '?')} | Tackles: {t1e.get('tackles_attempted', '?')}
- Defensive line breaks conceded: {t1e.get('defensive_line_breaks_conceded', '?')}
- Press beaten by single pass: {t1e.get('press_beaten_by_single_pass', '?')}
- Counter-attacks: {t1e.get('successful_counter_attacks', '?')}
- Wide players failed to track back: {t1e.get('wide_players_failed_to_track_back', '?')}
- Times entered opposition half: {t1e.get('times_entered_opposition_half', '?')}

Conditions: Pitch {gemini_events.get('physical_conditions', {}).get('pitch_quality', '?')}, Lighting {gemini_events.get('physical_conditions', {}).get('lighting', '?')}
Shirt numbers visible: {', '.join(gemini_events.get('shirt_numbers_visible', [])) or 'None identified'}"""
        else:
            event_stats_block = "Event counting data unavailable for this clip."

        # Build psychology block
        psych_entries = gemini_events.get("player_psychology", []) if gemini_events else []
        leaders = gemini_events.get("leadership_identified", []) if gemini_events else []
        mistakes = gemini_events.get("repeated_mistakes", []) if gemini_events else []
        if psych_entries or leaders or mistakes:
            psych_lines = []
            for p in psych_entries:
                psych_lines.append(f"- [{p.get('team', '?')}] {p.get('description', '?')}{' (#' + str(p['shirt_number']) + ')' if p.get('shirt_number') else ''}: {p.get('behaviour', '?')} — {p.get('interpretation', '?')} (Coaching: {p.get('coaching_note', '?')})")
            if leaders:
                psych_lines.append("\nLeadership identified:")
                for l in leaders:
                    psych_lines.append(f"- [{l.get('team', '?')}] {l.get('description', '?')}: {l.get('evidence', '?')}")
            if mistakes:
                psych_lines.append("\nRepeated mistakes:")
                for m in mistakes:
                    psych_lines.append(f"- [{m.get('team', '?')}] {m.get('player_description', '?')}: {m.get('mistake', '?')} (x{m.get('count', '?')}) — Fix: {m.get('coaching_point', '?')}")
            psychology_block = "\n".join(psych_lines)
        else:
            psychology_block = "Player psychology data unavailable for this clip."

        # Build injury block
        injury_flags = gemini_events.get("injury_risk_flags", []) if gemini_events else []
        if injury_flags:
            injury_lines = []
            for f in injury_flags:
                injury_lines.append(f"- [{f.get('team', '?')}] {f.get('description', '?')}: {f.get('observation', '?')}")
            injury_block = "\n".join(injury_lines)
        else:
            injury_block = "No injury or fatigue risk flags identified in this clip."

        prompt = f"""You are the world's best football coaching analyst. Your job is to write a comprehensive match analysis report that gives grassroots and semi-professional coaches the same quality of insight that top professional clubs pay tens of thousands of pounds for.

You have three sources of information:
1. VISUAL ANALYSIS — what an AI actually watched in the video
2. DATA AUDIT — what a second AI verified about the tracking data reliability
3. TRACKING DATA — computer-measured statistics (only trust what the audit approved)

---

VISUAL ANALYSIS (what was observed watching the footage):
{json.dumps(gemini_observations, indent=2, default=str)}

---

DATA AUDIT RESULT:
Overall Confidence: {audit_result.get('confidence', 'low')}
Reason: {audit_result.get('confidence_reason', 'insufficient data')}
Metrics NOT to trust: {audit_result.get('unreliable_metrics', [])}
Camera coverage: {audit_result.get('camera_coverage', 'unknown')}
Players tracked: {audit_result.get('players_tracked', 'unknown')} of estimated {audit_result.get('players_estimated_total', 'unknown')}

---

TRACKING DATA (only use metrics NOT flagged as unreliable):
{json.dumps(filtered_tracking_data, indent=2, default=str)}

---

Write a comprehensive coaching report with the following structure. Use markdown headers. Be specific, direct, and actionable. Never be vague. Never use phrases like "it appears" or "it seems" — state what you observed as fact. Never invent statistics. If data is unreliable, say so and use visual observations only.

Target audience: coaches at ALL levels from Sunday league to semi-professional. Write so a Sunday league coach understands but a semi-pro coach finds it useful.

Pitch dimensions for reference: 90-110m long, 45-90m wide.

---

# MATCH ANALYSIS REPORT

## EXECUTIVE SUMMARY
3-4 sentences. What was the dominant story of this clip? Who controlled it and why? What is the single most important takeaway?

---

# {team_0_name} — TEAM ANALYSIS

## Formation & Shape
What formation did they use? How did it shift between phases? Was it effective?

## Attacking Play
How did they build up? What patterns did they use? Where did they create danger? What worked and what didn't?

## Defensive Organisation
How did they defend? Was the shape solid? Where were they vulnerable?

## Pressing & Transitions
How did they press? Were transitions quick? What were their triggers?

## Set Pieces
What did you observe about their set piece organisation (if any were visible)?

## {team_0_name} — KEY STRENGTHS
- 3 specific bullet points with evidence from the clip

## {team_0_name} — AREAS TO IMPROVE
- 3 specific bullet points with evidence from the clip

## {team_0_name} — THIS WEEK'S TRAINING FOCUS
One specific training drill or session focus with exact instructions. Be precise enough that a coach can run the session from this description.

---

# {team_1_name} — TEAM ANALYSIS

## Formation & Shape
What formation did they use? How did it shift? Was it effective?

## Attacking Play
Build up, patterns, danger creation.

## Defensive Organisation
Shape, vulnerabilities, strengths.

## Pressing & Transitions
Press quality, transition speed, triggers.

## Set Pieces
Any observable set piece patterns.

## {team_1_name} — KEY STRENGTHS
- 3 specific bullet points

## {team_1_name} — AREAS TO IMPROVE
- 3 specific bullet points

## {team_1_name} — THIS WEEK'S TRAINING FOCUS
One specific training drill with exact instructions.

---

# INDIVIDUAL PLAYER OBSERVATIONS

For EVERY player observation from the visual analysis, write a player card:

**[Team] [Position] ([Kit Colour])**
- What they did well
- What needs improvement
- Specific coaching point for this player

If no individual observations are available, write: "Individual player analysis requires clearer camera footage or closer angles."

---

# HEAD-TO-HEAD TACTICAL COMPARISON

| Aspect | {team_0_name} | {team_1_name} |
|--------|--------------|--------------|
| Formation | ... | ... |
| Pressing Style | ... | ... |
| Build-up Method | ... | ... |
| Defensive Block | ... | ... |
| Transition Speed | ... | ... |
| Set Piece Threat | ... | ... |

---

# DATA CONFIDENCE REPORT

State clearly which metrics are reliable and which are not. Explain why in plain English that a coach will understand.

**Reliable metrics:** [list]
**Unreliable metrics:** [list with reason]
**Overall data confidence:** {audit_result.get('confidence', 'low').upper()}

---

# EVENT STATISTICS
{event_stats_block}

---

# PSYCHOLOGICAL & LEADERSHIP ANALYSIS
Based on player behaviour observed in the footage:

{psychology_block}

---

# INJURY & FATIGUE RISK FLAGS
{injury_block}

---

# FULL WEEK TRAINING PLAN
Based on everything observed, write a complete 3-session training week plan:

**Session 1 (Tuesday) — Technical Fix**
Focus on the primary technical weakness observed. Specific drills with duration, player numbers, and coaching points.

**Session 2 (Thursday) — Tactical Organisation**
Focus on the primary tactical weakness. Specific exercises that replicate the match situations where the team struggled.

**Session 3 (Friday/Pre-match) — Shape & Confidence**
Light session focused on shape reinforcement and building confidence. Specific exercises.

---

# OPPOSITION SCOUTING REPORT
If this is opponent footage, provide:
- Their tactical DNA in 3 bullet points
- How to exploit their defensive vulnerability
- How to neutralise their attacking strength
- Specific tactical instruction for your team before facing this opponent

---

# PARENT & PLAYER SUMMARY (Youth Football)
A shorter, positive, development-focused version of the key points suitable for sharing with young players and their parents. Focus on what they did well and specific improvements for next time. No jargon.

---

Rules you must follow:
- Never mention distances in metres if distances are flagged as unreliable
- Never mention speeds if speeds are flagged as unreliable
- Every coaching point must be specific enough to action — no generic advice
- If the clip is very short (under 30 seconds), note this limits the analysis depth
- Write in confident, direct language — coaches need clarity not academic writing
- Total report should be comprehensive — do not truncate or summarise prematurely
- The gap between professional and grassroots football is access to information, not intelligence — write as if this coach deserves the same insights as a Premier League analyst"""

        result = _call_claude(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8192,
            model="claude-sonnet-4-6",
        )
        report = result["content"][0]["text"]
        logger.info("Synthesis complete — %d chars", len(report))
        return report

    except Exception as e:
        logger.warning("Synthesis layer failed: %s", e)
        return _fallback_report(tracking_data, team_0_name, team_1_name)


# ---------------------------------------------------------------------------
# Fallback report — no API needed
# ---------------------------------------------------------------------------

def _fallback_report(tracking_data: dict, team_0_name: str, team_1_name: str) -> str:
    """Generate a basic report from tracking data alone when APIs fail."""
    phys = tracking_data.get("physical", {})
    shape = tracking_data.get("shape", {})
    possession = tracking_data.get("possession", {})

    lines = ["WHAT WE SAW", "AI analysis was unavailable for this clip.", ""]

    lines.append("WHAT THE DATA SHOWS")
    players = phys.get("players_analysed", "?")
    sprints = phys.get("total_sprints", "?")
    max_spd = phys.get("max_speed_kmh", "?")
    lines.append(f"- {players} players tracked, {sprints} total sprints")
    lines.append(f"- Top speed recorded: {max_spd} km/h")
    if possession.get("team_0_pct"):
        lines.append(f"- Possession: {team_0_name} {possession['team_0_pct']:.0f}% vs {team_1_name} {possession.get('team_1_pct', 0):.0f}%")
    lines.append("")

    lines.append("ONE THING TO FIX THIS WEEK")
    if isinstance(sprints, (int, float)) and sprints < 10:
        lines.append("Sprint count is low. Run repeated sprint drills (6x30m with 45s rest) to build match intensity.")
    elif shape.get("data_quality") == "ok" and isinstance(shape.get("avg_width_metres"), (int, float)) and shape["avg_width_metres"] < 35:
        lines.append(f"Team shape is narrow ({shape['avg_width_metres']:.0f}m width). Use wide channel games to encourage width in possession.")
    else:
        lines.append("Focus on maintaining team compactness during transitions. Use 5v5 transition games with counter-attack triggers.")
    lines.append("")

    lines.append("PLAYER TO WATCH")
    top_runner = phys.get("top_runner_id")
    if top_runner:
        lines.append(f"Player #{top_runner} — covered the most distance. Check if their effort is sustainable or if they're chasing the ball.")
    else:
        lines.append("Unable to identify standout players from available data.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def generate_brain_report(job_id: str, video_path: str, result: dict) -> str:
    """
    Run the three-layer AI brain and return a coaching report.

    Layer 1: Gemini watches the video
    Layer 2: Claude audits tracking data against key frames
    Layer 3: Claude synthesises the final coaching report
    """
    try:
        # Extract team names
        team_sep = result.get("team_separation", {})
        team_0_name = team_sep.get("team_0_colour_name", "Team A")
        team_1_name = team_sep.get("team_1_colour_name", "Team B")

        # LAYER 1a — Gemini watches the clip
        logger.info("[Brain L1a] Gemini watching clip for job %s", job_id)
        gemini_obs = gemini_watch_clip(video_path, team_0_name, team_1_name)

        # LAYER 1b — Gemini event counting and psychology
        logger.info("[Brain L1b] Gemini counting events for job %s", job_id)
        gemini_events = gemini_count_events(video_path, team_0_name, team_1_name)

        # Extract 8 key frames for audit
        logger.info("[Brain L2] Extracting key frames for audit")
        frames = _extract_key_frames(video_path, count=8)

        # Build tracking summary for the auditor
        phys = result.get("physical", {})
        tracking_summary = {
            "player_count": phys.get("players_analysed", 0),
            "team_0_count": result.get("tracking", {}).get("team_0_count", 0),
            "team_1_count": result.get("tracking", {}).get("team_1_count", 0),
            "team_0_name": team_0_name,
            "team_1_name": team_1_name,
            "total_sprints": phys.get("total_sprints", 0),
            "max_speed_kmh": phys.get("max_speed_kmh", 0),
            "ball_tracking_rate": result.get("ball", {}).get("tracking_rate", 0),
            "possession_team_0": result.get("possession", {}).get("team_0_pct", 0),
            "possession_team_1": result.get("possession", {}).get("team_1_pct", 0),
        }

        # LAYER 2 — Claude audits tracking vs frames
        logger.info("[Brain L2] Claude auditing tracking data for job %s", job_id)
        audit = claude_audit_tracking(frames, tracking_summary)

        # LAYER 3 — Claude synthesises the report
        logger.info("[Brain L3] Synthesising coaching report for job %s", job_id)
        report = synthesise_report(gemini_obs, audit, result, team_0_name, team_1_name, gemini_events=gemini_events)

        logger.info("[Brain] Report complete for job %s (%d chars)", job_id, len(report))
        return report

    except Exception as e:
        logger.error("[Brain] Failed for job %s: %s", job_id, e)
        return _fallback_report(result, result.get("team_separation", {}).get("team_0_colour_name", "Team A"), result.get("team_separation", {}).get("team_1_colour_name", "Team B"))
