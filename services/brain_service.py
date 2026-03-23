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

        model = genai.GenerativeModel("gemini-1.5-pro")

        prompt = f"""You are a professional football analyst watching a match clip.
The team in {team_0_name} kit vs {team_1_name} kit.

Watch this clip carefully and return a JSON object with exactly these fields:
{{
  "defensive_shape": "describe how the out-of-possession team organises",
  "pressing_pattern": "when and where does the team press",
  "ball_loss_zones": "where on the pitch are balls being lost and why",
  "key_moments": ["list up to 3 specific moments you saw with timestamp if visible"],
  "individual_observations": ["list up to 3 observations about specific players"],
  "tactical_vulnerability": "one specific area where the team is exposed",
  "one_thing_to_fix": "the single most important thing to work on"
}}

Return ONLY valid JSON. No markdown. No explanation."""

        response = model.generate_content([video_file, prompt])
        result = _parse_json_response(response.text)
        logger.info("Gemini analysis complete")
        return result

    except Exception as e:
        logger.warning("Gemini layer failed: %s", e)
        return {"error": f"Gemini unavailable: {e}", "one_thing_to_fix": "Analysis unavailable"}


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
            "text": f"""You are auditing football tracking data against actual match footage.

Tracking data claims:
{summary_text}

Look at these frames and answer:
1. Does the player count match what you see? (yes/no + actual count)
2. Do the team colours match the tracking labels? (yes/no)
3. Is the ball visible and does ball tracking seem accurate? (yes/no)
4. What metrics should NOT be trusted based on what you see?

Return JSON:
{{
  "player_count_valid": bool,
  "actual_player_count": int,
  "team_colours_valid": bool,
  "ball_tracking_valid": bool,
  "unreliable_metrics": ["list metrics that look wrong"],
  "confidence": "high/medium/low"
}}

Return ONLY valid JSON. No markdown. No explanation.""",
        })

        result = _call_claude(
            messages=[{"role": "user", "content": content}],
            max_tokens=512,
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
) -> str:
    """Combine all layers into a final coaching report."""
    try:
        # Filter tracking data — remove metrics flagged as unreliable
        unreliable = set(audit_result.get("unreliable_metrics", []))
        filtered_tracking = {
            k: v for k, v in tracking_data.items()
            if k not in unreliable
        }

        confidence = audit_result.get("confidence", "low")
        gemini_text = json.dumps(gemini_observations, indent=2, default=str)
        tracking_text = json.dumps(filtered_tracking, indent=2, default=str)

        prompt = f"""You are writing a coaching report for a grassroots football coach.
They have just uploaded a match clip for analysis.

WHAT THE AI SAW (visual analysis):
{gemini_text}

DATA AUDIT RESULT:
Confidence: {confidence}
Unreliable metrics: {list(unreliable)}

TRACKING DATA (only trust metrics NOT in unreliable list):
{tracking_text}

Write a coaching report with EXACTLY these four sections:

WHAT WE SAW
2-3 sentences describing what actually happened in the clip.
Be specific. Reference actual observations. No generic phrases.

WHAT THE DATA SHOWS
2-3 bullet points of metrics that ARE reliable.
If a metric is unreliable, do not mention it.
If all metrics are unreliable, say "Data confidence too low for statistical claims."

ONE THING TO FIX THIS WEEK
One specific, actionable coaching point.
Include a suggested drill or exercise.
Be direct. No hedging.

PLAYER TO WATCH
Name one player (by team colour and number/position if known) who stood out —
either positively or as needing attention.

Rules:
- Never invent statistics
- Never mention metrics marked as unreliable
- If confidence is low, lead with visual observations only
- Keep total length under 250 words
- Write for a coach who wants to improve their team, not impress anyone"""

        result = _call_claude(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
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

        # LAYER 1 — Gemini watches the clip
        logger.info("[Brain L1] Gemini watching clip for job %s", job_id)
        gemini_obs = gemini_watch_clip(video_path, team_0_name, team_1_name)

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
        report = synthesise_report(gemini_obs, audit, result, team_0_name, team_1_name)

        logger.info("[Brain] Report complete for job %s (%d chars)", job_id, len(report))
        return report

    except Exception as e:
        logger.error("[Brain] Failed for job %s: %s", job_id, e)
        return _fallback_report(result, result.get("team_separation", {}).get("team_0_colour_name", "Team A"), result.get("team_separation", {}).get("team_1_colour_name", "Team B"))
