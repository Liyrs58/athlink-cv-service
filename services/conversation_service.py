"""Gemini-powered Q&A interface for match data analysis.
"""

import os
import json
import logging
import urllib.request
from typing import Optional, List, Dict
from pathlib import Path
from services.memory_service import MEMORY_DIR, get_trend_analysis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Question classification
# ---------------------------------------------------------------------------

QUESTION_TYPES = [
    "PLAYER_QUESTION",
    "TEAM_QUESTION",
    "HISTORICAL_QUESTION",
    "TRAINING_QUESTION",
    "GENERAL_QUESTION",
]

# Keywords mapped to question type — checked in order, first match wins
_CLASSIFICATION_RULES = [
    ("PLAYER_QUESTION", [
        "player", "who is", "who's", "fastest", "slowest", "hardest working",
        "most sprints", "top scorer", "best", "worst", "number", "#",
        "striker", "midfielder", "defender", "goalkeeper", "sprinter",
        "runner", "distance covered", "lazy",
    ]),
    ("HISTORICAL_QUESTION", [
        "improved", "improvement", "trend", "over time", "last few",
        "previous match", "compared to", "getting better", "getting worse",
        "progress", "regression", "history", "across matches", "last 3",
        "last 5", "week on week", "match to match", "consistency",
    ]),
    ("TRAINING_QUESTION", [
        "training", "drill", "session", "practice", "work on",
        "focus on", "prepare", "preparation", "exercise", "warm up",
        "cool down", "improve", "fix", "weakness", "before saturday",
        "next match", "session plan", "what should we",
    ]),
    ("TEAM_QUESTION", [
        "team", "formation", "shape", "width", "depth", "compact",
        "spread", "narrow", "press", "pressing", "tactic", "system",
        "defend", "attack", "transition", "possession", "concede",
        "goal", "half", "first half", "second half", "set piece",
        "dead ball", "open play", "high press", "why did we",
    ]),
]


def classify_question(question: str) -> str:
    """Classify a coach's question into a type for prompt routing."""
    q = question.lower().strip()
    for qtype, keywords in _CLASSIFICATION_RULES:
        if any(kw in q for kw in keywords):
            return qtype
    return "GENERAL_QUESTION"


# ---------------------------------------------------------------------------
# Match data loading
# ---------------------------------------------------------------------------

def load_match(job_id: str) -> Optional[dict]:
    """Load a single match from disk."""
    path = MEMORY_DIR / "matches" / f"{job_id}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not load match %s: %s", job_id, e)
        return None


def load_all_matches(limit: int = 10) -> List[dict]:
    """Load most recent matches, newest first."""
    matches_dir = MEMORY_DIR / "matches"
    files = sorted(matches_dir.glob("*.json"), key=os.path.getmtime, reverse=True)
    matches = []
    for f in files[:limit]:
        try:
            with open(f) as fp:
                matches.append(json.load(fp))
        except Exception:
            continue
    return matches


def load_match_context(job_id: Optional[str] = None) -> dict:
    """
    Load match data relevant to a conversation.

    Returns a dict with:
      matches    — list of match dicts
      trends     — trend analysis across matches (if enough data)
      match_ids  — list of job_ids included
      data_depth — "single", "multiple", or "none"
    """
    if job_id:
        match = load_match(job_id)
        if match:
            return {
                "matches": [match],
                "trends": None,
                "match_ids": [job_id],
                "data_depth": "single",
            }
        return {"matches": [], "trends": None, "match_ids": [], "data_depth": "none"}

    matches = load_all_matches(limit=10)
    trends = get_trend_analysis()
    return {
        "matches": matches,
        "trends": trends,
        "match_ids": [m["job_id"] for m in matches],
        "data_depth": "multiple" if len(matches) > 1 else ("single" if matches else "none"),
    }


# ---------------------------------------------------------------------------
# Context serialisation — turn match data into prompt text
# ---------------------------------------------------------------------------

def _format_match_for_prompt(match: dict, index: int = 0) -> str:
    """Convert a stored match dict into a readable block for the prompt."""
    jid = match.get("job_id", "unknown")
    ts = match.get("timestamp", "")[:10]
    phys = match.get("physical", {})
    shape = match.get("shape", {})
    sits = match.get("situations", {})
    events = sits.get("events", []) if isinstance(sits, dict) else []
    analysis = match.get("analysis", "")

    lines = [f"=== MATCH {index + 1}: {jid} ({ts}) ==="]

    # Physical metrics
    lines.append("PHYSICAL:")
    lines.append(f"  Players analysed: {phys.get('players_analysed', '?')}")
    lines.append(f"  Total distance: {phys.get('total_distance_metres', '?')}m")
    lines.append(f"  Total sprints: {phys.get('total_sprints', '?')}")
    lines.append(f"  Max speed: {phys.get('max_speed_kmh', '?')} km/h")
    lines.append(f"  Top sprinter: Player #{phys.get('top_sprinter_id', '?')}")
    lines.append(f"  Top runner: Player #{phys.get('top_runner_id', '?')} ({phys.get('top_runner_distance', '?')}m)")

    # Shape metrics
    dq = shape.get("data_quality", "unavailable")
    if dq == "ok":
        lines.append("TEAM SHAPE:")
        lines.append(f"  Avg width: {shape.get('avg_width_metres', '?')}m")
        lines.append(f"  Avg depth: {shape.get('avg_depth_metres', '?')}m")
        lines.append(f"  Compactness: {shape.get('avg_compactness_metres', '?')}m")
        t0 = shape.get("team_0", {})
        t1 = shape.get("team_1", {})
        if t0 and t0.get("avg_width_metres") is not None:
            lines.append(f"  Team A width: {t0['avg_width_metres']}m, depth: {t0.get('avg_depth_metres', '?')}m")
        if t1 and t1.get("avg_width_metres") is not None:
            lines.append(f"  Team B width: {t1['avg_width_metres']}m, depth: {t1.get('avg_depth_metres', '?')}m")
    else:
        lines.append("TEAM SHAPE: unavailable for this match")

    # Situation timeline
    if events:
        lines.append("MATCH PHASES:")
        for ev in events:
            s = ev.get("start_time", 0)
            e = ev.get("end_time", 0)
            sit = ev.get("situation", "?")
            dur = ev.get("duration_seconds", 0)
            lines.append(f"  {s:.0f}s–{e:.0f}s: {sit} ({dur:.1f}s)")

    # Previous coaching analysis (if available and not an error)
    if analysis and not analysis.startswith("API error"):
        lines.append("COACHING REPORT (from match analysis):")
        # Truncate to keep prompt manageable
        if len(analysis) > 600:
            lines.append(f"  {analysis[:600]}…")
        else:
            lines.append(f"  {analysis}")

    return "\n".join(lines)


def _confidence_from_context(ctx: dict, question_type: str) -> str:
    """Determine answer confidence based on available data."""
    matches = ctx["matches"]
    if not matches:
        return "low"

    if question_type == "HISTORICAL_QUESTION":
        return "high" if len(matches) >= 3 else ("medium" if len(matches) >= 2 else "low")

    # Check data quality of most recent match
    latest = matches[0]
    phys = latest.get("physical", {})
    shape = latest.get("shape", {})

    has_physical = bool(phys.get("total_sprints") is not None and phys.get("players_analysed"))
    has_shape = shape.get("data_quality") == "ok"

    if question_type == "TEAM_QUESTION":
        return "high" if has_shape and has_physical else ("medium" if has_physical else "low")

    if question_type == "PLAYER_QUESTION":
        return "high" if has_physical and phys.get("players_analysed", 0) >= 10 else "medium"

    return "medium" if has_physical else "low"


# ---------------------------------------------------------------------------
# Prompt templates — one per question type
# ---------------------------------------------------------------------------

_SYSTEM_PREAMBLE = (
    "You are the Athlink Conversation Coach — a football coaching assistant "
    "that answers questions using ONLY the match data provided below.\n\n"
    "RULES:\n"
    "1. Every claim must reference a specific number, player, or event from the data.\n"
    "2. If the data doesn't support an answer, say so — never guess or use general football knowledge.\n"
    "3. Use direct coaching language. Talk like a coach to a coach.\n"
    "4. Keep answers concise — 3-8 sentences unless the question demands more.\n"
    "5. When referencing players, use 'Player #ID' format.\n"
    "6. When comparing matches, reference them by date.\n"
)

_PLAYER_TEMPLATE = (
    "The coach is asking about specific player performance.\n"
    "Rank or identify players using the match data. Include sprint counts, "
    "distances, and top speeds where relevant. Compare across matches if "
    "multiple are available.\n\n"
    "QUESTION: {question}\n\n"
    "MATCH DATA:\n{match_data}\n"
)

_TEAM_TEMPLATE = (
    "The coach is asking about team tactics, shape, or match events.\n"
    "Reference the situation timeline (OPEN_PLAY, DEAD_BALL, SET_PIECE, etc.) "
    "and team shape data (width, depth, compactness). Explain what happened "
    "and why, grounded in the numbers.\n\n"
    "QUESTION: {question}\n\n"
    "MATCH DATA:\n{match_data}\n"
)

_HISTORICAL_TEMPLATE = (
    "The coach is asking about trends across matches.\n"
    "Compare metrics between matches chronologically. Highlight improvements "
    "and regressions with specific numbers. If trend data is provided, use it.\n\n"
    "QUESTION: {question}\n\n"
    "MATCH DATA:\n{match_data}\n"
    "{trends}\n"
)

_TRAINING_TEMPLATE = (
    "The coach is asking what to work on in training.\n"
    "Based on the match data, identify the most important weaknesses and "
    "generate exactly 3 training drills.\n\n"
    "For each drill, provide:\n"
    "  DRILL NAME: [name]\n"
    "  DURATION: [X minutes]\n"
    "  SETUP: [players needed, equipment, field markings]\n"
    "  HOW IT WORKS: [step-by-step instructions]\n"
    "  COACHING POINT: [what specific weakness this fixes, referencing match data]\n"
    "  SUCCESS METRIC: [how the coach knows the drill is working]\n\n"
    "The drills MUST address weaknesses visible in the data — not generic fitness.\n"
    "For example, if team width was too narrow (under 40m), design a drill that "
    "forces wide play. If sprints were low, design a drill that builds repeated "
    "sprint endurance.\n\n"
    "QUESTION: {question}\n\n"
    "MATCH DATA:\n{match_data}\n"
)

_GENERAL_TEMPLATE = (
    "The coach is asking a general question about their team.\n"
    "Answer using the match data provided. Be specific and reference numbers.\n\n"
    "QUESTION: {question}\n\n"
    "MATCH DATA:\n{match_data}\n"
)

_TEMPLATES = {
    "PLAYER_QUESTION": _PLAYER_TEMPLATE,
    "TEAM_QUESTION": _TEAM_TEMPLATE,
    "HISTORICAL_QUESTION": _HISTORICAL_TEMPLATE,
    "TRAINING_QUESTION": _TRAINING_TEMPLATE,
    "GENERAL_QUESTION": _GENERAL_TEMPLATE,
}


def build_conversation_prompt(question: str, context: dict, question_type: str) -> str:
    """Build a Claude prompt grounded in real match data."""
    matches = context["matches"]

    # Serialise match data into text
    match_blocks = []
    for i, m in enumerate(matches):
        match_blocks.append(_format_match_for_prompt(m, i))
    match_data = "\n\n".join(match_blocks) if match_blocks else "(No match data available)"

    # Build trend string for historical questions
    trends_str = ""
    if context.get("trends"):
        t = context["trends"]
        parts = []
        if "width_trend" in t:
            parts.append(f"  Width trend: {t['width_trend']}")
        if "sprint_trend" in t:
            parts.append(f"  Sprint trend: {t['sprint_trend']}")
        if parts:
            trends_str = "TREND ANALYSIS:\n" + "\n".join(parts)

    template = _TEMPLATES.get(question_type, _GENERAL_TEMPLATE)
    body = template.format(
        question=question,
        match_data=match_data,
        trends=trends_str,
    )

    return _SYSTEM_PREAMBLE + "\n" + body


# ---------------------------------------------------------------------------
# Claude API call
# ---------------------------------------------------------------------------

def _call_claude(prompt: str, max_tokens: int = 1024) -> str:
    """Call Claude API and return the text response."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    data = json.dumps({
        "model": "claude-sonnet-4-6",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
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

    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
        return result["content"][0]["text"]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def ask_conversation(question: str, job_id: Optional[str] = None) -> dict:
    """
    Full conversation flow:
      1. Classify question
      2. Load match context
      3. Build prompt
      4. Call Claude
      5. Return grounded answer with sources and confidence

    Returns:
      {
        "answer": str,
        "sources": [job_id, ...],
        "confidence": "high" | "medium" | "low",
        "question_type": str,
      }
    """
    question_type = classify_question(question)
    context = load_match_context(job_id)

    if not context["matches"]:
        return {
            "answer": (
                "I don't have any match data to work with yet. "
                "Upload a match video to /api/v1/analyse first, "
                "then come back and ask me anything about it."
            ),
            "sources": [],
            "confidence": "low",
            "question_type": question_type,
        }

    confidence = _confidence_from_context(context, question_type)

    prompt = build_conversation_prompt(question, context, question_type)

    try:
        answer = _call_claude(prompt)
    except RuntimeError as e:
        # Missing API key — fall back to a data-grounded summary
        answer = _build_offline_answer(question, context, question_type)
    except Exception as e:
        logger.error("Claude API error in conversation: %s", e)
        answer = _build_offline_answer(question, context, question_type)

    return {
        "answer": answer,
        "sources": context["match_ids"],
        "confidence": confidence,
        "question_type": question_type,
    }


# ---------------------------------------------------------------------------
# Offline fallback — answers without Claude, using raw data
# ---------------------------------------------------------------------------

def _build_offline_answer(question: str, context: dict, question_type: str) -> str:
    """
    Generate a data-grounded answer without Claude API.
    Less eloquent but still references real numbers.
    """
    matches = context["matches"]
    if not matches:
        return "No match data available."

    latest = matches[0]
    phys = latest.get("physical", {})
    shape = latest.get("shape", {})
    sits = latest.get("situations", {})
    events = sits.get("events", []) if isinstance(sits, dict) else []
    jid = latest.get("job_id", "?")
    ts = latest.get("timestamp", "")[:10]

    lines = [f"Based on match {jid} ({ts}):"]

    if question_type == "PLAYER_QUESTION":
        lines.append(f"- {phys.get('players_analysed', '?')} players were tracked.")
        lines.append(f"- Hardest worker: Player #{phys.get('top_runner_id', '?')} "
                      f"covered {phys.get('top_runner_distance', '?')}m.")
        lines.append(f"- Most sprints: Player #{phys.get('top_sprinter_id', '?')} "
                      f"with the team total at {phys.get('total_sprints', '?')} sprints.")
        lines.append(f"- Top speed across the squad: {phys.get('max_speed_kmh', '?')} km/h.")

    elif question_type == "TEAM_QUESTION":
        if events:
            lines.append("Match phases:")
            for ev in events:
                lines.append(f"  {ev.get('start_time',0):.0f}s–{ev.get('end_time',0):.0f}s: "
                             f"{ev.get('situation','?')} ({ev.get('duration_seconds',0):.1f}s)")
        if shape.get("data_quality") == "ok":
            lines.append(f"Team width averaged {shape.get('avg_width_metres', '?')}m, "
                         f"depth {shape.get('avg_depth_metres', '?')}m.")

    elif question_type == "TRAINING_QUESTION":
        weaknesses = []
        if isinstance(phys.get("total_sprints"), (int, float)) and phys["total_sprints"] < 10:
            weaknesses.append(f"low sprint count ({phys['total_sprints']})")
        if shape.get("data_quality") == "ok":
            w = shape.get("avg_width_metres")
            if isinstance(w, (int, float)) and w < 40:
                weaknesses.append(f"narrow team shape ({w}m width)")
            elif isinstance(w, (int, float)) and w > 60:
                weaknesses.append(f"very wide shape ({w}m) — may need better compactness")
        if not weaknesses:
            weaknesses.append("maintaining intensity across the full match")
        lines.append(f"Key areas to work on: {', '.join(weaknesses)}.")
        lines.append("(Connect an API key for detailed drill recommendations.)")

    elif question_type == "HISTORICAL_QUESTION":
        if len(matches) >= 2:
            for i, m in enumerate(matches[:5]):
                p = m.get("physical", {})
                lines.append(f"  Match {m['job_id']} ({m['timestamp'][:10]}): "
                             f"sprints={p.get('total_sprints','?')}, "
                             f"speed={p.get('max_speed_kmh','?')}km/h")
            trends = context.get("trends")
            if trends:
                if "width_trend" in trends:
                    lines.append(f"Width trend: {trends['width_trend']}")
                if "sprint_trend" in trends:
                    lines.append(f"Sprint trend: {trends['sprint_trend']}")
        else:
            lines.append("Only 1 match in memory — need 2+ for comparison.")

    else:
        lines.append(f"- {phys.get('players_analysed', '?')} players tracked, "
                      f"{phys.get('total_sprints', '?')} total sprints, "
                      f"max speed {phys.get('max_speed_kmh', '?')} km/h.")
        if events:
            open_play = sum(1 for e in events if e.get("situation") == "OPEN_PLAY")
            dead_ball = sum(1 for e in events if e.get("situation") == "DEAD_BALL")
            lines.append(f"- {open_play} phases of open play, {dead_ball} dead ball periods.")

    return "\n".join(lines)
