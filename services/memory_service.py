"""Session memory and state persistence across analyses.
"""

import json
import os
from datetime import datetime
from pathlib import Path

MEMORY_DIR = Path("memory")

def _ensure_memory_dirs():
    MEMORY_DIR.mkdir(exist_ok=True)
    (MEMORY_DIR / "matches").mkdir(exist_ok=True)
    (MEMORY_DIR / "teams").mkdir(exist_ok=True)
    (MEMORY_DIR / "patterns").mkdir(exist_ok=True)

def store_match(job_id, tracking, situations, physical, shape, analysis, player_history=None):
    """Store complete match analysis to memory."""
    _ensure_memory_dirs()
    match_data = {
        "job_id": job_id,
        "timestamp": datetime.now().isoformat(),
        "tracking": tracking,
        "situations": situations,
        "physical": physical,
        "shape": shape,
        "analysis": analysis,
    }
    if player_history:
        match_data["player_history"] = player_history
    path = MEMORY_DIR / "matches" / f"{job_id}.json"
    with open(path, "w") as f:
        json.dump(match_data, f, indent=2)
    return path

def get_historical_context(limit=5):
    """Read last N matches and build context string for Claude."""
    matches_dir = MEMORY_DIR / "matches"
    files = sorted(matches_dir.glob("*.json"), key=os.path.getmtime, reverse=True)[:limit]

    if not files:
        return None

    history = []
    for f in files:
        with open(f) as fp:
            m = json.load(fp)
        physical = m.get("physical", {})
        shape = m.get("shape", {})
        history.append(
            f"Match {m['job_id']} ({m['timestamp'][:10]}): "
            f"width={shape.get('avg_width_metres')}m, "
            f"sprints={physical.get('total_sprints')}, "
            f"max_speed={physical.get('max_speed_kmh')}km/h"
        )

    return "Previous matches:\n" + "\n".join(history)

def get_player_history() -> dict:
    """Returns accumulated player appearance history across all stored matches."""
    matches_dir = MEMORY_DIR / "matches"
    files = sorted(matches_dir.glob("*.json"), key=os.path.getmtime)
    combined = {}
    for f in files:
        try:
            with open(f) as fp:
                m = json.load(fp)
            ph = m.get("player_history", {})
            for pid, appearances in ph.items():
                combined.setdefault(pid, []).extend(appearances)
        except Exception:
            continue
    # Deduplicate by match_id within each player
    for pid in combined:
        seen = set()
        deduped = []
        for a in combined[pid]:
            mid = a.get("match_id", "")
            if mid not in seen:
                seen.add(mid)
                deduped.append(a)
        combined[pid] = deduped
    return combined


def get_match_count():
    return len(list((MEMORY_DIR / "matches").glob("*.json")))

def clean_memory():
    """Delete stored matches with corrupted data from old pipeline runs."""
    matches_dir = MEMORY_DIR / "matches"
    files = list(matches_dir.glob("*.json"))

    deleted_count = 0
    for f in files:
        try:
            with open(f) as fp:
                m = json.load(fp)

            physical = m.get("physical", {})
            shape = m.get("shape", {})

            total_sprints = physical.get("total_sprints", 0)
            max_speed_kmh = physical.get("max_speed_kmh", 0)
            avg_width_m = shape.get("avg_width_metres", 0)

            # Delete if any metric exceeds thresholds from old pipeline
            if total_sprints > 50 or max_speed_kmh > 36 or avg_width_m > 70:
                f.unlink()
                deleted_count += 1
                print(f"Deleted corrupted match {f.name}: sprints={total_sprints}, speed={max_speed_kmh}km/h, width={avg_width_m}m")
        except Exception as e:
            print(f"Error checking {f.name}: {e}")

    if deleted_count > 0:
        print(f"Cleaned memory: deleted {deleted_count} corrupted matches")
    return deleted_count

def get_trend_analysis():
    """Analyse trends across all stored matches."""
    matches_dir = MEMORY_DIR / "matches"
    files = sorted(matches_dir.glob("*.json"), key=os.path.getmtime)

    if len(files) < 2:
        return None

    widths = []
    sprints = []
    for f in files:
        with open(f) as fp:
            m = json.load(fp)
        shape = m.get("shape", {})
        physical = m.get("physical", {})
        if shape.get("avg_width_metres"):
            widths.append(shape["avg_width_metres"])
        if physical.get("total_sprints"):
            sprints.append(physical["total_sprints"])

    trends = {}
    if len(widths) >= 2:
        trend = "increasing" if widths[-1] > widths[0] else "decreasing"
        trends["width_trend"] = f"{trend} ({widths[0]}m -> {widths[-1]}m)"
    if len(sprints) >= 2:
        trend = "increasing" if sprints[-1] > sprints[0] else "decreasing"
        trends["sprint_trend"] = f"{trend} ({sprints[0]} -> {sprints[-1]})"

    return trends
