import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

PITCH_W = 105.0
PITCH_H = 68.0
WINDOW_FRAMES = 75
MIN_PLAYERS_TEAM = 6
SHIFT_COOLDOWN = 150

KNOWN_FORMATIONS = [
    (4, 3, 3),
    (4, 4, 2),
    (4, 2, 4),   # 4-2-3-1 collapsed to 3 lines
    (3, 5, 2),
    (3, 4, 3),
    (5, 3, 2),
    (5, 4, 1),
    (4, 5, 1),
    (4, 1, 5),   # 4-1-4-1 collapsed to 3 lines
]

FORMATION_LABELS = {
    (4, 3, 3): "4-3-3",
    (4, 4, 2): "4-4-2",
    (4, 2, 4): "4-2-3-1",
    (3, 5, 2): "3-5-2",
    (3, 4, 3): "3-4-3",
    (5, 3, 2): "5-3-2",
    (5, 4, 1): "5-4-1",
    (4, 5, 1): "4-5-1",
    (4, 1, 5): "4-1-4-1",
}


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


def _nearest_formation(d, m, a):
    # type: (int, int, int) -> Tuple[str, Tuple[int, int, int]]
    best = None  # type: Optional[Tuple[int, int, int]]
    best_dist = float("inf")
    for f in KNOWN_FORMATIONS:
        dist = abs(f[0] - d) + abs(f[1] - m) + abs(f[2] - a)
        if dist < best_dist:
            best_dist = dist
            best = f
    if best is None:
        best = (4, 4, 2)
    return FORMATION_LABELS[best], best


def _classify_window(avg_positions, team_id):
    # type: (List[Dict[str, float]], int) -> Optional[Dict[str, Any]]
    """
    Classify formation for a set of players in one window.
    avg_positions: list of {"trackId": int, "x": float, "y": float}
    team_id: 0 (attacks right) or 1 (attacks left)
    Returns dict with formation info or None if not enough players.
    """
    if len(avg_positions) < MIN_PLAYERS_TEAM + 1:
        return None

    # GK detection
    if team_id == 0:
        # Team 0 attacks right, GK has lowest x
        gk_idx = min(range(len(avg_positions)), key=lambda i: avg_positions[i]["x"])
    else:
        # Team 1 attacks left, GK has highest x
        gk_idx = max(range(len(avg_positions)), key=lambda i: avg_positions[i]["x"])

    outfield = [p for i, p in enumerate(avg_positions) if i != gk_idx]

    if len(outfield) < MIN_PLAYERS_TEAM:
        return None

    # Sort by x for team 0 (deepest defender first),
    # by -x for team 1 (deepest defender first)
    if team_id == 0:
        outfield.sort(key=lambda p: p["x"])
    else:
        outfield.sort(key=lambda p: -p["x"])

    xs = np.array([p["x"] for p in outfield])

    # Split into lines using x-percentile thresholds
    p33 = float(np.percentile(xs, 33))
    p66 = float(np.percentile(xs, 66))

    # For team 1 sorted by -x, the percentiles are on the sorted array
    # which is already deepest-first, so same logic applies
    defenders = []
    midfielders = []
    attackers = []

    for p in outfield:
        px = p["x"]
        if team_id == 0:
            if px <= p33:
                defenders.append(p)
            elif px <= p66:
                midfielders.append(p)
            else:
                attackers.append(p)
        else:
            # Team 1: lower x = more attacking
            if px >= p66:
                defenders.append(p)
            elif px >= p33:
                midfielders.append(p)
            else:
                attackers.append(p)

    d = len(defenders)
    m = len(midfielders)
    a = len(attackers)

    formation_str, _ = _nearest_formation(d, m, a)

    # Stability score: fraction of players whose x falls within
    # their assigned line's x-range
    stable_count = 0
    if defenders:
        def_xs = [p["x"] for p in defenders]
        def_min, def_max = min(def_xs), max(def_xs)
        for p in defenders:
            if def_min <= p["x"] <= def_max:
                stable_count += 1
    if midfielders:
        mid_xs = [p["x"] for p in midfielders]
        mid_min, mid_max = min(mid_xs), max(mid_xs)
        for p in midfielders:
            if mid_min <= p["x"] <= mid_max:
                stable_count += 1
    if attackers:
        att_xs = [p["x"] for p in attackers]
        att_min, att_max = min(att_xs), max(att_xs)
        for p in attackers:
            if att_min <= p["x"] <= att_max:
                stable_count += 1

    stability_score = stable_count / len(outfield) if outfield else 0.0

    def_cx = float(np.mean([p["x"] for p in defenders])) if defenders else 0.0
    mid_cx = float(np.mean([p["x"] for p in midfielders])) if midfielders else 0.0
    att_cx = float(np.mean([p["x"] for p in attackers])) if attackers else 0.0
    compactness = abs(att_cx - def_cx)

    return {
        "formation": formation_str,
        "raw_d_m_a": [d, m, a],
        "stability_score": round(stability_score, 3),
        "def_centroid_x": round(def_cx, 2),
        "mid_centroid_x": round(mid_cx, 2),
        "att_centroid_x": round(att_cx, 2),
        "compactness_m": round(compactness, 2),
    }


def compute_formations(job_id):
    # type: (str) -> Dict[str, Any]
    """
    Detect tactical formations and track shape shifts over time.

    Reads:
        temp/{job_id}/pitch/pitch_map.json
        temp/{job_id}/tracking/team_results.json
        temp/{job_id}/tracking/track_results.json (for fps)
    """
    base = _base_job_id(job_id)
    candidates = [job_id]
    if base != job_id:
        candidates.append(base)

    # Load pitch_map.json
    pitch_data = None
    for jid in candidates:
        pitch_data = _load_json(Path("temp/{}/pitch/pitch_map.json".format(jid)))
        if pitch_data is not None:
            break
    if pitch_data is None:
        raise FileNotFoundError(
            "pitch_map.json not found for job '{}'".format(job_id)
        )

    # Load team_results.json
    team_map = {}  # type: Dict[int, int]
    for jid in candidates:
        team_data = _load_json(Path("temp/{}/tracking/team_results.json".format(jid)))
        if team_data is not None:
            tracks_list = team_data.get("tracks", team_data) if isinstance(team_data, dict) else team_data
            for t in tracks_list:
                team_map[t["trackId"]] = t.get("teamId", -1)
            break

    # Load track_results.json for fps
    fps = 25.0
    for jid in candidates:
        track_data = _load_json(Path("temp/{}/tracking/track_results.json".format(jid)))
        if track_data is not None:
            metadata = track_data.get("metadata", {})
            fps = float(metadata.get("fps", 25.0))
            if fps <= 0:
                fps = 25.0
            break

    # Collect per-player positions per frame
    # trackId -> list of (frameIndex, x, y)
    player_positions = {}  # type: Dict[int, List[Tuple[int, float, float]]]
    all_frames = set()  # type: set

    for player in pitch_data.get("players", []):
        tid = player["trackId"]
        for pt in player.get("trajectory2d", []):
            fi = int(pt["frameIndex"])
            x = float(pt["x"])
            y = float(pt["y"])
            if tid not in player_positions:
                player_positions[tid] = []
            player_positions[tid].append((fi, x, y))
            all_frames.add(fi)

    if not all_frames:
        return {
            "team_0": {
                "dominant_formation": "unknown",
                "formation_stability": 0.0,
                "avg_compactness_m": 0.0,
                "timeline": [],
            },
            "team_1": {
                "dominant_formation": "unknown",
                "formation_stability": 0.0,
                "avg_compactness_m": 0.0,
                "timeline": [],
            },
            "shape_shift_events": [],
        }

    min_frame = min(all_frames)
    max_frame = max(all_frames)

    # Build windows
    windows = []  # type: List[Tuple[int, int]]
    start = min_frame
    while start <= max_frame:
        end = start + WINDOW_FRAMES - 1
        if end > max_frame:
            end = max_frame
        windows.append((start, end))
        start = end + 1

    # For each window, compute avg position per player within that window
    team_timelines = {0: [], 1: []}  # type: Dict[int, List[Dict[str, Any]]]

    for w_start, w_end in windows:
        # Collect per-player avg positions in this window, grouped by team
        team_players = {0: [], 1: []}  # type: Dict[int, List[Dict[str, float]]]

        for tid, positions in player_positions.items():
            team_id = team_map.get(tid, -1)
            if team_id not in (0, 1):
                continue
            pts_in_window = [(fi, x, y) for fi, x, y in positions
                            if w_start <= fi <= w_end]
            if not pts_in_window:
                continue
            avg_x = float(np.mean([x for _, x, _ in pts_in_window]))
            avg_y = float(np.mean([y for _, _, y in pts_in_window]))
            team_players[team_id].append({
                "trackId": tid,
                "x": avg_x,
                "y": avg_y,
            })

        for team_id in (0, 1):
            result = _classify_window(team_players[team_id], team_id)
            if result is None:
                continue
            entry = {
                "frame_start": w_start,
                "frame_end": w_end,
            }
            entry.update(result)
            team_timelines[team_id].append(entry)

    # Dominant formation and stability per team
    team_results = {}  # type: Dict[str, Dict[str, Any]]
    for team_id in (0, 1):
        timeline = team_timelines[team_id]
        if not timeline:
            team_results["team_{}".format(team_id)] = {
                "dominant_formation": "unknown",
                "formation_stability": 0.0,
                "avg_compactness_m": 0.0,
                "timeline": [],
            }
            continue

        # Count formations
        formation_counts = {}  # type: Dict[str, int]
        for entry in timeline:
            f = entry["formation"]
            formation_counts[f] = formation_counts.get(f, 0) + 1

        dominant = max(formation_counts, key=formation_counts.get)
        stability = formation_counts[dominant] / len(timeline)
        avg_compact = float(np.mean([e["compactness_m"] for e in timeline]))

        team_results["team_{}".format(team_id)] = {
            "dominant_formation": dominant,
            "formation_stability": round(stability, 3),
            "avg_compactness_m": round(avg_compact, 2),
            "timeline": timeline,
        }

    # Shape shift events
    shift_events = []  # type: List[Dict[str, Any]]

    for team_id in (0, 1):
        timeline = team_timelines[team_id]
        if len(timeline) < 2:
            continue

        last_shift_frame = -SHIFT_COOLDOWN - 1

        for i in range(1, len(timeline)):
            prev = timeline[i - 1]
            curr = timeline[i]
            frame = curr["frame_start"]

            # Formation change
            if curr["formation"] != prev["formation"]:
                if frame - last_shift_frame >= SHIFT_COOLDOWN:
                    shift_events.append({
                        "frame": frame,
                        "team": team_id,
                        "from_formation": prev["formation"],
                        "to_formation": curr["formation"],
                        "type": "formation",
                    })
                    last_shift_frame = frame

            # Compactness change > 15m
            compact_diff = abs(curr["compactness_m"] - prev["compactness_m"])
            if compact_diff > 15.0:
                if frame - last_shift_frame >= SHIFT_COOLDOWN:
                    shift_events.append({
                        "frame": frame,
                        "team": team_id,
                        "from_formation": prev["formation"],
                        "to_formation": curr["formation"],
                        "type": "compactness",
                    })
                    last_shift_frame = frame

    # Sort shift events by frame
    shift_events.sort(key=lambda e: e["frame"])

    result = {
        "team_0": team_results.get("team_0", {
            "dominant_formation": "unknown",
            "formation_stability": 0.0,
            "avg_compactness_m": 0.0,
            "timeline": [],
        }),
        "team_1": team_results.get("team_1", {
            "dominant_formation": "unknown",
            "formation_stability": 0.0,
            "avg_compactness_m": 0.0,
            "timeline": [],
        }),
        "shape_shift_events": shift_events,
    }

    logger.info(
        "Formation analysis: team0=%s (%.1f%%), team1=%s (%.1f%%), %d shifts",
        result["team_0"]["dominant_formation"],
        result["team_0"]["formation_stability"] * 100,
        result["team_1"]["dominant_formation"],
        result["team_1"]["formation_stability"] * 100,
        len(shift_events),
    )

    return result
