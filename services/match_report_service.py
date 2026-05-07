"""Aggregate per-feature service outputs into a single match_report.json.

This is the unified document a frontend renders into the AthLink mood-board UI:
quality metrics, team formations, possession/xG/sprints, players, events, and
(later) tactics clip labels.

NOTE: this service is a pure aggregator — it does NOT recompute any analytics.
Each downstream service writes its own JSON under temp/{jobId}/...; this module
reads them, normalises shapes, and emits one document.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── default colours (mood board) ───────────────────────────────────────────
TEAM_HOME_HEX = "#1d63ff"
TEAM_AWAY_HEX = "#22c55e"


def _team_entries(team_data: Any) -> List[dict]:
    """Normalise team_results into a list of {trackId, teamId} dicts.
    team_service writes a bare list of track dicts; older paths may write a
    {tracks: [...]} envelope. Tolerate both."""
    if team_data is None:
        return []
    if isinstance(team_data, list):
        return [t for t in team_data if isinstance(t, dict)]
    if isinstance(team_data, dict):
        for key in ("tracks", "teams"):
            arr = team_data.get(key)
            if isinstance(arr, list):
                return [t for t in arr if isinstance(t, dict)]
    return []


def _safe_load(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _frames_processed(track_data: Optional[dict]) -> int:
    if not track_data:
        return 0
    return int(
        track_data.get("framesProcessed")
        or track_data.get("total_frames")
        or len(track_data.get("frames", []))
        or 0
    )


def _detect_video_meta(track_data: Optional[dict]) -> Dict[str, Any]:
    fps = 25.0
    if track_data:
        fps = float(track_data.get("fps", 25.0))
    frames = _frames_processed(track_data)
    return {
        "durationSec": round(frames / fps, 2) if fps else None,
        "fps": fps,
        "framesProcessed": frames,
    }


def _team_assignment_confidence(team_entries: List[dict]) -> Optional[float]:
    """Fraction of tracks that received any team label (0/1/2 = home/away/GK).
    -1 = unassigned counts as miss. Returns None if no entries."""
    if not team_entries:
        return None
    assigned = sum(1 for e in team_entries if int(e.get("teamId", e.get("team_id", -1))) in (0, 1, 2))
    return round(assigned / len(team_entries), 3)


def _quality(
    identity_metrics: Optional[dict],
    pitch_data: Optional[dict],
    track_data: Optional[dict],
    video_meta: Dict[str, Any],
    team_entries: Optional[List[dict]] = None,
) -> Dict[str, Any]:
    framesProcessed = video_meta["framesProcessed"]
    durationSec = video_meta["durationSec"] or 0.0
    if identity_metrics is None:
        identity_metrics = {}

    locks_created = float(identity_metrics.get("locks_created") or 0)
    locks_expired = float(identity_metrics.get("locks_expired") or 0)
    revived_count = float(identity_metrics.get("revived_count") or 0)
    identity_switches = float(identity_metrics.get("identity_switches") or 0)
    valid_id_coverage = float(identity_metrics.get("valid_id_coverage") or 0.0)
    unknown_boxes = int(identity_metrics.get("unknown_boxes") or 0)
    frames_in_collapse = int(identity_metrics.get("frames_in_collapse") or 0)
    lock_retention = float(identity_metrics.get("lock_retention_rate") or 0.0)

    # stableIds = locked-source share of total assignments (proxy)
    stableIds = round(min(1.0, lock_retention), 3)
    track_resurrection = round(
        revived_count / (revived_count + locks_expired) if (revived_count + locks_expired) else 0.0,
        3,
    )
    low_id_switches_per_min = round(
        identity_switches * 60.0 / durationSec if durationSec else 0.0,
        3,
    )

    pitch_coverage = 0.0
    short_tracks_filtered = 0
    if pitch_data and framesProcessed:
        h = pitch_data.get("homographies") or {}
        # pitch_service samples every Nth frame — use the actual stride to
        # measure coverage of intended sample points, not raw frames.
        stride = int(pitch_data.get("frameStride") or 5)
        expected = max(1, framesProcessed // stride)
        pitch_coverage = round(min(1.0, len(h) / expected), 3)
    elif pitch_data and pitch_data.get("homographyFound"):
        pitch_coverage = 1.0

    team_assignment_acc = _team_assignment_confidence(team_entries or [])

    soft_collapse_fraction = round(
        frames_in_collapse / framesProcessed if framesProcessed else 0.0,
        3,
    )

    # crude match confidence: weighted mix
    match_confidence = round(
        (
            0.35 * stableIds
            + 0.25 * valid_id_coverage
            + 0.20 * pitch_coverage
            + 0.20 * (1.0 - min(1.0, soft_collapse_fraction))
        ),
        3,
    )

    unique_ids = 0
    if track_data:
        seen = set()
        for fr in track_data.get("frames", []):
            for p in fr.get("players", []):
                pid = p.get("playerId") or p.get("displayId")
                if pid:
                    seen.add(str(pid))
        unique_ids = len(seen)

    # Count short / unassigned tracks for transparency in the dashboard
    if team_entries:
        short_tracks_filtered = sum(
            1 for e in team_entries
            if int(e.get("teamId", e.get("team_id", -1))) == -1
        )

    return {
        "stableIds": stableIds,
        "trackResurrection": track_resurrection,
        "lowIdSwitchesPerMin": low_id_switches_per_min,
        "teamAssignmentAccuracy": team_assignment_acc,
        "pitchDetectionCoverage": pitch_coverage,
        "matchConfidence": match_confidence,
        "validIdCoverage": valid_id_coverage,
        "softCollapseFraction": soft_collapse_fraction,
        "uniqueIds": unique_ids,
        "unknownBoxes": unknown_boxes,
        "shortTracksFiltered": short_tracks_filtered,
    }


def _teams_block(formation_data: Optional[dict], team_data: Optional[dict]) -> Dict[str, Any]:
    home: Dict[str, Any] = {"playerCount": 0, "formation": None, "formationConfidence": None, "color": TEAM_HOME_HEX}
    away: Dict[str, Any] = {"playerCount": 0, "formation": None, "formationConfidence": None, "color": TEAM_AWAY_HEX}

    if formation_data:
        for key, target in (("team_0", home), ("team_1", away)):
            block = formation_data.get(key) or {}
            target["formation"] = block.get("dominant_formation") or block.get("formation")
            conf = block.get("confidence") or block.get("formationConfidence")
            if conf is not None:
                try:
                    target["formationConfidence"] = round(float(conf), 3)
                except Exception:
                    pass

    if team_data:
        # team_service emits {0=home, 1=away, 2=goalkeeper, -2=official, -1=unassigned}.
        # The mood-board "playerCount" should mean tracks-on-this-team inclusive of GK.
        # We don't yet know which team a GK belongs to, so split GKs evenly until
        # team-half occupancy from pitch_map is wired (TODO).
        counts = defaultdict(int)
        gk_count = 0
        for entry in _team_entries(team_data):
            tid = entry.get("teamId", entry.get("team_id"))
            if tid is None:
                continue
            tid = int(tid)
            if tid in (0, 1):
                counts[tid] += 1
            elif tid == 2:
                gk_count += 1
        # Split GKs evenly: first to whichever team is short, then alternating
        if gk_count:
            if counts[0] <= counts[1]:
                counts[0] += min(1, gk_count); gk_count -= 1
            if gk_count:
                counts[1] += min(1, gk_count); gk_count -= 1
            # Anything left (rare) gets credited to home
            counts[0] += gk_count
        home["playerCount"] = counts.get(0, 0)
        away["playerCount"] = counts.get(1, 0)

    return {"home": home, "away": away}


def _metrics_block(
    pass_data: Optional[dict],
    pressing_data: Optional[dict],
    xg_data: Optional[dict],
    heatmap_data: Optional[dict],
    events_data: Optional[dict],
) -> Dict[str, Any]:
    def _pair(d: Optional[dict], key_a: str, key_b: str = None, default=0):
        if not d:
            return [default, default]
        a = d.get(key_a, default) if d else default
        b = d.get(key_b or key_a, default) if d else default
        if isinstance(a, list) and len(a) == 2:
            return [a[0], a[1]]
        return [a, b]

    possession = [0.0, 0.0]
    if events_data:
        f0 = events_data.get("possession_frames_team_0") or events_data.get("possession_team_0")
        f1 = events_data.get("possession_frames_team_1") or events_data.get("possession_team_1")
        try:
            f0, f1 = float(f0 or 0), float(f1 or 0)
            tot = f0 + f1
            if tot > 0:
                possession = [round(f0 / tot, 3), round(f1 / tot, 3)]
        except Exception:
            pass

    pass_acc = [0.0, 0.0]
    if pass_data:
        for i, key in enumerate(("team_0", "team_1")):
            blk = pass_data.get(key) or {}
            try:
                pass_acc[i] = round(float(blk.get("completion_pct") or blk.get("completionPct") or 0.0), 3)
            except Exception:
                pass

    pressures = [0, 0]
    if pressing_data:
        for i, key in enumerate(("team_0", "team_1")):
            blk = pressing_data.get(key) or {}
            try:
                pressures[i] = int(blk.get("pressures") or blk.get("count") or 0)
            except Exception:
                pass

    distance = [0.0, 0.0]
    sprints = [0, 0]
    if heatmap_data:
        for player in heatmap_data.get("players", []):
            team = int(player.get("teamId", -1)) if "teamId" in player else -1
            if team in (0, 1):
                try:
                    distance[team] += float(player.get("distance") or 0.0)
                    sprints[team] += int(player.get("sprint_count") or 0)
                except Exception:
                    pass
        distance = [round(distance[0] / 1000.0, 2), round(distance[1] / 1000.0, 2)]

    xg_pair = [0.0, 0.0]
    big_chances = [0, 0]
    shot_acc = [0.0, 0.0]
    if xg_data:
        shots_by_team: Dict[int, List[dict]] = defaultdict(list)
        for s in xg_data.get("shots", []):
            t = int(s.get("teamId", s.get("team_id", -1)))
            if t in (0, 1):
                shots_by_team[t].append(s)
        for t, shots in shots_by_team.items():
            xg_pair[t] = round(sum(float(s.get("xg") or 0.0) for s in shots), 2)
            big_chances[t] = sum(1 for s in shots if float(s.get("xg") or 0.0) >= 0.15)
            on_target = sum(1 for s in shots if s.get("on_target"))
            shot_acc[t] = round(on_target / max(len(shots), 1), 3)

    turnovers = [0, 0]
    if events_data:
        for e in events_data.get("turnover_events", []) or []:
            t = e.get("teamId", e.get("team_id"))
            if t in (0, 1):
                turnovers[int(t)] += 1

    return {
        "possessionPct": possession,
        "passAccuracy": pass_acc,
        "pressures": pressures,
        "distanceCoveredKm": distance,
        "sprints": sprints,
        "expectedGoals": xg_pair,
        "bigChances": big_chances,
        "shotAccuracy": shot_acc,
        "turnoversWon": turnovers,
    }


def _players_block(heatmap_data: Optional[dict], team_data: Optional[dict]) -> List[dict]:
    team_map: Dict[int, str] = {}
    if team_data:
        for entry in _team_entries(team_data):
            tid = entry.get("trackId") or entry.get("track_id")
            t = entry.get("teamId", entry.get("team_id"))
            if tid is not None and t in (0, 1):
                team_map[int(tid)] = "home" if t == 0 else "away"

    out: List[dict] = []
    if not heatmap_data:
        return out
    for p in heatmap_data.get("players", []):
        tid = p.get("trackId") or p.get("playerId")
        out.append({
            "playerId": p.get("playerId") or (f"P{tid}" if tid is not None else None),
            "team": team_map.get(int(tid)) if tid is not None else None,
            "name": None,
            "distanceM": round(float(p.get("distance") or 0.0), 1),
            "sprints": int(p.get("sprint_count") or 0),
            "topSpeedKmh": round(float(p.get("top_speed_kmh") or 0.0), 2),
        })
    return out


def _events_block(events_data: Optional[dict], fps: float) -> List[dict]:
    if not events_data:
        return []
    out: List[dict] = []
    for key in ("pass_events", "shot_events", "tackle_events", "turnover_events"):
        for e in events_data.get(key, []) or []:
            frame = e.get("frame") or e.get("frameIndex")
            if frame is None:
                continue
            ts = round(float(frame) / fps, 2) if fps else 0.0
            out.append({
                "frame": int(frame),
                "timestampSec": ts,
                "type": key.replace("_events", ""),
                "team": e.get("teamId", e.get("team_id")),
            })
    out.sort(key=lambda x: x["frame"])
    return out


def build_match_report(job_id: str) -> dict:
    """Read every per-feature JSON for `job_id` and return one match_report dict.
    Also writes temp/{job_id}/match_report.json next to the inputs."""
    base = Path(f"temp/{job_id}")

    track_data = _safe_load(base / "tracking" / "track_results.json")
    team_data = _safe_load(base / "tracking" / "team_results.json")
    pitch_data = _safe_load(base / "pitch" / "pitch_map.json")
    formation_data = _safe_load(base / "tactics" / "tactics_results.json") or _safe_load(base / "formation" / "formation.json")
    events_data = _safe_load(base / "events" / "event_timeline.json")
    pass_data = _safe_load(base / "pass_network" / "pass_network.json")
    pressing_data = _safe_load(base / "pressing" / "pressing.json")
    xg_data = _safe_load(base / "xg" / "xg_results.json")
    heatmap_data = _safe_load(base / "heatmap" / "heatmap.json")
    identity_metrics = _safe_load(base / "tracking" / "identity_metrics.json")

    video = _detect_video_meta(track_data)
    fps = video["fps"] or 25.0
    team_entries = _team_entries(team_data)

    report = {
        "jobId": job_id,
        "video": video,
        "scoreline": {"home": None, "away": None},
        "quality": _quality(identity_metrics, pitch_data, track_data, video, team_entries),
        "teams": _teams_block(formation_data, team_data),
        "metrics": _metrics_block(pass_data, pressing_data, xg_data, heatmap_data, events_data),
        "players": _players_block(heatmap_data, team_data),
        "events": _events_block(events_data, fps),
        "tacticsClips": [],   # populated by phase 3
    }

    out_path = base / "match_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report
