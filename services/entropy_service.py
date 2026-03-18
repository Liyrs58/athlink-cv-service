import math
from typing import List, Dict, Any, Optional

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
# Divide pitch into 6x4 zones for entropy calculation
ZONES_X = 6
ZONES_Y = 4


def _position_to_zone(wx: float, wy: float) -> int:
    """Maps world position to zone index 0-(ZONES_X*ZONES_Y-1)."""
    zx = min(int(wx / PITCH_LENGTH * ZONES_X), ZONES_X - 1)
    zy = min(int(wy / PITCH_WIDTH * ZONES_Y), ZONES_Y - 1)
    return zx * ZONES_Y + zy


def _shannon_entropy(zone_counts: Dict[int, int], n_players: int) -> float:
    """
    Computes Shannon entropy of player distribution across zones.
    H = -sum(p * log2(p))
    Max entropy = log2(n_zones) when perfectly spread.
    Returns normalised entropy 0-1.
    """
    if n_players == 0:
        return 0.0
    n_zones = ZONES_X * ZONES_Y
    entropy = 0.0
    for count in zone_counts.values():
        if count > 0:
            p = count / n_players
            entropy -= p * math.log2(p)
    max_entropy = math.log2(n_zones)
    return round(entropy / max_entropy, 3) if max_entropy > 0 else 0.0


def compute_team_entropy(
    tracks: List[Dict],
    frame_metadata: List[Dict],
    calibration: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Computes Shannon entropy of team shape over time.
    Low entropy = organised. High entropy = chaotic.
    Tracks entropy trend — rising entropy = losing shape.
    """
    frame_indices = sorted(set(m.get("frameIndex", 0) for m in frame_metadata))
    sampled = frame_indices[::5] if len(frame_indices) > 5 else frame_indices

    entropy_timeline = []

    for fi in sampled:
        team_positions = {0: [], 1: []}
        for t in tracks:
            if t.get("is_staff", False):
                continue
            team_id = t.get("teamId", -1)
            if team_id not in [0, 1]:
                continue
            traj = t.get("trajectory", [])
            entry = min(
                (e for e in traj if abs(e.get("frameIndex", 0) - fi) <= 4),
                key=lambda e: abs(e.get("frameIndex", 0) - fi),
                default=None,
            )
            if entry and "world_x" in entry:
                team_positions[team_id].append(
                    (entry["world_x"], entry["world_y"])
                )

        frame_entry = {"frame": fi}
        for tid in [0, 1]:
            positions = team_positions[tid]
            if len(positions) < 3:
                frame_entry[f"team_{tid}_entropy"] = None
                continue
            zone_counts: Dict[int, int] = {}
            for wx, wy in positions:
                zone = _position_to_zone(wx, wy)
                zone_counts[zone] = zone_counts.get(zone, 0) + 1
            frame_entry[f"team_{tid}_entropy"] = _shannon_entropy(
                zone_counts, len(positions)
            )
        entropy_timeline.append(frame_entry)

    if not entropy_timeline:
        return {"status": "insufficient_data"}

    # Average entropy per team
    def avg_entropy(team_id):
        vals = [
            f[f"team_{team_id}_entropy"]
            for f in entropy_timeline
            if f.get(f"team_{team_id}_entropy") is not None
        ]
        return round(sum(vals) / len(vals), 3) if vals else None

    # Entropy trend — compare first third vs last third
    def entropy_trend(team_id):
        vals = [
            f[f"team_{team_id}_entropy"]
            for f in entropy_timeline
            if f.get(f"team_{team_id}_entropy") is not None
        ]
        if len(vals) < 6:
            return "stable"
        third = len(vals) // 3
        first_avg = sum(vals[:third]) / third
        last_avg = sum(vals[-third:]) / third
        delta = last_avg - first_avg
        if delta > 0.05:
            return "rising"   # losing shape
        elif delta < -0.05:
            return "falling"  # gaining shape
        return "stable"

    # Detect shape collapse moments
    # Shape collapse = entropy spikes above 0.85 (very high disorder)
    collapse_frames = []
    for f in entropy_timeline:
        for tid in [0, 1]:
            e = f.get(f"team_{tid}_entropy")
            if e is not None and e > 0.85:
                collapse_frames.append({
                    "frame": f["frame"],
                    "team": tid,
                    "entropy": e,
                })

    return {
        "status": "ok",
        "team_0": {
            "avg_entropy": avg_entropy(0),
            "trend": entropy_trend(0),
            "interpretation": _interpret_entropy(avg_entropy(0), entropy_trend(0)),
        },
        "team_1": {
            "avg_entropy": avg_entropy(1),
            "trend": entropy_trend(1),
            "interpretation": _interpret_entropy(avg_entropy(1), entropy_trend(1)),
        },
        "shape_collapse_events": collapse_frames[:10],
        "frames_analysed": len(entropy_timeline),
    }


def _interpret_entropy(avg: Optional[float], trend: str) -> str:
    """Human-readable interpretation for coaches."""
    if avg is None:
        return "Insufficient data"
    if avg < 0.4:
        base = "Very organised, compact shape"
    elif avg < 0.6:
        base = "Structured shape with good coverage"
    elif avg < 0.75:
        base = "Moderate organisation, some gaps"
    else:
        base = "Spread out, limited structural cohesion"

    if trend == "rising":
        base += " — shape breaking down over time"
    elif trend == "falling":
        base += " — shape improving over time"

    return base
