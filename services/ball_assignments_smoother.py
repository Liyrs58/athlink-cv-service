"""Temporal smoother for noisy ball_assignments.json carrier_tid streams.

The upstream ball-assignment producer (external to this repo) picks the nearest
player to the ball per frame. With <30cm ball jitter and 2-3 players standing
within 3m of each other, the carrier_tid flips between them every 1-2 frames.
This module collapses that noise via vote-window + sticky-tid hysteresis.

Public:
    smooth_carrier_assignments(carrier_by_frame, window=6) -> (smoothed, stats)
"""

from collections import Counter, defaultdict
from typing import Dict, Tuple


def smooth_carrier_assignments(
    carrier_by_frame: Dict[int, dict],
    window: int = 6,
) -> Tuple[Dict[int, dict], Dict[str, float]]:
    """Smooth carrier_tid via vote-window + sticky hysteresis.

    Args:
        carrier_by_frame: {frameIndex: {carrier_tid, carrier_confidence,
            ball_source, ball_image, ball_world, grace_frames, ...}}
        window: half-window size in frames. Total window = 2*window + 1.

    Returns:
        smoothed: same shape as input with smoothed carrier_tid +
            recomputed carrier_confidence + smoother_votes/_was_switched.
        stats: diagnostic counts.
    """
    if not carrier_by_frame:
        return {}, {
            "total_frames": 0,
            "frames_with_carrier": 0,
            "tid_switches_in_raw": 0,
            "tid_switches_in_smoothed": 0,
            "noise_reduction_ratio": 0.0,
        }

    frames = sorted(carrier_by_frame.keys())

    # ── Pass 1: vote winner per frame ────────────────────────────────────
    pass1: Dict[int, Tuple[int, int, float]] = {}  # fi -> (winner_tid, votes, mean_conf)
    for fi in frames:
        votes: Counter = Counter()
        confs: Dict[int, list] = defaultdict(list)
        total_voters = 0
        for df in range(-window, window + 1):
            entry = carrier_by_frame.get(fi + df)
            if entry is None:
                continue
            if entry.get("ball_source") == "missing":
                continue
            tid_raw = entry.get("carrier_tid")
            if tid_raw is None:
                continue
            tid = int(tid_raw)
            votes[tid] += 1
            confs[tid].append(float(entry.get("carrier_confidence", 0) or 0))
            total_voters += 1
        if not votes:
            pass1[fi] = (-1, 0, 0.0)
            continue
        winner_tid, winner_votes = votes.most_common(1)[0]
        mean_conf = sum(confs[winner_tid]) / max(1, len(confs[winner_tid]))
        # Confidence policy:
        #   - If winner has clear plurality (share >= 0.5): use mean_conf as-is
        #     — successful vote-collapse IS a confidence signal.
        #   - If contested (share < 0.5): damp by 2x dominance so it lands
        #     in the [0, mean_conf] range proportional to share.
        dominance = winner_votes / max(1, total_voters)
        if dominance >= 0.5:
            damped_conf = mean_conf
        else:
            damped_conf = mean_conf * (2 * dominance)
        pass1[fi] = (winner_tid, winner_votes, damped_conf)

    # ── Pass 2: sticky hysteresis ────────────────────────────────────────
    smoothed: Dict[int, dict] = {}
    prev_tid = -1
    prev_votes = 0
    raw_switches = 0
    smoothed_switches = 0

    last_raw_tid = -1
    for fi in frames:
        # Track raw switches for stats
        raw_entry = carrier_by_frame[fi]
        raw_tid_val = raw_entry.get("carrier_tid")
        if raw_tid_val is not None:
            raw_tid = int(raw_tid_val)
            if last_raw_tid != -1 and raw_tid != last_raw_tid:
                raw_switches += 1
            last_raw_tid = raw_tid

        winner_tid, winner_votes, winner_conf = pass1[fi]
        was_switched = False

        if winner_tid == -1:
            # No data in window — passthrough
            chosen_tid = prev_tid if prev_tid != -1 else (
                int(raw_tid_val) if raw_tid_val is not None else -1
            )
            chosen_votes = 0
            chosen_conf = float(raw_entry.get("carrier_confidence", 0) or 0)
        elif winner_tid == prev_tid:
            chosen_tid = winner_tid
            chosen_votes = winner_votes
            chosen_conf = winner_conf
        elif prev_tid == -1:
            # First valid frame — accept the winner
            chosen_tid = winner_tid
            chosen_votes = winner_votes
            chosen_conf = winner_conf
            was_switched = True
        elif winner_votes >= prev_votes + 3:
            # Genuine handover: new carrier wins by ≥3 votes
            chosen_tid = winner_tid
            chosen_votes = winner_votes
            chosen_conf = winner_conf
            was_switched = True
        else:
            # Resist the flip — keep prev_tid
            chosen_tid = prev_tid
            chosen_votes = prev_votes
            # Pull current frame's confidence for prev_tid if it voted, else damp
            prev_in_window_confs = []
            valid_voters_here = 0
            for df in range(-window, window + 1):
                e = carrier_by_frame.get(fi + df)
                if e is None or e.get("ball_source") == "missing":
                    continue
                if e.get("carrier_tid") is None:
                    continue
                valid_voters_here += 1
                if int(e["carrier_tid"]) == prev_tid:
                    prev_in_window_confs.append(float(e.get("carrier_confidence", 0) or 0))
            if prev_in_window_confs:
                mean_prev = sum(prev_in_window_confs) / len(prev_in_window_confs)
                share = len(prev_in_window_confs) / max(1, valid_voters_here)
                chosen_conf = mean_prev * share
            else:
                chosen_conf = max(0.30, winner_conf * 0.5)  # coasting floor

        if was_switched and prev_tid != -1:
            smoothed_switches += 1

        out = dict(raw_entry)
        out["frameIndex"] = fi
        out["carrier_tid"] = chosen_tid if chosen_tid != -1 else None
        out["carrier_confidence"] = round(chosen_conf, 4)
        out["smoother_votes"] = chosen_votes
        out["smoother_was_switched"] = was_switched
        smoothed[fi] = out

        if chosen_tid != -1:
            prev_tid = chosen_tid
            prev_votes = chosen_votes

    # ── Stats ────────────────────────────────────────────────────────────
    frames_with_carrier = sum(
        1 for v in smoothed.values() if v.get("carrier_tid") is not None
    )
    noise_reduction = (
        1.0 - (smoothed_switches / raw_switches) if raw_switches > 0 else 0.0
    )
    stats = {
        "total_frames": len(frames),
        "frames_with_carrier": frames_with_carrier,
        "tid_switches_in_raw": raw_switches,
        "tid_switches_in_smoothed": smoothed_switches,
        "noise_reduction_ratio": round(noise_reduction, 3),
        "window": window,
    }
    return smoothed, stats
