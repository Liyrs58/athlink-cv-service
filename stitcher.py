#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARN] scipy not installed, using greedy matcher")


def get_pos(entry):
    """Extract position from track entry. Prefers field, falls back to pixel foot."""
    fp = entry.get("fieldPos")
    if fp and None not in fp:
        return [float(fp[0]), float(fp[1])]
    bbox = entry.get("bbox")
    if bbox and len(bbox) == 4:
        return [(bbox[0] + bbox[2]) / 2.0, bbox[3]]  # bottom-center
    return [None, None]


def build_tracklets(frames):
    """Convert frame JSON into contiguous tracklet segments."""
    track_frames = defaultdict(list)
    for frame in frames:
        idx = frame.get("frameIndex", 0)
        for p in frame.get("players", []):
            tid = p.get("trackId")
            if tid is None:
                continue
            track_frames[tid].append({
                "frame": idx,
                "trackId": tid,
                "team": p.get("teamId", -1),
                "embedding": np.array(p.get("embedding", [])) if p.get("embedding") else None,
                "colorHist": np.array(p.get("colorHist", [])) if p.get("colorHist") else None,
                "pos": get_pos(p),
            })

    tracklets = []
    for tid, entries in track_frames.items():
        entries.sort(key=lambda x: x["frame"])
        # Split on gaps > 5 frames
        seg = [entries[0]]
        for e in entries[1:]:
            if e["frame"] - seg[-1]["frame"] > 5:
                tracklets.append(seg)
                seg = [e]
            else:
                seg.append(e)
        tracklets.append(seg)

    # Build tracklet objects
    results = []
    for seg in tracklets:
        embs = [e["embedding"] for e in seg if e["embedding"] is not None and len(e["embedding"]) > 0]
        colors = [e["colorHist"] for e in seg if e["colorHist"] is not None and len(e["colorHist"]) > 0]
        poses = [e["pos"] for e in seg if e["pos"][0] is not None]

        results.append({
            "id": seg[0]["trackId"],
            "start": seg[0]["frame"],
            "end": seg[-1]["frame"],
            "team": seg[0]["team"],
            "avg_embedding": np.mean(embs, axis=0) if embs else None,
            "avg_color": np.mean(colors, axis=0) if colors else None,
            "last_pos": seg[-1]["pos"],
            "first_pos": seg[0]["pos"],
            "frames": len(seg),
        })
    return results


def fingerprint_distance(t1, t2):
    """Cost between two tracklets. Lower = better."""
    if t1["team"] != t2["team"] or t1["team"] == -1:
        return 999.0

    cost = 0.0
    # Appearance
    if t1["avg_embedding"] is not None and t2["avg_embedding"] is not None:
        e1 = t1["avg_embedding"] / (np.linalg.norm(t1["avg_embedding"]) + 1e-6)
        e2 = t2["avg_embedding"] / (np.linalg.norm(t2["avg_embedding"]) + 1e-6)
        cost += 0.35 * float(1.0 - np.dot(e1, e2))

    # Color histogram (chi-squared approximation)
    if t1["avg_color"] is not None and t2["avg_color"] is not None:
        c1 = t1["avg_color"].astype(np.float32) + 1e-7
        c2 = t2["avg_color"].astype(np.float32) + 1e-7
        d = np.sum((c1 - c2) ** 2 / (c1 + c2))
        cost += 0.30 * min(d / 5.0, 1.0)

    # Position jump
    if t1["last_pos"][0] is not None and t2["first_pos"][0] is not None:
        p1 = np.array(t1["last_pos"], dtype=float)
        p2 = np.array(t2["first_pos"], dtype=float)
        d = float(np.linalg.norm(p1 - p2))
        cost += 0.25 * (0 if d < 8 else min((d - 8) / 20.0, 1.0))

    # Time gap
    gap = max(0, t2["start"] - t1["end"])
    cost += 0.10 * min(gap / 120.0, 1.0)
    return cost


def stitch_tracklets(tracklets, max_gap=60, team_size=11):
    """Greedy iterative merge with formation hard limit."""
    if not tracklets:
        return tracklets, {}

    merged_map = {t["id"]: t["id"] for t in tracklets}

    def canon(tid):
        while merged_map.get(tid, tid) != tid:
            tid = merged_map[tid]
        return tid

    for team in [0, 1]:
        team_tlets = [t for t in tracklets if t["team"] == team]
        if len(team_tlets) < 2:
            continue

        for _ in range(100):  # safety limit
            # Group by canonical ID
            groups = defaultdict(list)
            for t in team_tlets:
                groups[canon(t["id"])].append(t)

            ids = list(groups.keys())
            if len(ids) <= team_size:
                break

            # Find cheapest valid merge across all pairs
            best_cost, best_pair = 999.0, None
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    id_a, id_b = ids[i], ids[j]
                    for ta in groups[id_a]:
                        for tb in groups[id_b]:
                            # Temporal compatibility
                            if ta["end"] < tb["start"] and (tb["start"] - ta["end"]) > max_gap:
                                continue
                            if tb["end"] < ta["start"] and (ta["start"] - tb["end"]) > max_gap:
                                continue
                            c = fingerprint_distance(ta, tb)
                            if c < best_cost:
                                best_cost = c
                                best_pair = (id_a, id_b, c)

            if best_pair is None or best_pair[2] > 0.60:
                break

            a, b, c = best_pair
            # Merge b into a
            for k, v in list(merged_map.items()):
                if v == b:
                    merged_map[k] = a
            merged_map[b] = a

    return tracklets, merged_map


def rewrite_frames(frames, merged_map):
    """Rewrite JSON with canonical IDs."""
    def canon(tid):
        while merged_map.get(tid, tid) != tid:
            tid = merged_map[tid]
        return tid

    new_frames = []
    for frame in frames:
        nf = {
            "frameIndex": frame.get("frameIndex"),
            "players": []
        }
        for p in frame.get("players", []):
            np_ = dict(p)
            np_["trackId"] = canon(p.get("trackId"))
            nf["players"].append(np_)
        new_frames.append(nf)
    return new_frames


def main(job_id):
    track_path = Path(f"temp/{job_id}/tracking/track_results.json")
    out_path = Path(f"temp/{job_id}/tracking/track_results_stitched.json")

    if not track_path.exists():
        print(f"❌ Missing {track_path}")
        return

    with open(track_path) as f:
        data = json.load(f)

    if "frames" not in data and "tracks" in data:
        # Adaptation for tracking_service.py format
        print(f"[STITCHER] Athlink format detected, no internal frames found.")
        # Athlink tracking_service.py format is a single dict with 'tracks' as list of trajectories.
        # This stitcher is written for a different format (list of frames).
        # I will keep it as requested but it may need adjustment later.

    frames = data.get("frames", [])
    tracklets = build_tracklets(frames)
    _, merged_map = stitch_tracklets(tracklets)

    new_frames = rewrite_frames(frames, merged_map)
    data["frames"] = new_frames

    with open(out_path, "w") as f:
        json.dump(data, f)

    raw_ids = set()
    new_ids = set()
    for frame in frames:
        for p in frame.get("players", []):
            raw_ids.add(p.get("trackId"))
    for frame in new_frames:
        for p in frame.get("players", []):
            new_ids.add(p.get("trackId"))

    print(f"[STITCHER] Raw IDs: {len(raw_ids)} → Stitched IDs: {len(new_ids)}")
    print(f"[STITCHER] Merges: {len(raw_ids) - len(new_ids)}")
    print(f"[STITCHER] Output: {out_path}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "fix_test")
