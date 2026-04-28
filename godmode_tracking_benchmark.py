"""
╔══════════════════════════════════════════════════════════════════╗
║   ATHLINK — GODMODE TRACKING BENCHMARK                          ║
║   RF-DETR + OSNet ReID × 4 trackers — head-to-head race        ║
║                                                                  ║
║   Runs ALL 4 trackers on your clip and prints a winner table    ║
║   The metric that matters for football: unique_ids ≈ 22-26      ║
║   Lower unique_ids = better Re-ID = fewer ghost players         ║
╚══════════════════════════════════════════════════════════════════╝

INSTALL (one-time):
    pip install boxmot rfdetr supervision opencv-python Pillow

RUN:
    python godmode_tracking_benchmark.py

OUTPUT:
    - 4 annotated videos  →  ./runs/<tracker_name>/output.mp4
    - JSON results        →  ./runs/benchmark_results.json
    - Terminal leaderboard at the end
"""

import cv2
import numpy as np
import json
import time
import os
from pathlib import Path
from PIL import Image as PILImage

# ─── CONFIG ──────────────────────────────────────────────────────
VIDEO_PATH = "/Users/rudra/Downloads/Aston villa vs Psg clip 1.mov"
OUTPUT_DIR = Path("./runs")
MAX_FRAMES  = 500          # ~20 seconds at 25fps — stress test for ID persistence
CONF_THRESH = 0.60         # Increased to 0.60 to eliminate crowd (requires Large model)
PERSON_CLASS = 1           # RF-DETR class 1 = person

# ReID model — auto-downloads on first run (~5MB)
REID_MODEL = "osnet_x0_25_msmt17.pt"

# The 4 trackers to race — ranked by expected football performance
TRACKERS = [
    {
        "name": "botsort",
        "label": "BoT-SORT + OSNet",
        "note": "Re-ID features are critical for remembering players out of frame",
        "max_age": 90,
    },
]
# ─────────────────────────────────────────────────────────────────


def load_model():
    """Initialises RF-DETR strictly on CPU to prevent backend crashes on Mac."""
    from rfdetr import RFDETRLarge
    print("[GODMODE] Loading RF-DETR Large model (CPU)...")
    model = RFDETRLarge(device="cpu")
    return model


def build_tracker(name: str, max_age: int):
    """Returns a BoxMOT tracker instance (boxmot 12.x API)."""
    import boxmot
    reid = Path(REID_MODEL)
    device = "cpu"  # change to "mps" or "cuda:0" if available

    tracker_map = {
        "botsort":    lambda: boxmot.BotSort(reid, device, False),
    }

    if name not in tracker_map:
        raise ValueError(f"Unknown tracker: {name}")

    return tracker_map[name]()


def rfdetr_to_numpy(detections, frame_w, frame_h):
    """
    Converts sv.Detections → N×6 numpy array [x1,y1,x2,y2,conf,cls]
    which is what BoxMOT trackers expect.
    """
    if detections is None or len(detections) == 0:
        return np.empty((0, 6))

    mask = detections.class_id == PERSON_CLASS
    if not np.any(mask):
        return np.empty((0, 6))

    boxes = detections.xyxy[mask]
    confs = detections.confidence[mask]
    cls   = detections.class_id[mask]

    dets = np.column_stack([boxes, confs, cls]).astype(float)
    return dets


def draw_frame(frame, tracks, tracker_name, frame_idx, fps, unique_ids, total_frames):
    """Annotate frame with bounding boxes, IDs, trails."""
    annotated = frame.copy()

    if tracks is not None and len(tracks) > 0:
        for track in tracks:
            x1, y1, x2, y2 = track[:4].astype(int)
            tid = int(track[4])
            # conf = float(track[5]) if len(track) > 5 else 0.0

            # Colour per ID (deterministic)
            color_idx = tid % 20
            colors = [
                (255,80,80),(80,255,80),(80,80,255),(255,255,80),(255,80,255),
                (80,255,255),(255,160,0),(0,255,160),(160,0,255),(0,160,255),
                (255,100,150),(100,255,150),(150,100,255),(255,200,100),(200,100,255),
                (100,200,255),(200,255,100),(255,150,200),(150,200,255),(200,150,100),
            ]
            color = colors[color_idx]

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"#{tid}"
            cv2.putText(annotated, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # HUD
    hud = (f"{tracker_name}  |  frame {frame_idx}/{total_frames}"
           f"  |  tracks: {len(tracks) if tracks is not None else 0}"
           f"  |  unique IDs so far: {unique_ids}"
           f"  |  {fps:.1f} FPS")
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(annotated, hud, (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return annotated


def run_tracker(model, tracker_cfg):
    name  = tracker_cfg["name"]
    label = tracker_cfg["label"]
    out_dir = OUTPUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "output.mp4"

    print(f"\n{'═'*60}")
    print(f"  {label}")
    print(f"  {tracker_cfg['note']}")
    print(f"{'═'*60}")

    tracker = build_tracker(name, tracker_cfg["max_age"])

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps_src  = cap.get(cv2.CAP_PROP_FPS)
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total    = min(MAX_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_src, (width, height)
    )

    unique_ids    = set()
    id_switches   = 0
    prev_ids      = set()
    frame_times   = []
    det_counts    = []
    track_counts  = []
    id_history    = {}   # tid → list of frame indices seen

    for frame_idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()

        # ── Detect ────────────────────────────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = PILImage.fromarray(rgb)
        detections = model.predict(pil, threshold=CONF_THRESH)
        dets_np = rfdetr_to_numpy(detections, width, height)
        det_counts.append(len(dets_np))

        # ── Track ─────────────────────────────────────────────────
        if len(dets_np) > 0:
            tracks = tracker.update(dets_np, frame)
        else:
            tracks = tracker.update(np.empty((0, 6)), frame)

        # ── Stats ──────────────────────────────────────────────────
        curr_ids = set()
        if tracks is not None and len(tracks) > 0:
            for t in tracks:
                tid = int(t[4])
                curr_ids.add(tid)
                unique_ids.add(tid)
                if tid not in id_history:
                    id_history[tid] = []
                id_history[tid].append(frame_idx)

        # crude ID switch count: IDs that were present, disappeared, reappeared
        new_ids = curr_ids - prev_ids
        if frame_idx > 0:
            id_switches += len(new_ids - (unique_ids - curr_ids))
        prev_ids = curr_ids
        track_counts.append(len(curr_ids))

        elapsed = time.perf_counter() - t0
        frame_times.append(elapsed)
        avg_fps = 1.0 / (sum(frame_times[-20:]) / min(len(frame_times), 20))

        # ── Annotate & write ──────────────────────────────────────
        annotated = draw_frame(frame, tracks, label, frame_idx+1,
                               avg_fps, len(unique_ids), total)
        writer.write(annotated)

        if frame_idx % 25 == 0:
            print(f"  [{frame_idx+1:>3}/{total}]"
                  f"  detections: {len(dets_np):>2}"
                  f"  active tracks: {len(curr_ids):>2}"
                  f"  unique IDs: {len(unique_ids):>3}"
                  f"  {avg_fps:>5.1f} fps")

    cap.release()
    writer.release()

    # Compute track fragmentation: how many tracks cover the same region
    # (rough proxy for ID explosion)
    avg_track_len = (
        np.mean([len(v) for v in id_history.values()])
        if id_history else 0
    )

    result = {
        "tracker":          label,
        "name":             name,
        "unique_ids":       len(unique_ids),
        "avg_active_tracks": float(np.mean(track_counts)) if track_counts else 0,
        "avg_track_length_frames": float(avg_track_len),
        "avg_fps":          float(1.0 / np.mean(frame_times)),
        "avg_detections":   float(np.mean(det_counts)) if det_counts else 0,
        "output":           str(out_path),
        "note":             tracker_cfg["note"],
    }

    print(f"\n  ✓ Done  |  unique IDs: {result['unique_ids']}"
          f"  |  avg track len: {result['avg_track_length_frames']:.1f} frames"
          f"  |  {result['avg_fps']:.1f} FPS")
    return result


def print_leaderboard(results):
    # Score: lower unique_ids + longer avg track length = better
    # Target: ~22-26 unique IDs for a full 11v11 clip
    TARGET = 24

    print(f"\n{'═'*68}")
    print(f"  GODMODE LEADERBOARD — target unique IDs ≈ 22-26")
    print(f"{'═'*68}")
    print(f"  {'Tracker':<28}  {'IDs':>5}  {'Δ target':>8}  {'Avg len':>8}  {'FPS':>6}")
    print(f"  {'-'*28}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*6}")

    sorted_r = sorted(results, key=lambda r: abs(r["unique_ids"] - TARGET))

    for i, r in enumerate(sorted_r):
        medal = ["🥇", "🥈", "🥉", " 4"][min(i, 3)]
        delta = r["unique_ids"] - TARGET
        delta_s = f"+{delta}" if delta >= 0 else str(delta)
        print(f"  {medal} {r['tracker']:<26}  {r['unique_ids']:>5}  "
              f"{delta_s:>8}  {r['avg_track_length_frames']:>7.1f}f  "
              f"{r['avg_fps']:>5.1f}")

    winner = sorted_r[0]
    print(f"\n  WINNER → {winner['tracker']}")
    print(f"  {winner['unique_ids']} unique IDs | "
          f"{winner['avg_track_length_frames']:.0f} frames avg track length")
    print(f"\n  WHAT TO DO NEXT:")
    print(f"  1. Open ./runs/{winner['name']}/output.mp4 — watch for stable IDs")
    print(f"  2. If IDs still > 30 → run:  python godmode_tuner.py")
    print(f"  3. If IDs ≈ 22-26  → integrate '{winner['name']}' into your backend")
    print(f"{'═'*68}\n")


def main():
    if not Path(VIDEO_PATH).exists():
        print(f"\n[ERROR] Video not found: {VIDEO_PATH}")
        print("        Update VIDEO_PATH at the top of this script.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'═'*60}")
    print(f"  ATHLINK GODMODE TRACKING BENCHMARK")
    print(f"  4 trackers × RF-DETR + OSNet Re-ID")
    print(f"  {MAX_FRAMES} frames  |  target IDs ≈ 22-26")
    print(f"{'═'*60}")

    model = load_model()
    results = []

    for cfg in TRACKERS:
        try:
            r = run_tracker(model, cfg)
            results.append(r)
        except Exception as e:
            print(f"\n[SKIP] {cfg['label']} failed: {e}")
            import traceback
            traceback.print_exc()

    # Save JSON
    json_path = OUTPUT_DIR / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[GODMODE] Results saved → {json_path}")

    if results:
        print_leaderboard(results)


if __name__ == "__main__":
    main()
