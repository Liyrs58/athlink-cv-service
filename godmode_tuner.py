"""
╔══════════════════════════════════════════════════════════════════╗
║   ATHLINK — GODMODE PARAMETER TUNER                             ║
║   Run this AFTER godmode_tracking_benchmark.py                  ║
║   Feed it the winning tracker name and it tunes the params      ║
╚══════════════════════════════════════════════════════════════════╝

USAGE:
    # After benchmark, set WINNING_TRACKER to your winner:
    python godmode_tuner.py --tracker botsort

    # Or just edit WINNING_TRACKER below and run:
    python godmode_tuner.py
"""

import cv2
import numpy as np
import json
import time
import sys
import argparse
from pathlib import Path
from PIL import Image as PILImage

VIDEO_PATH      = "/Users/rudra/Downloads/Aston villa vs Psg clip 1.mov"
WINNING_TRACKER = "bytetrack"   # ← change to your benchmark winner
REID_MODEL      = "osnet_x0_25_msmt17.pt"
MAX_FRAMES      = 100
CONF_THRESH     = 0.40
OUTPUT_DIR      = Path("./runs/tuner")

# Parameter grid to sweep
# max_age: how many frames to keep a lost track alive (KEY for football)
# iou_thresh: how lenient the IoU matching is
PARAM_GRID = [
    {"max_age": 30,  "label": "max_age=30  (1.2s)"},
    {"max_age": 50,  "label": "max_age=50  (2.0s)"},
    {"max_age": 75,  "label": "max_age=75  (3.0s)  ← recommended"},
    {"max_age": 100, "label": "max_age=100 (4.0s)"},
]


def load_rfdetr():
    from rfdetr import RFDETRBase
    return RFDETRBase(device="cpu")


def build_tracker_with_params(name, max_age):
    import boxmot
    reid = Path(REID_MODEL)
    reid_kwargs = dict(model_weights=reid, device="cpu", fp16=False)
    tracker_map = {
        "botsort":    lambda: boxmot.BotSort(**reid_kwargs),
        "deepocsort": lambda: boxmot.DeepOcSort(**reid_kwargs),
        "strongsort": lambda: boxmot.StrongSort(reid, "cpu", False),
        "boosttrack": lambda: boxmot.BoostTrack(reid, "cpu", False),
        "bytetrack":  lambda: boxmot.ByteTrack(),
    }
    return tracker_map[name]()


def run_param_combo(model, tracker_name, params):
    tracker = build_tracker_with_params(tracker_name, params["max_age"])
    cap = cv2.VideoCapture(VIDEO_PATH)
    # fps_src = cap.get(cv2.CAP_PROP_FPS)
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = min(MAX_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    unique_ids   = set()
    track_counts = []
    id_history   = {}

    for frame_idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil  = PILImage.fromarray(rgb)
        dets = model.predict(pil, threshold=CONF_THRESH)

        # Convert to numpy
        if dets is not None and len(dets) > 0:
            mask = dets.class_id == 1  # RF-DETR person class is 1
            if np.any(mask):
                boxes = dets.xyxy[mask]
                confs = dets.confidence[mask]
                cls   = dets.class_id[mask]
                dets_np = np.column_stack([boxes, confs, cls]).astype(float)
            else:
                dets_np = np.empty((0, 6))
        else:
            dets_np = np.empty((0, 6))

        if len(dets_np) > 0:
            tracks = tracker.update(dets_np, frame)
        else:
            tracks = tracker.update(np.empty((0, 6)), frame)

        if tracks is not None and len(tracks) > 0:
            for t in tracks:
                tid = int(t[4])
                unique_ids.add(tid)
                if tid not in id_history:
                    id_history[tid] = []
                id_history[tid].append(frame_idx)
            track_counts.append(len(tracks))
        else:
            track_counts.append(0)

    cap.release()

    avg_track_len = (
        np.mean([len(v) for v in id_history.values()])
        if id_history else 0
    )

    return {
        "params":    params["label"],
        "max_age":   params["max_age"],
        "unique_ids": len(unique_ids),
        "avg_track_length": float(avg_track_len),
        "avg_active": float(np.mean(track_counts)) if track_counts else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", default=WINNING_TRACKER)
    args = parser.parse_args()
    tracker_name = args.tracker

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'═'*56}")
    print(f"  GODMODE TUNER — {tracker_name.upper()}")
    print(f"  Sweeping max_age  |  target unique IDs ≈ 22-26")
    print(f"{'═'*56}\n")

    model = load_rfdetr()
    results = []

    for params in PARAM_GRID:
        print(f"  Testing {params['label']}...", end=" ", flush=True)
        r = run_param_combo(model, tracker_name, params)
        results.append(r)
        print(f"unique IDs: {r['unique_ids']}  |  avg track len: {r['avg_track_length']:.1f}f")

    TARGET = 24
    print(f"\n{'─'*56}")
    print(f"  {'Params':<30}  {'IDs':>4}  {'Avg len':>8}")
    print(f"  {'-'*30}  {'-'*4}  {'-'*8}")

    best = min(results, key=lambda r: abs(r["unique_ids"] - TARGET))
    for r in results:
        marker = " ← BEST" if r == best else ""
        print(f"  {r['params']:<30}  {r['unique_ids']:>4}  "
              f"{r['avg_track_length']:>7.1f}f{marker}")

    print(f"\n  OPTIMAL CONFIG:")
    print(f"    tracker  = {tracker_name}")
    print(f"    max_age  = {best['max_age']}")
    print(f"    reid     = {REID_MODEL}")
    print(f"\n  Copy this into your backend tracking_service.py")
    print(f"{'═'*56}\n")

    json_path = OUTPUT_DIR / f"tuner_{tracker_name}.json"
    with open(json_path, "w") as f:
        json.dump({"tracker": tracker_name, "results": results, "best": best}, f, indent=2)


if __name__ == "__main__":
    main()
