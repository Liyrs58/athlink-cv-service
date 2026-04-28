"""
Athlink — RF-DETR + OC-SORT Standalone Test
============================================
Run this COMPLETELY independently of your FastAPI backend.
It reads your .mov clip, runs RF-DETR detection + OC-SORT tracking,
and writes an annotated output video so you can visually validate
before touching a single line of your pipeline.

Usage:
    pip install rfdetr supervision opencv-python
    python test_rfdetr_ocsort.py

Output:
    ./athlink_tracking_test_output.mp4
"""

import cv2
import numpy as np
import supervision as sv
from rfdetr import RFDETRLarge
from pathlib import Path
import time
import torch

device = "cpu"
print(f"[Athlink] Forcing device: {device}")

VIDEO_PATH = "/Users/rudra/Downloads/Aston villa vs Psg clip 1.mov"
OUTPUT_PATH = "./athlink_tracking_test_output.mp4"
DATA_PATH = "./athlink_tracking_data.json"

# RFDETRLarge = 129M params, best accuracy for football occlusion
# Swap to RFDETRBase (29M) if you want faster inference on weaker GPU
MODEL_SIZE = "base"

# Only track 'person' class from COCO (class_id = 0)
# Set to None to track everything (ball, etc.)
TRACK_CLASS_IDS = None

# Confidence threshold — 0.4 catches more players, 0.6 is stricter
CONFIDENCE_THRESHOLD = 0.20

# How many frames to process (None = full video)
MAX_FRAMES = None

# ─── SETUP ───────────────────────────────────────────────────────────────────

def load_model(size: str):
    print(f"\n[Athlink] Loading RF-DETR {size.upper()}...")
    if size == "large":
        from rfdetr import RFDETRLarge
        return RFDETRLarge(device="cpu")
    else:
        from rfdetr import RFDETRBase
        return RFDETRBase(device="cpu")

def build_tracker():
    """
    Using supervision's ByteTrack with version-appropriate arguments.
    """
    print("[Athlink] Using ByteTrack tracker")
    return sv.ByteTrack(
        track_activation_threshold=0.45,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
    )

# ─── ANNOTATORS ──────────────────────────────────────────────────────────────

def build_annotators():
    """Distinct colours per player ID, trajectory trails included."""
    box = sv.BoxAnnotator(
        thickness=2,
    )
    label = sv.LabelAnnotator(
        text_scale=0.5,
        text_thickness=1,
        text_padding=3,
    )
    trace = sv.TraceAnnotator(
        thickness=2,
        trace_length=40,   # frames of trail to draw — shows movement paths
    )
    return box, label, trace

# ─── MAIN LOOP ───────────────────────────────────────────────────────────────

def run_test():
    # Validate input
    if not Path(VIDEO_PATH).exists():
        print(f"\n[ERROR] Video not found at: {VIDEO_PATH}")
        print("        Update VIDEO_PATH at the top of this script.")
        return

    model = load_model(MODEL_SIZE)
    tracker = build_tracker()
    box_ann, label_ann, trace_ann = build_annotators()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {VIDEO_PATH}")
        return

    fps        = cap.get(cv2.CAP_PROP_FPS)
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    process_n  = total if MAX_FRAMES is None else min(MAX_FRAMES, total)

    print(f"\n[Athlink] Video:  {Path(VIDEO_PATH).name}")
    print(f"          Size:   {width}x{height}  |  FPS: {fps:.1f}")
    print(f"          Frames: {process_n} / {total}")
    print(f"          Output: {OUTPUT_PATH}\n")

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_times = []
    active_ids  = set()
    tracking_data = [] # List of frame-level detection data

    for frame_idx in range(process_n):
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()

        # ── Detection ──────────────────────────────────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        from PIL import Image as PILImage
        pil_frame = PILImage.fromarray(rgb)

        detections = model.predict(pil_frame, threshold=CONFIDENCE_THRESHOLD)

        # Filter to person class only (skip referee bounding boxes later)
        if TRACK_CLASS_IDS is not None and len(detections) > 0:
            mask = np.isin(detections.class_id, TRACK_CLASS_IDS)
            detections = detections[mask]

        # ── Tracking ───────────────────────────────────────────────────────
        tracked = tracker.update_with_detections(detections)

        # Collect active IDs for stats
        if tracked.tracker_id is not None:
            active_ids.update(tracked.tracker_id.tolist())
            
            # Save raw tracking data for this frame
            for i in range(len(tracked)):
                bbox = tracked.xyxy[i].tolist()
                tracking_data.append({
                    "frame": frame_idx,
                    "id": int(tracked.tracker_id[i]),
                    "bbox": bbox, # [x1, y1, x2, y2]
                    "conf": float(tracked.confidence[i]),
                    "class": int(tracked.class_id[i])
                })

        # ── Annotation ─────────────────────────────────────────────────────
        labels = [
            f"#{tid}  {conf:.0%}"
            for tid, conf in zip(
                tracked.tracker_id if tracked.tracker_id is not None else [],
                tracked.confidence if tracked.confidence is not None else [],
            )
        ]

        annotated = frame.copy()
        annotated = trace_ann.annotate(annotated, tracked)
        annotated = box_ann.annotate(annotated, tracked)
        annotated = label_ann.annotate(annotated, tracked, labels)

        # ── HUD overlay ────────────────────────────────────────────────────
        elapsed = time.perf_counter() - t0
        frame_times.append(elapsed)
        avg_fps = 1.0 / (sum(frame_times[-30:]) / min(len(frame_times), 30))

        cv2.putText(
            annotated,
            f"RF-DETR + OC-SORT  |  frame {frame_idx+1}/{process_n}"
            f"  |  detections: {len(tracked)}"
            f"  |  unique IDs so far: {len(active_ids)}"
            f"  |  {avg_fps:.1f} FPS",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(annotated)

        # Progress every 30 frames
        if frame_idx % 30 == 0:
            print(
                f"  frame {frame_idx+1:>4}/{process_n}"
                f"  |  tracked: {len(tracked):>2}"
                f"  |  unique IDs: {len(active_ids):>3}"
                f"  |  {avg_fps:>5.1f} fps"
            )

    cap.release()
    writer.release()

    avg_inf = sum(frame_times) / len(frame_times) * 1000
    print(f"\n[Athlink] Done!")
    print(f"          Avg inference+tracking: {avg_inf:.1f} ms/frame")
    print(f"          Total unique player IDs assigned: {len(active_ids)}")
    print(f"          Output saved → {OUTPUT_PATH}")

    # Save to JSON
    import json
    with open(DATA_PATH, "w") as f:
        json.dump(tracking_data, f, indent=4)
    print(f"          Tracking data saved → {DATA_PATH}")
    print(
        f"\n[Athlink] WHAT TO CHECK IN OUTPUT:"
        f"\n   ✓  Player IDs stay stable when players run (no flicker)"
        f"\n   ✓  IDs survive brief occlusion (players crossing)"
        f"\n   ✓  Trail lines show clean movement paths"
        f"\n   ✓  Total unique IDs ≈ number of players visible (no explosion of IDs)"
        f"\n   ✗  If IDs keep switching → lower iou_threshold in build_tracker()"
        f"\n   ✗  If phantom detections → raise CONFIDENCE_THRESHOLD to 0.55+"
    )


if __name__ == "__main__":
    run_test()
