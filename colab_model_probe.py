#!/usr/bin/env python3
"""
CRITICAL: Probe roboflow_players.pt vs yolov8m.pt on Villa/PSG .mov
to decide which model to use for the acceptance run.

Run in Colab BEFORE running full tracking.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

print("=" * 70)
print("MODEL PROBE: roboflow_players.pt vs yolov8m.pt")
print("=" * 70)

# Verify video exists
video_path = "/content/Aston villa vs Psg clip 1.mov"
if not os.path.exists(video_path):
    print(f"ERROR: Video not found at {video_path}")
    sys.exit(1)

print(f"\n✓ Video: {video_path}")

# Open video and sample frames
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"  Resolution: {w}x{h}")
print(f"  FPS: {fps}")
print(f"  Total frames: {frame_count}")

# Sample frames at stride=5: indices 0, 3, 6, 9, 12
sample_indices = [0, 3, 6, 9, 12]
frames = {}

for target_idx in sample_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
    ret, frame = cap.read()
    if ret:
        frames[target_idx] = frame
        print(f"  ✓ Frame {target_idx} loaded")
    else:
        print(f"  ✗ Frame {target_idx} failed to load")

cap.release()

if not frames:
    print("ERROR: No frames loaded")
    sys.exit(1)

print(f"\n{'='*70}")
print("LOADING MODELS")
print(f"{'='*70}\n")

# Load roboflow
print("⏳ Loading roboflow_players.pt...")
try:
    roboflow_model = YOLO("/content/roboflow_players.pt")
    print(f"✓ roboflow_players.pt loaded")
    print(f"  names: {roboflow_model.names}")
    print(f"  num_classes: {len(roboflow_model.names)}")
except Exception as e:
    print(f"✗ Failed to load roboflow: {e}")
    roboflow_model = None

# Load COCO yolov8m
print("\n⏳ Loading yolov8m.pt (COCO)...")
try:
    coco_model = YOLO("yolov8m.pt")
    print(f"✓ yolov8m.pt loaded")
    print(f"  names (first 10): {dict(list(coco_model.names.items())[:10])}")
except Exception as e:
    print(f"✗ Failed to load yolov8m: {e}")
    coco_model = None

print(f"\n{'='*70}")
print("FRAME PROBE")
print(f"{'='*70}\n")

if roboflow_model:
    print("ROBOFLOW_PLAYERS.PT")
    print("-" * 70)
    print(f"{'Frame':<8} {'Raw Boxes':<12} {'Class Hist':<40} {'Player Count':<12}")
    print("-" * 70)

    roboflow_stats = {}
    for frame_idx in sample_indices:
        if frame_idx not in frames:
            continue

        frame = frames[frame_idx]
        results = roboflow_model.predict(frame, conf=0.05, verbose=False)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            roboflow_stats[frame_idx] = {"raw_boxes": 0, "class_hist": {}, "player_count": 0}
            print(f"{frame_idx:<8} {0:<12} {'none':<40} {0:<12}")
            continue

        # Build class histogram
        class_hist = {}
        player_count = 0
        for box in boxes:
            cls = int(box.cls.item())
            class_hist[cls] = class_hist.get(cls, 0) + 1
            if cls in (1, 2, 3):  # roboflow player/goalkeeper/referee
                player_count += 1

        hist_str = ", ".join([f"{k}:{v}" for k, v in sorted(class_hist.items())])
        print(f"{frame_idx:<8} {len(boxes):<12} {hist_str:<40} {player_count:<12}")
        roboflow_stats[frame_idx] = {
            "raw_boxes": len(boxes),
            "class_hist": class_hist,
            "player_count": player_count
        }

if coco_model:
    print(f"\n{'='*70}")
    print("YOLOV8M.PT (COCO FALLBACK)")
    print("-" * 70)
    print(f"{'Frame':<8} {'Raw Boxes':<12} {'Class Hist':<40} {'Person Count':<12}")
    print("-" * 70)

    coco_stats = {}
    for frame_idx in sample_indices:
        if frame_idx not in frames:
            continue

        frame = frames[frame_idx]
        results = coco_model.predict(frame, conf=0.05, verbose=False)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            coco_stats[frame_idx] = {"raw_boxes": 0, "class_hist": {}, "person_count": 0}
            print(f"{frame_idx:<8} {0:<12} {'none':<40} {0:<12}")
            continue

        # Build class histogram
        class_hist = {}
        person_count = 0
        for box in boxes:
            cls = int(box.cls.item())
            class_hist[cls] = class_hist.get(cls, 0) + 1
            if cls == 0:  # COCO person
                person_count += 1

        hist_str = ", ".join([f"{k}:{v}" for k, v in sorted(class_hist.items())])
        print(f"{frame_idx:<8} {len(boxes):<12} {hist_str:<40} {person_count:<12}")
        coco_stats[frame_idx] = {
            "raw_boxes": len(boxes),
            "class_hist": class_hist,
            "person_count": person_count
        }

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}\n")

if roboflow_model and coco_model:
    roboflow_player_total = sum(s.get("player_count", 0) for s in roboflow_stats.values())
    coco_person_total = sum(s.get("person_count", 0) for s in coco_stats.values())

    print(f"Roboflow player detections (classes 1/2/3):  {roboflow_player_total}")
    print(f"COCO person detections (class 0):            {coco_person_total}")

    if roboflow_player_total == 0 and coco_person_total > 5:
        print("\n🔴 HYPOTHESIS H2 WINS: .mov is out-of-distribution for roboflow")
        print("   → USE COCO FALLBACK (yolov8m.pt)")
        print("   → Remap COCO class 0 → internal class 2")
        recommendation = "COCO"
    elif roboflow_player_total > 5:
        print("\n🟢 ROBOFLOW WORKS: Classes 1/2/3 detected normally")
        print("   → USE ROBOFLOW_PLAYERS.PT")
        recommendation = "ROBOFLOW"
    else:
        print("\n🟡 INCONCLUSIVE: Both models weak on this clip")
        print("   → CHECK VIDEO CODEC / LIGHTING")
        recommendation = "CHECK_CODEC"

    print(f"\nRECOMMENDATION: {recommendation}")
else:
    print("Could not load one or both models")

print(f"\n{'='*70}")
