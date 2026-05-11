#!/usr/bin/env python3
"""
Inline model probe — paste directly into Colab cell.
No file dependencies.
"""

import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

print("=" * 70)
print("MODEL PROBE: roboflow_players.pt vs yolov8m.pt")
print("=" * 70)

video_path = "/content/Aston villa vs Psg clip 1.mov"
if not os.path.exists(video_path):
    print(f"ERROR: Video not found at {video_path}")
    sys.exit(1)

print(f"\n✓ Video: {video_path}")

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"  Resolution: {w}x{h}")
print(f"  FPS: {fps}")
print(f"  Total frames: {frame_count}")

sample_indices = [0, 3, 6, 9, 12]
frames = {}

for target_idx in sample_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
    ret, frame = cap.read()
    if ret:
        frames[target_idx] = frame
        print(f"  ✓ Frame {target_idx} loaded")

cap.release()

print(f"\n{'='*70}")
print("LOADING MODELS")
print(f"{'='*70}\n")

print("⏳ Loading roboflow_players.pt...")
roboflow_model = YOLO("/content/roboflow_players.pt")
print(f"✓ roboflow_players.pt")
print(f"  names: {roboflow_model.names}")

print("\n⏳ Loading yolov8m.pt...")
coco_model = YOLO("yolov8m.pt")
print(f"✓ yolov8m.pt")

print(f"\n{'='*70}")
print("ROBOFLOW_PLAYERS.PT")
print("-" * 70)
print(f"{'Frame':<8} {'Raw':<6} {'Class Hist':<35} {'Players (1/2/3)':<6}")
print("-" * 70)

roboflow_total = 0
for frame_idx in sample_indices:
    if frame_idx not in frames:
        continue

    frame = frames[frame_idx]
    results = roboflow_model.predict(frame, conf=0.05, verbose=False)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        print(f"{frame_idx:<8} {0:<6} {'none':<35} {0:<6}")
        continue

    class_hist = {}
    player_count = 0
    for box in boxes:
        cls = int(box.cls.item())
        class_hist[cls] = class_hist.get(cls, 0) + 1
        if cls in (1, 2, 3):
            player_count += 1

    roboflow_total += player_count
    hist_str = ", ".join([f"{k}:{v}" for k, v in sorted(class_hist.items())])
    print(f"{frame_idx:<8} {len(boxes):<6} {hist_str:<35} {player_count:<6}")

print(f"\nTotal roboflow players (classes 1/2/3): {roboflow_total}")

print(f"\n{'='*70}")
print("YOLOV8M.PT (COCO)")
print("-" * 70)
print(f"{'Frame':<8} {'Raw':<6} {'Class Hist':<35} {'Persons (0)':<6}")
print("-" * 70)

coco_total = 0
for frame_idx in sample_indices:
    if frame_idx not in frames:
        continue

    frame = frames[frame_idx]
    results = coco_model.predict(frame, conf=0.05, verbose=False)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        print(f"{frame_idx:<8} {0:<6} {'none':<35} {0:<6}")
        continue

    class_hist = {}
    person_count = 0
    for box in boxes:
        cls = int(box.cls.item())
        class_hist[cls] = class_hist.get(cls, 0) + 1
        if cls == 0:
            person_count += 1

    coco_total += person_count
    hist_str = ", ".join([f"{k}:{v}" for k, v in sorted(class_hist.items())])
    print(f"{frame_idx:<8} {len(boxes):<6} {hist_str:<35} {person_count:<6}")

print(f"\nTotal COCO persons (class 0): {coco_total}")

print(f"\n{'='*70}")
print("DECISION")
print(f"{'='*70}\n")

if roboflow_total > 5:
    print("✓ ROBOFLOW WORKS: Use roboflow_players.pt")
    print(f"  Player detections across 5 frames: {roboflow_total}")
elif coco_total > 5:
    print("⚠ ROBOFLOW FAILS, COCO WORKS: Switch to yolov8m.pt")
    print(f"  Roboflow players: {roboflow_total}")
    print(f"  COCO persons: {coco_total}")
    print("\n  ACTION: Wire fallback into tracker_core.run_tracking()")
else:
    print("❌ BOTH WEAK: Check video codec or lighting")
    print(f"  Roboflow players: {roboflow_total}")
    print(f"  COCO persons: {coco_total}")
