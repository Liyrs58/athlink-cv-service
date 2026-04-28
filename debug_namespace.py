import sys
import os
import torch
import cv2
import numpy as np
from pathlib import Path

# Mock detect_device for standalone test
def _detect_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"

def test_minimal():
    print("--- Testing Ultralytics YOLO Minimal ---")
    from ultralytics import YOLO
    try:
        model = YOLO("yolo11n.pt")
        # Try a single dummy inference
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model(dummy, verbose=False)
        print(f"Minimal YOLO Success. Results type: {type(results)}")
        if isinstance(results, list):
            print(f"Result[0] type: {type(results[0])}")
    except Exception as e:
        print(f"Minimal YOLO Failed: {e}")
        import traceback
        traceback.print_exc()

def test_validation():
    print("\n--- Testing Validation Logic ---")
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        
        print("Testing model(dummy)...")
        results_list = model(dummy, verbose=False, conf=0.20, classes=[0])
        
        n_persons = 0
        if isinstance(results_list, list) and len(results_list) > 0:
            res = results_list[0]
            print(f"Result item type: {type(res)}")
            if hasattr(res, "boxes") and res.boxes is not None:
                n_persons = len(res.boxes)
                print(f"Found {n_persons} persons.")
        print("Validation logic Success.")
    except Exception as e:
        print(f"Validation logic Failed: {e}")
        import traceback
        traceback.print_exc()

def test_boxmot():
    print("\n--- Testing BoxMOT BotSort Initialization ---")
    try:
        import boxmot
        tracker = boxmot.BotSort(
            reid_weights=Path("osnet_x0_25_msmt17.pt"),
            device=_detect_device(),
            half=False,
        )
        print("BoxMOT BotSort Init Success.")
    except Exception as e:
        print(f"BoxMOT BotSort Init Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_minimal()
    test_validation()
    test_boxmot()
