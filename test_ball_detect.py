"""Test if ball detection works on a real frame."""
import sys
import cv2
sys.path.insert(0, '/Users/rudra/athlink-cv-service/athlink-cv-service')

from services.ball_tracking_service import BallTracker, MIN_CONF

print("Loading video...")
cap = cv2.VideoCapture('/tmp/test_5s.mp4')

print("Creating and loading BallTracker...")
tracker = BallTracker()
tracker.load_model()

print(f"Model: {tracker.model is not None}")
print(f"Model type: {tracker.model_type}")
print(f"MIN_CONF: {MIN_CONF}")
print()

# Test on first 10 frames
detection_count = 0
for frame_idx in range(10):
    ret, frame = cap.read()
    if not ret:
        break

    print(f"Frame {frame_idx}...")
    result = tracker.detect(frame, frame_idx)

    if result and 'x' in result:
        detection_count += 1
        print(f"  ✓ Detected at ({result['x']:.0f}, {result['y']:.0f}), conf={result['confidence']:.3f}")
    else:
        print(f"  ✗ No detection")

cap.release()

print()
print(f"Total detections in 10 frames: {detection_count}")
if detection_count > 0:
    print("SUCCESS!")
else:
    print("FAIL: No detections found")
