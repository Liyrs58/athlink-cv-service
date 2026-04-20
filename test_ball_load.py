"""Test if ball model loads correctly in the pipeline."""
import sys
sys.path.insert(0, '/Users/rudra/athlink-cv-service/athlink-cv-service')

from services.ball_tracking_service import BallTracker

print("Creating BallTracker...")
tracker = BallTracker()

print("Loading model...")
tracker.load_model()

print(f"Model loaded: {tracker.model is not None}")
print(f"Model type: {tracker.model_type}")

if tracker.model:
    print(f"Model classes: {tracker.model.names}")
    print("SUCCESS: Ball model loaded!")
else:
    print("FAIL: Ball model is None!")
