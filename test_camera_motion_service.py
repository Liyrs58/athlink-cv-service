#!/usr/bin/env python3
"""
Test Phase 1: Camera Motion Detection Service.

Verify ORB+RANSAC and ECC fallback work correctly.
"""

import cv2
import numpy as np
from services.camera_motion_service import CameraMotionDetector, log_camera_motion


def test_detect_translation():
    """Test 1: Synthetic translation detection."""
    print("\n" + "=" * 80)
    print("TEST 1: SYNTHETIC TRANSLATION DETECTION")
    print("=" * 80)

    detector = CameraMotionDetector()

    # Create synthetic frame with texture
    frame1 = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    cv2.putText(frame1, "FRAME 1", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.circle(frame1, (320, 240), 50, (0, 255, 0), -1)
    cv2.circle(frame1, (200, 200), 30, (255, 0, 0), -1)
    cv2.circle(frame1, (450, 350), 40, (0, 0, 255), -1)

    # Translate by known amount
    dx_true, dy_true = 50.0, 30.0
    M_true = np.float32([[1, 0, dx_true], [0, 1, dy_true]])
    frame2 = cv2.warpAffine(frame1, M_true, (640, 480))

    # Detect motion
    motion1 = detector.estimate(frame1, 0)
    motion2 = detector.estimate(frame2, 1)

    print(f"Ground truth: dx={dx_true:.1f}, dy={dy_true:.1f}")
    print(log_camera_motion(1, motion2))

    # Check error
    dx_error = abs(motion2["dx"] - dx_true)
    dy_error = abs(motion2["dy"] - dy_true)
    motion_error = np.sqrt((motion2["dx"] - dx_true)**2 + (motion2["dy"] - dy_true)**2)

    print(f"\nDetected: dx={motion2['dx']:.1f}, dy={motion2['dy']:.1f}")
    print(f"Error: dx_err={dx_error:.1f}, dy_err={dy_error:.1f}, motion_err={motion_error:.1f} px")

    if motion_error < 5.0 and motion2["motion_class"] == "pan":
        print("✓ PASS: Translation detected with <5px error, classified as pan")
        return True
    else:
        print(f"✗ FAIL: motion_error={motion_error:.1f} px, class={motion2['motion_class']}")
        return False


def test_fast_pan_classification():
    """Test 2: Fast pan classification threshold."""
    print("\n" + "=" * 80)
    print("TEST 2: FAST PAN CLASSIFICATION")
    print("=" * 80)

    detector = CameraMotionDetector()

    # Create synthetic frame
    frame1 = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    cv2.putText(frame1, "FRAME", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    for i in range(10):
        cv2.circle(frame1, (100 + i*50, 200 + i*30), 20, (255, 255, 255), 1)

    # Large translation for fast_pan (>80px)
    dx_true, dy_true = 100.0, 20.0
    M_true = np.float32([[1, 0, dx_true], [0, 1, dy_true]])
    frame2 = cv2.warpAffine(frame1, M_true, (640, 480))

    detector.estimate(frame1, 0)
    motion2 = detector.estimate(frame2, 1)

    print(log_camera_motion(1, motion2))
    print(f"Motion magnitude: {motion2['motion_px']:.1f} px")

    if motion2["motion_class"] == "fast_pan":
        print("✓ PASS: Fast pan detected (>80px)")
        return True
    else:
        print(f"✗ FAIL: Expected fast_pan, got {motion2['motion_class']}")
        return False


def test_stable_frame():
    """Test 3: Stable frame (minimal motion)."""
    print("\n" + "=" * 80)
    print("TEST 3: STABLE FRAME DETECTION")
    print("=" * 80)

    detector = CameraMotionDetector()

    # Create frame with texture
    frame1 = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    for i in range(10):
        cv2.circle(frame1, (100 + i*50, 200 + i*30), 20, (255, 255, 255), 1)

    # Add small Gaussian noise (stable, no translation)
    frame2 = frame1.copy() + np.random.normal(0, 5, frame1.shape).astype(np.uint8)

    detector.estimate(frame1, 0)
    motion2 = detector.estimate(frame2, 1)

    print(log_camera_motion(1, motion2))
    print(f"Motion magnitude: {motion2['motion_px']:.1f} px")

    if motion2["motion_class"] == "stable" or motion2["motion_px"] < 20:
        print("✓ PASS: Stable frame detected (<20px)")
        return True
    else:
        print(f"✗ FAIL: Expected stable, got {motion2['motion_class']} with motion={motion2['motion_px']:.1f}")
        return False


def test_cut_detection():
    """Test 4: Scene cut detection (very large motion)."""
    print("\n" + "=" * 80)
    print("TEST 4: SCENE CUT DETECTION")
    print("=" * 80)

    detector = CameraMotionDetector()

    # Create two completely different frames (simulating a cut)
    frame1 = np.full((480, 640, 3), 50, dtype=np.uint8)
    cv2.putText(frame1, "SCENE 1", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    frame2 = np.full((480, 640, 3), 200, dtype=np.uint8)
    cv2.putText(frame2, "SCENE 2", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    detector.estimate(frame1, 0)
    motion2 = detector.estimate(frame2, 1)

    print(log_camera_motion(1, motion2))
    print(f"Confidence: {motion2['confidence']:.2f}")

    if motion2["motion_class"] == "cut" or motion2["confidence"] < 0.25:
        print("✓ PASS: Cut detected (low confidence or very large motion)")
        return True
    else:
        print(f"✗ FAIL: Expected cut, got {motion2['motion_class']} with conf={motion2['confidence']:.2f}")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("PHASE 1 TEST SUITE: CAMERA MOTION DETECTION")
    print("=" * 80)

    results = []
    results.append(("Translation Detection", test_detect_translation()))
    results.append(("Fast Pan Classification", test_fast_pan_classification()))
    results.append(("Stable Frame Detection", test_stable_frame()))
    results.append(("Cut Detection", test_cut_detection()))

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n✓✓✓ PHASE 1 COMPLETE: Camera motion detection working ✓✓✓")
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
