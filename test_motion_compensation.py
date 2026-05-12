#!/usr/bin/env python3
"""
Test Phase 2: Position compensation during camera motion.

Verify that when a slot moves due to camera pan, compensating the position
before Hungarian matching reduces the cost and improves matching.
"""

import numpy as np
from services.identity_core import IdentityCore


def test_position_compensation_improves_cost():
    """Test that position compensation reduces cost during pan."""
    print("\n" + "=" * 80)
    print("TEST: POSITION COMPENSATION IMPROVES MATCHING COST")
    print("=" * 80)

    # Create identity core
    identity = IdentityCore()
    identity.frame_id = 0

    # Scenario: player at (200, 300) in previous frame
    # Camera pans right by +50px, down by +30px
    # So slot.last_position = (200, 300)
    # Detection appears at (250, 330) — should match after compensation

    slot = identity.slots[0]
    slot.embedding = np.random.randn(512).astype(np.float32)
    slot.embedding /= np.linalg.norm(slot.embedding)
    slot.last_position = (200.0, 300.0)
    slot.state = "active"
    slot.team_id = 1

    # Create a track with matching embedding at compensated position
    track_embedding = slot.embedding.copy() + 0.01 * np.random.randn(512).astype(np.float32)
    track_embedding /= np.linalg.norm(track_embedding)

    # Track detection position: slot_pos + camera_motion
    # If slot was at (200, 300) and camera panned (+50, +30),
    # the track should be at (250, 330) to appear stationary
    track_position = (250.0, 330.0)

    # Simulate camera pan motion
    camera_motion_no = {
        "dx": 0.0,
        "dy": 0.0,
        "motion_px": 0.0,
        "affine": None,
        "confidence": 1.0,
        "motion_class": "stable",
    }

    camera_motion_pan = {
        "dx": 50.0,
        "dy": 30.0,
        "motion_px": np.sqrt(50.0**2 + 30.0**2),
        "affine": [[1.0, 0.0, 50.0], [0.0, 1.0, 30.0]],
        "confidence": 1.0,
        "motion_class": "pan",
    }

    # Cost WITHOUT motion compensation
    # (slot at 200,300 matched against detection at 250,330 — far apart)
    cost_without = identity._slot_cost(
        slot, track_embedding, track_position, tid=10, camera_motion=None
    )

    # Cost WITH motion compensation
    # (slot compensated to 250,330 matched against detection at 250,330 — identical!)
    cost_with = identity._slot_cost(
        slot, track_embedding, track_position, tid=10, camera_motion=camera_motion_pan
    )

    print(f"\nSlot position (original): {slot.last_position}")
    print(f"Track position (detected): {track_position}")
    print(f"Camera motion: dx={camera_motion_pan['dx']:.0f}, dy={camera_motion_pan['dy']:.0f}")
    print(f"\nCost WITHOUT compensation: {cost_without:.4f}")
    print(f"Cost WITH compensation:    {cost_with:.4f}")
    print(f"Cost improvement: {cost_without - cost_with:.4f}")

    if cost_with < cost_without:
        improvement_pct = 100.0 * (cost_without - cost_with) / max(cost_without, 0.001)
        print(f"\n✓ PASS: Compensation reduced cost by {improvement_pct:.1f}%")
        return True
    else:
        print(f"\n✗ FAIL: Compensation did not reduce cost")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("PHASE 2 TEST: MOTION COMPENSATION")
    print("=" * 80)

    result = test_position_compensation_improves_cost()

    if result:
        print("\n✓✓✓ PHASE 2 PARTIAL: Position compensation working ✓✓✓")
    else:
        print("\n✗ PHASE 2 TEST FAILED")
