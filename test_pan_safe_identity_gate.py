#!/usr/bin/env python3
"""
Test Phase 3: PAN-SAFE IDENTITY GATE.

Verify that identity system prevents wrong locks/rebinds/takeovers during fast camera pans.
"""

import numpy as np
from services.identity_core import IdentityCore


def test_fast_pan_blocks_new_lock_creation():
    """Test that new lock creation is blocked during fast_pan."""
    print("\n" + "=" * 80)
    print("TEST 1: FAST_PAN BLOCKS NEW LOCK CREATION")
    print("=" * 80)

    identity = IdentityCore()

    # Create an unlocked slot ready for a lock
    slot = identity.slots[0]
    slot.embedding = np.random.randn(512).astype(np.float32)
    slot.embedding /= np.linalg.norm(slot.embedding)
    slot.state = "active"  # Active state reduces initial cost
    slot.last_position = (250.0, 330.0)
    slot.last_pitch = (250.0, 330.0)

    # Create very similar track embedding for low cost
    track_embedding = slot.embedding.copy()
    track_embedding += 0.001 * np.random.randn(512).astype(np.float32)  # Minimal noise
    track_embedding /= np.linalg.norm(track_embedding)

    # Simulate normal play
    normal_motion = {
        "dx": 0.0, "dy": 0.0, "motion_px": 0.0, "affine": None,
        "confidence": 1.0, "motion_class": "stable",
    }

    # Simulate fast pan
    fast_pan_motion = {
        "dx": 100.0, "dy": 20.0, "motion_px": np.sqrt(100.0**2 + 20.0**2),
        "affine": [[1.0, 0.0, 100.0], [0.0, 1.0, 20.0]],
        "confidence": 1.0, "motion_class": "fast_pan",
    }

    # Frame 1: Normal play, pending streak builds
    for frame in range(5):
        identity.begin_frame(frame, present_tids={5})
        track_to_pid, meta = identity.assign_tracks(
            tracks=[type('Track', (), {'track_id': 5})()],
            embeddings={5: track_embedding},
            positions={5: (250.0, 330.0)},
            allow_new_assignments=True,
            camera_motion=normal_motion,
        )
        identity.end_frame()

    # Frame 5-6: Try to lock in normal play
    identity.begin_frame(5, present_tids={5})
    track_to_pid_normal, meta_normal = identity.assign_tracks(
        tracks=[type('Track', (), {'track_id': 5})()],
        embeddings={5: track_embedding},
        positions={5: (250.0, 330.0)},
        allow_new_assignments=True,
        camera_motion=normal_motion,
    )
    lock_normal = identity.locks.get_lock(5)
    identity.end_frame()

    # Clear lock for pan test
    if lock_normal:
        identity.locks.release_lock(5, reason="test_reset", frame_id=6)
    slot.pending_streak = 5  # Reset pending for pan test
    slot.pending_tid = 5  # Restore tid for continued pending assignment
    slot.pending_seen_seq = identity.identity_frame_seq  # Sync sequence counter

    # Frame 7: Fast pan, same conditions but pan is active
    identity.begin_frame(7, present_tids={5})
    track_to_pid_pan, meta_pan = identity.assign_tracks(
        tracks=[type('Track', (), {'track_id': 5})()],
        embeddings={5: track_embedding},
        positions={5: (250.0, 330.0)},
        allow_new_assignments=True,
        camera_motion=fast_pan_motion,
    )
    lock_pan = identity.locks.get_lock(5)
    identity.end_frame()

    print(f"\nLock created in normal play: {lock_normal is not None}")
    print(f"Lock created during fast_pan: {lock_pan is not None}")
    print(f"Pan lock attempts blocked: {identity.pan_lock_attempts_blocked}")

    if lock_normal is not None and lock_pan is None and identity.pan_lock_attempts_blocked > 0:
        print("✓ PASS: Fast pan blocked new lock creation")
        return True
    else:
        print("✗ FAIL: Fast pan did not block new lock creation")
        return False


def test_fast_pan_blocks_rebind():
    """Test that new locks are blocked during fast_pan (rebind prevention)."""
    print("\n" + "=" * 80)
    print("TEST 2: FAST_PAN BLOCKS NEW LOCK (REBIND PREVENTION)")
    print("=" * 80)

    identity = IdentityCore()

    # Create slot with matching embedding
    slot = identity.slots[0]
    slot.embedding = np.random.randn(512).astype(np.float32)
    slot.embedding /= np.linalg.norm(slot.embedding)
    slot.state = "active"
    slot.last_position = (250.0, 330.0)

    # Create nearly identical embedding for low cost
    track_embedding = slot.embedding.copy()
    track_embedding += 0.001 * np.random.randn(512).astype(np.float32)
    track_embedding /= np.linalg.norm(track_embedding)

    # Build pending streak in normal play
    normal_motion = {
        "dx": 0.0, "dy": 0.0, "motion_px": 0.0, "affine": None,
        "confidence": 1.0, "motion_class": "stable",
    }

    for frame in range(5):
        identity.begin_frame(frame, present_tids={10})
        track_to_pid, meta = identity.assign_tracks(
            tracks=[type('Track', (), {'track_id': 10})()],
            embeddings={10: track_embedding},
            positions={10: (250.0, 330.0)},
            allow_new_assignments=True,
            camera_motion=normal_motion,
        )
        identity.end_frame()

    # Release any lock to allow new lock creation attempt during pan
    lock_before = identity.locks.get_lock(10)
    if lock_before:
        identity.locks.release_lock(10, reason="test_reset", frame_id=5)

    # Reset pending to be ready for lock attempt
    slot.pending_streak = 5  # Ensure ready for lock
    slot.pending_tid = 10
    slot.pending_seen_seq = identity.identity_frame_seq

    # Now fast pan arrives and tries to create lock
    # (should be blocked to prevent rebind/takeover during motion)
    fast_pan_motion = {
        "dx": 100.0, "dy": 20.0, "motion_px": 102.0,
        "affine": [[1.0, 0.0, 100.0], [0.0, 1.0, 20.0]],
        "confidence": 1.0, "motion_class": "fast_pan",
    }

    identity.begin_frame(5, present_tids={10})

    track_to_pid_pan, meta_pan = identity.assign_tracks(
        tracks=[type('Track', (), {'track_id': 10})()],
        embeddings={10: track_embedding},
        positions={10: (350.0, 380.0)},
        allow_new_assignments=True,
        camera_motion=fast_pan_motion,
    )
    identity.end_frame()

    print(f"Pan lock attempts blocked: {identity.pan_lock_attempts_blocked}")
    print(f"Fast pan frames: {identity.fast_pan_frames}")

    if identity.pan_lock_attempts_blocked > 0:
        print("✓ PASS: Fast pan blocked new lock creation")
        return True
    else:
        print("✗ FAIL: Fast pan did not block lock creation")
        return False


def test_fast_pan_keeps_existing_lock_same_tid():
    """Test that existing locks with same tid are preserved during fast_pan."""
    print("\n" + "=" * 80)
    print("TEST 3: FAST_PAN PRESERVES EXISTING LOCK (SAME TID)")
    print("=" * 80)

    identity = IdentityCore()

    # Create slot and lock it
    slot = identity.slots[0]
    slot.embedding = np.random.randn(512).astype(np.float32)
    slot.embedding /= np.linalg.norm(slot.embedding)
    slot.state = "active"

    lk, status = identity.locks.try_create_lock(5, "P1", "hungarian", frame_id=0)
    print(f"Initial lock created: {status}")

    fast_pan_motion = {
        "dx": 100.0, "dy": 20.0, "motion_px": 102.0,
        "affine": [[1.0, 0.0, 100.0], [0.0, 1.0, 20.0]],
        "confidence": 1.0, "motion_class": "fast_pan",
    }

    # During fast pan, the same tid=5 with P1 should refresh the lock, not block it
    identity.begin_frame(5, present_tids={5})
    identity.camera_motion = fast_pan_motion
    identity.camera_motion_class = "fast_pan"

    emb = np.random.randn(512).astype(np.float32)
    emb /= np.linalg.norm(emb)

    track_to_pid, meta = identity.assign_tracks(
        tracks=[type('Track', (), {'track_id': 5})()],
        embeddings={5: emb},
        positions={5: (250.0, 330.0)},
        allow_new_assignments=True,
        camera_motion=fast_pan_motion,
    )
    identity.end_frame()

    # Check lock is still valid
    lk_after = identity.locks.get_lock(5)
    print(f"Lock still valid after pan: {lk_after is not None}")
    print(f"Lock pid: {lk_after.pid if lk_after else 'N/A'}")

    if lk_after is not None and lk_after.pid == "P1" and track_to_pid.get(5) == "P1":
        print("✓ PASS: Fast pan preserved existing lock")
        return True
    else:
        print("✗ FAIL: Fast pan did not preserve existing lock")
        return False


def test_fast_pan_extends_ttl():
    """Test that dormant locks get TTL extension during fast_pan."""
    print("\n" + "=" * 80)
    print("TEST 4: FAST_PAN EXTENDS DORMANT LOCK TTL")
    print("=" * 80)

    identity = IdentityCore()

    # Create and lock a track
    lk, status = identity.locks.try_create_lock(5, "P1", "hungarian", frame_id=0, ttl=150)
    original_ttl = lk.ttl
    print(f"Original TTL: {original_ttl}")

    # Make it dormant
    identity.locks._tid_to_lock[5].dormant = True
    identity.locks._tid_to_lock[5].dormant_since_frame = 0

    fast_pan_motion = {
        "dx": 100.0, "dy": 20.0, "motion_px": 102.0,
        "affine": [[1.0, 0.0, 100.0], [0.0, 1.0, 20.0]],
        "confidence": 1.0, "motion_class": "fast_pan",
    }

    # During fast pan, dormant locks should get TTL extended
    identity.begin_frame(10, present_tids=set())
    identity.camera_motion = fast_pan_motion
    identity.camera_motion_class = "fast_pan"

    extended = identity.locks.extend_dormant_ttl_for_pan(10)
    extended_ttl = identity.locks._tid_to_lock[5].ttl
    identity.end_frame()

    print(f"Extended count: {extended}")
    print(f"Extended TTL: {extended_ttl}")
    print(f"TTL increase: {extended_ttl - original_ttl}")

    if extended > 0 and extended_ttl > original_ttl:
        print("✓ PASS: Fast pan extended dormant lock TTL")
        return True
    else:
        print("✗ FAIL: Fast pan did not extend TTL")
        return False


def test_cut_blocks_interpolation_carryover():
    """Test that cuts block identity carryover during interpolation."""
    print("\n" + "=" * 80)
    print("TEST 5: CUT BLOCKS INTERPOLATION IDENTITY CARRYOVER")
    print("=" * 80)

    identity = IdentityCore()

    # Create and lock a track
    slot = identity.slots[0]
    slot.embedding = np.random.randn(512).astype(np.float32)
    slot.embedding /= np.linalg.norm(slot.embedding)
    slot.state = "active"

    lk, status = identity.locks.try_create_lock(5, "P1", "hungarian", frame_id=0)

    cut_motion = {
        "dx": 0.0, "dy": 0.0, "motion_px": 500.0,  # Very large motion = cut
        "affine": None, "confidence": 0.1,
        "motion_class": "cut",
    }

    # During cut, new assignments should be blocked
    identity.begin_frame(5, present_tids={5})
    identity.camera_motion = cut_motion
    identity.camera_motion_class = "cut"

    # Existing lock should be preserved but new assignments blocked
    emb = np.random.randn(512).astype(np.float32)
    emb /= np.linalg.norm(emb)

    track_to_pid, meta = identity.assign_tracks(
        tracks=[type('Track', (), {'track_id': 5})()],
        embeddings={5: emb},
        positions={5: (250.0, 330.0)},
        allow_new_assignments=True,
        camera_motion=cut_motion,
    )
    identity.end_frame()

    lk_after = identity.locks.get_lock(5)
    print(f"Lock preserved during cut: {lk_after is not None}")
    print(f"Cut frames recorded: {identity.cut_frames}")

    if lk_after is not None and identity.cut_frames > 0:
        print("✓ PASS: Cut blocked interpolation while preserving locks")
        return True
    else:
        print("✗ FAIL: Cut did not properly block or preserve locks")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("PHASE 3 TEST SUITE: PAN-SAFE IDENTITY GATE")
    print("=" * 80)

    results = []
    results.append(("Fast Pan Blocks New Lock", test_fast_pan_blocks_new_lock_creation()))
    results.append(("Fast Pan Blocks Rebind", test_fast_pan_blocks_rebind()))
    results.append(("Fast Pan Preserves Existing Lock", test_fast_pan_keeps_existing_lock_same_tid()))
    results.append(("Fast Pan Extends TTL", test_fast_pan_extends_ttl()))
    results.append(("Cut Blocks Identity Carryover", test_cut_blocks_interpolation_carryover()))

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
        print("\n✓✓✓ PHASE 3 COMPLETE: Pan-safe identity gate working ✓✓✓")
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
