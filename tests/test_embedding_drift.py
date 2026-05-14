import numpy as np
import pytest

def test_cosine_similarity_identical():
    """Identical embeddings should have similarity 1.0"""
    from services.embedding_drift import compute_cosine_similarity
    emb = np.array([1.0, 0.0, 0.0])
    sim = compute_cosine_similarity(emb, emb)
    assert abs(sim - 1.0) < 1e-6

def test_cosine_similarity_orthogonal():
    """Orthogonal embeddings should have similarity ~0.0"""
    from services.embedding_drift import compute_cosine_similarity
    emb1 = np.array([1.0, 0.0, 0.0])
    emb2 = np.array([0.0, 1.0, 0.0])
    sim = compute_cosine_similarity(emb1, emb2)
    assert abs(sim - 0.0) < 1e-6

def test_cosine_similarity_opposite():
    """Opposite embeddings should have similarity -1.0"""
    from services.embedding_drift import compute_cosine_similarity
    emb1 = np.array([1.0, 0.0, 0.0])
    emb2 = np.array([-1.0, 0.0, 0.0])
    sim = compute_cosine_similarity(emb1, emb2)
    assert abs(sim - (-1.0)) < 1e-6

def test_drift_tracker_init():
    """DriftTracker should initialize with empty state"""
    from services.embedding_drift import DriftTracker
    tracker = DriftTracker(drift_threshold=0.70)
    assert tracker.drift_threshold == 0.70
    assert len(tracker.pid_anchors) == 0
    assert len(tracker.pid_history) == 0

def test_drift_tracker_anchor_creation():
    """Creating anchor should store embedding"""
    from services.embedding_drift import DriftTracker
    tracker = DriftTracker(drift_threshold=0.70)
    emb = np.array([1.0, 0.0, 0.0])
    tracker.create_anchor("P1", emb)
    assert "P1" in tracker.pid_anchors
    assert np.allclose(tracker.pid_anchors["P1"], emb)

def test_end_to_end_drift_tracking():
    """Full flow: anchor → update → drift → vlm decision → report"""
    from services.embedding_drift import DriftTracker

    tracker = DriftTracker(drift_threshold=0.70)

    # Simulate 3 players over 10 frames
    pids = ["P1", "P2", "P3"]

    # Create anchors (frame 0)
    anchors = {
        "P1": np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "P2": np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "P3": np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    }
    for pid, emb in anchors.items():
        tracker.create_anchor(pid, emb)

    # Update over frames, with P1 drifting significantly, P2/P3 stable
    for frame in range(1, 11):
        # P1: strong gradual drift (moves increasingly toward opposite direction)
        drift_factor = frame / 10.0
        emb_p1 = np.array([
            1.0 - drift_factor*0.8,  # Original component fades
            drift_factor*0.8,          # Opposite component grows
            0.0, 0.0, 0.0
        ], dtype=np.float32)

        # P2, P3: very stable
        emb_p2 = np.array([0.02, 0.98, 0.0, 0.0, 0.0], dtype=np.float32)
        emb_p3 = np.array([0.0, 0.0, 0.99, 0.01, 0.0], dtype=np.float32)

        tracker.update_drift("P1", emb_p1)
        tracker.update_drift("P2", emb_p2)
        tracker.update_drift("P3", emb_p3)

        # Log VLM decisions
        for pid in pids:
            history = tracker.pid_history[pid]
            sim = history[-1] if history else None
            triggered = tracker.should_trigger_vlm(pid, sim)
            tracker.log_vlm_decision(pid, frame, sim, triggered)

    # Verify report
    report = tracker.export_report()
    assert report["players"]["P1"]["total_frames"] == 10
    assert report["players"]["P2"]["total_frames"] == 10
    assert report["players"]["P3"]["total_frames"] == 10

    # P1 should have triggered VLM (drifted below 0.70)
    assert report["players"]["P1"]["drift_triggered_count"] > 0, \
        f"P1 triggered {report['players']['P1']['drift_triggered_count']} times, expected > 0"

    # P2, P3 should NOT trigger (stable)
    assert report["players"]["P2"]["drift_triggered_count"] == 0
    assert report["players"]["P3"]["drift_triggered_count"] == 0

    # Verify final stability scores
    p1_final = report["players"]["P1"]["final_stability"]
    p2_final = report["players"]["P2"]["final_stability"]

    # P1 should have low final stability (drifted)
    assert p1_final < 0.65, f"P1 final_stability {p1_final} should be low"

    # P2 should have high final stability (stable)
    assert p2_final > 0.95, f"P2 final_stability {p2_final} should be high"
