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
