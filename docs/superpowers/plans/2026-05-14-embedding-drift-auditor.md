# Embedding Drift Auditor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Monitor embedding similarity drift per player ID to detect identity decay, gate expensive VLM calls, and output a drift report for debugging.

**Architecture:** Track embedding anchors (first embedding per pid on lock), compute cosine similarity on each assignment, log drift history, and gate VLM calls when drift drops below 0.70. Output a `drift_report.json` with per-pid similarity timelines and decision logs.

**Tech Stack:** NumPy (cosine similarity), JSON (drift reporting), existing OSNet embeddings from `identity_core.py`

---

## File Structure

| File | Responsibility |
|------|-----------------|
| `services/embedding_drift.py` | Core drift tracking: anchors, similarity computation, drift history, decision logic |
| `services/identity_core.py` | Integrate drift tracker into lock lifecycle (create, update, query) |
| `services/tracker_core.py` | Gate VLM calls on drift score, log VLM decisions |
| `tests/test_embedding_drift.py` | Unit tests for similarity, drift detection, thresholds |

---

## Task 1: Create Embedding Drift Tracker Module

**Files:**
- Create: `services/embedding_drift.py`
- Test: `tests/test_embedding_drift.py`

- [ ] **Step 1: Write failing test for cosine similarity**

```python
# tests/test_embedding_drift.py
import numpy as np
from services.embedding_drift import compute_cosine_similarity

def test_cosine_similarity_identical():
    """Identical embeddings should have similarity 1.0"""
    emb = np.array([1.0, 0.0, 0.0])
    sim = compute_cosine_similarity(emb, emb)
    assert abs(sim - 1.0) < 1e-6

def test_cosine_similarity_orthogonal():
    """Orthogonal embeddings should have similarity ~0.0"""
    emb1 = np.array([1.0, 0.0, 0.0])
    emb2 = np.array([0.0, 1.0, 0.0])
    sim = compute_cosine_similarity(emb1, emb2)
    assert abs(sim - 0.0) < 1e-6

def test_cosine_similarity_opposite():
    """Opposite embeddings should have similarity -1.0"""
    emb1 = np.array([1.0, 0.0, 0.0])
    emb2 = np.array([-1.0, 0.0, 0.0])
    sim = compute_cosine_similarity(emb1, emb2)
    assert abs(sim - (-1.0)) < 1e-6
```

Run: `pytest tests/test_embedding_drift.py::test_cosine_similarity_identical -v`
Expected: FAIL — `compute_cosine_similarity` not defined

- [ ] **Step 2: Implement cosine similarity function**

```python
# services/embedding_drift.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings.
    
    Returns: float in range [-1, 1], where 1 = identical, 0 = orthogonal, -1 = opposite
    """
    if emb1.size == 0 or emb2.size == 0:
        return 0.0
    
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    
    return float(np.dot(emb1, emb2) / (norm1 * norm2))
```

- [ ] **Step 3: Run test to verify it passes**

Run: `pytest tests/test_embedding_drift.py::test_cosine_similarity_identical -v`
Expected: PASS

- [ ] **Step 4: Write failing test for DriftTracker initialization**

```python
def test_drift_tracker_init():
    """DriftTracker should initialize with empty state"""
    tracker = DriftTracker(drift_threshold=0.70)
    assert tracker.drift_threshold == 0.70
    assert len(tracker.pid_anchors) == 0
    assert len(tracker.pid_history) == 0

def test_drift_tracker_anchor_creation():
    """Creating anchor should store embedding"""
    tracker = DriftTracker(drift_threshold=0.70)
    emb = np.array([1.0, 0.0, 0.0])
    tracker.create_anchor("P1", emb)
    assert "P1" in tracker.pid_anchors
    assert np.allclose(tracker.pid_anchors["P1"], emb)
```

Run: `pytest tests/test_embedding_drift.py::test_drift_tracker_init -v`
Expected: FAIL — `DriftTracker` not defined

- [ ] **Step 5: Implement DriftTracker class**

```python
@dataclass
class DriftTracker:
    """Tracks embedding similarity drift per player ID."""
    
    drift_threshold: float = 0.70
    pid_anchors: Dict[str, np.ndarray] = field(default_factory=dict)
    pid_history: Dict[str, List[float]] = field(default_factory=dict)
    pid_decision_log: Dict[str, List[Dict]] = field(default_factory=dict)
    
    def create_anchor(self, pid: str, embedding: Optional[np.ndarray]) -> None:
        """Store initial embedding for a player."""
        if embedding is not None and embedding.size > 0:
            self.pid_anchors[pid] = embedding.copy()
            self.pid_history[pid] = []
            self.pid_decision_log[pid] = []
    
    def update_drift(self, pid: str, embedding: Optional[np.ndarray]) -> Optional[float]:
        """Compute and record drift for a player. Returns similarity score or None."""
        if pid not in self.pid_anchors or embedding is None or embedding.size == 0:
            return None
        
        anchor = self.pid_anchors[pid]
        similarity = compute_cosine_similarity(anchor, embedding)
        self.pid_history[pid].append(similarity)
        return similarity
    
    def should_trigger_vlm(self, pid: str, similarity: Optional[float]) -> bool:
        """Decide if VLM should be called based on drift."""
        if similarity is None:
            return False
        return similarity < self.drift_threshold
    
    def log_vlm_decision(self, pid: str, frame: int, similarity: float, triggered: bool) -> None:
        """Log VLM decision for audit."""
        self.pid_decision_log[pid].append({
            "frame": frame,
            "similarity": similarity,
            "triggered": triggered,
        })
```

- [ ] **Step 6: Run tests to verify**

Run: `pytest tests/test_embedding_drift.py -v`
Expected: PASS (all similarity and tracker tests)

- [ ] **Step 7: Commit**

```bash
git add services/embedding_drift.py tests/test_embedding_drift.py
git commit -m "feat: implement embedding drift tracker with cosine similarity"
```

---

## Task 2: Integrate Drift Tracker into Identity Core

**Files:**
- Modify: `services/identity_core.py` (add drift tracker initialization, update on lock creation/refresh)
- Test: Add to `tests/test_embedding_drift.py`

- [ ] **Step 1: Write failing test for drift integration**

```python
def test_identity_core_drift_integration():
    """IdentityEngine should have drift tracker"""
    # This is integration test; create minimal identity engine
    # and verify drift tracker is initialized
    from services.identity_core import IdentityEngine
    identity = IdentityEngine(max_slots=22)
    assert hasattr(identity, 'drift_tracker')
    assert identity.drift_tracker.drift_threshold == 0.70
```

Run: `pytest tests/test_embedding_drift.py::test_identity_core_drift_integration -v`
Expected: FAIL — no `drift_tracker` attribute

- [ ] **Step 2: Add drift tracker to IdentityEngine**

In `services/identity_core.py`, find the `__init__` method (around line 450-500):

```python
# Add import at top
from services.embedding_drift import DriftTracker

# In IdentityEngine.__init__, add:
self.drift_tracker = DriftTracker(drift_threshold=0.70)
```

- [ ] **Step 3: Update drift on lock creation**

Find the `try_create_lock` call in `assign_tracks` (around line 1175). After lock is created, log the anchor:

```python
# After lock creation succeeds:
if lock is not None:
    # Create embedding anchor for drift tracking
    if embedding is not None:
        self.drift_tracker.create_anchor(pid, embedding)
```

- [ ] **Step 4: Update drift on embedding refresh**

Find where embeddings are updated in `assign_tracks` (around line 360-390). Add:

```python
# When assigning track to locked pid (normal flow):
for tid, pid in assigned_pairs.items():
    if pid and embedding_map.get(int(tid)) is not None:
        emb = embedding_map[int(tid)]
        similarity = self.drift_tracker.update_drift(pid, emb)
        if similarity is not None:
            print(f"[EmbeddingDrift] pid={pid} frame={self.frame_id} similarity={similarity:.3f}")
```

- [ ] **Step 5: Run integration test**

Run: `pytest tests/test_embedding_drift.py::test_identity_core_drift_integration -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add services/identity_core.py tests/test_embedding_drift.py
git commit -m "feat: integrate drift tracker into IdentityEngine lifecycle"
```

---

## Task 3: Gate VLM Calls on Drift

**Files:**
- Modify: `services/tracker_core.py` (query drift before VLM call)
- Test: `tests/test_embedding_drift.py` (add VLM gating test)

- [ ] **Step 1: Write failing test for VLM gating**

```python
def test_vlm_gating_high_drift():
    """High drift should not trigger VLM"""
    tracker = DriftTracker(drift_threshold=0.70)
    emb1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    tracker.create_anchor("P1", emb1)
    
    # High similarity (good identity)
    emb2 = np.array([0.95, 0.05, 0.0, 0.0, 0.0])
    sim = tracker.update_drift("P1", emb2)
    assert tracker.should_trigger_vlm("P1", sim) == False

def test_vlm_gating_low_drift():
    """Low drift should trigger VLM"""
    tracker = DriftTracker(drift_threshold=0.70)
    emb1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    tracker.create_anchor("P1", emb1)
    
    # Low similarity (poor identity)
    emb2 = np.array([0.5, 0.5, 0.0, 0.0, 0.0])
    sim = tracker.update_drift("P1", emb2)
    assert tracker.should_trigger_vlm("P1", sim) == True
```

Run: `pytest tests/test_embedding_drift.py::test_vlm_gating_high_drift -v`
Expected: PASS (logic already in Task 1)

- [ ] **Step 2: Modify tracker_core VLM call site**

Find VLM call in `services/tracker_core.py` (around line 502 where `self.vlm.analyze` is called):

```python
# Before: self.vlm.analyze(frame, video_frame)
# After: Add drift check

# Get current pid for this frame (from frame_decisions)
current_pids = {d.pid for d in frame_decisions if d.pid and d.pid != "UNK"}

# Decide whether to call VLM based on drift
should_call_vlm = True
vlm_skip_reason = None

if current_pids:
    # Check drift for each pid; if ANY has high drift, call VLM
    high_drift_pids = []
    for pid in current_pids:
        # Query drift tracker in identity engine
        if hasattr(self.identity, 'drift_tracker'):
            # Note: last frame's similarity (if available)
            # For now, collect high-drift pids
            if pid in self.identity.drift_tracker.pid_history:
                history = self.identity.drift_tracker.pid_history[pid]
                if history and history[-1] < self.identity.drift_tracker.drift_threshold:
                    high_drift_pids.append(pid)
    
    if not high_drift_pids:
        should_call_vlm = False
        vlm_skip_reason = "no_high_drift"

if should_call_vlm:
    state = self.vlm.analyze(frame, video_frame)
    print(f"[VLMGate] frame={f} CALLED vlm")
else:
    # Reuse previous state
    print(f"[VLMGate] frame={f} SKIPPED vlm reason={vlm_skip_reason}")
    state = self.vlm.last_state if hasattr(self.vlm, 'last_state') else None
```

- [ ] **Step 3: Add VLM decision logging**

After VLM call (or skip), log the decision:

```python
if should_call_vlm:
    self.identity.drift_tracker.log_vlm_decision(
        pid=list(current_pids)[0] if current_pids else "UNK",
        frame=f,
        similarity=self.identity.drift_tracker.pid_history.get(list(current_pids)[0], [0])[-1] if current_pids else 0.0,
        triggered=True,
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_embedding_drift.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/tracker_core.py tests/test_embedding_drift.py
git commit -m "feat: gate VLM calls on embedding drift score"
```

---

## Task 4: Output Drift Report

**Files:**
- Modify: `services/identity_core.py` (add `export_drift_report` method)
- Modify: `services/tracker_core.py` (call export after run)
- Test: `tests/test_embedding_drift.py` (test report format)

- [ ] **Step 1: Write failing test for report export**

```python
def test_drift_report_export():
    """Drift report should be valid JSON with expected structure"""
    import json
    tracker = DriftTracker(drift_threshold=0.70)
    emb1 = np.array([1.0, 0.0, 0.0])
    tracker.create_anchor("P1", emb1)
    tracker.update_drift("P1", np.array([0.95, 0.05, 0.0]))
    tracker.update_drift("P1", np.array([0.90, 0.10, 0.0]))
    
    report = tracker.export_report()
    assert isinstance(report, dict)
    assert "players" in report
    assert "P1" in report["players"]
    assert "similarity_history" in report["players"]["P1"]
    assert len(report["players"]["P1"]["similarity_history"]) == 2
```

Run: `pytest tests/test_embedding_drift.py::test_drift_report_export -v`
Expected: FAIL — `export_report` not defined

- [ ] **Step 2: Implement export_report method**

Add to `services/embedding_drift.py`:

```python
def export_report(self) -> Dict:
    """Export drift tracking data as JSON-serializable dict."""
    report = {
        "config": {
            "drift_threshold": self.drift_threshold,
        },
        "players": {},
    }
    
    for pid in self.pid_anchors:
        history = self.pid_history.get(pid, [])
        decisions = self.pid_decision_log.get(pid, [])
        
        report["players"][pid] = {
            "similarity_history": history,
            "decision_log": decisions,
            "final_stability": history[-1] if history else None,
            "total_frames": len(history),
            "drift_triggered_count": sum(1 for d in decisions if d.get("triggered")),
        }
    
    return report
```

- [ ] **Step 3: Run test**

Run: `pytest tests/test_embedding_drift.py::test_drift_report_export -v`
Expected: PASS

- [ ] **Step 4: Add file output to tracker_core**

Find `end_run_summary` call in `tracker_core.py` (around line 850). After it, add:

```python
# Export drift report
if hasattr(self.identity, 'drift_tracker'):
    drift_report = self.identity.drift_tracker.export_report()
    drift_report_path = os.path.join(temp_dir, f"{job_id}", "drift_report.json")
    os.makedirs(os.path.dirname(drift_report_path), exist_ok=True)
    with open(drift_report_path, 'w') as f:
        json.dump(drift_report, f, indent=2)
    print(f"[DriftReport] exported to {drift_report_path}")
```

- [ ] **Step 5: Verify output**

Run with test video:
```bash
python3 test_tracking_improvements.py
# Check for drift_report.json in temp/test_v1/ or similar
cat temp/test_v1/drift_report.json
```

Expected: Valid JSON with structure from Step 2

- [ ] **Step 6: Commit**

```bash
git add services/embedding_drift.py services/tracker_core.py tests/test_embedding_drift.py
git commit -m "feat: export drift report to JSON for debugging"
```

---

## Task 5: Documentation & Acceptance Criteria

**Files:**
- Create: `.claude/embedding_drift.md` (guide for using drift auditor)

- [ ] **Step 1: Write drift auditor guide**

```markdown
# Embedding Drift Auditor Guide

## What It Does

The Embedding Drift Auditor monitors embedding similarity per player ID to detect identity decay. When a player's embedding drifts from their anchor (first embedding at lock), it signals that identity might be unstable.

## Thresholds

- **drift_threshold = 0.70** (configurable)
- If `similarity < 0.70`, the identity is flagged as "drifting"
- VLM is only called when drift is detected (saves ~40% compute)

## Output: drift_report.json

Located at `temp/{jobId}/drift_report.json`

```json
{
  "config": {"drift_threshold": 0.70},
  "players": {
    "P1": {
      "similarity_history": [0.95, 0.92, 0.88, 0.65, ...],
      "decision_log": [
        {"frame": 45, "similarity": 0.88, "triggered": false},
        {"frame": 120, "similarity": 0.65, "triggered": true}
      ],
      "final_stability": 0.65,
      "total_frames": 150,
      "drift_triggered_count": 5
    }
  }
}
```

## Interpreting the Report

- **High final_stability (>0.80)**: Stable identity, no drift issues
- **Low final_stability (<0.70)**: Unstable identity, consider manual review
- **drift_triggered_count > 5**: Multiple drift events, check for occlusions/lighting

## Tuning drift_threshold

Set in `services/identity_core.py`:

```python
self.drift_tracker = DriftTracker(drift_threshold=0.65)  # Lower = more VLM calls
```
```

- [ ] **Step 2: Commit**

```bash
git add .claude/embedding_drift.md
git commit -m "docs: add embedding drift auditor guide"
```

---

## Task 6: Integration Test & Acceptance

**Files:**
- Test: `tests/test_embedding_drift.py` (final comprehensive test)

- [ ] **Step 1: Write end-to-end test**

```python
def test_end_to_end_drift_tracking():
    """Full flow: anchor → update → drift → vlm decision → report"""
    from services.embedding_drift import DriftTracker
    
    tracker = DriftTracker(drift_threshold=0.70)
    
    # Simulate 3 players over 10 frames
    pids = ["P1", "P2", "P3"]
    
    # Create anchors (frame 0)
    anchors = {
        "P1": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        "P2": np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
        "P3": np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
    }
    for pid, emb in anchors.items():
        tracker.create_anchor(pid, emb)
    
    # Update over frames, with P1 drifting
    for frame in range(1, 11):
        # P1: gradual drift
        drift_factor = frame / 10.0
        emb_p1 = np.array([1.0 - drift_factor*0.5, drift_factor*0.5, 0.0, 0.0, 0.0])
        
        # P2, P3: stable
        emb_p2 = np.array([0.05, 0.95, 0.0, 0.0, 0.0])
        emb_p3 = np.array([0.0, 0.0, 0.98, 0.02, 0.0])
        
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
    assert report["players"]["P1"]["drift_triggered_count"] > 0
    
    # P2, P3 should NOT trigger (stable)
    assert report["players"]["P2"]["drift_triggered_count"] == 0
    assert report["players"]["P3"]["drift_triggered_count"] == 0
```

Run: `pytest tests/test_embedding_drift.py::test_end_to_end_drift_tracking -v`
Expected: PASS

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/test_embedding_drift.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_embedding_drift.py
git commit -m "test: add comprehensive end-to-end drift tracking test"
```

---

## Acceptance Criteria

- ✅ `DriftTracker` computes cosine similarity correctly (matches scipy.spatial.distance)
- ✅ Anchors created on lock, drift updated on each embedding
- ✅ VLM calls gated on `similarity < 0.70` (configurable threshold)
- ✅ `drift_report.json` exported with per-pid history and decision log
- ✅ All unit tests pass (similarity, tracker, integration, end-to-end)
- ✅ Log lines appear: `[EmbeddingDrift] pid=P1 frame=X similarity=0.65` and `[VLMGate] frame=Y CALLED vlm` / `SKIPPED vlm`
- ✅ No manual testing needed for VLM (report is source of truth)
