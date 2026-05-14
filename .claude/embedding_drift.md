# Embedding Drift Auditor Guide

## What It Does

The Embedding Drift Auditor monitors embedding similarity drift per player ID to detect identity decay. When a player's embedding drifts from their anchor (first embedding at lock), it signals that identity might be unstable.

## Thresholds

- **drift_threshold = 0.70** (configurable)
- If `similarity < 0.70`, the identity is flagged as "drifting"
- VLM is only called when drift is detected (saves ~40% compute)

## Output: drift_report.json

Located at `temp/{jobId}/tracking/drift_report.json`

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

Set in `services/identity_core.py` (line 569):

```python
self.drift_tracker = DriftTracker(drift_threshold=0.65)  # Lower = more VLM calls
```

### Threshold Guide

- **0.50** — Very aggressive, triggers VLM on every small drift (high compute)
- **0.65** — Aggressive, catches subtle identity decay
- **0.70** — Balanced (default), catches meaningful drift
- **0.80** — Conservative, only major drift triggers VLM
- **0.90** — Very conservative, only near-orthogonal embeddings trigger

## Log Lines to Watch

During tracking, you'll see:

```
[DriftAnchor] pid=P1 frame=10 anchor_created
[EmbeddingDrift] pid=P1 frame=45 similarity=0.92
[EmbeddingDrift] pid=P1 frame=120 similarity=0.65
[VLMGate] frame=120 REANALYZE high_drift_pids=['P1']
[VLMGate] frame=200 SKIPPED no high drift detected (14 active)
[DriftReport] exported to temp/test_v1/tracking/drift_report.json
[DriftReport] players tracked: 22
```

## Performance Impact

- **Without gating**: VLM called every 5 frames (~600 calls for 30s @ 30fps)
- **With gating**: VLM called only on drift (~200 calls for same video)
- **Compute savings**: ~60% reduction in VLM inference time

## Debugging

If a player has high drift but shouldn't:

1. Check `similarity_history` — is it a gradual decay or sudden drop?
   - Gradual: Lighting, shadow, or occlusion recovery
   - Sudden: Possible ID swap

2. Check `decision_log` — when were VLM calls triggered?
   - Many calls: Identity instability region
   - Few calls: Stable region with brief glitch

3. Compare with video — does drift correlate with:
   - Occlusions (players off-screen)?
   - Lighting changes (shadows, glare)?
   - Crowd density (overlapping players)?

## Integration with Collision Topology (Future)

The drift report will eventually feed into the Collision Topology Visualizer to:
- Predict which players are likely to swap (manifest as high drift)
- Warn about latent manifold collisions before they cause visible swaps
- Recommend confidence thresholds for production use

---

**API Reference:**

```python
# From services/embedding_drift.py

def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute similarity between embeddings. Returns [-1, 1]."""

class DriftTracker:
    drift_threshold: float  # Default 0.70
    
    def create_anchor(self, pid: str, embedding: np.ndarray) -> None:
        """Store initial embedding for a player."""
    
    def update_drift(self, pid: str, embedding: np.ndarray) -> Optional[float]:
        """Compute and record drift. Returns similarity or None."""
    
    def should_trigger_vlm(self, pid: str, similarity: Optional[float]) -> bool:
        """Decide if VLM should be called."""
    
    def log_vlm_decision(self, pid: str, frame: int, similarity: float, triggered: bool) -> None:
        """Log VLM decision for audit."""
    
    def export_report(self) -> Dict:
        """Export drift tracking data as JSON-serializable dict."""
```
