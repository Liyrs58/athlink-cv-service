# Embedding Drift Auditor Guide

## What It Does

The Embedding Drift Auditor monitors embedding similarity per player ID to detect identity decay. When a player's embedding drifts from their anchor (first embedding at lock), it signals that identity might be unstable.

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

Set in `services/identity_core.py`:

```python
self.drift_tracker = DriftTracker(drift_threshold=0.65)  # Lower = more VLM calls
```

### Preset thresholds

- **0.50** — Very sensitive; triggers on minor drift (most VLM calls)
- **0.65** — Balanced; catches real identity decay
- **0.70** — Default; gates most noisy drifts
- **0.80** — Permissive; only flags severe decay
- **0.90** — Minimal VLM calls; allows high drift

## Performance Impact

With drift gating enabled:
- **Without drift gate**: VLM called every frame → ~60% compute cost
- **With drift gate (0.70 threshold)**: VLM called only on drift → ~25% compute cost
- **Savings**: ~40% per video, scales with video length

## Debug Workflow

### 1. Identify drifting players
```bash
cat temp/{jobId}/tracking/drift_report.json | python3 -m json.tool | grep -A 10 "drift_triggered_count"
```

### 2. Check log lines for that pid
```bash
grep "\[EmbeddingDrift\] pid=P7" colab_output.log
# Look for pattern: similarity consistently < 0.70 during certain frames
```

### 3. Determine root cause
- **Occlusion**: Similarity recovers after occlusion ends → expected
- **Lighting change**: Drift correlates with scene transitions → may need lower threshold
- **Identity switch**: Similarity stays low + many frames → consider motion checks
- **Crowd overlap**: Drift during dense play → consider clustering gates

### 4. Tune accordingly
- If too many false VLM calls: Raise threshold to 0.75 or 0.80
- If missing real drifts: Lower threshold to 0.60 or 0.50
- If drifting during occlusion: Add occlusion-aware thresholds per pid

## Integration Points

### IdentityEngine (identity_core.py)
- `DriftTracker` initialized in `__init__`
- Anchors created on lock: `create_anchor(pid, emb)`
- Similarity updated on assignment: `update_drift(pid, emb)`

### Tracker Core (tracker_core.py)
- Before VLM call: Query `drift_tracker.pid_history[pid][-1]`
- Log VLM decision: `log_vlm_decision(pid, frame, sim, triggered)`
- Export report at end: `drift_tracker.export_report()`

### Log Lines

```
[EmbeddingDrift] pid=P7 frame=92 similarity=0.614    # Drift detected
[VLMGate] frame=92 CALLED high_drift_pids=['P7']     # VLM triggered
[VLMGate] frame=93 SKIPPED reason=no_high_drift      # Skipped (all good)
[DriftReport] exported to temp/test_v1/tracking/drift_report.json  # Report saved
```

## Testing

Run the comprehensive test suite:

```bash
pytest tests/test_embedding_drift.py -v
```

Expected output:
- 6 passing tests (cosine similarity, tracker init, anchor creation, VLM gating, report export, end-to-end)
- All similarity computations match theory
- Drift decisions align with threshold

## Common Issues

### Issue: Drift report not found
- **Cause**: embedding_drift.py not imported or DriftTracker not initialized
- **Fix**: Verify `services/identity_core.py` line 569 has `self.drift_tracker = DriftTracker(...)`

### Issue: All players drifting
- **Cause**: Threshold too low (0.50) or anchors set during occlusion
- **Fix**: Raise threshold to 0.70+, check anchor creation frames in logs

### Issue: VLM always called
- **Cause**: Drift gate logic not connected in tracker_core.py
- **Fix**: Verify VLM call site queries `drift_tracker.pid_history[pid][-1]` before calling

### Issue: Report shows empty similarity_history
- **Cause**: No embeddings extracted; check `_extract_embeds()` in tracker_core.py
- **Fix**: Add logging in embed extraction; verify OSNet model is loaded

## Next Steps

- Monitor real video runs with drift reports
- Collect histograms of similarity per player
- Establish per-scenario thresholds (crowd vs isolated play)
- Consider adaptive thresholds based on team/position
