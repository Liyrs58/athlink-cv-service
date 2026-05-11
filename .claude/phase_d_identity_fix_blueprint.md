# Phase D: Identity Engine Fix Blueprint

## Status

**Phase C Acceptance: ✓ PASSED**
- Detector: working (51 players detected, 81 in 5-frame probe)
- BoT-SORT: working (19 unique track IDs, 13-15 active per frame, 93.3% coverage)
- Identity engine: **disabled** (soft snapshot architecture issue under investigation)

**Next sprint:** Fix identity engine + re-enable + Phase B validator hardening

---

## Root Cause Analysis

### The Problem

Identity engine completely non-functional when enabled:
- 0 locks created (target: 20-40)
- 0 valid identity frames (should be ~80%+)
- All 352 detections marked "unknown" (missing P-IDs)
- Soft recovery loops infinitely from frame 8 onward

### Why It Happens

**Soft snapshot timing mismatch:**

1. **Frames 0-8 (early non-play period):** 
   - Detector returns 0 boxes (dark/cutaway content)
   - BoT-SORT gets empty input
   - `current_count = 0`, `baseline = 8.0` (from prior run)
   - State: NORMAL (not yet play)

2. **Frame 8 (collapse detection):**
   ```
   [SoftCollapse] Frame 8: current=0 baseline=8.0 saved=0
   ```
   - `current=0 < 0.5*baseline` → collapse triggered
   - **Calls `identity.snapshot_soft()` to save healthy state**
   - But healthy state is already lost (all tracks pruned when detections went to 0)
   - Result: `saved=0` (no dormant slots to save)

3. **Frame 15 (recovery entry):**
   ```
   [SoftRecovery] Frame 15: current=13 → entering recovery mode
   [SoftRecoveryCause] F15 no snapshot or consumed early
   ```
   - Now we have 13 players again
   - But no snapshot to revive from (`snapshot_slots=0`)
   - Recovery mode blocked by `allow_new_assignments=False`
   - Result: all 13 detections marked "unknown" (can't create new locks during recovery)

4. **Frames 15+ (infinite recovery loop):**
   - Recovery window lasts 60 frames minimum
   - Every collapse resets `_soft_recovery_frames = 60`
   - Pattern repeats: collapse → no snapshot → recovery → collapse → …

### Root Cause

**Snapshot timing is reactive, not proactive.** It fires AFTER collapse (when healthy state is already lost), not BEFORE collapse (when healthy state is available).

---

## Solution: Proactive Snapshot Architecture

### New State Machine

Add a **proactive snapshot** phase that runs independently of collapse detection:

```python
# In TrackerCore.__init__:
self._snapshot_window = (15, 30)  # snapshot every 15-30 frames while healthy
self._last_proactive_snapshot = -100
self._proactive_snapshot_threshold = 0.75  # 75% of baseline
```

### Logic

In `on_frame()` (before any recovery/collapse logic):

```python
# PROACTIVE snapshot: capture healthy state while active_count is high
if (current_count >= self._proactive_snapshot_threshold * self._active_baseline
    and video_frame - self._last_proactive_snapshot >= 15):
    
    saved_count = self.identity.snapshot_soft(video_frame)
    if saved_count > 0:
        self._last_proactive_snapshot = video_frame
        print(f"[ProactiveSnapshot] F{video_frame}: saved {saved_count} slots "
              f"(current={current_count:.1f} baseline={self._active_baseline:.1f})")
```

### Key Differences from v1 (Prior Attempted Fix)

**v1 (failed):**
- Ran only inside `if is_play` block → never triggered during early non-play frames
- Threshold too high (90%) → no state to save when baseline=8 and current=0
- Only fired at collapse time (reactive)

**v2 (this fix):**
- Runs at **frame entry**, before any state checks → always available
- Threshold moderate (75%) → fires during healthy stable play, not during collapse
- Runs continuously → builds up a rolling snapshot of the last healthy period
- Scene-gated: snapshot only during PLAY state, not CUTAWAY

### Implementation Location

**File:** `services/tracker_core.py:on_frame()` method

**Insertion point:** Line ~500, just after `_update_scene_state()` call, before recovery/collapse checks

```python
def on_frame(self, frame_idx, video_frame_idx, frame, detections):
    # ... existing setup ...
    
    current_count = len([t for t in self._trk.tracks if t.hits >= 2])
    is_play = self._is_play_frame(video_frame_idx)
    
    # NEW: Proactive snapshot (before any collapse/recovery logic)
    if is_play:
        if (current_count >= 0.75 * max(self._active_baseline, 1.0)
            and video_frame_idx - self._last_proactive_snapshot >= 15):
            saved = self.identity.snapshot_soft(video_frame_idx)
            if saved > 0:
                self._last_proactive_snapshot = video_frame_idx
    
    # ... existing collapse/recovery detection (unchanged) ...
```

---

## Phase D Sprint Plan

### Step 1: Proactive Snapshot (Day 1)
- [ ] Add `_last_proactive_snapshot`, `_snapshot_window`, `_proactive_snapshot_threshold` to `__init__`
- [ ] Insert proactive snapshot logic at frame entry
- [ ] Test on Villa/PSG clip: verify `saved > 0` in logs
- [ ] Acceptance gate: ≥ 3 proactive snapshots per 30-frame run

### Step 2: Re-enable Identity (Day 2)
- [ ] Change line 230: `self.identity = IdentityCore()`
- [ ] Disable `_NoOpIdentity()` passthrough
- [ ] Run full pipeline: expect 20-40 locks, ≥ 80% identity coverage
- [ ] Acceptance gates:
  - `collapse_lock_creations = 0` (soft collapse blocking works)
  - `recovery_normal_assignments = 0` (recovery window blocks new assignments)
  - `locks_created >= 20`
  - `lock_retention_rate >= 0.65`

### Step 3: Phase B Validator Hardening (Day 3)
- [ ] Create `services/tactical_failure_reasons.py` (enum of failure strings)
- [ ] Extend `services/story_validators.py` with v2 validators:
  - `validate_global_truth()`
  - `validate_pressing_triangle_v2()`
  - `validate_pressing_trap_v2()`
  - `validate_1v1_pressure()`
  - `validate_recovery_run()`
- [ ] Wire into `render_performance_zone.py:render_story()`
- [ ] Acceptance gate: manifest includes `perFrameValidity` with reasons

### Step 4: Renderer Gate Strengthening (Day 4)
- [ ] Add stable-frames-before-show hysteresis (5-8 frames by story type)
- [ ] Add fade-after-3-invalid-frames transition
- [ ] Hard global gate: `carrier_conf >= 0.65`, `ball_to_carrier <= 1.8m`
- [ ] Acceptance gate: `overlay_drawn_ratio >= 0.80`

### Step 5: End-to-End Verification (Day 5)
- [ ] Run full Phase A/B/C pipeline with all fixes
- [ ] Verify manifest gates:
  - `collapse_lock_creations = 0`
  - `locks_created >= 20`
  - `overlay_drawn_ratio >= 0.80`
  - `story_outcome = RENDERED` (or NONE if no tactical pattern)
  - `ballToCarrier_max <= 1.8m`
  - `triangle_edge_max <= 8m`
- [ ] Visual inspection: no drifting ball marker, no cross-pitch triangles, smooth transitions

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Snapshot still fires during non-play → dilutes recovery buffer | Add `if is_play` gate (simple, already planned) |
| Snapshot fires but state still empty (early in run) | Accept: threshold is 75%, so only fires when healthy |
| Phase B validators too strict → `overlay_drawn_ratio` drops below 0.80 | All new validators are additions; geometry validators unchanged |
| Re-enabling identity breaks something else | Run on same Villa/PSG clip; if gates pass, ship it |

---

## Verification Checklist

### Pre-Phase D (Complete ASAP)
- [x] Phase C gates pass (93.3% coverage, 19 IDs, 15 per frame)
- [x] Detector working (51 dets, 81 in probe)
- [ ] Disable identity engine (already done)

### Post-Phase D (Per-step acceptance)
- [ ] Proactive snapshot fires ≥ 3 times: `[ProactiveSnapshot]` in logs
- [ ] Identity re-enabled: `self.identity = IdentityCore()`
- [ ] Locks created: ≥ 20 (check `locks_created` in metrics)
- [ ] Identity coverage: ≥ 80% (check `valid_id_coverage` in metrics)
- [ ] Manifest includes `perFrameValidity` with reasons
- [ ] Overlay drawn: ≥ 80% of frames
- [ ] Ball-to-carrier: ≤ 1.8m max (gates report this)
- [ ] Triangle edge: ≤ 8m max (geometry validators)
- [ ] Visual: no stale geometry, smooth transitions, ball marker stable

---

## Files to Modify

| File | Changes | Complexity |
|------|---------|-----------|
| `services/tracker_core.py` | Add proactive snapshot state + logic at frame entry | LOW |
| `services/tracker_core.py` | Line 230: re-enable IdentityCore | TRIVIAL |
| **NEW:** `services/tactical_failure_reasons.py` | Enum of failure strings | LOW |
| `services/story_validators.py` | Add 5 new v2 validators + constants | MEDIUM |
| `services/render_performance_zone.py` | Wire validators + stable-frames hysteresis + hard gates | MEDIUM |

---

## Success Criteria

Phase D is **COMPLETE** when:

1. **Detector** ✓ (already proven in Phase A)
2. **Identity** — locks created ≥ 20, coverage ≥ 80%
3. **Rendering** — overlay_drawn_ratio ≥ 0.80, geometry valid
4. **Manifest** — all gates pass, includes perFrameValidity with reasons

Target: All criteria pass on Villa/PSG clip within 5 working days.
