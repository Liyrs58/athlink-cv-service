# Project Context

This is a Python computer vision pipeline for grassroots football player tracking.

**Stack:** YOLO, BoT-SORT (boxmot), OpenCV, NumPy, FastAPI

## Context Boundaries

- **DO NOT** index or read `/Users/rudra/ruflo/`. It is an external agent framework (TypeScript/Node) unrelated to this Python CV pipeline.
- **DO NOT** traverse into `node_modules/`, `.git/`, or `temp/` directories.
- **Focus only on:** `services/`, `models/`, `tracker_config/`, and test scripts in root.

## Code Style

- Python 3.11+, type hints optional but appreciated
- NumPy vectorized operations preferred over loops
- All CV logic must run on CPU and MPS (Apple Silicon)

## Key Files & Architecture

### Track Results JSON Schema

The primary output format is `track_results.json` with this structure:

```json
{
  "jobId": "string",
  "videoPath": "string",
  "frameStride": int,
  "framesProcessed": int,
  "trackCount": int,
  "tracks": [
    {
      "trackId": int,
      "hits": int,
      "firstSeen": int,
      "lastSeen": int,
      "teamId": int,
      "confidence_score": float,
      "trajectory": [
        {
          "frameIndex": int,
          "bbox": [x1, y1, x2, y2],
          "confidence": float,
          "world_x": float,
          "world_y": float
        }
      ]
    }
  ]
}
```

Key points:
- Top-level `tracks` array (NOT `frames`)
- Each track contains full `trajectory` across all frames
- Team assignment via `teamId` per track
- World coordinates from pitch homography transformation

### Service Layer

- `tracking_service.run_tracking()` — Main tracking pipeline (YOLO + BoT-SORT + fingerprinting)
- `team_service.assign_teams()` — Team classification (dominant color)
- `pitch_service.map_pitch()` — Pitch homography detection
- `fingerprint.FingerprintDB` — Track resurrection after occlusions

### Verify Scripts (standalone debugging)

- `verify_tracking.py <jobId> [--render]` — Track stability visualization
- `verify_teams.py <jobId> [--render]` — Team assignment validation
- `verify_pitch.py <jobId>` — Top-down pitch diagram

### Test Suite

- `RUN_TESTS.sh /path/to/video.mp4` — Automated test runner
- `test_tracking_improvements.py` — Validation against 8 metrics
- `GEMINI_TEST_PROMPT.md` — Comprehensive testing instructions for external testing

## Recent Improvements

**BoT-SORT Parameter Tuning:**
- `match_thresh`: 0.8 → 0.35 (fixes motion association with frameStride=5)
- `appearance_thresh`: 0.25 → 0.55 (prevents cross-team ID swaps)
- Detection confidence: 0.20 → 0.10 (enables BYTE low-confidence recovery)

**FingerprintDB Track Resurrection:**
- Multi-signal matching: color histogram (30%) + embedding (35%) + position (25%) + time_gap (10%)
- Team-gating prevents cross-team resurrection
- Max age: 120 frames (4 seconds @ 30fps)

**Infrastructure:**
- Kalman coasting on invalid frames (cutaway shots)
- Foot-point-based world coordinate projection (foot of player, not center)
- Wider pitch polygon margins (3m → 5m)
- Stricter ReID stitching (0.40 → 0.30 threshold, team-gated)

**Identity Validity Gate (latest):**

Core invariant: P-IDs only emitted for LOCKED or REVIVED states. PROVISIONAL is internal only.

Key changes made to enforce this:

1. **`identity_locks.py`** — Hard block on Hungarian lock creation during collapse.
   `try_create_lock` returns `(None, "blocked_collapse")` immediately when `in_collapse=True` and `source="hungarian"`. Previously only printed a warning and continued. `collapse_lock_creations` still counts for audit.

2. **`tracker_core.py`** — `allow_new_assignments=False` passed to `assign_tracks` whenever `_soft_recovery_frames > 0 OR in_scene_recovery OR _soft_collapse`. This gates the entire Hungarian path during recovery windows; only locked pairs pass through, all others become UNKNOWN.

3. **`identity_core.py`** — Boot eager-accept path (`_slot_cost` returning 0.30 for empty lost slots) disabled during any recovery/collapse flag. Previously this fired whenever `_recovery_frames_left > 0` regardless of mode, feeding cheap PROVISIONAL assignments through.

4. **`tracker_core.py`** — Calls `identity.end_run_summary()` at end of run (which prints acceptance criteria). Added `unknown_boxes` and `valid_id_coverage` metrics.

**Acceptance criteria (run-level):**
- `collapse_lock_creations = 0`
- `recovery_normal_assignments = 0` during recovery windows
- `locks_created <= 40`
- `lock_retention_rate >= 0.65`
- `unknown_boxes` may increase — expected and acceptable
- `identity_switches=0` alone is NOT a success signal

**Expected log lines after fix:**
```
[RecoveryAssign] F240 locked=9 revived=3 normal=0 blocked=6
[CollapseBlock] Hungarian lock BLOCKED during collapse frame=... tid=... pid=...
[IdentityMetrics]
  collapse_lock_creations      = 0  OK
  recovery_normal_assignments  = 0  OK
```

**Expected Outcomes:**
- Unique IDs: 26 → ≤18 (for 14 visible players)
- ID switches: < 5 per 10-second clip
- Team balance: 6-8 per team
- Occlusion recovery: Tracks survive 2-3 second occlusions
