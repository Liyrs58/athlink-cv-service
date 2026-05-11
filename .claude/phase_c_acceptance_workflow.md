# Phase C Acceptance Workflow

## Overview

Phase C is the final acceptance test for the broadcast tactical render pipeline on the Villa/PSG `.mov` clip. It verifies that the **detector → BoT-SORT tracking → pitch mapping → team assignment → render_story pipeline** produces:

1. Valid BoT-SORT tracks (≥10 active tracks per wide-pitch frame)
2. Correct pitch homography mapping
3. Proper team assignment (balanced 6-8 per team)
4. Tactical overlay rendering with valid geometry and confidence gates

## Current Status

**Phase A (Detector Recovery):** ✓ **PASSED**
- roboflow_players.pt confirmed detecting 81 players across 5 sampled frames
- No codec issues; no OOD failures
- Ready to proceed to Phase C

**Phase B (Validator Hardening):** ⏳ **DEFERRED** to Phase D
- Tactical failure reasons enum not yet implemented
- Per-story validators (PRESSING_TRIANGLE, TRAP, 1V1, RECOVERY) would be new code
- Can proceed with Phase C using current lenient geometry validators

**Phase C (Renderer Acceptance):** 🔄 **IN PROGRESS**
- Identity engine disabled (line 228 of tracker_core.py: `self.identity = _NoOpIdentity()`)
- Tracking confirmed working: 13-19 unique track IDs per run
- Ready to execute full pipeline: pitch → teams → render

## Execution in Colab

Copy/paste the entire content of `colab_phase_c_acceptance.py` into a Colab cell:

```python
exec(open('/content/athlink-cv-service/colab_phase_c_acceptance.py').read())
```

This will:

### Step 1: Tracking (13.6s)
- Load roboflow_players.pt (classes: 0=ball, 1=goalkeeper, 2=player, 3=referee)
- YOLO predict at conf=0.05 on all frames
- BoT-SORT tracking (match_thresh=0.70, new_track_thresh=0.40)
- Output: `temp/phase_c_acceptance/tracking/track_results.json`
- Expected result: 13-19 unique track IDs across 30 processed frames

### Step 2: Pitch Mapping (~30s)
- Corner detection via HSV green-field masking
- Homography computation from 4 corners
- Output: `temp/phase_c_acceptance/pitch/pitch_map.json`
- Expected result: `homographyFound=True`, `homographyConfidence > 0.85`

### Step 3: Team Assignment (~10s)
- Color histogram clustering (K-means, k=2)
- Team-gated ReID (OSNet unavailable in Colab, uses color fallback)
- Output: `temp/phase_c_acceptance/tracking/team_results.json`
- Expected result: 6-8 players per team (balanced)

### Step 4: Render Story (~20s)
- Load story JSON (if available) or auto-detect from analysis
- Validate carrier, presser, cover (geometry only, no new v2 validators)
- Draw tactical overlay on every frame where geometry valid
- Smooth carrier transitions (5-frame hysteresis on carrier ID change)
- Output: `temp/phase_c_acceptance/render/render_manifest.json` + annotated video
- Expected result: `overlay_drawn_ratio >= 0.80`, `storyOutcome=RENDERED` or `NONE`

## Acceptance Gates (Phase C)

All of the following must pass:

| Gate | Threshold | Rationale |
|------|-----------|-----------|
| `overlay_drawn_ratio` | ≥ 0.80 | At least 80% of frames have valid geometry |
| `storyOutcome` | `RENDERED` or `NONE` | Never `FAILED:<REASON>` (indicates crash) |
| `ballToCarrier_max` | ≤ 1.8m | Ball never more than 1.8m from carrier in rendered frames |
| `triangle_edge_max` | ≤ 8.0m | Largest triangle edge ≤ 8 meters |
| `homography_confidence` | ≥ 0.85 | Pitch mapping confidence (if homography used) |
| `validFrames_ratio` | ≥ 0.75 | At least 75% of all frames satisfy geometry constraints |

## Manifest Structure

Output JSON at `temp/phase_c_acceptance/render/render_manifest.json`:

```json
{
  "jobId": "phase_c_acceptance",
  "framesProcessed": 30,
  "storyType": "PRESSING_TRIANGLE | TRAP | 1V1 | RECOVERY | NONE",
  "storyOutcome": "RENDERED | FAILED:<REASON> | NONE",
  "overlayDrawnRatio": 0.87,
  "framesWithOverlay": [0, 1, 2, 3, ...],
  "perFrameValidity": [
    {
      "frameIndex": 0,
      "valid": true,
      "reasons": []
    },
    {
      "frameIndex": 1,
      "valid": false,
      "reasons": ["CARRIER_CONFIDENCE_LOW", "BALL_TOO_FAR"]
    }
  ],
  "perFrameEvidence": [
    {
      "frameIndex": 0,
      "ballToCarrierM": 0.7,
      "triangleEdgeMaxM": 6.2,
      "homographyConfidence": 0.91,
      ...
    }
  ],
  "thresholdsUsed": {
    "GLOBAL_MIN_CARRIER_CONFIDENCE": 0.65,
    "GLOBAL_MAX_BALL_TO_CARRIER_M": 1.8,
    ...
  }
}
```

## Visual Verification

After Phase C execution in Colab, a rendered MP4 should be available at:

```
/content/athlink-cv-service/temp/phase_c_acceptance/render/annotated_*.mp4
```

Download and inspect in browser (or via agent-browser screenshot):

- **Triangle overlay:** Should not span pitch-wide (edge ≤ 8m)
- **Ball marker:** Should stay within ~30-50px of carrier centerpoint
- **Carrier line:** Should point from carrier to ball (not drift)
- **Overlay stability:** Should fade smoothly (not snap on/off)
- **Anchor geometry:** Should not jump between frames (smooth hysteresis)

## Failure Modes

If a gate fails, check the manifest `perFrameValidity` for reasons:

| Reason | Fix | Phase |
|--------|-----|-------|
| `BALL_TOO_FAR_FROM_CARRIER` | Detector is losing ball; increase conf_min | A (re-run probe) |
| `CARRIER_CONFIDENCE_LOW` | Track ID unstable; check smoother thresholds | D (tune smoother) |
| `TRIANGLE_TOO_LARGE` | Presser/cover too far; use stricter geometry | B (new validators) |
| `HOMOGRAPHY_CONFIDENCE_LOW` | Pitch corners not detected; try manual corners | Triage |
| `TRACK_ID_UNSTABLE` | Identity engine issue (currently disabled) | D (fix identity) |

## If Phase C Passes

1. Disable the `debug_allow_invalid_story` flag (if any) in render_performance_zone.py
2. Verify that `story.json` is auto-detected correctly (or manually provide it)
3. Document the passing acceptance gates in the PR description
4. Plan Phase D sprint: fix identity engine with proper snapshot timing

## If Phase C Fails

1. Check which gate(s) failed
2. If detector issue: re-run Phase A diagnostics (model probe)
3. If geometry issue: enable debug visualization in render_performance_zone.py (draw all attempted triangles, not just rendered ones)
4. If tracker issue: check BoT-SORT hyperparameters in tracking_service.py:806–813
5. If identity issue: defer to Phase D (currently disabled anyway)

## Next Steps

1. Copy `colab_phase_c_acceptance.py` content into a Colab cell and run
2. Inspect the manifest for gate pass/fail
3. Download and visually inspect the annotated MP4
4. Report results: gates passed, visual quality, any anomalies
5. Plan Phase D (identity fix + Phase B validator hardening) if all gates pass
