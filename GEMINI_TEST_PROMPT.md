# Gemini Testing Prompt for Tracking Improvements

Use this prompt with Gemini to run comprehensive testing of the football player tracking fixes.

---

## Context

I've just implemented 3 major improvements to a football player tracking pipeline using YOLO11 + BoT-SORT:

### Changes Made:

1. **BoT-SORT parameter tuning:**
   - `match_thresh`: 0.8 → 0.35 (fixes motion association with frame stride=5)
   - `appearance_thresh`: 0.25 → 0.55 (prevents cross-team ID swaps)
   - `track_high_thresh`: 0.3 → 0.5
   - `new_track_thresh`: 0.35 → 0.4
   - `track_buffer`: 300 → 150 frames
   - Detection confidence: 0.20 → 0.10 (enables BYTE low-confidence recovery)

2. **Infrastructure improvements:**
   - Kalman coasting on invalid frames (cutaway shots)
   - Foot-point-based world coordinate projection (pixel_to_world)
   - Wider pitch polygon margins (3m → 5m)
   - Torso-only color histogram (25-65% vertical crop)
   - Stricter ReID stitching (0.40 → 0.30 threshold, team-gated)

3. **FingerprintDB track resurrection system (optional, for future implementation):**
   - Multi-signal fingerprinting (color histogram, embedding, position)
   - Team-gated matching (never merges across teams)
   - Resurrects lost tracks when new detection matches fingerprint (score < 0.40)

### Expected Outcomes:

- **ID fragmentation:** 26 unique IDs → ≤18 (for 14 visible players)
- **ID switches:** Significant reduction (< 5 per 10-second clip)
- **Team balance:** 6-8 players per team (roughly equal)
- **Track stability:** Consistent bounding boxes, no color flickering
- **Visual stability:** Players near sideline/throw-in tracked (wider polygon margin working)

---

## Testing Instructions

### Step 1: Setup

```bash
cd /Users/rudra/athlink-cv-service/athlink-cv-service
source .venv/bin/activate
```

### Step 2: Run Full Test Suite

```bash
./RUN_TESTS.sh /path/to/test_video.mp4
```

This automatically:
1. Runs tracking with the fixed parameters
2. Validates 8 metrics (ID count, switches, team balance, track lengths, etc.)
3. Generates test_results.json
4. Prints summary to console

### Step 3: Manual Testing (if RUN_TESTS.sh doesn't work)

**Start the server:**

```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

**In another terminal, run tracking:**

```bash
curl -s -X POST http://localhost:8001/api/v1/track/players-with-teams \
  -H "Content-Type: application/json" \
  -d '{
    "jobId":"fix_test",
    "videoPath":"/path/to/video.mp4",
    "frameStride":5,
    "maxFrames":300
  }' | python3 -m json.tool > temp/fix_test/tracking/track_results.json
```

**Run validation:**

```bash
python3 test_tracking_improvements.py \
  --results temp/fix_test/tracking/track_results.json \
  --output test_results.json
```

**Generate visuals:**

```bash
python3 verify_tracking.py fix_test --render
python3 verify_teams.py fix_test --render
```

---

## Validation Checklist

### Quantitative Metrics (from test_results.json)

- [ ] **Unique ID Count** ≤ 18 (was 26)
- [ ] **ID Switches** < 5 (was higher)
- [ ] **Team Balance** 6-8 per team (no lopsided assignment)
- [ ] **Cross-Team Merges** = 0 (team gate working)
- [ ] **Track Lengths** min ≥ 8 detections (MIN_TRACK_DETECTIONS)
- [ ] **Confidence Score** avg > 0.6 (high-quality tracks)
- [ ] **Valid Frames** ≥ 75% (enough gameplay data)

### Qualitative Validation (from verify_tracking.py and verify_teams.py outputs)

#### verify_tracking/ subdirectory:

- [ ] Bounding boxes are stable (same player = same color across frames)
- [ ] No rapid color flicker (ID swapping would show as color change)
- [ ] Players near sideline/throw-in are tracked (wider polygon margin working)

#### verify_teams/ subdirectory:

- [ ] Team 0 players have consistent team color
- [ ] Team 1 players have consistent team color
- [ ] No mix-ups (team jumping to opposite team or vice versa)
- [ ] Rough balance between team sizes visible in frames

#### Logs (from terminal output):

- [ ] No excessive errors in tracking pipeline
- [ ] Kalman smoothing applied to world coordinates
- [ ] Team classification completed for all tracks

---

## Debug Commands (if issues arise)

### Check config is correct:

```bash
cat tracker_config/botsort_football.yaml
# Should show: match_thresh: 0.35, appearance_thresh: 0.55, etc.
```

### Check that detection confidence is set to 0.10:

```bash
grep -n "YOLO_CONF\|det_conf" services/tracking_service.py
```

### Check pixel_to_world is using foot-point:

```bash
grep -n "def pixel_to_world" services/tracking_service.py
grep -A 5 "foot" services/tracking_service.py
```

### Test on a short clip first:

```bash
ffmpeg -i input.mp4 -t 10 -c copy short_test.mp4  # 10 second clip
./RUN_TESTS.sh short_test.mp4
```

---

## Expected Output Format (test_results.json)

```json
{
  "summary": {
    "total_tracks": 14,
    "total_frames": 300,
    "job_id": "fix_test",
    "tests_passed": "8/8"
  },
  "tests": [
    {
      "test": "Unique ID Count",
      "metric": 16,
      "expected_max": 18,
      "passed": true,
      "improvement": "✓ ID fragmentation fixed"
    },
    {
      "test": "ID Switches",
      "metric": 3,
      "expected_max": 5,
      "passed": true,
      "comment": "Lower is better; indicates stable tracking"
    }
  ]
}
```

---

## Troubleshooting

**"track_results.json not found"**
→ Tracking failed. Check logs for YOLO/BoT-SORT errors. Ensure video path is correct.

**"Too many unique IDs still (>18)"**
→ Parameters may not have been applied. Verify `botsort_football.yaml` was updated. Restart the server.

**"Team balance is off (e.g., 15 vs 8)"**
→ Check team classification logic in `team_service.py`. The team assignment stage may need tuning.

**"Kalman coasting not working (tracks die on cutaways)"**
→ Verify `model.track()` is called with empty frame on invalid frames in tracking_service.py.

---

## Success Criteria

**Test passes when:**

1. ✓ At least 6/8 validation tests pass
2. ✓ Unique ID count ≤ 18 (improvement from 26)
3. ✓ Team balance within 3 players per side
4. ✓ Visual verification shows stable colors (no flicker)
5. ✓ Tracks remain visible on sidelines (wider polygon margin)

**If all criteria met:** Implementation is complete and validated.
**If some fail:** Debug using commands above; most issues are configuration-related.

---

## When Done

1. Report the test_results.json summary
2. Share screenshots of verify_tracking/ showing stable colors
3. Confirm tests pass
4. Ready to commit and push to GitHub

Good luck! 🚀
