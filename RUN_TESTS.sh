#!/bin/bash
# Test suite runner for tracking improvements
# Usage: ./RUN_TESTS.sh /path/to/video.mp4

set -e

VIDEO_PATH="${1:-}"
JOB_ID="tracking_test_$(date +%s)"
TEMP_DIR="temp/$JOB_ID"

if [ -z "$VIDEO_PATH" ]; then
    echo "Usage: ./RUN_TESTS.sh /path/to/video.mp4"
    echo ""
    echo "Test suite validates:"
    echo "  ✓ ID fragmentation fix (match_thresh 0.35, appearance_thresh 0.55)"
    echo "  ✓ Kalman coasting on invalid frames"
    echo "  ✓ Team classification accuracy"
    echo "  ✓ Track stability metrics"
    exit 1
fi

if [ ! -f "$VIDEO_PATH" ]; then
    echo "✗ Video not found: $VIDEO_PATH"
    exit 1
fi

echo "=========================================="
echo "TRACKING IMPROVEMENTS TEST SUITE"
echo "=========================================="
echo "Video:       $VIDEO_PATH"
echo "Job ID:      $JOB_ID"
echo "Output dir:  $TEMP_DIR"
echo ""

# Step 1: Set up environment
echo "[1/4] Setting up environment..."
source .venv/bin/activate || { echo "✗ venv not found"; exit 1; }
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
mkdir -p "$TEMP_DIR"

# Step 2: Run tracking with fixed parameters
echo "[2/4] Running tracking (match_thresh=0.35, appearance_thresh=0.55)..."
python3 << 'EOF'
import sys
import json
from pathlib import Path
from services.tracking_service import run_tracking

video_path = sys.argv[1]
job_id = sys.argv[2]

try:
    result = run_tracking(
        video_path=video_path,
        job_id=job_id,
        frame_stride=5,
        max_frames=300,  # Cap at 300 frames (~10s @ 30fps)
        max_track_age=90,
    )
    print(f"✓ Tracking complete: {result['trackCount']} tracks, {result['framesProcessed']} frames")
except Exception as e:
    print(f"✗ Tracking failed: {e}")
    sys.exit(1)
EOF "$VIDEO_PATH" "$JOB_ID"

# Step 3: Run validation tests
echo "[3/4] Running validation tests..."
python3 test_tracking_improvements.py \
    --results "temp/$JOB_ID/tracking/track_results.json" \
    --output "temp/$JOB_ID/test_results.json"

# Step 4: Summary
echo "[4/4] Generating summary..."
RESULTS_FILE="temp/$JOB_ID/test_results.json"
if [ -f "$RESULTS_FILE" ]; then
    echo ""
    echo "=========================================="
    echo "TEST RESULTS"
    echo "=========================================="
    python3 -m json.tool "$RESULTS_FILE" | tail -50
    echo ""
    echo "Full results: $RESULTS_FILE"
fi

echo ""
echo "✓ Test suite complete!"
echo ""
echo "Next steps:"
echo "  1. Review test results above"
echo "  2. Run with different videos to validate generalization"
echo ""
