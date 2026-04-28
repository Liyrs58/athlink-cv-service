#!/bin/bash
JOB_ID=${1:-v2_test}
VIDEO=${2:-/Users/rudra/Downloads/1b16c594_villa_psg_40s_new.mp4}
cd /Users/rudra/athlink-cv-service/athlink-cv-service

# Ensure PYTHONPATH is set so modules can find each other
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Start backend if not already running
if ! pgrep -f uvicorn > /dev/null; then
  python3 -m uvicorn main:app --host 0.0.0.0 --port 8001 &
  UVICORN_PID=$!
  sleep 5
fi

# Submit job
curl -s -X POST http://localhost:8001/api/v1/track/players-with-teams \
  -H "Content-Type: application/json" \
  -d "{\"jobId\":\"$JOB_ID\",\"videoPath\":\"$VIDEO\",\"frameStride\":5,\"maxFrames\":100}" | python3 -m json.tool

sleep 5

# Post-process
python3 validate_v3.py $JOB_ID
python3 stitcher.py $JOB_ID
python3 validate_v3.py $JOB_ID

# Cleanup if we started it
if [ ! -z "$UVICORN_PID" ]; then
  kill $UVICORN_PID
fi
