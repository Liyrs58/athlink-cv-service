# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Start the server (port 8001 avoids macOS AirPlay conflict on 5000/7000)
cd /Users/rudra/Desktop/athlink-cv-service
source .venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# Full pipeline test (run in order)
curl -s -X POST http://localhost:8001/api/v1/track/players-with-teams \
  -H "Content-Type: application/json" \
  -d '{"jobId":"test_v1","videoPath":"/path/to/video.mp4","frameStride":3,"maxFrames":150}' | python3 -m json.tool

python3 verify_tracking.py test_v1 --render
python3 verify_teams.py test_v1 --render

curl -s -X POST http://localhost:8001/api/v1/pitch/map \
  -H "Content-Type: application/json" \
  -d '{"jobId":"test_v1","videoPath":"/path/to/video.mp4"}' | python3 -m json.tool

python3 verify_pitch.py test_v1

curl -s -X POST http://localhost:8001/api/v1/tactics/analyze \
  -H "Content-Type: application/json" \
  -d '{"jobId":"test_v1"}' | python3 -m json.tool
```

## Architecture

### Processing Pipeline (ordered)

```
Video → /api/v1/track/players-with-teams
      → /api/v1/pitch/map
      → /api/v1/tactics/analyze
```

Each step depends on the output of the previous. All intermediate outputs are saved to `temp/{jobId}/`:

| Step | Service | Output |
|------|---------|--------|
| Tracking | `tracking_service.run_tracking()` | `tracking/track_results.json` |
| Teams | `team_service.assign_teams()` | `tracking/team_results.json` |
| Pitch | `pitch_service.map_pitch()` | `pitch/pitch_map.json` |
| Tactics | `tactics_service.analyze_tactics()` | `tactics/tactics_results.json` |

### Service Architecture Rule

- `frame_service.py` uses a **class** (`FrameService` with static methods)
- All other services use **plain module-level functions** — do not convert them to classes

### Key Design Decisions

**Portrait-mode video handling**: Phone-recorded videos (e.g. `RPReplay_*.MP4`) are typically 1792×828 portrait. YOLO fails to detect upright persons in sideways frames. All services check `raw_h > raw_w` and apply `cv2.ROTATE_90_COUNTERCLOCKWISE`. This must be consistent across tracking, pitch, and all verify scripts.

**No sklearn**: All ML operations (clustering, heatmaps, formation detection) use numpy only.

**No homography on dirt pitches**: `pitch_service` detects pitch corners via HSV green-field masking. Dirt/sand pitches return `homographyFound=False` and fall back to proportional normalization (`px / frame_w * 105`). Fallback coords are valid but less accurate.

**Model caching**: YOLO model is lazy-loaded once and cached globally. `tracking_service._model` and `detection_service._detection_service` are module-level singletons.

**Track filtering**: `run_tracking()` only keeps tracks with `hits >= 2` in the output. Tracks created early and pruned by `max_track_age` are lost (not accumulated in a completed-tracks list — this is by design).

### Routes Structure

All routes are thin — they validate input, delegate to a service function, and return a Pydantic response model.

```
routes/health.py     → GET  /api/v1/health
routes/analyze.py    → POST /api/v1/analyze
routes/video.py      → POST /api/v1/video/inspect
routes/frames.py     → POST /api/v1/video/sample-frames
routes/detect.py     → POST /api/v1/detect/players
routes/track.py      → POST /api/v1/track/players
                     → POST /api/v1/track/players-with-teams
routes/pitch.py      → POST /api/v1/pitch/map
routes/tactics.py    → POST /api/v1/tactics/analyze
```

Note: `routes/detection.py` exists but is **not registered** in `main.py`. Use `routes/detect.py` instead.

### Pydantic Models

Two model files exist with some overlap:
- `models/analysis.py` — primary, used by most routes; includes `FrameSampleRequest`, `PlayerDetectionRequest`, `Detection`, `BoundingBox`
- `models/detection.py` — older duplicate, used by `routes/detection.py` only (which is inactive)

### Environment Variables

| Variable | Default | Used in |
|----------|---------|---------|
| `YOLO_MODEL_PATH` | `yolov8n.pt` | `tracking_service.py` |
| `YOLO_CONF` | `0.35` | `tracking_service.py` |
| `YOLO_IOU` | `0.45` | `tracking_service.py` |
| `TRACK_IOU_THRESHOLD` | `0.3` | `tracking_service.py` |
| `FRAME_DARK_THRESHOLD` | `20.0` | `frame_service.py` |

### Verify Scripts

Standalone scripts for visual debugging — not part of the API:

- `verify_tracking.py <jobId> [--render]` — prints track table, optionally renders annotated frames to `temp/{jobId}/verify_tracking/`
- `verify_teams.py <jobId> [--render]` — prints team assignment table, optionally renders team-coloured bboxes to `temp/{jobId}/verify_teams/`
- `verify_pitch.py <jobId>` — renders top-down pitch diagram with player positions to `temp/{jobId}/verify_pitch/`
- `verify_detections.py <jobId>` — renders YOLO detection bboxes on sampled frames
