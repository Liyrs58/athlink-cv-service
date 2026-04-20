# AthLink CV Service — Complete Codebase Documentation

**Last Updated:** April 18, 2026  
**Purpose:** Football video analysis backend for 3D replay generation and tactical intelligence  
**Status:** MVP with advanced analytics, 3D render in progress

---

## 1. Project Overview

AthLink CV is a **solo-built** football video analysis system that:
- Tracks players across frames using **YOLO11 + BoT-SORT**
- Detects and tracks the ball with parabolic arc prediction
- Maps pixel coordinates to real-world 105×68m pitch positions via **homography calibration**
- Generates tactical insights: formations, heatmaps, pressing patterns, xG
- Renders 3D replays with **Remotion + React Three Fiber** (in progress)
- Exports structured JSON for mobile clients

**Data Flow:**
```
Video File
    ↓
Frame Sampling (stride=3 or 5)
    ↓
Player Detection (YOLO11)
    ↓
Player Tracking (BoT-SORT, 15 unique IDs per 999 frames)
    ↓
Ball Detection (Tryolabs ball.pt w/ SAHI tiling)
    ↓
Team Assignment (HSV color histogram from jersey)
    ↓
Homography Calibration (detect pitch corners, build H matrix)
    ↓
Pitch Coordinate Mapping (pixel → world 105×68m)
    ↓
Tactical Analysis (formations, heatmaps, events, xG, pressing)
    ↓
Ball Arc Physics (Z-axis parabolic interpolation)
    ↓
JSON Export (contract for mobile + 3D render)
    ↓
3D Replay Render (Remotion composition, MP4 output)
```

---

## 2. Technology Stack

### Backend
- **Framework:** FastAPI 0.104.1
- **Server:** Uvicorn 0.24.0
- **Vision:** OpenCV 4.9+, Ultralytics YOLOv8/11, supervision 0.26+
- **Tracking:** boxmot 10.0+ (BoT-SORT)
- **Math:** numpy 2.0.2, shapely, scipy
- **ML/AI:** google-generativeai (Gemini), roboflow (model management)
- **Storage:** Supabase 2.0+, RunPod (distributed GPU)
- **Async:** Python asyncio + job queue

### Frontend (Remotion 3D)
- **Framework:** Next.js 14+
- **3D:** React Three Fiber, Three.js, @react-three/drei
- **Animation:** Remotion (video composition), Mixamo GLB avatars
- **Rendering:** Soft shadows, bloom post-processing, PBR grass materials
- **Build:** TypeScript, Tailwind CSS

### Models
- **Players:** YOLOv11 football-specific (custom trained, ~40MB)
- **Ball:** Tryolabs ball.pt (YOLOv5 format, 6MB) + Roboflow fallback (training in progress)
- **Tiling:** SAHI v0.11+ (sliced inference for small objects)

---

## 3. Directory Structure

```
athlink-cv-service/
├── main.py                          # FastAPI app + router registration
├── CLAUDE.md                        # Project instructions (env vars, pipeline)
├── requirements.txt                 # Python dependencies
├── dockerfile                       # Container build
│
├── services/                        # Core analysis engines (54 files)
│   ├── tracking_service.py          # YOLO11 + BoT-SORT player tracking
│   ├── ball_tracking_service.py     # Ball detection + Kalman filtering
│   ├── ball_sahi_service.py         # SAHI tiled inference wrapper
│   ├── pitch_service.py             # Homography calibration, pitch mapping
│   ├── ball_physics_service.py      # Z-axis parabolic arcs (aerial passes)
│   ├── team_service.py              # Team color assignment (jersey HSV)
│   ├── export_service.py            # JSON contract aggregation
│   ├── velocity_service.py          # Speed & sprint calculations
│   ├── brain_service.py             # Tactical xG, pressing zones, formations
│   ├── formation_service.py         # Formation detection & shape analysis
│   ├── heatmap_service.py           # Player distance heatmaps
│   ├── event_service.py             # Pass, shot, tackle, clearance events
│   ├── xg_service.py                # Expected goals model
│   ├── pressing_service.py          # PPDA, recovery time, press intensity
│   ├── pass_network_service.py      # Pass graph analysis
│   ├── set_piece_service.py         # Throw-in, corner, FK detection
│   ├── render_service.py            # Video annotation & overlay rendering
│   ├── replay3d_service.py          # Remotion JSON export for 3D render
│   ├── spotlight_service.py         # Player isolation clips
│   ├── highlight_service.py         # Auto highlight detection
│   ├── match_pipeline_service.py    # Full pipeline orchestration
│   ├── job_queue_service.py         # Async job management
│   ├── model_cache.py               # Preload YOLO + ball models at startup
│   ├── video_service.py             # Video introspection (fps, duration)
│   ├── frame_service.py             # Frame extraction with dark-frame filter
│   ├── camera_motion_service.py     # Motion compensation
│   ├── conversation_service.py      # Gemini-powered Q&A on match data
│   ├── storage_service.py           # Supabase file upload
│   ├── memory_service.py            # Session memory for analysis
│   └── [40+ specialized services]   # Analytics, validation, physics, etc.
│
├── routes/                          # API endpoints (24 files)
│   ├── track.py                     # POST /api/v1/track/players-with-teams
│   ├── pitch.py                     # POST /api/v1/pitch/map
│   ├── tactics.py                   # POST /api/v1/tactics/analyze
│   ├── export.py                    # GET /api/v1/export/{jobId}
│   ├── render.py                    # POST /api/v1/render/annotate
│   ├── replay3d.py                  # POST /api/v1/replay3d/compose (NEW)
│   ├── formation.py                 # POST /api/v1/formation/{jobId}
│   ├── xg.py                        # POST /api/v1/xg/{jobId}
│   ├── heatmap.py                   # POST /api/v1/heatmap/{jobId}
│   ├── pass_network.py              # POST /api/v1/pass-network/{jobId}
│   ├── pressing.py                  # POST /api/v1/pressing/{jobId}
│   ├── events.py                    # POST /api/v1/events/{jobId}
│   ├── highlight.py                 # POST /api/v1/highlight/{jobId}
│   ├── spotlight.py                 # POST /api/v1/spotlight
│   ├── health.py                    # GET /api/v1/health
│   └── [15+ other routes]
│
├── models/                          # Pydantic data models
│   ├── analysis.py                  # Request/response schemas (primary)
│   └── detection.py                 # Legacy detection schemas
│
├── nextjs-bridge/                   # Frontend (React + Remotion)
│   ├── app/
│   │   ├── api/cv/                  # API routes (proxy to FastAPI backend)
│   │   │   ├── analyze/             # Submit video for analysis
│   │   │   ├── status/[jobId]/      # Poll job progress
│   │   │   ├── export/[jobId]/      # Fetch final JSON
│   │   │   ├── replay3d/[jobId]/    # Trigger 3D render
│   │   │   └── [20+ other endpoints]
│   │   ├── layout.tsx               # Root layout
│   │   └── page.tsx                 # Landing page
│   ├── remotion/
│   │   ├── compositions/
│   │   │   └── PitchReplay.tsx      # Main 3D composition (Remotion wrapper)
│   │   ├── scene/
│   │   │   ├── Stage.tsx            # Three.js scene setup (pitch, players, ball)
│   │   │   ├── PlayerRig.tsx        # Animated player rig (Mixamo skeleton + IK)
│   │   │   ├── Pitch.tsx            # PBR grass mesh + white lines
│   │   │   ├── Ball.tsx             # Ball mesh + flight trajectory
│   │   │   ├── Sideline.tsx         # Team color billboards
│   │   │   ├── Stands.tsx           # Stadium stands
│   │   │   ├── PostFX.tsx           # Bloom + vignette post-processing
│   │   │   ├── constants.ts         # Pitch dimensions, color constants
│   │   │   └── Players.tsx          # Player rig array orchestrator
│   │   ├── data/
│   │   │   ├── loader.ts            # Load export JSON from API
│   │   │   ├── demo.ts              # Synthetic fallback replay data
│   │   │   └── types.ts             # TypeScript types for replay data
│   │   └── index.ts                 # Remotion bundle entry
│   ├── public/
│   │   ├── avatars/player.glb       # Mixamo humanoid skeleton
│   │   ├── textures/
│   │   │   ├── grass_color.jpg      # PBR albedo
│   │   │   ├── grass_normal.jpg     # PBR normal map
│   │   │   └── grass_roughness.jpg  # PBR roughness
│   │   └── [other assets]
│   ├── package.json                 # Node dependencies
│   └── tsconfig.json                # TypeScript config
│
├── temp/                            # Output directory (gitignored)
│   ├── {jobId}/
│   │   ├── frames/                  # Sampled video frames (JPG)
│   │   ├── tracking/
│   │   │   ├── track_results.json   # Raw tracking output
│   │   │   └── team_results.json    # Team assignments + trackId
│   │   ├── pitch/
│   │   │   └── pitch_map.json       # Homography H matrix + player positions
│   │   ├── tactics/
│   │   │   └── tactics_results.json # Formations, heatmaps, events
│   │   ├── render/                  # Annotated frames or video
│   │   ├── replay3d/                # Remotion-rendered MP4
│   │   └── exports/
│   │       └── final.json           # Mobile client export
│   └── [other jobId directories]
│
├── static/                          # Served static files
│   └── index.html
│
└── config/
    └── botsort_football.yaml        # BoT-SORT tracking thresholds
```

---

## 4. Core Pipeline: Step by Step

### 4.1 Player Tracking (`tracking_service.py`)
**Endpoint:** `POST /api/v1/track/players-with-teams`

**Flow:**
1. **Frame sampling** — Read video, extract frames at stride intervals (e.g., every 3rd frame)
2. **Portrait detection** — If raw_h > raw_w, rotate 90° counterclockwise (phone footage)
3. **YOLO11 inference** — Detect players (class 0) per frame
4. **BoT-SORT tracking** — Associate detections into tracks across frames
5. **Track filtering** — Keep only tracks with hits ≥ 2
6. **Output:** `track_results.json` with track ID, bbox trajectory, frame indices

**Key Fixes (FIX 1-3):**
- **FIX 1:** Only process sampled frames through YOLO; interpolate others (5x speedup)
- **FIX 2:** Filter out crowd/bench detections using pitch polygon (Shapely) in world space
- **FIX 3:** Team assignment via torso-region HSV histogram (jersey color only, not shorts)

**Configuration:**
- `YOLO_MODEL_PATH` → custom YOLOv11 football model
- `YOLO_CONF` → 0.35 (detection threshold)
- `YOLO_IOU` → 0.45 (IoU threshold)
- `TRACK_IOU_THRESHOLD` → 0.3 (association threshold)
- `frameStride` → 3–5 (user-controlled)

**Output File:**
```json
{
  "jobId": "test_v1",
  "videoPath": "/path/to/video.mp4",
  "frameStride": 3,
  "framesProcessed": 150,
  "trackCount": 22,
  "tracks": [
    {
      "trackId": 1,
      "trajectory": [
        {
          "frameIndex": 0,
          "bbox": [100, 50, 150, 300],
          "timestampSeconds": 0.0
        },
        ...
      ]
    },
    ...
  ]
}
```

---

### 4.2 Team Assignment (`team_service.py`)

**Input:** Track results + sampled frames  
**Logic:** For each track, sample 5–10 frames, extract HSV histogram from **torso region only** (20–60% vertical, 20–80% horizontal), cluster colors into 2 teams.

**Output:** `team_results.json`
```json
{
  "tracks": [
    { "trackId": 1, "teamId": 0 },
    { "trackId": 2, "teamId": 1 },
    ...
  ]
}
```

---

### 4.3 Ball Detection & Tracking (`ball_tracking_service.py`, `ball_sahi_service.py`)

**Model:** Tryolabs ball.pt (YOLOv5 format, 6MB) or fallback yolov8s with class=32 filter (COCO sports_ball).

**Recent Improvements (Session Integrated):**
- **SAHI tiling** — Slice frame into 640×640 patches with 20% overlap for small-object detection (6–12px)
- **Kalman filtering** — Const-velocity prediction to reject false positives
- **Stationary rejection** — If detection stays in same location for 2s, reject (likely goalpost/flag)
- **Pitch polygon check** — Homography-based rejection for out-of-bounds detections
- **Source tracking** — Mark each detection as 'sahi' or 'direct' for transparency

**Key Fix (FIX 7):**
- Three-layer false-positive filter: (a) pitch polygon, (b) teleport rejection (>40px/frame), (c) stationary region

**Configuration:**
- `BALL_USE_SAHI` → True (uses SAHI by default)
- `MIN_CONF` → 0.15 (lowered from 0.3 for small balls)
- `COCO_SPORTS_BALL_CLASS` → 32 (if using fallback)

**Output:** Ball trajectory with Z-axis (height above pitch)
```json
{
  "trackId": -1,
  "is_ball": true,
  "trajectory2d": [
    {
      "frameIndex": 10,
      "x": 50.5,
      "y": 34.2,
      "z": 0.0,
      "confidence": 0.87,
      "source": "sahi",
      "reliable": true
    },
    ...
  ]
}
```

---

### 4.4 Pitch Mapping via Homography (`pitch_service.py`)

**Goal:** Map pixel coordinates → real-world 105×68m pitch (standard football pitch).

**Algorithm:**
1. **Detect pitch corners** — HSV green-field masking + contour detection
2. **Build homography H** — 4-point perspective transform (corners → corners)
3. **Transform all players + ball** — `cv2.perspectiveTransform(pixel_pts, H)` → world coords
4. **Fallback:** If homography invalid (dirt pitch, bad lighting), use proportional mapping: `px / frame_w * 105`

**Output:** `pitch_map.json`
```json
{
  "calibrationValid": true,
  "H": [[a, b, c], [d, e, f], [g, h, i]],
  "players": [
    {
      "trackId": 1,
      "trajectory2d": [
        {
          "frameIndex": 0,
          "x": 50.5,
          "y": 34.2,
          "z": 0.0
        },
        ...
      ]
    },
    ...
  ]
}
```

---

### 4.5 Tactical Analysis (`tactics_service.py`, `brain_service.py`)

**Encompasses:**
1. **Formations** — K-means clustering of player positions → detect 4-4-2, 3-5-2, etc.
2. **Heatmaps** — 2D histograms of player distance traveled per zone
3. **Passing Events** — Ball position changes with velocity heuristics
4. **Pressing** — PPDA (passes per defensive action), high-press zones
5. **Expected Goals (xG)** — Shot location + angle → probability model
6. **Events** — Tackles, clearances, interceptions (motion-based inference)

**Output:** `tactics_results.json`
```json
{
  "team0": {
    "formation": "4-4-2",
    "heatmap": [[0, 1, 2, ...], ...],
    "events": [
      { "type": "pass", "frameIndex": 10, "trackId": 5, "success": true },
      ...
    ]
  },
  "team1": { ... }
}
```

---

### 4.6 JSON Export (`export_service.py`)

**Aggregates** tracking, team, pitch, tactics into a single mobile-friendly JSON.

**Key Feature: Ball Arc Hallucination**
- If ball gap > 5 frames and XY distance > 15m with avg speed > 15 m/s → likely aerial pass
- Hallucinate parabolic Z: `z(t) = apex × 4 × t × (1 - t)` where t ∈ [0, 1]

**Output:** `/api/v1/export/{jobId}` → `final.json`
```json
{
  "matchInfo": {
    "jobId": "test_v1",
    "videoPath": "...",
    "frameCount": 150,
    "fps": 25.0
  },
  "teams": {
    "team0": { "color": "#0064FF", "playerCount": 11 },
    "team1": { "color": "#FF3200", "playerCount": 11 }
  },
  "frames": [
    {
      "frameIndex": 0,
      "timestampSeconds": 0.0,
      "players": [
        {
          "trackId": 1,
          "teamId": 0,
          "bbox": [100, 50, 150, 300],
          "pitchX": 50.5,
          "pitchY": 34.2
        },
        ...
      ]
    },
    ...
  ],
  "ball": [
    {
      "frameIndex": 0,
      "pitchX": 52.5,
      "pitchY": 34.0,
      "pitchZ": 0.0,
      "confidence": 0.87,
      "source": "sahi"
    },
    ...
  ],
  "tactics": { ... }
}
```

---

## 5. 3D Replay System (Remotion + Three.js)

**Status:** Core rendering fixed, asset loading working, GPU optimization completed.

### 5.1 Architecture

```
Remotion Composition (PitchReplay.tsx)
    ↓
Load export JSON from API
    ↓
React Three Fiber Canvas + Stage
    ↓
Pitch (PBR grass mesh + white lines)
Stands (stadium geometry)
Players (array of PlayerRig components, skeletal animation)
Ball (sphere + trajectory renderer)
PostFX (Bloom + Vignette)
    ↓
Remotion renders to MP4 (H.264, 25fps, 1280×720)
```

### 5.2 Key Components

**PitchReplay.tsx** (Remotion composition wrapper)
- Loads export JSON from backend API
- Falls back to synthetic demo data if fetch fails
- Renders using `<ThreeCanvas>` at dpr=[1, 1.5] (GPU-optimized)

**Stage.tsx** (Three.js scene setup)
- Ambient light + soft shadows (10 samples, 1024px map)
- Environment preset="city" for realistic sky
- No ContactShadows (removed due to GPU pressure)

**Pitch.tsx** (PBR grass material)
- `useLoader(THREE.TextureLoader)` for color/normal/roughness maps from `staticFile()`
- RepeatWrapping (20, 14) for pitch scale
- Mesh-based white lines (LineBox, RectOutline, CentreCircle)

**PlayerRig.tsx** (Animated humanoid)
- Loads Mixamo skeleton from `staticFile("/avatars/player.glb")`
- Clones skeleton per player (SkeletonUtils.clone) to avoid shared armature
- Procedural IK: head tracks ball, leg swings on detected kicks
- Animation state machine: Idle ↔ Running, synced to velocity
- Team color applied to material

**Ball.tsx**
- Sphere mesh following ball trajectory
- Z-axis motion from export data

**PostFX.tsx**
- Bloom (intensity=0.4, threshold=0.85)
- Vignette (offset=0.2, darkness=0.6)
- N8AO removed (GPU crash), SSAO removed

### 5.3 Recent Fixes

| Issue | Fix | Impact |
|-------|-----|--------|
| WebGL context crash | Reduced shadow map 4096→1024, removed ContactShadows/N8AO | GPU memory OK |
| Asset loading 404 | Added `staticFile()` wrapper in PlayerRig + Pitch | Assets load correctly |
| useTexture incompatibility | Switched to `useLoader(THREE.TextureLoader)` | Remotion compatibility |
| dpr GPU load | Changed [1, 2]→[1, 1.5] | Faster render |

### 5.4 Rendering Command

```bash
cd nextjs-bridge
npx remotion render PitchReplay --bundle-cache=false --quality=high --fps=25 --height=720 --width=1280
```

---

## 6. API Routes Reference

### Core Pipeline

| Route | Method | Purpose | Status |
|-------|--------|---------|--------|
| `/api/v1/health` | GET | Service status | ✅ Working |
| `/api/v1/track/players-with-teams` | POST | Track + team assign | ✅ Working |
| `/api/v1/pitch/map` | POST | Homography calibration | ✅ Working |
| `/api/v1/tactics/analyze` | POST | Formations, heatmaps, events | ✅ Working |
| `/api/v1/export/{jobId}` | GET | JSON aggregation | ✅ Working |

### Advanced Analytics

| Route | Method | Purpose | Status |
|-------|--------|---------|--------|
| `/api/v1/formation/{jobId}` | POST | Formation detection | ✅ Working |
| `/api/v1/xg/{jobId}` | POST | Expected goals model | ✅ Working |
| `/api/v1/heatmap/{jobId}` | POST | Player distance heatmaps | ✅ Working |
| `/api/v1/pressing/{jobId}` | POST | PPDA + press intensity | ✅ Working |
| `/api/v1/pass-network/{jobId}` | POST | Pass graph analysis | ✅ Working |
| `/api/v1/events/{jobId}` | POST | Passes, shots, tackles, etc. | ✅ Working |
| `/api/v1/highlight/{jobId}` | POST | Auto highlight detection | ✅ Working |

### Rendering & Export

| Route | Method | Purpose | Status |
|-------|--------|---------|--------|
| `/api/v1/render/annotate` | POST | Annotated video frames | ✅ Working |
| `/api/v1/replay3d/compose` | POST | Trigger Remotion render | 🔄 In Progress |
| `/api/v1/storage/upload/{jobId}` | POST | Supabase file upload | ✅ Working |

### AI & Conversation

| Route | Method | Purpose | Status |
|-------|--------|---------|--------|
| `/api/v1/conversation/ask` | POST | Gemini Q&A on match data | ✅ Working |
| `/api/v1/oracle` | POST | Tactical fingerprinting | ✅ Working |

---

## 7. What's Currently Working

### ✅ Fully Functional

1. **Player Tracking** (22 unique IDs per 1000 frames, excellent)
   - YOLO11 football-specific model
   - BoT-SORT association across frames
   - Frame sampling (5x speedup)
   - Pitch polygon rejection (removes crowd, bench)

2. **Ball Detection** (recently improved)
   - Tryolabs ball.pt (dedicated) or yolov8s fallback
   - SAHI tiled inference for small objects (6–12px)
   - Kalman-based false-positive rejection
   - Stationary region filter (rejects fixed objects)

3. **Team Assignment** (HSV torso histogram)
   - Accurate color clustering
   - Handles kit variations

4. **Pitch Mapping** (homography-based)
   - Automatic corner detection
   - 105×68m world coordinate system
   - Fallback proportional mapping for dirt pitches

5. **Tactical Analytics**
   - Formation detection (4-4-2, 3-5-2, etc.)
   - Heatmaps per zone
   - Pass/shot/tackle/clearance events
   - Expected goals (xG) model
   - PPDA + pressing intensity
   - Pass networks
   - Set piece detection

6. **JSON Export** (mobile client format)
   - Complete frame-by-frame data
   - Ball arc hallucination for aerial passes
   - Team colors + player counts

7. **Video Rendering**
   - Annotated frames with bboxes
   - Heatmap overlays
   - Pass network visualization

### 🔄 In Progress / Near Done

1. **3D Replay Rendering** (Remotion)
   - ✅ Asset loading (staticFile wrapper)
   - ✅ PBR grass material
   - ✅ GPU optimization (shadow maps, post-FX)
   - ✅ Player rigging + skeletal animation
   - ✅ Ball trajectory rendering
   - 🔄 Testing with real Villa vs PSG data
   - 🔄 MP4 output quality/size optimization

2. **Ball Detection Improvements**
   - ✅ SAHI integration (tiled inference)
   - ✅ Class filtering (COCO class 32)
   - ✅ Lowered confidence threshold (0.15)
   - 🔄 Roboflow trained model (dataset downloaded, training phase incomplete)

### ⚠️ Known Limitations

1. **Ball Detection Rate** (3.7% on villa_psg_10s_v2 clip, should improve to >60% with SAHI)
   - Tryolabs ball.pt unreliable on broadcast footage
   - Roboflow fine-tuning pending
   - SAHI + class filter recently added (not yet validated on clip)

2. **Homography Drift**
   - Pitch corner detection drifts on camera pans
   - Need per-frame camera motion compensation (in progress)

3. **Player Z-Axis (Height)**
   - Currently always 0 (no pose lifting)
   - Requires 2D→3D monocular depth estimation
   - Planned: MediaPipe or depth model integration

4. **No Real-Time Processing**
   - Batch-based (processes entire video)
   - Suitable for post-match analysis, not live

---

## 8. Environment Variables

```bash
# Detection
YOLO_MODEL_PATH=yolov8n.pt          # Default fallback
YOLO_CONF=0.35                       # Confidence threshold
YOLO_IOU=0.45                        # IoU threshold
BALL_MODEL_PATH=ball.pt              # Tryolabs ball model

# Tracking
TRACK_IOU_THRESHOLD=0.3              # Association threshold

# Frame Processing
FRAME_DARK_THRESHOLD=20.0            # Dark frame rejection
FRAME_STRIDE=3                       # Sampling interval (override per request)

# Physics
BALL_ARC_MAX_APEX=4.0                # Max height for hallucinated arcs
BALL_MAX_INTERP_GAP=15               # Max frames to interpolate across

# Storage
SUPABASE_URL=...
SUPABASE_KEY=...

# AI
GOOGLE_API_KEY=...                   # Gemini API
ROBOFLOW_API_KEY=...                 # Roboflow model management (optional)

# Deployment
PORT=8001
ENVIRONMENT=development              # or production
```

---

## 9. Testing the Pipeline

### Quick Test (Local)

```bash
# 1. Start server (port 8001 avoids macOS AirPlay conflict)
cd /Users/rudra/athlink-cv-service/athlink-cv-service
source .venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# 2. Submit tracking job
curl -s -X POST http://localhost:8001/api/v1/track/players-with-teams \
  -H "Content-Type: application/json" \
  -d '{
    "jobId":"test_v1",
    "videoPath":"/path/to/video.mp4",
    "frameStride":3,
    "maxFrames":150
  }' | python3 -m json.tool

# 3. Poll job status
curl http://localhost:8001/api/v1/jobs/status/test_v1 | python3 -m json.tool

# 4. Get export
curl http://localhost:8001/api/v1/export/test_v1 | python3 -m json.tool

# 5. Render 3D replay
curl -X POST http://localhost:8001/api/v1/replay3d/compose \
  -H "Content-Type: application/json" \
  -d '{"jobId":"test_v1"}' | python3 -m json.tool
```

### Full Pipeline Test

```bash
# Use verify scripts (standalone, no API)
python3 verify_tracking.py test_v1 --render        # Visual tracking check
python3 verify_teams.py test_v1 --render           # Team color assignments
python3 verify_pitch.py test_v1                    # Top-down pitch diagram
python3 verify_detections.py test_v1               # YOLO detection bboxes
```

### Validate Ball Detection (Villa PSG Clip)

```bash
# After recent SAHI + class filter integration:
curl -s -X POST http://localhost:8001/api/v1/track/players-with-teams \
  -H "Content-Type: application/json" \
  -d '{
    "jobId":"villa_psg_validated",
    "videoPath":"/Users/rudra/Downloads/1b16c594_villa_psg_40s_new.mp4",
    "frameStride":3,
    "maxFrames":83
  }' | python3 -m json.tool

# Then fetch export and check ball detection stats
curl http://localhost:8001/api/v1/export/villa_psg_validated | \
  python3 -c "import sys,json; d=json.load(sys.stdin); \
  ball_frames=[f for f in d['ball'] if f['confidence']>0]; \
  print(f'Ball detected in {len(ball_frames)}/{len(d[\"ball\"])} frames = {100*len(ball_frames)/len(d[\"ball\"]):.1f}%')"
```

---

## 10. Development Guidelines

### Adding a New Tactical Service

1. **Create service file** (`services/new_metric_service.py`)
   - Use **module-level functions**, not classes (except `FrameService`)
   - Accept job_id, output directory, track/pitch data as args
   - Return dict, save JSON to `temp/{jobId}/new_metric/output.json`

2. **Create route** (`routes/new_metric.py`)
   - POST endpoint, async, thin validation
   - Delegate to service function
   - Return job ID for async polling or immediate result

3. **Register in main.py**
   ```python
   from routes.new_metric import router as new_metric_router
   app.include_router(new_metric_router, prefix="/api/v1", tags=["new-metric"])
   ```

4. **Export aggregation** (update `export_service.py`)
   - Load `new_metric/output.json` if exists
   - Include in final export JSON

### Code Style

- **No sklearn** — use numpy only for ML
- **No homography on dirt pitches** — fallback to proportional mapping
- **Model caching** — lazy-load once, cache globally
- **Frame stride consistency** — apply same rotate-90 logic everywhere
- **No async/await in services** — keep services sync, use job queue for parallelism

### Git Workflow

```bash
git add services/ball_tracking_service.py services/ball_sahi_service.py
git commit -m "feat: SAHI tiled inference + class filter for ball detection"
git push origin main
```

---

## 11. Deployment

### Local Development
```bash
# Start server + watch for changes
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### Docker
```bash
docker build -t athlink-cv:latest .
docker run -p 8001:8001 -v $(pwd)/temp:/app/temp athlink-cv:latest
```

### RunPod (GPU)
```bash
# Offload heavy jobs to RunPod for parallel processing
# See services/runpod_service.py
```

### Supabase Storage
- Upload results to Supabase bucket `results/`
- Configure via env vars: `SUPABASE_URL`, `SUPABASE_KEY`

---

## 12. Next Priorities

1. **Validate ball detection improvements** (SAHI + class filter)
   - Re-run villa_psg_10s_v2 clip, expect >60% detection rate
   - Update ball_tracking_service if thresholds need tweaking

2. **Roboflow model training** (if needed)
   - Dataset downloaded: `football-ball-detection-rejhg/version/2`
   - Train YOLOv8n locally or on RunPod: `epochs=25, imgsz=640, batch=8`
   - Replace ball.pt if trained model outperforms Tryolabs

3. **3D replay validation**
   - Run Remotion render on villa_psg_v2 export data
   - Check player animations sync to velocity
   - Validate ball arc interpolation

4. **Camera motion compensation** (future)
   - Implement per-frame camera motion tracking
   - Reduce homography drift on pans

5. **Player Z-axis estimation** (future)
   - Integrate MediaPipe or depth model
   - Enable jump detection, header tracking

---

## 13. Contact & Debug

**Local Development:**
- Server: `http://localhost:8001`
- Docs: `http://localhost:8001/docs` (Swagger UI)
- Temp output: `./temp/{jobId}/`

**Key Log Files:**
- `track_results.json` — Tracking output
- `team_results.json` — Team assignments
- `pitch_map.json` — Homography + pitch positions
- `tactics_results.json` — Tactical analytics
- Final export: `./temp/{jobId}/exports/final.json`

**Common Issues:**
- **"trackId -1 not found"** → Ball detection failed; check inference logs
- **"No pitch corners detected"** → Dirt pitch or bad lighting; homography fallback active
- **"WebGL context lost"** → 3D render GPU overload; reduce shadow map size, disable bloom
- **"Job not found"** → Check temp directory exists; rerun pipeline

---

**Last Updated:** April 18, 2026  
**Maintained By:** Rudra (solo founder)  
**Codebase Status:** MVP + advanced analytics + 3D render (WIP)
