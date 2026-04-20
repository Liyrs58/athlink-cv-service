# Codebase Audit Report

**Generated:** April 18, 2026  
**Scope:** Services and routes analysis  
**Status:** 82% service utilization, 93% route utilization

---

## Executive Summary

| Metric | Count | Status |
|--------|-------|--------|
| Total Services | 63 | - |
| **Active** (called by routes) | 52 | ✅ 82% |
| **Utility** (called by services only) | 6 | ⚙️ 9% |
| **Orphaned** (never called) | 5 | ❌ 7% |
| Total Routes | 33 | - |
| **Registered** in main.py | 31 | ✅ 93% |
| **Dead Code** (unregistered) | 2 | ❌ 7% |

---

## Services by Category

### ✅ ACTIVE SERVICES (52) — Called by routes

These services are wired up and actively used:

| Service | Called By | Purpose |
|---------|-----------|---------|
| `analytics_overlay_service` | analytics_overlay.py | Overlay renderings |
| `analytics_service` | analytics.py | EPL-style analytics |
| `ball_tracking_service` | analyse.py | Ball detection + Kalman |
| `brain_service` | analyse.py | Tactical analysis engine |
| `camera_compensator` | analyse.py | Motion compensation |
| `confidence_service` | analyse.py | Detection confidence filters |
| `conversation_service` | conversation.py | Gemini Q&A |
| `counter_press_service` | counter_press.py | Counter-press metrics |
| `defensive_line_service` | defensive_line.py | Defensive alignment |
| `detection_service` | detect.py | YOLO player detection |
| `entropy_service` | analyse.py | Entropy-based analytics |
| `event_service` | events.py | Pass/shot/tackle events |
| `export_service` | export.py | JSON export aggregation |
| `fatigue_clock_service` | analyse.py | Player fatigue tracking |
| `formation_service` | formation.py | Formation detection |
| `frame_service` | frames.py, detect.py | Frame extraction |
| `game_brain` | analyse.py | Game state analysis |
| `heatmap_service` | heatmap.py | Player heatmaps |
| `highlight_service` | highlight.py | Auto highlight detection |
| `homography_service` | analyse.py | Pitch homography |
| `interpretation_service` | analyse.py | High-level interpretation |
| `job_queue_service` | 10 routes (analyse, highlight, jobs, match_pipeline, oracle, pitch, render, spotlight, stream, tactics, track) | Async job queue |
| `job_service` | analyze.py | Job management |
| `match_oracle_service` | oracle.py | Tactical fingerprinting |
| `match_pipeline_service` | match_pipeline.py | Full pipeline orchestration |
| `memory_service` | analyse.py | Session memory |
| `multi_pass_validator` | analyse.py | Multi-pass validation |
| `observer_brain` | analyse.py | Observer-based tactics |
| `pass_network_service` | pass_network.py, report_cards.py | Pass graphs |
| `physics_corrector` | analyse.py | Physics correction |
| `pitch_service` | pitch.py | Homography calibration |
| `pressing_service` | pressing.py | PPDA + pressing |
| `reid_service` | analyse.py | Re-identification |
| `render_service` | render.py | Video annotation |
| `report_card_service` | report_cards.py | Player report cards |
| `runpod_service` | analyse.py | RunPod integration |
| `set_piece_service` | set_pieces.py | Set piece detection |
| `shape_service` | analyse.py | Shape analysis |
| `speed_estimator` | analyse.py | Speed estimation |
| `spotlight_service` | spotlight.py | Player clips |
| `storage_service` | storage.py, track.py | Supabase upload |
| `tactics_service` | tactics.py | Tactical analysis |
| `team_separation_service` | analyse.py | Team separation |
| `team_service` | track.py | Team color assignment |
| `tracking_service` | analyse.py, health.py, track.py | YOLO11 + BoT-SORT |
| `trajectory_service` | analyse.py | Trajectory analysis |
| `validation_service` | track.py | Content validation |
| `velocity_service` | analyse.py | Velocity calculations |
| `video_annotator` | analyse.py | Video frame annotation |
| `video_service` | video.py | Video introspection |
| `voronoi_service` | analyse.py | Voronoi diagrams |
| `xg_service` | xg.py | Expected goals |

---

### ⚙️ UTILITY SERVICES (6) — Called by services only

These services support other services but have no direct route:

| Service | Called By | Purpose |
|---------|-----------|---------|
| `ball_physics_service` | pitch_service.py | Ball arc physics (Z-axis) |
| `ball_sahi_service` | ball_tracking_service.py | SAHI tiled inference |
| `model_cache` | ball_tracking_service.py, tracking_service.py | Model preloading |
| `pnlcalib_service` | homography_service.py, tracking_service.py | Pitch corner calibration |
| `sanity_service` | analytics_service.py | Output validation |
| `scene_classifier` | tracking_service.py | Scene type detection |

**Status:** ✅ These are properly used as internal utilities. No action needed.

---

### ❌ ORPHANED SERVICES (5) — Never called

These services are defined but **nothing calls them**:

| Service | Notes |
|---------|-------|
| `ball_tracker_service` | Likely a duplicate of ball_tracking_service |
| `camera_motion_service` | Camera motion compensation (planned but not wired) |
| `replay3d_service` | 3D replay export (unfinished, see route status) |
| `stream_tracker_service` | Stream-based tracking (not implemented) |
| `visual_intelligence_service` | Visual analysis (not implemented) |

**Recommendation:**
- Remove or document their intended purpose
- If needed, integrate into pipeline (camera_motion_service)
- Check if replay3d_service should be wired up

---

## Routes by Status

### ✅ REGISTERED ROUTES (31) — Wired up in main.py

All these routes are properly imported and registered:

```
Core Pipeline:
  ✅ track.py              → /api/v1/track/players-with-teams    [5 services]
  ✅ pitch.py              → /api/v1/pitch/map                   [2 services]
  ✅ tactics.py            → /api/v1/tactics/analyze             [2 services]
  ✅ export.py             → /api/v1/export/{jobId}              [1 service]
  ✅ health.py             → /api/v1/health                      [1 service]

Detection & Frames:
  ✅ frames.py             → /api/v1/video/sample-frames         [1 service]
  ✅ detect.py             → /api/v1/detect/players              [2 services]
  ✅ video.py              → /api/v1/video/inspect               [1 service]

Analytics (single-service routes):
  ✅ analytics.py          → /api/v1/analytics                   [1 service]
  ✅ counter_press.py      → /api/v1/counter-press               [1 service]
  ✅ defensive_line.py     → /api/v1/defensive-line              [1 service]
  ✅ events.py             → /api/v1/events/{jobId}              [1 service]
  ✅ formation.py          → /api/v1/formation/{jobId}           [1 service]
  ✅ heatmap.py            → /api/v1/heatmap/{jobId}             [1 service]
  ✅ highlight.py          → /api/v1/highlight/{jobId}           [2 services]
  ✅ pass_network.py       → /api/v1/pass-network/{jobId}        [1 service]
  ✅ pressing.py           → /api/v1/pressing/{jobId}            [1 service]
  ✅ set_pieces.py         → /api/v1/set-pieces/{jobId}          [1 service]
  ✅ xg.py                 → /api/v1/xg/{jobId}                  [1 service]

Advanced Features:
  ✅ analyse.py            → /api/v1/analyse                     [24 services] ← MONOLITH
  ✅ analytics_overlay.py  → /api/v1/analytics-overlay           [2 services]
  ✅ conversation.py       → /api/v1/conversation/ask            [1 service]
  ✅ match_pipeline.py     → /api/v1/match/pipeline              [2 services]
  ✅ oracle.py             → /api/v1/oracle                      [2 services]
  ✅ render.py             → /api/v1/render/annotate             [2 services]
  ✅ report_cards.py       → /api/v1/reports                     [2 services]
  ✅ spotlight.py          → /api/v1/spotlight                   [2 services]
  ✅ stream.py             → /api/v1/stream                      [1 service]
  ✅ storage.py            → /api/v1/storage/upload              [1 service]

Job Management:
  ✅ analyze.py            → /api/v1/analyze                     [1 service]
  ✅ jobs.py               → /api/v1/jobs/status/{jobId}         [1 service]
```

---

### ❌ UNREGISTERED ROUTES (2) — Dead code

These route files exist but are **not imported in main.py**:

| Route | Issue | Status |
|-------|-------|--------|
| `detection.py` | Legacy duplicate of detect.py | ❌ Remove |
| `replay3d.py` | New 3D replay feature (in development) | ⏳ Needs integration |

**Location:** `/routes/detection.py` and `/routes/replay3d.py`

**Action:**
- `detection.py` → Delete (replaced by detect.py)
- `replay3d.py` → Either integrate or move to staging

---

## Critical Findings

### 🔴 Issue #1: The `analyse.py` Monolith

**Problem:** The route `analyse.py` calls **24 services directly**, making it a catch-all endpoint.

**Services used by analyse.py:**
```
ball_tracking_service, brain_service, camera_compensator,
confidence_service, entropy_service, fatigue_clock_service,
game_brain, homography_service, interpretation_service,
job_queue_service, memory_service, multi_pass_validator,
observer_brain, physics_corrector, reid_service,
runpod_service, shape_service, speed_estimator,
team_separation_service, tracking_service, trajectory_service,
velocity_service, video_annotator, voronoi_service
```

**Why this matters:**
- `analyse.py` is the full pipeline orchestrator
- Any bug affects 24 downstream services
- Hard to test individual components
- Difficult to scale or parallelize

**Recommendation:**
- ✅ This is intentional (it's the end-to-end pipeline route)
- Keep as-is but document the dependency graph
- Consider adding checkpoints for partial re-runs

---

### 🟡 Issue #2: Orphaned Services Need Decisions

**5 services are unused:**
- `ball_tracker_service` — Check if duplicate
- `camera_motion_service` — Planned feature (camera compensation)
- `replay3d_service` — Should be integrated
- `stream_tracker_service` — Not implemented
- `visual_intelligence_service` — Not implemented

**Action Required:**
1. **Immediate:** Check if `ball_tracker_service` is a duplicate → delete
2. **Short-term:** Integrate `replay3d_service` into route
3. **Future:** Plan `camera_motion_service` + `stream_tracker_service`

---

### 🟡 Issue #3: Dead Route Files

**2 route files are not wired up:**
- `detection.py` → Legacy, replaced by `detect.py`
- `replay3d.py` → Exists but route not registered in main.py

**Action Required:**
1. Delete `routes/detection.py`
2. Check `routes/replay3d.py` — does it have a working handler?
   - If yes, add to main.py: `app.include_router(replay3d_router, prefix="/api/v1/replay3d")`
   - If no, move to staging or delete

---

## Summary Table

### Services
| Category | Count | Health | Notes |
|----------|-------|--------|-------|
| Active | 52 | ✅ Good | Wired up via routes |
| Utility | 6 | ✅ Good | Helper functions |
| Orphaned | 5 | ❌ Action needed | Remove or integrate |
| **Total** | **63** | **82%** | 5 services unused |

### Routes
| Category | Count | Health | Notes |
|----------|-------|--------|-------|
| Registered | 31 | ✅ Good | In main.py |
| Dead code | 2 | ❌ Action needed | Not in main.py |
| **Total** | **33** | **93%** | 2 routes unused |

---

## Recommended Actions (Priority Order)

### 🔴 Priority 1: Delete Dead Code (5 min)
```bash
rm routes/detection.py              # Legacy duplicate
```

### 🟡 Priority 2: Audit Orphaned Services (10 min)
```bash
# Check if ball_tracker_service is duplicate
grep -r "ball_tracker_service" services/
grep -r "ball_tracking_service" services/

# Check orphaned service implementations
wc -l services/ball_tracker_service.py
wc -l services/camera_motion_service.py
wc -l services/replay3d_service.py
```

### 🟡 Priority 3: Decide on Unregistered Routes (15 min)
- **replay3d.py:** Check if it's complete, then either:
  - ✅ Register in main.py if ready
  - ❌ Move to staging/ if incomplete
  - ❌ Delete if no longer needed

### 🟢 Priority 4: Document Dependencies (30 min)
- Add comments to `analyse.py` listing all 24 service dependencies
- Create a dependency graph visualization (optional)

---

## Files to Review

```
files_to_check = [
    "services/ball_tracker_service.py",
    "services/camera_motion_service.py",
    "services/replay3d_service.py",
    "services/stream_tracker_service.py",
    "services/visual_intelligence_service.py",
    "routes/detection.py",
    "routes/replay3d.py",
]
```

---

## Conclusion

**Codebase Health: 82/100** ✅

The codebase is **well-organized** with high utilization:
- ✅ 82% of services are actively used
- ✅ 93% of routes are registered
- ⚠️ 5 orphaned services need decisions
- ⚠️ 2 unregistered routes need cleanup

**Recommended:** Delete dead code (detection.py), audit orphaned services, and document the analyse.py monolith.

---

**Generated by:** Codebase audit script  
**Analysis Date:** April 18, 2026
