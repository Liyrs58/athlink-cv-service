from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from routes.health import router as health_router
from routes.analyze import router as analyze_router
from routes.analyse import router as analyse_router
from routes.video import router as video_router
from routes.frames import router as frames_router
from routes import detect
from routes.track import router as track_router
from routes.pitch import router as pitch_router
from routes.tactics import router as tactics_router
from routes.jobs import router as jobs_router
from routes.export import router as export_router
from routes.render import router as render_router
from routes.spotlight import router as spotlight_router
from routes.highlight import router as highlight_router
from routes.pass_network import router as pass_network_router
from routes.xg import router as xg_router
from routes.heatmap import router as heatmap_router
from routes.formation import router as formation_router
from routes.pressing import router as pressing_router
from routes.events import router as events_router
from routes.defensive_line import router as defensive_line_router
from routes.counter_press import router as counter_press_router
from routes.set_pieces import router as set_pieces_router
from routes.report_cards import router as report_cards_router
from routes.storage import router as storage_router
from routes.analytics import router as analytics_router
from routes.match_pipeline import router as match_pipeline_router
from routes.analytics_overlay import router as analytics_overlay_router
from routes.conversation import router as conversation_router
from services.memory_service import clean_memory

TAGS_METADATA = [
    {"name": "health", "description": "Service health and device info"},
    {"name": "analysis", "description": "Submit a video for analysis"},
    {"name": "video", "description": "Video inspection and frame sampling"},
    {"name": "frames", "description": "Extract sample frames from video"},
    {"name": "detect", "description": "YOLO player detection on sampled frames"},
    {"name": "track", "description": "Player tracking with BoT-SORT and team assignment"},
    {"name": "pitch", "description": "Homography-based pitch coordinate mapping"},
    {"name": "tactics", "description": "Formation detection, heatmaps, events, and space occupation"},
    {"name": "jobs", "description": "Async job status polling and listing"},
    {"name": "export", "description": "Aggregated JSON export for mobile clients"},
    {"name": "render", "description": "Annotated video render with overlays and minimap"},
    {"name": "spotlight", "description": "Player selection, spotlight rendering, and clip export"},
    {"name": "highlight", "description": "Automatic highlight detection from player movements"},
    {"name": "pass-network", "description": "Pass network graph analysis"},
    {"name": "xg", "description": "Expected goals (xG) model from shot analysis"},
    {"name": "heatmap", "description": "Player distance heatmaps and sprint analysis"},
    {"name": "formation", "description": "Tactical formation detection and shape shifts"},
    {"name": "pressing", "description": "Pressing intensity, PPDA, and recovery time"},
    {"name": "storage", "description": "Supabase storage upload and URL retrieval"},
    {"name": "analytics", "description": "Unified EPL analytics report"},
    {"name": "match", "description": "Full match pipeline with checkpointing"},
    {"name": "analytics-overlay", "description": "Analytics overlay video rendering"},
    {"name": "conversation", "description": "Conversation Coach — ask questions about your match data"},
]

@asynccontextmanager
async def lifespan(app):
    """Clean corrupted match data at startup."""
    clean_memory()
    yield

app = FastAPI(
    lifespan=lifespan,
    title="AthLink CV Service",
    description=(
        "Football video analysis backend. Tracks players across frames using "
        "YOLOv8 + BoT-SORT, assigns team colors, maps positions to a 105\u00d768 m "
        "pitch coordinate system, and produces tactical analytics (formations, "
        "heatmaps, passing lanes, events). All heavy endpoints are async \u2014 "
        "submit a job, then poll GET /api/v1/jobs/status/{jobId} until complete."
    ),
    version="2.0.0",
    openapi_tags=TAGS_METADATA,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router, prefix="/api/v1", tags=["health"])
app.include_router(analyze_router, prefix="/api/v1", tags=["analysis"])
app.include_router(analyse_router, prefix="/api/v1", tags=["analysis"])
app.include_router(video_router, prefix="/api/v1", tags=["video"])
app.include_router(frames_router, prefix="/api/v1", tags=["frames"])
app.include_router(detect.router, prefix="/api/v1/detect", tags=["detect"])
app.include_router(track_router, prefix="/api/v1/track", tags=["track"])
app.include_router(pitch_router, prefix="/api/v1/pitch", tags=["pitch"])
app.include_router(tactics_router, prefix="/api/v1/tactics", tags=["tactics"])
app.include_router(jobs_router, prefix="/api/v1/jobs", tags=["jobs"])
app.include_router(export_router, prefix="/api/v1/export", tags=["export"])
app.include_router(render_router, prefix="/api/v1/render", tags=["render"])
app.include_router(spotlight_router, prefix="/api/v1/spotlight", tags=["spotlight"])
app.include_router(highlight_router, prefix="/api/v1/highlight", tags=["highlight"])
app.include_router(pass_network_router, prefix="/api/v1", tags=["pass-network"])
app.include_router(xg_router, prefix="/api/v1", tags=["xg"])
app.include_router(heatmap_router, prefix="/api/v1", tags=["heatmap"])
app.include_router(formation_router, prefix="/api/v1", tags=["formation"])
app.include_router(pressing_router, prefix="/api/v1", tags=["pressing"])
app.include_router(events_router, prefix="/api/v1/events", tags=["events"])
app.include_router(defensive_line_router, prefix="/api/v1/defensive-line", tags=["defensive-line"])
app.include_router(counter_press_router, prefix="/api/v1/counter-press", tags=["counter-press"])
app.include_router(set_pieces_router, prefix="/api/v1/set-pieces", tags=["set-pieces"])
app.include_router(report_cards_router, prefix="/api/v1/reports", tags=["reports"])
app.include_router(storage_router, prefix="/api/v1/storage", tags=["storage"])
app.include_router(analytics_router, prefix="/api/v1", tags=["analytics"])
app.include_router(match_pipeline_router, prefix="/api/v1/match", tags=["match"])
app.include_router(analytics_overlay_router, prefix="/api/v1/analytics-overlay", tags=["analytics-overlay"])
app.include_router(conversation_router, prefix="/api/v1", tags=["conversation"])

@app.get("/")
async def root():
    return {
        "message": "AthLink CV Service",
        "description": "Football video analysis backend service",
        "version": "2.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    import os

    # Startup checks — catch import errors before uvicorn binds
    try:
        import cv2
        print(f"[startup] cv2 OK: {cv2.__version__}")
    except ImportError as e:
        print(f"[startup] FATAL: {e}")
        raise SystemExit(1)

    port = int(os.getenv("PORT", "8005"))
    print(f"[startup] Binding to 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
