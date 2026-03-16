from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.health import router as health_router
from routes.analyze import router as analyze_router
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
]

app = FastAPI(
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
