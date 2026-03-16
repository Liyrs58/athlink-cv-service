"""
Minimal FastAPI app for Railway testing
"""
from fastapi import FastAPI
import os

app = FastAPI(title="AthLink CV Service - Minimal")

@app.get("/")
async def root():
    return {
        "message": "AthLink CV Service - Minimal",
        "status": "ok",
        "port": os.environ.get("PORT", "not set")
    }

@app.get("/api/v1/health")
async def health():
    return {
        "status": "ok",
        "service": "athlink-cv-service-minimal",
        "port": os.environ.get("PORT", "not set")
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
