from fastapi import FastAPI
import os

app = FastAPI()

@app.get("/api/v1/health")
async def health():
    return {"status": "ok", "service": "athlink-cv-service"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
