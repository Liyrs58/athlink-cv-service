import runpod
import os
import base64
import tempfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def handler(job):
    try:
        job_input = job.get("input", {})
        video_b64 = job_input.get("video_base64")
        filename = job_input.get("filename", "video.mp4")
        job_id = job_input.get("job_id", "runpod_job")

        if not video_b64:
            return {"error": "No video provided"}

        video_bytes = base64.b64decode(video_b64)
        video_path = f"/tmp/{job_id}_{filename}"
        with open(video_path, "wb") as f:
            f.write(video_bytes)
        logger.info("Video saved to %s", video_path)

        os.makedirs(f"temp/{job_id}", exist_ok=True)

        from services.tracking_service import run_tracking
        from services.velocity_service import compute_all_velocities
        from services.interpretation_service import generate_interpretation
        from routes.analyse import _run_analysis_pipeline

        result = _run_analysis_pipeline(job_id, video_path)
        return {"output": result}

    except Exception as e:
        import traceback
        logger.error("Handler error: %s", traceback.format_exc())
        return {"error": str(e), "traceback": traceback.format_exc()}

runpod.serverless.start({"handler": handler})
