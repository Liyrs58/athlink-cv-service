import runpod
import os
import base64
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def handler(job):
    try:
        job_input = job.get("input", {})
        video_url = job_input.get("video_url")
        video_b64 = job_input.get("video_base64")
        filename = job_input.get("filename", "video.mp4")
        job_id = job_input.get("job_id", "runpod_job")

        video_path = f"/tmp/{job_id}_{filename}"

        if video_url:
            # Download video from Supabase URL
            logger.info("Downloading video from URL: %s", video_url)
            resp = requests.get(video_url, timeout=300)
            resp.raise_for_status()
            with open(video_path, "wb") as f:
                f.write(resp.content)
            logger.info("Video downloaded: %s bytes", len(resp.content))
        elif video_b64:
            # Fallback: decode base64
            video_bytes = base64.b64decode(video_b64)
            with open(video_path, "wb") as f:
                f.write(video_bytes)
            logger.info("Video decoded from base64: %s bytes", len(video_bytes))
        else:
            return {"error": "No video provided (need video_url or video_base64)"}

        os.makedirs(f"temp/{job_id}", exist_ok=True)

        from routes.analyse import _run_analysis_pipeline
        result = _run_analysis_pipeline(job_id, video_path)
        return {"output": result}

    except Exception as e:
        import traceback
        logger.error("Handler error: %s", traceback.format_exc())
        return {"error": str(e), "traceback": traceback.format_exc()}

runpod.serverless.start({"handler": handler})
