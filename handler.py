import runpod
import os
import base64
import requests
import logging
import sys
import traceback

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

def handler(job):
    job_id_str = job.get("id", "unknown")
    logger.info(f"[JOB {job_id_str}] Handler received job")

    try:
        job_input = job.get("input", {})
        video_url = job_input.get("video_url")
        video_b64 = job_input.get("video_base64")
        filename = job_input.get("filename", "video.mp4")
        job_id = job_input.get("job_id", "runpod_job")

        logger.info(f"[JOB {job_id_str}] Input: video_url={bool(video_url)}, video_b64={bool(video_b64)}, filename={filename}, job_id={job_id}")

        video_path = f"/tmp/{job_id}_{filename}"

        if video_url:
            # Download video from Supabase URL
            logger.info(f"[JOB {job_id_str}] Downloading video from URL: {video_url}")
            resp = requests.get(video_url, timeout=300)
            resp.raise_for_status()
            with open(video_path, "wb") as f:
                f.write(resp.content)
            logger.info(f"[JOB {job_id_str}] Video downloaded: {len(resp.content)} bytes")
        elif video_b64:
            # Fallback: decode base64
            logger.info(f"[JOB {job_id_str}] Decoding video from base64")
            video_bytes = base64.b64decode(video_b64)
            with open(video_path, "wb") as f:
                f.write(video_bytes)
            logger.info(f"[JOB {job_id_str}] Video decoded from base64: {len(video_bytes)} bytes")
        else:
            error_msg = "No video provided (need video_url or video_base64)"
            logger.error(f"[JOB {job_id_str}] {error_msg}")
            return {"error": error_msg}

        os.makedirs(f"temp/{job_id}", exist_ok=True)

        logger.info(f"[JOB {job_id_str}] Importing analysis pipeline...")
        try:
            from routes.analyse import _run_analysis_pipeline
            logger.info(f"[JOB {job_id_str}] Successfully imported _run_analysis_pipeline")
        except ImportError as ie:
            logger.error(f"[JOB {job_id_str}] Failed to import: {ie}")
            logger.error(f"[JOB {job_id_str}] Python path: {sys.path}")
            logger.error(f"[JOB {job_id_str}] Current directory: {os.getcwd()}")
            raise

        logger.info(f"[JOB {job_id_str}] Starting analysis pipeline...")
        result = _run_analysis_pipeline(job_id, video_path)
        logger.info(f"[JOB {job_id_str}] Analysis completed successfully")
        return {"output": result}

    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"[JOB {job_id_str}] Handler error: {error_traceback}")
        return {"error": str(e), "traceback": error_traceback}

logger.info("Starting RunPod serverless handler...")
runpod.serverless.start({"handler": handler})
