import os, requests, time, logging
logger = logging.getLogger(__name__)

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "1qpoq5tzl69dri")

def is_runpod_available():
    available = bool(RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID)
    logger.info("RunPod check: API_KEY=%s, ENDPOINT=%s, available=%s",
                bool(RUNPOD_API_KEY), bool(RUNPOD_ENDPOINT_ID), available)
    return available

def run_on_runpod(video_path: str, job_id: str) -> dict:
    """Upload video to Supabase, send URL to RunPod for GPU processing."""
    from services.storage_service import upload_file_from_path

    # Upload video to Supabase so RunPod can download it
    remote_path = f"runpod/{job_id}/{os.path.basename(video_path)}"
    video_url = upload_file_from_path("match-videos", remote_path, video_path)
    if not video_url:
        raise Exception("Failed to upload video to Supabase storage")
    logger.info("Video uploaded to Supabase: %s", video_url)

    # Send URL to RunPod (not the video itself)
    resp = requests.post(
        f"https://api.runpod.io/v2/{RUNPOD_ENDPOINT_ID}/run",
        headers={"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"},
        json={"input": {"video_url": video_url, "filename": os.path.basename(video_path), "job_id": job_id}},
        timeout=60,
    )
    resp.raise_for_status()
    runpod_job_id = resp.json()["id"]
    logger.info("RunPod job submitted: %s", runpod_job_id)

    for _ in range(200):
        time.sleep(5)
        status_resp = requests.get(
            f"https://api.runpod.io/v2/{RUNPOD_ENDPOINT_ID}/status/{runpod_job_id}",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
            timeout=30,
        )
        status_resp.raise_for_status()
        data = status_resp.json()
        status = data.get("status")
        logger.info("RunPod status: %s", status)
        if status == "COMPLETED":
            output = data.get("output", {})
            if "error" in output:
                raise Exception(f"RunPod error: {output['error']}")
            return output.get("output", output)
        if status == "FAILED":
            raise Exception(f"RunPod job failed: {data}")

    raise Exception("RunPod job timed out after 1000s")
