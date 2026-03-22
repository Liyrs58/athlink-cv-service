import os, requests, base64, time, logging
logger = logging.getLogger(__name__)

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "1qpoq5tzl69dri")

def is_runpod_available():
    return bool(RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID)

def run_on_runpod(video_path: str, job_id: str) -> dict:
    with open(video_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode("utf-8")

    resp = requests.post(
        f"https://api.runpod.io/v2/{RUNPOD_ENDPOINT_ID}/run",
        headers={"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"},
        json={"input": {"video_base64": video_b64, "filename": os.path.basename(video_path), "job_id": job_id}},
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
