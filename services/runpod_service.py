import os
import requests
import base64
import time
import logging

logger = logging.getLogger(__name__)

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")

def is_runpod_available() -> bool:
    return bool(RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID)

def submit_to_runpod(video_path: str, job_id: str) -> str:
    """
    Submit video to RunPod GPU for processing.
    Returns RunPod job ID.
    """
    # Read and encode video
    with open(video_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode("utf-8")

    filename = os.path.basename(video_path)

    response = requests.post(
        f"https://api.runpod.io/v2/{RUNPOD_ENDPOINT_ID}/run",
        headers={
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "input": {
                "video_base64": video_b64,
                "filename": filename,
                "job_id": job_id,
            }
        },
        timeout=30,
    )

    response.raise_for_status()
    data = response.json()
    return data["id"]  # RunPod job ID


def poll_runpod_result(runpod_job_id: str,
                        timeout_seconds: int = 600) -> dict:
    """
    Poll RunPod until job completes or times out.
    Returns the result dict.
    """
    start = time.time()

    while time.time() - start < timeout_seconds:
        response = requests.get(
            f"https://api.runpod.io/v2/{RUNPOD_ENDPOINT_ID}/status/{runpod_job_id}",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        status = data.get("status")
        logger.info(f"RunPod job {runpod_job_id} status: {status}")

        if status == "COMPLETED":
            output = data.get("output", {})
            if "error" in output:
                raise Exception(f"RunPod error: {output['error']}")
            return output.get("output", output)

        elif status == "FAILED":
            raise Exception(f"RunPod job failed: {data}")

        elif status in ("IN_QUEUE", "IN_PROGRESS"):
            time.sleep(5)

        else:
            time.sleep(5)

    raise Exception(f"RunPod job timed out after {timeout_seconds}s")


def run_on_runpod(video_path: str, job_id: str) -> dict:
    """
    Full flow: submit to RunPod, poll for result, return.
    """
    logger.info(f"Submitting job {job_id} to RunPod GPU")
    runpod_job_id = submit_to_runpod(video_path, job_id)
    logger.info(f"RunPod job ID: {runpod_job_id}")
    result = poll_runpod_result(runpod_job_id)
    logger.info(f"RunPod job {runpod_job_id} completed")
    return result
