import logging
import time
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2)
_jobs: dict = {}  # jobId → job state dict


def create_job(job_id: str) -> dict:
    job = {
        "jobId": job_id,
        "status": "queued",
        "createdAt": time.time(),
        "startedAt": None,
        "completedAt": None,
        "error": None,
        "result": None,
    }
    _jobs[job_id] = job
    return job


def get_job(job_id: str):
    return _jobs.get(job_id)


def submit_job(job_id: str, fn, *args, **kwargs):
    def _wrapper():
        job = _jobs[job_id]
        job["status"] = "processing"
        job["startedAt"] = time.time()
        try:
            result = fn(*args, **kwargs)
            job["status"] = "completed"
            job["completedAt"] = time.time()
            job["result"] = result
        except BaseException as e:
            logger.error("Job %s failed: %s", job_id, e, exc_info=True)
            job["status"] = "failed"
            job["completedAt"] = time.time()
            job["error"] = str(e)

    _executor.submit(_wrapper)


def list_jobs():
    return sorted(_jobs.values(), key=lambda j: j["createdAt"], reverse=True)
