import logging
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

logger = logging.getLogger(__name__)


def _numpy_safe(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: _numpy_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_numpy_safe(i) for i in obj]
    return obj

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
            job["result"] = _numpy_safe(result)
        except BaseException as e:
            logger.error("Job %s failed: %s", job_id, e, exc_info=True)
            job["status"] = "failed"
            job["completedAt"] = time.time()
            job["error"] = str(e)

    _executor.submit(_wrapper)


def list_jobs():
    return sorted(_jobs.values(), key=lambda j: j["createdAt"], reverse=True)
