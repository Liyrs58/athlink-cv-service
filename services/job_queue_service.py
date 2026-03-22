import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

logger = logging.getLogger(__name__)

_JOBS_DIR = os.environ.get("JOBS_DIR", "/tmp/athlink_jobs")
os.makedirs(_JOBS_DIR, exist_ok=True)

_executor = ThreadPoolExecutor(max_workers=2)
_jobs: dict = {}  # in-memory cache
_lock = threading.Lock()


def _numpy_safe(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: _numpy_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_numpy_safe(i) for i in obj]
    return obj


def _job_path(job_id: str) -> str:
    return os.path.join(_JOBS_DIR, f"{job_id}.json")


def _save_job(job: dict):
    try:
        with open(_job_path(job["jobId"]), "w") as f:
            json.dump(_numpy_safe(job), f)
    except Exception as e:
        logger.warning("Failed to persist job %s: %s", job.get("jobId"), e)


def _load_job_from_disk(job_id: str) -> dict | None:
    path = _job_path(job_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load job %s from disk: %s", job_id, e)
        return None


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
    with _lock:
        _jobs[job_id] = job
    _save_job(job)
    return job


def get_job(job_id: str):
    with _lock:
        if job_id in _jobs:
            return _jobs[job_id]
    # Not in memory — check disk (survives restarts)
    job = _load_job_from_disk(job_id)
    if job is not None:
        with _lock:
            _jobs[job_id] = job
    return job


def submit_job(job_id: str, fn, *args, **kwargs):
    def _wrapper():
        with _lock:
            job = _jobs[job_id]
        job["status"] = "processing"
        job["startedAt"] = time.time()
        _save_job(job)
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
        finally:
            _save_job(job)

    _executor.submit(_wrapper)


def list_jobs():
    # Merge disk jobs into memory
    try:
        for fname in os.listdir(_JOBS_DIR):
            if fname.endswith(".json"):
                jid = fname[:-5]
                if jid not in _jobs:
                    job = _load_job_from_disk(jid)
                    if job:
                        with _lock:
                            _jobs[jid] = job
    except Exception:
        pass
    with _lock:
        jobs = list(_jobs.values())
    return sorted(jobs, key=lambda j: j.get("createdAt", 0), reverse=True)
