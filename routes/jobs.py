from typing import List

from fastapi import APIRouter, HTTPException

from services.job_queue_service import get_job, list_jobs
from models.analysis import JobStatusResponse

router = APIRouter()


@router.get("/status/{job_id}", response_model=JobStatusResponse,
             summary="Poll job status")
def job_status(job_id: str):
    """Returns current state of a queued/processing/completed/failed job."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job


@router.get("/list", response_model=List[JobStatusResponse],
             summary="List all jobs")
def jobs_list():
    """Returns all jobs sorted by creation time (newest first)."""
    return list_jobs()
