from typing import List

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from services.job_queue_service import get_job, list_jobs
from routes.analyse import numpy_safe

router = APIRouter()


@router.get("/status/{job_id}",
             summary="Poll job status")
def job_status(job_id: str):
    """Returns current state of a queued/processing/completed/failed job."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return JSONResponse(content=numpy_safe(job))


@router.get("/list",
             summary="List all jobs")
def jobs_list():
    """Returns all jobs sorted by creation time (newest first)."""
    return JSONResponse(content=numpy_safe(list_jobs()))
