from fastapi import APIRouter, HTTPException
from pathlib import Path
import logging

from services.storage_service import upload_job_results, get_public_url

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload/{jobId}",
             summary="Upload all job outputs to Supabase storage")
async def upload_job(jobId: str):
    """Upload tracking, pitch, tactics, and render outputs for a completed job."""
    if not Path("temp", jobId).is_dir():
        raise HTTPException(status_code=404, detail="Job directory not found: {}".format(jobId))
    try:
        result = upload_job_results(jobId)
        return result
    except Exception as e:
        logger.exception("Storage upload failed for job %s", jobId)
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.get("/url/{jobId}/{bucket}/{path:path}",
            summary="Get public URL for an uploaded file")
async def get_url(jobId: str, bucket: str, path: str):
    """Return the public URL for a file in Supabase storage."""
    url = get_public_url(bucket, path)
    return {"url": url}
