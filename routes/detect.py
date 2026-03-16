from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import logging
import os

from services.frame_service import FrameService
from services.detection_service import run_detection_on_frames

router = APIRouter()
logger = logging.getLogger(__name__)


class DetectPlayersRequest(BaseModel):
    jobId: str
    videoPath: str
    sampleCount: int = Field(8, ge=1, le=50)


class FrameDetection(BaseModel):
    frameIndex: int
    timestampSeconds: Optional[float]
    imagePath: str
    detections: list


class DetectPlayersResponse(BaseModel):
    jobId: str
    sampleCount: int
    frames: List[FrameDetection]


@router.post("/players", response_model=DetectPlayersResponse,
              summary="Detect players on sampled frames")
async def detect_players(req: DetectPlayersRequest):
    """Sample frames from video and run YOLO person detection on each."""
    if not os.path.exists(req.videoPath):
        raise HTTPException(status_code=400, detail=f"Video not found: {req.videoPath}")

    try:
        sampled = FrameService.sample_frames(
            video_path=req.videoPath,
            job_id=req.jobId,
            sample_count=req.sampleCount,
            skip_dark=True,
        )
    except Exception as e:
        logger.exception("Frame sampling failed for job %s", req.jobId)
        raise HTTPException(status_code=500, detail=f"Frame sampling failed: {e}")

    frame_paths = [f["imagePath"] for f in sampled]
    output_dir = f"temp/{req.jobId}/detections"

    try:
        detection_results = run_detection_on_frames(
            frame_paths=frame_paths,
            job_id=req.jobId,
            output_dir=output_dir,
        )
    except Exception as e:
        logger.exception("Detection failed for job %s", req.jobId)
        raise HTTPException(status_code=500, detail=f"Detection failed: {e}")

    # Merge timestamp/brightness from sampled frames into detection results
    ts_map = {f["imagePath"]: f["timestampSeconds"] for f in sampled}
    for frame in detection_results:
        frame["timestampSeconds"] = ts_map.get(frame["imagePath"])

    return DetectPlayersResponse(
        jobId=req.jobId,
        sampleCount=len(detection_results),
        frames=[FrameDetection(**f) for f in detection_results],
    )
