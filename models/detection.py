from pydantic import BaseModel
from typing import List


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class Detection(BaseModel):
    className: str
    confidence: float
    bbox: BoundingBox


class FrameDetectionResult(BaseModel):
    frameIndex: int
    timestampSeconds: float
    imagePath: str
    detections: List[Detection]


class PlayerDetectionRequest(BaseModel):
    jobId: str
    videoPath: str
    sampleCount: int = 3


class PlayerDetectionResponse(BaseModel):
    jobId: str
    sampleCount: int
    totalDetections: int
    frames: List[FrameDetectionResult]
    detectionsPath: str
