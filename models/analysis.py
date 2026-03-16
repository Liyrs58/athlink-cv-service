from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class AnalysisRequest(BaseModel):
    """
    Request model for video analysis job submission.
    """
    jobId: str = Field(..., description="Unique identifier for the analysis job")
    videoPath: str = Field(..., description="Path to the video file to analyze")
    
    class Config:
        json_schema_extra = {
            "example": {
                "jobId": "job_123",
                "videoPath": "/path/to/video.mp4"
            }
        }

class VideoMetadata(BaseModel):
    """
    Video file metadata model.
    """
    path: str = Field(..., description="Absolute path to the video file")
    filename: str = Field(..., description="Filename without path")
    extension: str = Field(..., description="File extension (e.g., .mp4)")
    sizeBytes: int = Field(..., description="File size in bytes")
    createdAt: Optional[datetime] = Field(None, description="File creation timestamp")
    modifiedAt: Optional[datetime] = Field(None, description="File modification timestamp")
    width: Optional[int] = Field(None, description="Video width in pixels")
    height: Optional[int] = Field(None, description="Video height in pixels")
    fps: Optional[float] = Field(None, description="Frames per second")
    frameCount: Optional[int] = Field(None, description="Total number of frames")
    durationSeconds: Optional[float] = Field(None, description="Video duration in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "path": "/tmp/test.mp4",
                "filename": "test.mp4",
                "extension": ".mp4",
                "sizeBytes": 123456,
                "createdAt": "2024-01-01T12:00:00Z",
                "modifiedAt": "2024-01-01T12:00:00Z",
                "width": 1920,
                "height": 1080,
                "fps": 25.0,
                "frameCount": 500,
                "durationSeconds": 20.0
            }
        }

class AnalysisResponse(BaseModel):
    """
    Response model for video analysis job submission.
    """
    jobId: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Job status (queued, processing, completed, failed)")
    message: str = Field(..., description="Status message or description")
    video: Optional[VideoMetadata] = Field(None, description="Video file metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "jobId": "job_123",
                "status": "queued",
                "message": "video accepted for analysis",
                "video": {
                    "path": "/tmp/test.mp4",
                    "filename": "test.mp4",
                    "extension": ".mp4",
                    "sizeBytes": 123456
                }
            }
        }

class VideoInspectResponse(BaseModel):
    """
    Response model for video inspection endpoint.
    """
    jobId: str = Field(..., description="Job identifier")
    video: VideoMetadata = Field(..., description="Video file metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "jobId": "job_123",
                "video": {
                    "path": "/tmp/test.mp4",
                    "filename": "test.mp4",
                    "extension": ".mp4",
                    "sizeBytes": 123456,
                    "width": 1920,
                    "height": 1080,
                    "fps": 25.0,
                    "frameCount": 500,
                    "durationSeconds": 20.0
                }
            }
        }

class FrameSampleRequest(BaseModel):
    """
    Request model for frame sampling from video.
    """
    jobId: str = Field(..., description="Unique identifier for the sampling job")
    videoPath: str = Field(..., description="Path to the video file to sample")
    sampleCount: int = Field(default=3, ge=1, le=100, description="Number of frames to sample (1-100)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "jobId": "job_123",
                "videoPath": "/absolute/path/to/video.mp4",
                "sampleCount": 3
            }
        }

class FrameInfo(BaseModel):
    """
    Information about an extracted frame.
    """
    frameIndex: int = Field(..., description="Index of the frame in the video")
    timestampSeconds: float = Field(..., description="Timestamp of the frame in seconds")
    imagePath: str = Field(..., description="Path to the saved image file")
    
    class Config:
        json_schema_extra = {
            "example": {
                "frameIndex": 0,
                "timestampSeconds": 0.0,
                "imagePath": "temp/job_123/frames/frame_000000.jpg"
            }
        }

class FrameSampleResponse(BaseModel):
    """
    Response model for frame sampling from video.
    """
    jobId: str = Field(..., description="Job identifier")
    sampleCount: int = Field(..., description="Number of frames extracted")
    frames: List[FrameInfo] = Field(..., description="List of extracted frame information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "jobId": "job_123",
                "sampleCount": 3,
                "frames": [
                    {
                        "frameIndex": 0,
                        "timestampSeconds": 0.0,
                        "imagePath": "temp/job_123/frames/frame_000000.jpg"
                    },
                    {
                        "frameIndex": 250,
                        "timestampSeconds": 10.0,
                        "imagePath": "temp/job_123/frames/frame_000250.jpg"
                    }
                ]
            }
        }

class BoundingBox(BaseModel):
    """
    Bounding box for detected objects.
    """
    x: int = Field(..., description="X coordinate of top-left corner")
    y: int = Field(..., description="Y coordinate of top-left corner")
    width: int = Field(..., description="Width of the bounding box")
    height: int = Field(..., description="Height of the bounding box")
    
    class Config:
        json_schema_extra = {
            "example": {
                "x": 120,
                "y": 80,
                "width": 32,
                "height": 90
            }
        }

class Detection(BaseModel):
    """
    Single detection result.
    """
    className: str = Field(..., description="Class name of detected object")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")
    
    class Config:
        json_schema_extra = {
            "example": {
                "className": "person",
                "confidence": 0.91,
                "bbox": {
                    "x": 120,
                    "y": 80,
                    "width": 32,
                    "height": 90
                }
            }
        }

class FrameDetectionResult(BaseModel):
    """
    Detection results for a single frame.
    """
    frameIndex: int = Field(..., description="Index of the frame in the video")
    timestampSeconds: float = Field(..., description="Timestamp of the frame in seconds")
    imagePath: str = Field(..., description="Path to the frame image file")
    detections: List[Detection] = Field(..., description="List of detections in this frame")
    
    class Config:
        json_schema_extra = {
            "example": {
                "frameIndex": 0,
                "timestampSeconds": 0.0,
                "imagePath": "temp/job_123/frames/frame_000000.jpg",
                "detections": [
                    {
                        "className": "person",
                        "confidence": 0.91,
                        "bbox": {
                            "x": 120,
                            "y": 80,
                            "width": 32,
                            "height": 90
                        }
                    }
                ]
            }
        }

class PlayerDetectionRequest(BaseModel):
    """
    Request model for player detection on video frames.
    """
    jobId: str = Field(..., description="Unique identifier for the detection job")
    videoPath: str = Field(..., description="Path to the video file to analyze")
    sampleCount: int = Field(default=3, ge=1, le=100, description="Number of frames to sample (1-100)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "jobId": "job_123",
                "videoPath": "/absolute/path/to/video.mp4",
                "sampleCount": 3
            }
        }

class PlayerDetectionResponse(BaseModel):
    """
    Response model for player detection results.
    """
    jobId: str = Field(..., description="Job identifier")
    sampleCount: int = Field(..., description="Number of frames processed")
    frames: List[FrameDetectionResult] = Field(..., description="Detection results for each frame")
    
    class Config:
        json_schema_extra = {
            "example": {
                "jobId": "job_123",
                "sampleCount": 3,
                "frames": [
                    {
                        "frameIndex": 0,
                        "timestampSeconds": 0.0,
                        "imagePath": "temp/job_123/frames/frame_000000.jpg",
                        "detections": [
                            {
                                "className": "person",
                                "confidence": 0.91,
                                "bbox": {
                                    "x": 120,
                                    "y": 80,
                                    "width": 32,
                                    "height": 90
                                }
                            }
                        ]
                    }
                ]
            }
        }

class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.
    """
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    device: str = Field(..., description="Compute device (cuda, mps, or cpu)")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "service": "athlink-cv-service",
                "device": "mps"
            }
        }


class QueuedJobResponse(BaseModel):
    """
    Standard response for async jobs submitted to the queue.
    """
    jobId: str = Field(..., description="Job identifier (may include suffix like _pitch, _tactics, _render)")
    status: str = Field(..., description="Initial status, always 'queued'")
    message: str = Field(..., description="Human-readable message with poll URL")

    class Config:
        json_schema_extra = {
            "example": {
                "jobId": "job_123_pitch",
                "status": "queued",
                "message": "Job queued. Poll GET /api/v1/jobs/status/job_123_pitch"
            }
        }


class JobStatusResponse(BaseModel):
    """
    Response for job status polling.
    """
    jobId: str = Field(..., description="Job identifier")
    status: str = Field(..., description="queued | processing | completed | failed")
    createdAt: float = Field(..., description="Unix timestamp when job was created")
    startedAt: Optional[float] = Field(None, description="Unix timestamp when processing started")
    completedAt: Optional[float] = Field(None, description="Unix timestamp when job finished")
    error: Optional[str] = Field(None, description="Error message if status is 'failed'")
    result: Optional[Dict[str, Any]] = Field(None, description="Job result if status is 'completed'")

    class Config:
        json_schema_extra = {
            "example": {
                "jobId": "job_123",
                "status": "completed",
                "createdAt": 1710000000.0,
                "startedAt": 1710000001.0,
                "completedAt": 1710000045.0,
                "error": None,
                "result": {"trackCount": 22}
            }
        }


class ErrorResponse(BaseModel):
    """
    Standard error response body.
    """
    detail: str = Field(..., description="Error description")
