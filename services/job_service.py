"""Job creation and status management.
"""

from typing import Dict, Any
from models.analysis import AnalysisResponse, VideoMetadata
from services.video_service import VideoService

class JobService:
    """
    Service layer for managing video analysis jobs.
    
    Currently handles basic job submission. Future implementations will include:
    - Job queue management
    - Progress tracking
    - Result storage
    - Error handling
    """
    
    def __init__(self):
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
    
    async def submit_analysis_job(self, job_id: str, video_path: str) -> AnalysisResponse:
        """
        Submit a new video analysis job.
        
        Args:
            job_id: Unique identifier for the job
            video_path: Path to the video file
            
        Returns:
            AnalysisResponse: Job submission confirmation with video metadata
            
        Raises:
            ValueError: If job_id already exists or video_path is invalid
            FileNotFoundError: If video file doesn't exist
            OSError: If there's an error accessing the video file
        """
        # Validate job ID
        if not job_id or not job_id.strip():
            raise ValueError("Job ID cannot be empty")
        
        # Check if job already exists
        if job_id in self.active_jobs:
            raise ValueError(f"Job {job_id} already exists")
        
        # Validate video file and get metadata
        video_metadata = VideoService.validate_and_get_metadata(video_path)
        
        # Store job information (in-memory for now)
        self.active_jobs[job_id] = {
            "video_path": video_path,
            "status": "queued",
            "created_at": "2024-01-01T00:00:00Z",  # Placeholder
            "progress": 0,
            "video_metadata": video_metadata.dict()
        }
        
        # Return response with video metadata
        return AnalysisResponse(
            jobId=job_id,
            status="queued",
            message="video accepted for analysis",
            video=video_metadata
        )
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a specific job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job status information
            
        Raises:
            ValueError: If job doesn't exist
        """
        if job_id not in self.active_jobs:
            raise ValueError(f"Job {job_id} not found")
        
        return self.active_jobs[job_id]
