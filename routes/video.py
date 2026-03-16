from fastapi import APIRouter, HTTPException
from models.analysis import AnalysisRequest, VideoInspectResponse
from services.video_service import VideoService

router = APIRouter()

@router.post("/video/inspect", response_model=VideoInspectResponse)
async def inspect_video(request: AnalysisRequest) -> VideoInspectResponse:
    """
    Inspect a video file and return detailed metadata without creating a job.
    
    Args:
        request: Analysis request containing jobId and videoPath
        
    Returns:
        VideoInspectResponse: Detailed video metadata
        
    Raises:
        HTTPException: If request validation fails or file not found
    """
    try:
        # Validate video file and get metadata
        video_metadata = VideoService.validate_and_get_metadata(request.videoPath)
        
        # Return inspection response
        return VideoInspectResponse(
            jobId=request.jobId,
            video=video_metadata
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except OSError as e:
        raise HTTPException(status_code=422, detail=f"File access error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
