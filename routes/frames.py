from fastapi import APIRouter, HTTPException
from models.analysis import FrameSampleRequest, FrameSampleResponse
from services.frame_service import FrameService

router = APIRouter()

@router.post("/video/sample-frames", response_model=FrameSampleResponse)
async def sample_frames(request: FrameSampleRequest) -> FrameSampleResponse:
    """
    Extract and sample frames from a video file.
    
    Args:
        request: Frame sampling request with job info and parameters
        
    Returns:
        FrameSampleResponse: Information about extracted frames
        
    Raises:
        HTTPException: If request validation fails or frame extraction fails
    """
    try:
        result = FrameService.sample_frames_from_video(request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except OSError as e:
        raise HTTPException(status_code=422, detail=f"File system error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
