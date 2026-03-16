from fastapi import APIRouter, HTTPException
from models.analysis import AnalysisRequest, AnalysisResponse
from services.job_service import JobService

router = APIRouter()

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_video(request: AnalysisRequest) -> AnalysisResponse:
    """
    Submit a video for analysis.
    
    Args:
        request: Analysis request containing jobId and videoPath
        
    Returns:
        AnalysisResponse: Job status and video metadata
        
    Raises:
        HTTPException: If request validation fails or file not found
    """
    try:
        job_service = JobService()
        result = await job_service.submit_analysis_job(
            job_id=request.jobId,
            video_path=request.videoPath
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except OSError as e:
        raise HTTPException(status_code=422, detail=f"File access error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
