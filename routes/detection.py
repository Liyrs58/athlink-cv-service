from fastapi import APIRouter, HTTPException
from models.detection import PlayerDetectionRequest, PlayerDetectionResponse
from services.detection_service import run_detection_on_frames
from services.frame_service import FrameService

router = APIRouter()

@router.post("/players", response_model=PlayerDetectionResponse)
async def detect_players(request: PlayerDetectionRequest) -> PlayerDetectionResponse:
    """
    Run player detection on sampled video frames.
    
    Args:
        request: Player detection request with job info and parameters
        
    Returns:
        PlayerDetectionResponse: Detection results for all frames
        
    Raises:
        HTTPException: If request validation fails or detection fails
    """
    try:
        # First, sample frames from the video
        from models.analysis import FrameSampleRequest
        frame_request = FrameSampleRequest(
            jobId=request.jobId,
            videoPath=request.videoPath,
            sampleCount=request.sampleCount
        )
        
        frame_result = FrameService.sample_frames_from_video(frame_request)
        
        # Extract frame paths from the sampled frames
        frame_paths = [frame.imagePath for frame in frame_result.frames]
        
        # Run detection on the frames
        output_dir = f"temp/{request.jobId}/detections"
        detection_results = run_detection_on_frames(frame_paths, request.jobId, output_dir)
        
        # Convert detection results to response format
        from models.detection import FrameDetectionResult, Detection, BoundingBox
        
        frame_detection_results = []
        for i, frame_data in enumerate(detection_results):
            detections = []
            for det in frame_data["detections"]:
                bbox = det["bbox"]  # [x, y, width, height]
                detection = Detection(
                    className=det["className"],
                    confidence=det["confidence"],
                    bbox=BoundingBox(
                        x=bbox[0],
                        y=bbox[1],
                        width=bbox[2],
                        height=bbox[3]
                    )
                )
                detections.append(detection)
            
            frame_result = FrameDetectionResult(
                frameIndex=frame_data["frameIndex"],
                timestampSeconds=frame_data["timestampSeconds"] or 0.0,
                imagePath=frame_data["imagePath"],
                detections=detections
            )
            frame_detection_results.append(frame_result)
        
        return PlayerDetectionResponse(
            jobId=request.jobId,
            sampleCount=len(frame_detection_results),
            frames=frame_detection_results
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=422, detail=f"Missing dependencies: {str(e)}")
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=f"Detection error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
