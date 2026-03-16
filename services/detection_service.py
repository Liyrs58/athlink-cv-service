import os
import json
import cv2
from pathlib import Path
from typing import List, Dict, Any, Optional
from models.analysis import (
    PlayerDetectionRequest, 
    PlayerDetectionResponse, 
    FrameDetectionResult, 
    Detection, 
    BoundingBox,
    FrameSampleRequest
)
from services.frame_service import FrameService

class DetectionService:
    """
    Service for player detection using YOLO models.
    """
    
    def __init__(self):
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """
        Load YOLO model for person detection.
        Uses YOLOv8n (nano) model for lightweight inference.
        Auto-detects GPU (cuda/mps) when available.
        """
        try:
            from ultralytics import YOLO
            from services.tracking_service import _detect_device
            device = _detect_device()
            self.model = YOLO('yolov8n.pt')
            self.model.to(device)
        except ImportError:
            raise ImportError("ultralytics package not found. Install with: pip install ultralytics")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {str(e)}")
    
    def _yolo_to_detection(self, result) -> List[Detection]:
        """
        Convert YOLO results to Detection objects.
        
        Args:
            result: YOLO detection result
            
        Returns:
            List of Detection objects
        """
        detections = []
        
        if result.boxes is not None:
            for box in result.boxes:
                # Get class name and confidence
                class_id = int(box.cls)
                confidence = float(box.conf)
                
                # Filter for person class (class_id = 0 in COCO dataset)
                if class_id == 0 and confidence > 0.25:  # Person detection with confidence threshold
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Convert to our format
                    bbox = BoundingBox(
                        x=int(x1),
                        y=int(y1),
                        width=int(x2 - x1),
                        height=int(y2 - y1)
                    )
                    
                    detection = Detection(
                        className="person",
                        confidence=confidence,
                        bbox=bbox
                    )
                    
                    detections.append(detection)
        
        return detections
    
    def _save_detection_results(self, job_id: str, detection_results: List[FrameDetectionResult]) -> None:
        """
        Save detection results to JSON file.
        
        Args:
            job_id: Job identifier
            detection_results: Detection results to save
        """
        # Create detections directory
        base_temp_dir = Path("temp")
        job_dir = base_temp_dir / job_id
        detections_dir = job_dir / "detections"
        detections_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON file
        output_file = detections_dir / "sample_detections.json"
        
        # Convert to serializable format
        serializable_results = [
            {
                "frameIndex": frame.frameIndex,
                "timestampSeconds": frame.timestampSeconds,
                "imagePath": frame.imagePath,
                "detections": [
                    {
                        "className": det.className,
                        "confidence": det.confidence,
                        "bbox": {
                            "x": det.bbox.x,
                            "y": det.bbox.y,
                            "width": det.bbox.width,
                            "height": det.bbox.height
                        }
                    }
                    for det in frame.detections
                ]
            }
            for frame in detection_results
        ]
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def _ensure_frames_exist(self, request: PlayerDetectionRequest) -> List[FrameDetectionResult]:
        """
        Ensure sampled frames exist, create them if needed.
        
        Args:
            request: Player detection request
            
        Returns:
            List of frame info for existing or newly created frames
        """
        # Check if frames directory exists
        frames_dir = Path("temp") / request.jobId / "frames"
        
        if not frames_dir.exists():
            # Create frames first
            frame_request = FrameSampleRequest(
                jobId=request.jobId,
                videoPath=request.videoPath,
                sampleCount=request.sampleCount
            )
            
            frame_result = FrameService.sample_frames_from_video(frame_request)
            
            # Convert FrameInfo to FrameDetectionResult (without detections initially)
            frame_results = []
            for frame_info in frame_result.frames:
                frame_result = FrameDetectionResult(
                    frameIndex=frame_info.frameIndex,
                    timestampSeconds=frame_info.timestampSeconds,
                    imagePath=frame_info.imagePath,
                    detections=[]  # No detections yet
                )
                frame_results.append(frame_result)
            
            return frame_results
        else:
            # Load existing frame info
            frame_results = []
            frame_files = sorted(frames_dir.glob("frame_*.jpg"))
            
            for frame_file in frame_files:
                # Extract frame index from filename
                frame_index = int(frame_file.stem.split('_')[1])
                
                frame_result = FrameDetectionResult(
                    frameIndex=frame_index,
                    timestampSeconds=0.0,  # Will be calculated later if needed
                    imagePath=str(frame_file),
                    detections=[]  # Will be filled by detection
                )
                frame_results.append(frame_result)
            
            return frame_results[:request.sampleCount]  # Limit to requested count
    
    def detect_players_on_frames(self, request: PlayerDetectionRequest) -> PlayerDetectionResponse:
        """
        Run player detection on sampled video frames.
        
        Args:
            request: Player detection request
            
        Returns:
            PlayerDetectionResponse: Detection results for all frames
            
        Raises:
            ValueError: For invalid parameters or detection failures
            FileNotFoundError: If video file doesn't exist
            RuntimeError: If model loading or inference fails
        """
        # Validate input
        if not request.jobId or not request.jobId.strip():
            raise ValueError("jobId cannot be empty")
        
        if not request.videoPath or not request.videoPath.strip():
            raise ValueError("videoPath cannot be empty")
        
        if request.sampleCount <= 0:
            raise ValueError("sampleCount must be greater than 0")
        
        # Ensure model is loaded
        if self.model is None:
            self._load_model()
        
        try:
            # Ensure frames exist
            frame_results = self._ensure_frames_exist(request)
            
            # Run detection on each frame
            detection_results = []
            
            for frame_result in frame_results:
                # Load frame image
                frame_image = cv2.imread(frame_result.imagePath)
                
                if frame_image is None:
                    raise ValueError(f"Cannot read frame image: {frame_result.imagePath}")
                
                # Run YOLO detection
                yolo_results = self.model(frame_image, verbose=False)
                
                # Convert results to our format
                detections = []
                for result in yolo_results:
                    detections.extend(self._yolo_to_detection(result))
                
                # Update frame result with detections
                frame_result.detections = detections
                detection_results.append(frame_result)
            
            # Save detection results to JSON
            self._save_detection_results(request.jobId, detection_results)
            
            return PlayerDetectionResponse(
                jobId=request.jobId,
                sampleCount=len(detection_results),
                frames=detection_results
            )
            
        except Exception as e:
            if isinstance(e, (ValueError, FileNotFoundError, RuntimeError)):
                raise
            else:
                raise RuntimeError(f"Detection failed: {str(e)}")


# Module-level singleton + convenience function (used by routes/detect.py)
_detection_service: Optional["DetectionService"] = None


def run_detection_on_frames(
    frame_paths: List[str],
    job_id: str,
    output_dir: str,
) -> List[Dict[str, Any]]:
    """Run YOLO person detection on a list of frame image paths.

    Returns a list of dicts with keys:
        frameIndex, timestampSeconds, imagePath, detections
    """
    global _detection_service
    if _detection_service is None:
        _detection_service = DetectionService()

    model = _detection_service.model
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for path in frame_paths:
        # Derive frame index from filename (frame_000042.jpg → 42)
        stem = Path(path).stem
        try:
            frame_index = int(stem.split("_")[-1])
        except ValueError:
            frame_index = len(results)

        img = cv2.imread(path)
        if img is None:
            continue

        yolo_out = model(img, verbose=False, conf=0.35, iou=0.45, classes=[0])
        detections = _detection_service._yolo_to_detection(yolo_out[0])

        results.append({
            "frameIndex": frame_index,
            "timestampSeconds": None,
            "imagePath": path,
            "detections": [
                {
                    "className": d.className,
                    "confidence": d.confidence,
                    "bbox": [d.bbox.x, d.bbox.y, d.bbox.width, d.bbox.height],
                }
                for d in detections
            ],
        })

    # Persist
    out_file = Path(output_dir) / "sample_detections.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    return results
