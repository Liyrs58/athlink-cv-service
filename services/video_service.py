import os
import cv2
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from models.analysis import VideoMetadata

class VideoService:
    """
    Service for video file validation and metadata collection.
    """
    
    ALLOWED_EXTENSIONS = {'.mp4', '.mov', '.m4v', '.avi', '.webm'}
    
    @staticmethod
    def validate_video_path(video_path: str) -> None:
        """
        Validate that the video path is not empty.
        
        Args:
            video_path: Path to the video file
            
        Raises:
            ValueError: If path is empty or only whitespace
        """
        if not video_path or not video_path.strip():
            raise ValueError("videoPath cannot be empty")
    
    @staticmethod
    def validate_file_exists(video_path: str) -> None:
        """
        Validate that the video file exists on disk.
        
        Args:
            video_path: Path to the video file
            
        Raises:
            FileNotFoundError: If file does not exist
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if not os.path.isfile(video_path):
            raise ValueError(f"Path is not a file: {video_path}")
    
    @staticmethod
    def validate_video_extension(video_path: str) -> None:
        """
        Validate that the file has an allowed video extension.
        
        Args:
            video_path: Path to the video file
            
        Raises:
            ValueError: If file extension is not allowed
        """
        path = Path(video_path)
        extension = path.suffix.lower()
        
        if extension not in VideoService.ALLOWED_EXTENSIONS:
            allowed_str = ", ".join(VideoService.ALLOWED_EXTENSIONS)
            raise ValueError(f"Invalid video extension: {extension}. Allowed extensions: {allowed_str}")
    
    @staticmethod
    def extract_video_metadata(video_path: str) -> Dict[str, Any]:
        """
        Extract video metadata using OpenCV.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dict containing video metadata (width, height, fps, frame_count, duration)
            
        Raises:
            ValueError: If OpenCV cannot read the video file
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"OpenCV cannot open video file: {video_path}")
            
            # Get basic video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate duration
            if fps > 0:
                duration_seconds = frame_count / fps
            else:
                duration_seconds = 0.0
            
            # Clean up
            cap.release()
            
            return {
                "width": width if width > 0 else None,
                "height": height if height > 0 else None,
                "fps": fps if fps > 0 else None,
                "frameCount": frame_count if frame_count > 0 else None,
                "durationSeconds": duration_seconds if duration_seconds > 0 else None
            }
            
        except Exception as e:
            raise ValueError(f"Error extracting video metadata: {str(e)}")
    
    @staticmethod
    def calculate_frame_samples(frame_count: int, sample_count: int = 3) -> List[int]:
        """
        Calculate frame indices for sampling (first, middle, last, etc.).
        
        Args:
            frame_count: Total number of frames in the video
            sample_count: Number of frames to sample
            
        Returns:
            List of frame indices to sample
        """
        if frame_count <= 0:
            return []
        
        if sample_count == 1:
            return [0]  # First frame only
        elif sample_count == 2:
            return [0, frame_count - 1]  # First and last
        elif sample_count == 3:
            middle = frame_count // 2
            return [0, middle, frame_count - 1]  # First, middle, last
        else:
            # Distribute samples evenly throughout the video
            step = max(1, frame_count // (sample_count - 1))
            return [min(i * step, frame_count - 1) for i in range(sample_count)]
    
    @staticmethod
    def get_video_metadata(video_path: str) -> VideoMetadata:
        """
        Collect basic and video metadata for a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            VideoMetadata: Complete file and video metadata object
            
        Raises:
            OSError: If there's an error accessing the file
            ValueError: If OpenCV cannot read the video file
        """
        path = Path(video_path)
        
        try:
            # Basic file info
            stat = path.stat()
            
            # Extract video metadata using OpenCV
            video_meta = VideoService.extract_video_metadata(video_path)
            
            # Calculate frame samples for future pipeline use
            frame_samples = []
            if video_meta["frameCount"]:
                frame_samples = VideoService.calculate_frame_samples(
                    video_meta["frameCount"], 
                    sample_count=3
                )
            
            # Combine all metadata
            metadata = VideoMetadata(
                path=str(path.absolute()),
                filename=path.name,
                extension=path.suffix.lower(),
                sizeBytes=stat.st_size,
                createdAt=datetime.fromtimestamp(stat.st_ctime) if hasattr(stat, 'st_ctime') else None,
                modifiedAt=datetime.fromtimestamp(stat.st_mtime) if hasattr(stat, 'st_mtime') else None,
                width=video_meta["width"],
                height=video_meta["height"],
                fps=video_meta["fps"],
                frameCount=video_meta["frameCount"],
                durationSeconds=video_meta["durationSeconds"]
            )
            
            return metadata
            
        except OSError as e:
            raise OSError(f"Error accessing video file: {e}")
        except ValueError as e:
            # Re-raise OpenCV errors as ValueError
            raise ValueError(str(e))
    
    @staticmethod
    def validate_and_get_metadata(video_path: str) -> VideoMetadata:
        """
        Validate video file and return metadata.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            VideoMetadata: Validated file metadata
            
        Raises:
            ValueError: For validation errors
            FileNotFoundError: If file doesn't exist
            OSError: For file access errors
        """
        # Run all validations
        VideoService.validate_video_path(video_path)
        VideoService.validate_file_exists(video_path)
        VideoService.validate_video_extension(video_path)
        
        # Return metadata
        return VideoService.get_video_metadata(video_path)
