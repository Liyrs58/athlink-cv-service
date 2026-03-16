import os
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from models.analysis import FrameSampleRequest, FrameSampleResponse, FrameInfo

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dark frame threshold
DARK_THRESHOLD = float(os.getenv('FRAME_DARK_THRESHOLD', '20.0'))

class FrameService:
    """
    Service for extracting and sampling frames from video files.
    """
    
    @staticmethod
    def create_job_output_directory(job_id: str) -> Path:
        """
        Create output directory for a job.
        
        Args:
            job_id: Unique identifier for the job
            
        Returns:
            Path: Path to the created frames directory
        """
        base_temp_dir = Path("temp")
        job_dir = base_temp_dir / job_id
        frames_dir = job_dir / "frames"
        
        # Create directories if they don't exist
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        return frames_dir
    
    @staticmethod
    def calculate_frame_indices(frame_count: int, sample_count: int) -> List[int]:
        """
        Calculate frame indices for sampling.
        
        Args:
            frame_count: Total number of frames in the video
            sample_count: Number of frames to sample
            
        Returns:
            List of frame indices to sample
            
        Raises:
            ValueError: If sample_count is invalid
        """
        if sample_count <= 0:
            raise ValueError("sampleCount must be greater than 0")
        
        if sample_count == 1:
            return [0]  # First frame only
        elif sample_count == 2:
            return [0, frame_count - 1]  # First and last
        elif sample_count == 3:
            middle = frame_count // 2
            return [0, middle, frame_count - 1]  # First, middle, last
        else:
            # Distribute samples evenly throughout the video
            if sample_count > frame_count:
                sample_count = frame_count
            
            step = max(1, frame_count // (sample_count - 1))
            indices = []
            for i in range(sample_count):
                idx = min(i * step, frame_count - 1)
                indices.append(idx)
            
            return indices
    
    @staticmethod
    def extract_frame_at_index(video_path: str, frame_index: int) -> Optional[Any]:
        """
        Extract a specific frame from video using OpenCV.
        
        Args:
            video_path: Path to the video file
            frame_index: Index of the frame to extract
            
        Returns:
            OpenCV image array or None if extraction fails
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return None
            
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            
            # Read frame
            ret, frame = cap.read()
            
            cap.release()
            
            if ret and frame is not None:
                return frame
            else:
                return None
                
        except Exception:
            return None
    
    @staticmethod
    def save_frame_as_image(frame: Any, output_path: Path, quality: int = 95) -> bool:
        """
        Save OpenCV frame as image file.
        
        Args:
            frame: OpenCV image array
            output_path: Path where to save the image
            quality: JPEG quality (1-100)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save frame as JPEG
            return cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        except Exception:
            return False
    
    @staticmethod
    def get_frame_timestamp(video_path: str, frame_index: int, fps: float) -> float:
        """
        Calculate timestamp for a given frame index.
        
        Args:
            video_path: Path to the video file
            frame_index: Index of the frame
            fps: Frames per second of the video
            
        Returns:
            float: Timestamp in seconds
        """
        if fps <= 0:
            return 0.0
        
        return frame_index / fps
    
    @staticmethod
    def get_frame_brightness(frame: Any) -> float:
        """
        Calculate mean brightness of a frame.
        
        Args:
            frame: OpenCV image array
            
        Returns:
            float: Mean pixel brightness (0-255)
        """
        return float(np.mean(frame))
    
    @staticmethod
    def find_bright_frame(video_path: str, start_index: int, max_search_frames: int, total_frames: int) -> Optional[tuple]:
        """
        Find a frame with brightness above DARK_THRESHOLD.
        
        Args:
            video_path: Path to the video file
            start_index: Starting frame index for search
            max_search_frames: Maximum number of frames to search forward
            total_frames: Total number of frames in the video
            
        Returns:
            tuple: (frame_index, frame, brightness) or None if not found
        """
        for offset in range(max_search_frames):
            frame_index = start_index + offset
            
            # Check bounds
            if frame_index >= total_frames:
                break
                
            frame = FrameService.extract_frame_at_index(video_path, frame_index)
            
            if frame is not None:
                brightness = FrameService.get_frame_brightness(frame)
                if brightness >= DARK_THRESHOLD:
                    return frame_index, frame, brightness
        
        return None
    
    @staticmethod
    def sample_frames(
        video_path: str,
        job_id: str,
        sample_count: int = 10,
        skip_dark: bool = True,
    ) -> list[dict]:
        """
        Extract and sample frames from a video file.
        
        Args:
            video_path: Path to the video file
            job_id: Unique identifier for the job
            sample_count: Number of frames to sample
            skip_dark: Whether to skip dark frames (default True)
            
        Returns:
            list[dict]: List of frame dictionaries with metadata
            
        Raises:
            ValueError: For invalid parameters or frame extraction failures
            FileNotFoundError: If video file doesn't exist
            OSError: For file system errors
            RuntimeError: If no usable frames are found
        """
        # Validate input
        if not job_id or not job_id.strip():
            raise ValueError("jobId cannot be empty")
        
        if not video_path or not video_path.strip():
            raise ValueError("videoPath cannot be empty")
        
        if sample_count <= 0:
            raise ValueError("sampleCount must be greater than 0")
        
        # Check if video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if not os.path.isfile(video_path):
            raise ValueError(f"Path is not a file: {video_path}")
        
        # Create output directory
        frames_dir = FrameService.create_job_output_directory(job_id)
        
        try:
            # Open video to get properties
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            # Get video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if frame_count <= 0:
                raise ValueError("Video appears to have no frames")
            
            cap.release()
            
            # Calculate margins to skip intro/outro (2% of frames)
            margin = max(1, int(frame_count * 0.02))
            usable_start = margin
            usable_end = frame_count - margin
            usable_frame_count = usable_end - usable_start
            
            if usable_frame_count <= 0:
                raise ValueError("Video is too short after applying margins")
            
            # Calculate target frame indices within usable range
            target_indices = FrameService.calculate_frame_indices(usable_frame_count, sample_count)
            actual_indices = [usable_start + idx for idx in target_indices]
            
            extracted_frames = []
            max_search_frames = max(1, int(frame_count * 0.05))  # 5% search window
            
            # Extract each frame
            for slot, target_index in enumerate(actual_indices):
                frame_index = target_index
                frame = None
                brightness = 0.0
                skipped = False
                
                if skip_dark:
                    # Search for a bright frame within the search window
                    result = FrameService.find_bright_frame(
                        video_path, 
                        target_index, 
                        max_search_frames,
                        frame_count
                    )
                    
                    if result is not None:
                        frame_index, frame, brightness = result
                        if frame_index != target_index:
                            skipped = True
                    else:
                        # No bright frame found, use the original target
                        if target_index >= frame_count:
                            raise ValueError(f"Target frame index {target_index} exceeds video frame count {frame_count}")
                        
                        frame = FrameService.extract_frame_at_index(video_path, target_index)
                        if frame is not None:
                            brightness = FrameService.get_frame_brightness(frame)
                        else:
                            logger.warning(f"Skipping unreadable frame at index {target_index}")
                            continue
                else:
                    # No brightness filtering, just extract the target frame
                    if target_index >= frame_count:
                        raise ValueError(f"Target frame index {target_index} exceeds video frame count {frame_count}")
                        
                    frame = FrameService.extract_frame_at_index(video_path, target_index)
                    if frame is None:
                        logger.warning(f"Skipping unreadable frame at index {target_index}")
                        continue
                    brightness = FrameService.get_frame_brightness(frame)
                
                # Generate output filename using actual frame index
                frame_filename = f"frame_{frame_index:06d}.jpg"
                frame_path = frames_dir / frame_filename
                
                # Save frame
                if not FrameService.save_frame_as_image(frame, frame_path):
                    raise ValueError(f"Failed to save frame at index {frame_index}")
                
                # Calculate timestamp
                timestamp = FrameService.get_frame_timestamp(video_path, frame_index, fps)
                
                # Create frame info
                frame_info = {
                    "frameIndex": frame_index,
                    "timestampSeconds": timestamp,
                    "imagePath": str(frame_path),
                    "brightness": round(brightness, 2),
                    "skipped": skipped
                }
                
                extracted_frames.append(frame_info)
                
                # Log the sampled frame
                logger.info(
                    f"Sampled frame {slot + 1}/{sample_count}: "
                    f"index={frame_index}, brightness={brightness:.1f}, "
                    f"skipped={skipped}, path={frame_path.name}"
                )
            
            if not extracted_frames:
                raise RuntimeError("No usable frames were found in the video")
            
            return extracted_frames
            
        except Exception as e:
            # Clean up created directory on error
            try:
                import shutil
                parent_dir = frames_dir.parent
                if parent_dir.exists():
                    shutil.rmtree(parent_dir)
            except:
                pass
            
            raise e
