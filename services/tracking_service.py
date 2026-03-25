import cv2


def run_tracking(job_id: str, video_path: str, frame_stride: int = 2, progress_path: str = None):
    """Minimal compatibility shim to unblock imports and runtime while full tracking implementation is restored."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    cap.release()

    frame_metadata = []
    if total_frames > 0:
        step = max(int(frame_stride), 1)
        for idx in range(0, total_frames, step):
            frame_metadata.append({
                "frameIndex": idx,
                "frameWidth": frame_width,
                "frameHeight": frame_height,
                "fps": fps,
                "analysis_valid": True,
                "scene_cut": False,
                "tracks_active": 0,
                "ball_detected": False,
                "ball_source": None,
            })

    return {
        "jobId": job_id,
        "tracks": [],
        "frame_metadata": frame_metadata,
        "framesProcessed": len(frame_metadata),
    }