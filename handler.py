"""
RunPod Serverless Handler for Athlink CV
Wraps the existing analysis pipeline for GPU serverless execution.
"""
import runpod
import os
import base64
import tempfile
import json

def handler(job):
    """
    RunPod job handler.
    
    Input:
    {
        "input": {
            "video_base64": "...",  # base64 encoded video
            "filename": "match.mp4",
            "job_id": "abc123"
        }
    }
    
    Output: full analysis result dict
    """
    try:
        job_input = job.get("input", {})
        
        video_b64 = job_input.get("video_base64")
        filename = job_input.get("filename", "video.mp4")
        job_id = job_input.get("job_id", "runpod_job")
        
        if not video_b64:
            return {"error": "No video provided"}
        
        # Decode video to temp file
        video_bytes = base64.b64decode(video_b64)
        temp_path = f"/tmp/{job_id}_{filename}"
        
        with open(temp_path, "wb") as f:
            f.write(video_bytes)
        
        # Run the full analysis pipeline
        from routes.analyse import _run_analysis_pipeline
        result = _run_analysis_pipeline(
            job_id=job_id,
            temp_path=temp_path,
            skip_cleanup=False
        )
        
        return {"output": result}
        
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
