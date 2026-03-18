from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List
import shutil
import uuid
import os
import logging

from services.job_queue_service import create_job, submit_job
from routes.analyse import _run_analysis_pipeline, numpy_safe
from services.match_oracle_service import MatchOracle

router = APIRouter()
logger = logging.getLogger(__name__)


def _run_oracle_pipeline(job_id: str, temp_paths: list, opponent_name: str):
    """
    Runs the full analysis pipeline on each clip,
    then synthesises with Match Oracle.
    """
    try:
        analyses = []
        for i, path in enumerate(temp_paths):
            clip_id = f"{job_id}_clip{i}"
            result = _run_analysis_pipeline(
                job_id=clip_id,
                temp_path=path,
                skip_cleanup=True,
            )
            analyses.append(result)

        oracle = MatchOracle()
        oracle_result = oracle.synthesise(analyses, opponent_name=opponent_name)

        return numpy_safe({
            "job_id": job_id,
            "type": "match_oracle",
            "opponent": opponent_name,
            "clips_analysed": len(analyses),
            "oracle": oracle_result,
            "individual_analyses": analyses,
        })
    finally:
        for path in temp_paths:
            if os.path.exists(path):
                os.remove(path)


@router.post("/oracle/analyse")
async def oracle_analyse(
    clips: List[UploadFile] = File(...),
    opponent_name: str = Form(default="Opponent"),
):
    """
    Submit 2-3 video clips of the same opponent.
    Returns a Tactical Fingerprint synthesised across all clips.
    """
    if len(clips) < 2:
        return JSONResponse(
            {"error": "Minimum 2 clips required for opponent analysis"},
            status_code=400,
        )
    if len(clips) > 3:
        return JSONResponse(
            {"error": "Maximum 3 clips accepted"},
            status_code=400,
        )

    job_id = str(uuid.uuid4())[:8]
    temp_paths = []

    for i, clip in enumerate(clips):
        temp_path = f"/tmp/{job_id}_clip{i}_{clip.filename}"
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(clip.file, f)
        temp_paths.append(temp_path)

    create_job(job_id)
    submit_job(job_id, _run_oracle_pipeline, job_id, temp_paths, opponent_name)

    return JSONResponse({
        "job_id": job_id,
        "status": "processing",
        "clips_submitted": len(clips),
        "opponent": opponent_name,
        "poll_url": f"/api/v1/jobs/status/{job_id}",
        "message": f"Analysing {len(clips)} clips of {opponent_name}. Takes 3-5 minutes.",
    })
