"""GET /api/v1/match-report/{jobId} — unified match report JSON + PDF."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from services.match_report_service import build_match_report

router = APIRouter()


@router.get("/match-report/{job_id}")
def get_match_report(job_id: str):
    base = Path(f"temp/{job_id}")
    if not base.exists():
        raise HTTPException(status_code=404, detail=f"job {job_id} has no outputs")
    try:
        report = build_match_report(job_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"build failed: {e}") from e
    return JSONResponse(content=report)


@router.get("/match-report/{job_id}/pdf")
def get_match_report_pdf(job_id: str):
    """Return a PDF rendered from match_report.json. Falls back to JSON if
    the PDF generator (services.report_card_service) cannot run for this job."""
    base = Path(f"temp/{job_id}")
    if not base.exists():
        raise HTTPException(status_code=404, detail=f"job {job_id} has no outputs")

    # Make sure the JSON is fresh
    build_match_report(job_id)

    pdf_path = base / "reports" / f"match_report_{job_id}.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from services.report_card_service import build_match_pdf  # type: ignore
        build_match_pdf(job_id, str(pdf_path))
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="report_card_service.build_match_pdf not implemented yet",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"pdf build failed: {e}") from e

    if not pdf_path.exists():
        raise HTTPException(status_code=500, detail="pdf file was not produced")
    return FileResponse(str(pdf_path), media_type="application/pdf", filename=pdf_path.name)
