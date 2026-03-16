from fastapi import APIRouter

from models.analysis import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse,
             summary="Service health check")
async def health_check():
    """
    Health check endpoint to verify service is running.

    Returns:
        dict: Health status information
    """
    from services.tracking_service import _detect_device
    device = _detect_device()
    return {
        "status": "ok",
        "service": "athlink-cv-service",
        "device": device,
    }
