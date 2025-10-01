"""
Health check API endpoints.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from datetime import datetime

from app.models.schemas import HealthResponse
from app.config import settings

# Create router
router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify API status.

    Returns:
        HealthResponse: Health status information
    """
    return HealthResponse(status="healthy", version=settings.app_version)


@router.get("/health/detailed")
async def detailed_health_check():
    """
    Detailed health check with system information.

    Returns:
        dict: Detailed health information
    """
    import psutil
    import sys

    try:
        # Get system information
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        health_info = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": settings.app_version,
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "python_version": sys.version,
                "platform": sys.platform,
            },
            "configuration": {
                "debug_mode": settings.debug,
                "s3_configured": bool(settings.s3_bucket_name),
                "openai_configured": bool(settings.openai_api_key),
            },
        }

        return JSONResponse(content=health_info, status_code=200)

    except Exception as e:
        return JSONResponse(
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
            },
            status_code=503,
        )
