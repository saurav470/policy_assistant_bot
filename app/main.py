"""
Main FastAPI application with proper structure and best practices.
"""

import logging
import signal
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import settings
from app.api import chat_history, health, insurance

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)




def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured FastAPI application
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Healthcare Backend API for document processing and chat history management",
        debug=settings.debug,
        # lifespan=lifespan,
        # docs_url="/docs" if settings.debug else None,
        # redoc_url="/redoc" if settings.debug else None,
    )

    # Add middleware
    setup_middleware(app)

    # Add exception handlers
    setup_exception_handlers(app)

    # Include routers
    setup_routers(app)

    return app


def setup_middleware(app: FastAPI) -> None:
    """
    Setup middleware for the application.

    Args:
        app (FastAPI): FastAPI application instance
    """
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )



def setup_exception_handlers(app: FastAPI) -> None:
    """
    Setup exception handlers for the application.

    Args:
        app (FastAPI): FastAPI application instance
    """

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request, exc):
        """Handle HTTP exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "status_code": exc.status_code,
                "path": str(request.url),
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc):
        """Handle request validation errors."""
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation error",
                "details": exc.errors(),
                "path": str(request.url),
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        """Handle general exceptions."""
        logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": (
                    str(exc) if settings.debug else "An unexpected error occurred"
                ),
                "path": str(request.url),
            },
        )


def setup_routers(app: FastAPI) -> None:
    """
    Setup API routers for the application.

    Args:
        app (FastAPI): FastAPI application instance
    """
    # Include API routers
    app.include_router(health.router)
    app.include_router(chat_history.router)
    app.include_router(insurance.router)

    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "message": f"Welcome to {settings.app_name}",
            "version": settings.app_version,
            "docs_url": (
                "/docs"
                if settings.debug
                else "Documentation not available in production"
            ),
            "health_check": "/api/v1/health",
            "insurance_chatbot": "/api/v1/insurance/chat",
        }


def setup_signal_handlers() -> None:
    """
    Setup signal handlers for graceful shutdown.
    """

    def shutdown_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        raise SystemExit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)


# Create the application instance
app = create_app()

# Setup signal handlers
setup_signal_handlers()


