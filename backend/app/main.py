"""
FastAPI Main Application Entry Point with Lifespan Model Startup and CORS Middleware.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.routes.prediction import router as prediction_router
from app.services.inference import model_service
from app.utils.logging_config import setup_logging

logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to load ML model once on startup
    and perform cleanup on shutdown.
    """
    logger.info("Initializing application startup...")
    try:
        model_service.load_model()
        logger.info("ML Model and locations list loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load ML model on startup: {e}")
        raise e
    yield
    logger.info("Application shutdown completed.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routes
app.include_router(prediction_router, prefix=settings.API_PREFIX)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
