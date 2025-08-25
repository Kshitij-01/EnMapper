"""
EnMapper FastAPI Application - Main Entry Point

Phase 0: Foundation with health endpoints and basic structure
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from settings import get_settings, Settings
from core.health import HealthChecker
from core.policy import PolicyEngine
from core.models import RunContract
from core.providers import ModelRegistry


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    # Startup
    logger.info("🚀 Starting EnMapper API")
    
    settings = get_settings()
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")
    
    # Initialize core components
    try:
        # Initialize health checker
        health_checker = HealthChecker()
        app.state.health_checker = health_checker
        
        # Initialize policy engine
        policy_engine = PolicyEngine()
        app.state.policy_engine = policy_engine
        
        # Initialize model registry
        model_registry = ModelRegistry()
        await model_registry.initialize()
        app.state.model_registry = model_registry
        
        logger.info("✅ Core components initialized")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize components: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down EnMapper API")


# Create FastAPI app
def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="EnMapper API",
        description="AI-Powered Data Mapping and Migration Platform",
        version="1.0.0",
        lifespan=lifespan,
        debug=settings.debug
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


app = create_app()


# === HEALTH & STATUS ENDPOINTS ===

@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for container orchestration.
    Returns basic service status.
    """
    settings = get_settings()
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "enmapper-api",
        "version": "1.0.0",
        "environment": settings.environment.value
    }


@app.get("/readiness", tags=["Health"])
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check endpoint for container orchestration.
    Validates all dependencies are available.
    """
    try:
        health_checker: HealthChecker = app.state.health_checker
        results = await health_checker.check_all()
        
        # Determine overall readiness
        is_ready = all(component["healthy"] for component in results["components"].values())
        
        return {
            "status": "ready" if is_ready else "not_ready",
            "timestamp": datetime.utcnow().isoformat(),
            **results
        }
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")


@app.get("/status", tags=["Health"])
async def service_status() -> Dict[str, Any]:
    """
    Detailed service status including configuration and provider health.
    """
    try:
        settings = get_settings()
        health_checker: HealthChecker = app.state.health_checker
        model_registry: ModelRegistry = app.state.model_registry
        
        # Get component health
        health_results = await health_checker.check_all()
        
        # Get provider status
        provider_status = await model_registry.get_provider_health()
        
        # Get configuration validation
        config_validation = settings.validate_configuration()
        
        return {
            "service": {
                "name": "enmapper-api",
                "version": "1.0.0",
                "environment": settings.environment.value,
                "debug": settings.debug
            },
            "health": health_results,
            "providers": provider_status,
            "configuration": config_validation,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail="Status check failed")


# === PHASE 0 API ENDPOINTS ===

@app.get("/", tags=["Root"])
async def root() -> Dict[str, str]:
    """Root endpoint with API information."""
    return {
        "message": "EnMapper API - AI-Powered Data Mapping and Migration Platform",
        "version": "1.0.0",
        "phase": "Phase 0 - Foundation",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/info", tags=["Info"])
async def api_info() -> Dict[str, Any]:
    """API information and capabilities."""
    settings = get_settings()
    
    return {
        "api": {
            "name": "EnMapper",
            "version": "1.0.0",
            "phase": "Phase 0 - Foundation",
            "environment": settings.environment.value
        },
        "capabilities": {
            "providers": ["openai", "anthropic", "groq", "ollama"],
            "modes": ["metadata_only", "data_mode"],
            "lanes": ["interactive", "flex", "batch"]
        },
        "endpoints": {
            "health": "/health",
            "readiness": "/readiness", 
            "status": "/status",
            "docs": "/docs"
        }
    }


# === ERROR HANDLERS ===

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested endpoint was not found",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Custom 500 handler."""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An internal error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


def get_settings_dependency() -> Settings:
    """Dependency to get settings instance."""
    return get_settings()


# === DEVELOPMENT SERVER ===

if __name__ == "__main__":
    settings = get_settings()
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug"
    )
