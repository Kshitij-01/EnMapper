"""
Health Check System for EnMapper

Monitors database, Redis, and external service dependencies.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import httpx
import redis.asyncio as redis
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine

from settings import get_settings


logger = logging.getLogger(__name__)


class HealthChecker:
    """Health checker for all EnMapper dependencies."""
    
    def __init__(self):
        self.settings = get_settings()
        self._db_engine: Optional[sa.engine.Engine] = None
        self._redis_client: Optional[redis.Redis] = None
    
    async def check_database(self) -> Dict[str, Any]:
        """Check PostgreSQL database connectivity."""
        try:
            # Convert database URL for async usage
            db_url = self.settings.database.postgres_url
            if db_url.startswith("postgresql://"):
                db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
            
            # Create async engine if not exists
            if not self._db_engine:
                self._db_engine = create_async_engine(db_url, echo=False)
            
            # Test connection
            async with self._db_engine.connect() as conn:
                result = await conn.execute(sa.text("SELECT 1"))
                await result.fetchone()
            
            return {
                "healthy": True,
                "status": "connected",
                "timestamp": datetime.utcnow().isoformat(),
                "database": "postgresql"
            }
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "healthy": False,
                "status": "connection_failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "database": "postgresql"
            }
    
    async def check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity."""
        try:
            # Create Redis client if not exists
            if not self._redis_client:
                self._redis_client = redis.from_url(self.settings.database.redis_url)
            
            # Test connection
            await self._redis_client.ping()
            
            return {
                "healthy": True,
                "status": "connected",
                "timestamp": datetime.utcnow().isoformat(),
                "cache": "redis"
            }
            
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return {
                "healthy": False,
                "status": "connection_failed", 
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "cache": "redis"
            }
    
    async def check_llm_providers(self) -> Dict[str, Any]:
        """Check LLM provider availability."""
        providers_status = {}
        
        # Check Anthropic
        if self.settings.llm.anthropic.api_key:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{self.settings.llm.anthropic.base_url}/v1/messages",
                        headers={"x-api-key": self.settings.llm.anthropic.api_key.get_secret_value()},
                        timeout=5.0
                    )
                    providers_status["anthropic"] = {
                        "healthy": response.status_code in [200, 401, 400],  # 401/400 means API is responding
                        "status": "available",
                        "response_code": response.status_code
                    }
            except Exception as e:
                providers_status["anthropic"] = {
                    "healthy": False,
                    "status": "unavailable",
                    "error": str(e)
                }
        
        # Check OpenAI
        if self.settings.llm.openai.api_key:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{self.settings.llm.openai.base_url}/models",
                        headers={"Authorization": f"Bearer {self.settings.llm.openai.api_key.get_secret_value()}"},
                        timeout=5.0
                    )
                    providers_status["openai"] = {
                        "healthy": response.status_code in [200, 401],
                        "status": "available",
                        "response_code": response.status_code
                    }
            except Exception as e:
                providers_status["openai"] = {
                    "healthy": False,
                    "status": "unavailable", 
                    "error": str(e)
                }
        
        # Check Groq
        if self.settings.llm.groq.api_key:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{self.settings.llm.groq.base_url}/models",
                        headers={"Authorization": f"Bearer {self.settings.llm.groq.api_key.get_secret_value()}"},
                        timeout=5.0
                    )
                    providers_status["groq"] = {
                        "healthy": response.status_code in [200, 401],
                        "status": "available",
                        "response_code": response.status_code
                    }
            except Exception as e:
                providers_status["groq"] = {
                    "healthy": False,
                    "status": "unavailable",
                    "error": str(e)
                }
        
        # Check Ollama (local)
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.settings.llm.ollama.base_url}/api/tags",
                    timeout=5.0
                )
                providers_status["ollama"] = {
                    "healthy": response.status_code == 200,
                    "status": "available" if response.status_code == 200 else "unavailable",
                    "response_code": response.status_code
                }
        except Exception as e:
            providers_status["ollama"] = {
                "healthy": False,
                "status": "unavailable",
                "error": str(e)
            }
        
        return {
            "providers": providers_status,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def check_all(self) -> Dict[str, Any]:
        """Run all health checks concurrently."""
        try:
            # Run checks concurrently
            db_task = asyncio.create_task(self.check_database())
            redis_task = asyncio.create_task(self.check_redis())
            providers_task = asyncio.create_task(self.check_llm_providers())
            
            db_health, redis_health, providers_health = await asyncio.gather(
                db_task, redis_task, providers_task
            )
            
            return {
                "components": {
                    "database": db_health,
                    "redis": redis_health
                },
                "llm_providers": providers_health,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "components": {
                    "database": {"healthy": False, "error": "health_check_failed"},
                    "redis": {"healthy": False, "error": "health_check_failed"}
                },
                "llm_providers": {"providers": {}, "error": "health_check_failed"},
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def close(self):
        """Clean up connections."""
        if self._db_engine:
            await self._db_engine.dispose()
        if self._redis_client:
            await self._redis_client.close()
