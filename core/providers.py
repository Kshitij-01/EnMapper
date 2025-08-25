"""
Model Registry and LLM Provider Adapters for EnMapper

Phase 0: Skeleton implementation with uniform adapter interface.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional, AsyncGenerator
import httpx
import asyncio

from settings import get_settings, LLMProvider


logger = logging.getLogger(__name__)


class LLMAdapter(ABC):
    """Abstract base class for LLM provider adapters."""
    
    @abstractmethod
    async def infer(self, prompt: str, **kwargs) -> str:
        """Generate text completion."""
        pass
    
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate text embeddings."""
        pass
    
    @abstractmethod
    async def token_estimate(self, text: str) -> int:
        """Estimate token count for text."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check provider health and availability."""
        pass


class AnthropicAdapter(LLMAdapter):
    """Anthropic Claude adapter."""
    
    def __init__(self):
        self.settings = get_settings()
        self.config = self.settings.llm.anthropic
        self.client = None
    
    async def _get_client(self):
        """Get or create HTTP client."""
        if not self.client:
            self.client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers={
                    "x-api-key": self.config.api_key.get_secret_value() if self.config.api_key else "",
                    "content-type": "application/json",
                    "anthropic-version": "2023-06-01"
                },
                timeout=self.config.timeout
            )
        return self.client
    
    async def infer(self, prompt: str, **kwargs) -> str:
        """Generate completion using Claude."""
        if not self.config.api_key:
            raise ValueError("Anthropic API key not configured")
        
        client = await self._get_client()
        
        payload = {
            "model": kwargs.get("model", self.config.default_model),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "messages": [{"role": "user", "content": prompt}]
        }
        
        try:
            response = await client.post("/v1/messages", json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result["content"][0]["text"]
            
        except Exception as e:
            logger.error(f"Anthropic inference failed: {e}")
            raise
    
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings (placeholder - Anthropic doesn't have embedding API)."""
        # Phase 0: Return placeholder
        # Later phases will use alternative embedding service
        return [0.0] * 1536  # OpenAI embedding dimension
    
    async def token_estimate(self, text: str) -> int:
        """Estimate token count using simple heuristic."""
        # Rough estimate: ~4 characters per token for English
        return len(text) // 4
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Anthropic API health."""
        try:
            if not self.config.api_key:
                return {"healthy": False, "error": "API key not configured"}
            
            client = await self._get_client()
            response = await client.get("/v1/messages", timeout=5.0)
            
            return {
                "healthy": response.status_code in [200, 400, 401],  # 400/401 means API is responding
                "status_code": response.status_code,
                "provider": "anthropic",
                "model": self.config.default_model,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "provider": "anthropic",
                "timestamp": datetime.utcnow().isoformat()
            }


class OpenAIAdapter(LLMAdapter):
    """OpenAI GPT adapter."""
    
    def __init__(self):
        self.settings = get_settings()
        self.config = self.settings.llm.openai
        self.client = None
    
    async def _get_client(self):
        """Get or create HTTP client."""
        if not self.client:
            headers = {
                "Authorization": f"Bearer {self.config.api_key.get_secret_value()}" if self.config.api_key else "",
                "Content-Type": "application/json"
            }
            if self.config.organization:
                headers["OpenAI-Organization"] = self.config.organization
            
            self.client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers=headers,
                timeout=self.config.timeout
            )
        return self.client
    
    async def infer(self, prompt: str, **kwargs) -> str:
        """Generate completion using GPT."""
        if not self.config.api_key:
            raise ValueError("OpenAI API key not configured")
        
        client = await self._get_client()
        
        payload = {
            "model": kwargs.get("model", self.config.default_model),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "messages": [{"role": "user", "content": prompt}]
        }
        
        try:
            response = await client.post("/chat/completions", json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"OpenAI inference failed: {e}")
            raise
    
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI."""
        if not self.config.api_key:
            raise ValueError("OpenAI API key not configured")
        
        client = await self._get_client()
        
        payload = {
            "model": "text-embedding-ada-002",
            "input": text
        }
        
        try:
            response = await client.post("/embeddings", json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result["data"][0]["embedding"]
            
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise
    
    async def token_estimate(self, text: str) -> int:
        """Estimate token count using simple heuristic."""
        # Rough estimate: ~4 characters per token for English
        return len(text) // 4
    
    async def health_check(self) -> Dict[str, Any]:
        """Check OpenAI API health."""
        try:
            if not self.config.api_key:
                return {"healthy": False, "error": "API key not configured"}
            
            client = await self._get_client()
            response = await client.get("/models", timeout=5.0)
            
            return {
                "healthy": response.status_code == 200,
                "status_code": response.status_code,
                "provider": "openai",
                "model": self.config.default_model,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "provider": "openai",
                "timestamp": datetime.utcnow().isoformat()
            }


class GroqAdapter(LLMAdapter):
    """Groq fast inference adapter."""
    
    def __init__(self):
        self.settings = get_settings()
        self.config = self.settings.llm.groq
        self.client = None
    
    async def _get_client(self):
        """Get or create HTTP client."""
        if not self.client:
            self.client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers={
                    "Authorization": f"Bearer {self.config.api_key.get_secret_value()}" if self.config.api_key else "",
                    "Content-Type": "application/json"
                },
                timeout=self.config.timeout
            )
        return self.client
    
    async def infer(self, prompt: str, **kwargs) -> str:
        """Generate completion using Groq."""
        if not self.config.api_key:
            raise ValueError("Groq API key not configured")
        
        client = await self._get_client()
        
        payload = {
            "model": kwargs.get("model", self.config.default_model),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "messages": [{"role": "user", "content": prompt}]
        }
        
        try:
            response = await client.post("/chat/completions", json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"Groq inference failed: {e}")
            raise
    
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings (placeholder - Groq focused on inference)."""
        # Phase 0: Return placeholder
        return [0.0] * 1536
    
    async def token_estimate(self, text: str) -> int:
        """Estimate token count using simple heuristic."""
        return len(text) // 4
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Groq API health."""
        try:
            if not self.config.api_key:
                return {"healthy": False, "error": "API key not configured"}
            
            client = await self._get_client()
            response = await client.get("/models", timeout=5.0)
            
            return {
                "healthy": response.status_code == 200,
                "status_code": response.status_code,
                "provider": "groq", 
                "model": self.config.default_model,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "provider": "groq",
                "timestamp": datetime.utcnow().isoformat()
            }


class OllamaAdapter(LLMAdapter):
    """Ollama local model adapter."""
    
    def __init__(self):
        self.settings = get_settings()
        self.config = self.settings.llm.ollama
        self.client = None
    
    async def _get_client(self):
        """Get or create HTTP client."""
        if not self.client:
            self.client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )
        return self.client
    
    async def infer(self, prompt: str, **kwargs) -> str:
        """Generate completion using Ollama."""
        client = await self._get_client()
        
        payload = {
            "model": kwargs.get("model", self.config.default_model),
            "prompt": prompt,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens)
            }
        }
        
        try:
            response = await client.post("/api/generate", json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            
        except Exception as e:
            logger.error(f"Ollama inference failed: {e}")
            raise
    
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings using Ollama."""
        client = await self._get_client()
        
        payload = {
            "model": kwargs.get("model", "nomic-embed-text"),
            "prompt": text
        }
        
        try:
            response = await client.post("/api/embeddings", json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result.get("embedding", [0.0] * 768)  # Default dimension
            
        except Exception as e:
            logger.error(f"Ollama embedding failed: {e}")
            # Return placeholder on failure
            return [0.0] * 768
    
    async def token_estimate(self, text: str) -> int:
        """Estimate token count using simple heuristic."""
        return len(text) // 4
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Ollama service health."""
        try:
            client = await self._get_client()
            response = await client.get("/api/tags", timeout=5.0)
            
            return {
                "healthy": response.status_code == 200,
                "status_code": response.status_code,
                "provider": "ollama",
                "model": self.config.default_model,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "provider": "ollama",
                "timestamp": datetime.utcnow().isoformat()
            }


class ModelRegistry:
    """
    Model Registry for managing LLM providers and routing.
    
    Phase 0: Basic adapter management and health checking.
    Later phases will add sophisticated routing, load balancing, and fallback logic.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.adapters: Dict[str, LLMAdapter] = {}
        self._health_cache = {}
        self._last_health_check = {}
    
    async def initialize(self):
        """Initialize all configured adapters."""
        logger.info("🔧 Initializing Model Registry")
        
        # Initialize adapters for configured providers
        if self.settings.llm.anthropic.api_key:
            self.adapters["anthropic"] = AnthropicAdapter()
            logger.info("✅ Anthropic adapter initialized")
        
        if self.settings.llm.openai.api_key:
            self.adapters["openai"] = OpenAIAdapter()
            logger.info("✅ OpenAI adapter initialized")
        
        if self.settings.llm.groq.api_key:
            self.adapters["groq"] = GroqAdapter()
            logger.info("✅ Groq adapter initialized")
        
        # Ollama doesn't require API key
        self.adapters["ollama"] = OllamaAdapter()
        logger.info("✅ Ollama adapter initialized")
        
        logger.info(f"🚀 Model Registry ready with {len(self.adapters)} adapters")
    
    def get_adapter(self, provider: str) -> Optional[LLMAdapter]:
        """Get adapter for a specific provider."""
        return self.adapters.get(provider)
    
    async def get_provider_health(self) -> Dict[str, Any]:
        """Get health status for all providers."""
        health_results = {}
        
        # Run health checks concurrently
        tasks = {
            provider: adapter.health_check()
            for provider, adapter in self.adapters.items()
        }
        
        try:
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            
            for (provider, _), result in zip(tasks.items(), results):
                if isinstance(result, Exception):
                    health_results[provider] = {
                        "healthy": False,
                        "error": str(result),
                        "provider": provider,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                else:
                    health_results[provider] = result
            
        except Exception as e:
            logger.error(f"Provider health check failed: {e}")
            for provider in self.adapters.keys():
                health_results[provider] = {
                    "healthy": False,
                    "error": "health_check_failed",
                    "provider": provider,
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        return {
            "providers": health_results,
            "total_providers": len(self.adapters),
            "healthy_providers": sum(1 for h in health_results.values() if h.get("healthy", False)),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def route_request(self, provider_hint: Optional[str] = None) -> str:
        """
        Route request to best available provider.
        
        Phase 0: Simple routing based on configured default.
        Later phases will implement sophisticated routing logic.
        """
        # Use hint if provided and available
        if provider_hint and provider_hint in self.adapters:
            return provider_hint
        
        # Use configured default
        default_provider = self.settings.llm.default_provider.value
        if default_provider in self.adapters:
            return default_provider
        
        # Fall back to first available
        if self.adapters:
            return next(iter(self.adapters.keys()))
        
        raise ValueError("No LLM providers available")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        return list(self.adapters.keys())
    
    async def close(self):
        """Clean up adapters and connections."""
        for adapter in self.adapters.values():
            if hasattr(adapter, 'client') and adapter.client:
                await adapter.client.aclose()
        logger.info("🛑 Model Registry closed")
