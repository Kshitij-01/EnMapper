"""
LLM Configuration Manager - Loads settings for different LLM providers
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path

from core.generic_llm_agent import LLMProvider
from settings import get_settings

class LLMConfigManager:
    """Manages LLM provider configurations"""
    
    def __init__(self):
        self.settings = get_settings()
    
    def get_provider_config(self, provider: LLMProvider) -> Dict[str, Any]:
        """Get configuration for a specific LLM provider"""
        
        if provider == LLMProvider.CLAUDE:
            return self._get_claude_config()
        elif provider == LLMProvider.OPENAI:
            return self._get_openai_config()
        elif provider == LLMProvider.GROQ:
            return self._get_groq_config()
        elif provider == LLMProvider.OLLAMA:
            return self._get_ollama_config()
        else:
            return {}
    
    def _get_claude_config(self) -> Dict[str, Any]:
        """Get Claude/Anthropic configuration"""
        return {
            "api_key": self.settings.llm.anthropic.api_key.get_secret_value() if self.settings.llm.anthropic.api_key else None,
            "base_url": self.settings.llm.anthropic.base_url,
            "model": self.settings.llm.anthropic.default_model,
            "max_tokens": self.settings.llm.anthropic.max_tokens,
            "temperature": self.settings.llm.anthropic.temperature,
            "timeout": self.settings.llm.anthropic.timeout
        }
    
    def _get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI configuration"""
        return {
            "api_key": self.settings.llm.openai.api_key.get_secret_value() if self.settings.llm.openai.api_key else None,
            "base_url": self.settings.llm.openai.base_url,
            "model": self.settings.llm.openai.default_model,
            "max_tokens": self.settings.llm.openai.max_tokens,
            "temperature": self.settings.llm.openai.temperature,
            "timeout": self.settings.llm.openai.timeout
        }
    
    def _get_groq_config(self) -> Dict[str, Any]:
        """Get Groq configuration"""
        return {
            "api_key": self.settings.llm.groq.api_key.get_secret_value() if self.settings.llm.groq.api_key else None,
            "base_url": self.settings.llm.groq.base_url,
            "model": self.settings.llm.groq.default_model,
            "max_tokens": self.settings.llm.groq.max_tokens,
            "temperature": self.settings.llm.groq.temperature,
            "timeout": self.settings.llm.groq.timeout
        }
    
    def _get_ollama_config(self) -> Dict[str, Any]:
        """Get Ollama configuration"""
        return {
            "base_url": self.settings.llm.ollama.base_url,
            "model": self.settings.llm.ollama.default_model,
            "max_tokens": self.settings.llm.ollama.max_tokens,
            "temperature": self.settings.llm.ollama.temperature,
            "timeout": self.settings.llm.ollama.timeout
        }
    
    def get_default_provider(self) -> LLMProvider:
        """Get the default LLM provider from settings"""
        
        provider_map = {
            "anthropic": LLMProvider.CLAUDE,
            "openai": LLMProvider.OPENAI,
            "groq": LLMProvider.GROQ,
            "ollama": LLMProvider.OLLAMA
        }
        
        default = self.settings.llm.default_provider.value.lower()
        return provider_map.get(default, LLMProvider.MOCK)
    
    def is_provider_configured(self, provider: LLMProvider) -> bool:
        """Check if a provider is properly configured"""
        
        config = self.get_provider_config(provider)
        
        if provider == LLMProvider.CLAUDE:
            return config.get("api_key") is not None
        elif provider == LLMProvider.OPENAI:
            return config.get("api_key") is not None
        elif provider == LLMProvider.GROQ:
            return config.get("api_key") is not None
        elif provider == LLMProvider.OLLAMA:
            return config.get("base_url") is not None
        elif provider == LLMProvider.MOCK:
            return True
        
        return False
    
    def get_best_available_provider(self) -> LLMProvider:
        """Get the best available and configured provider"""
        
        # Priority order
        priority_providers = [
            LLMProvider.CLAUDE,
            LLMProvider.OPENAI,
            LLMProvider.GROQ,
            LLMProvider.OLLAMA,
            LLMProvider.MOCK  # Always available as fallback
        ]
        
        for provider in priority_providers:
            if self.is_provider_configured(provider):
                return provider
        
        return LLMProvider.MOCK
    
    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all providers"""
        
        status = {}
        
        for provider in LLMProvider:
            config = self.get_provider_config(provider)
            status[provider.value] = {
                "configured": self.is_provider_configured(provider),
                "has_api_key": bool(config.get("api_key")) if provider != LLMProvider.MOCK else True,
                "model": config.get("model", "N/A"),
                "base_url": config.get("base_url", "N/A")
            }
        
        return status
