"""
Advanced Settings Management for EnMapper - LLM-Powered Data Integration Platform

This module provides a comprehensive, hierarchical settings system that supports:
- Environment variables
- JSON file injection
- Dynamic runtime configuration
- Validation and type safety
- Feature flags and toggles
- Provider-specific configurations
"""

import json
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import Field
from pydantic.types import SecretStr
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ProcessingMode(str, Enum):
    """Data processing modes."""
    METADATA_ONLY = "metadata_only"
    DATA_MODE = "data_mode"


class ProcessingLane(str, Enum):
    """Processing lane types."""
    INTERACTIVE = "interactive"
    FLEX = "flex"
    BATCH = "batch"


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    OLLAMA = "ollama"


class RoutingProfile(str, Enum):
    """Provider routing profiles."""
    QUALITY_FIRST = "quality_first"
    COST_FIRST = "cost_first"
    LATENCY_FIRST = "latency_first"
    OFFLINE_FIRST = "offline_first"


# === LLM Provider Settings ===

class OpenAISettings(BaseSettings):
    """OpenAI-specific configuration."""
    api_key: Optional[SecretStr] = Field(None, env="OPENAI_API_KEY")
    organization: Optional[str] = Field(None, env="OPENAI_ORG_ID")
    base_url: str = Field("https://api.openai.com/v1", env="OPENAI_BASE_URL")
    default_model: str = Field("gpt-4", env="OPENAI_DEFAULT_MODEL")
    max_tokens: int = Field(4096, env="OPENAI_MAX_TOKENS")
    temperature: float = Field(0.7, env="OPENAI_TEMPERATURE")
    timeout: int = Field(60, env="OPENAI_TIMEOUT")


class AnthropicSettings(BaseSettings):
    """Anthropic-specific configuration."""
    api_key: Optional[SecretStr] = Field(None, env="ANTHROPIC_API_KEY")
    base_url: str = Field("https://api.anthropic.com", env="ANTHROPIC_BASE_URL")
    default_model: str = Field("claude-3-5-sonnet-20241022", env="ANTHROPIC_DEFAULT_MODEL")
    max_tokens: int = Field(4096, env="ANTHROPIC_MAX_TOKENS")
    temperature: float = Field(0.7, env="ANTHROPIC_TEMPERATURE")
    timeout: int = Field(60, env="ANTHROPIC_TIMEOUT")


class GroqSettings(BaseSettings):
    """Groq-specific configuration."""
    api_key: Optional[SecretStr] = Field(None, env="GROQ_API_KEY")
    base_url: str = Field("https://api.groq.com/openai/v1", env="GROQ_BASE_URL")
    default_model: str = Field("mixtral-8x7b-32768", env="GROQ_DEFAULT_MODEL")
    max_tokens: int = Field(4096, env="GROQ_MAX_TOKENS")
    temperature: float = Field(0.7, env="GROQ_TEMPERATURE")
    timeout: int = Field(30, env="GROQ_TIMEOUT")


class OllamaSettings(BaseSettings):
    """Ollama local model configuration."""
    base_url: str = Field("http://localhost:11434", env="OLLAMA_BASE_URL")
    default_model: str = Field("llama2", env="OLLAMA_DEFAULT_MODEL")
    max_tokens: int = Field(4096, env="OLLAMA_MAX_TOKENS")
    temperature: float = Field(0.7, env="OLLAMA_TEMPERATURE")
    timeout: int = Field(120, env="OLLAMA_TIMEOUT")
    gpu_layers: int = Field(-1, env="OLLAMA_GPU_LAYERS")


class LLMProviderSettings(BaseSettings):
    """Consolidated LLM provider settings."""
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    groq: GroqSettings = Field(default_factory=GroqSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    
    # Global provider settings
    default_provider: LLMProvider = Field(LLMProvider.ANTHROPIC, env="DEFAULT_LLM_PROVIDER")
    routing_profile: RoutingProfile = Field(RoutingProfile.QUALITY_FIRST, env="ROUTING_PROFILE")
    fallback_chain: List[LLMProvider] = Field(
        [LLMProvider.ANTHROPIC, LLMProvider.OPENAI, LLMProvider.GROQ],
        env="FALLBACK_CHAIN"
    )
    retry_attempts: int = Field(3, env="LLM_RETRY_ATTEMPTS")
    circuit_breaker_threshold: int = Field(5, env="CIRCUIT_BREAKER_THRESHOLD")


# === Main Settings Class ===

class Settings(BaseSettings):
    """Main application settings."""
    
    # Application metadata
    app_name: str = Field("EnMapper", env="APP_NAME")
    version: str = Field("1.0.0", env="APP_VERSION")
    environment: Environment = Field(Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(True, env="DEBUG")
    
    # Component settings
    llm: LLMProviderSettings = Field(default_factory=LLMProviderSettings)
    
    # JSON override support
    settings_json_path: Optional[str] = Field(None, env="SETTINGS_JSON_PATH")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        env_nested_delimiter = "__"  # Support nested env vars like LLM__OPENAI__API_KEY
        extra = "ignore"  # Ignore extra environment variables
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_json_overrides()
    
    def _load_json_overrides(self):
        """Load settings from JSON file if specified."""
        if self.settings_json_path and os.path.exists(self.settings_json_path):
            try:
                with open(self.settings_json_path, 'r') as f:
                    json_settings = json.load(f)
                
                # Apply JSON overrides recursively
                self._apply_json_overrides(json_settings)
                
                print(f"✅ Loaded settings from {self.settings_json_path}")
            except Exception as e:
                print(f"⚠️  Failed to load settings from {self.settings_json_path}: {e}")
    
    def _apply_json_overrides(self, json_data: Dict[str, Any], prefix: str = ""):
        """Recursively apply JSON settings overrides."""
        for key, value in json_data.items():
            if isinstance(value, dict):
                # Recurse into nested dictionaries
                self._apply_json_overrides(value, f"{prefix}{key}.")
            else:
                # Set the value on the appropriate nested object
                target = self
                path_parts = prefix.split('.') if prefix else []
                path_parts.append(key)
                
                # Navigate to the correct nested object
                for part in path_parts[:-1]:
                    if hasattr(target, part):
                        target = getattr(target, part)
                    else:
                        break
                else:
                    # Set the final value
                    if hasattr(target, path_parts[-1]):
                        setattr(target, path_parts[-1], value)
    
    def get_active_providers(self) -> List[LLMProvider]:
        """Get list of active LLM providers based on API keys."""
        active = []
        
        if self.llm.openai.api_key:
            active.append(LLMProvider.OPENAI)
        if self.llm.anthropic.api_key:
            active.append(LLMProvider.ANTHROPIC)
        if self.llm.groq.api_key:
            active.append(LLMProvider.GROQ)
        
        # Ollama doesn't require API key (local)
        active.append(LLMProvider.OLLAMA)
        
        return active
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate the current configuration and return status."""
        validation = {
            "timestamp": datetime.now().isoformat(),
            "environment": self.environment,
            "debug": self.debug,
            "providers": {},
            "warnings": [],
            "errors": []
        }
        
        # Validate providers
        active_providers = self.get_active_providers()
        validation["providers"]["active"] = [p.value for p in active_providers]
        validation["providers"]["default"] = self.llm.default_provider.value
        
        if not active_providers:
            validation["errors"].append("No LLM providers configured")
        
        if self.llm.default_provider not in active_providers:
            validation["warnings"].append(f"Default provider {self.llm.default_provider} not active")
        
        return validation


# === Settings Factory ===

_settings_instance = None

def get_settings(reload: bool = False) -> Settings:
    """Get singleton settings instance."""
    global _settings_instance
    
    if _settings_instance is None or reload:
        _settings_instance = Settings()
    
    return _settings_instance


def load_settings_from_json(json_path: str) -> Settings:
    """Load settings with JSON override."""
    os.environ["SETTINGS_JSON_PATH"] = json_path
    return Settings()


# === Utilities ===

def create_example_settings_json() -> str:
    """Create an example settings.json file."""
    example = {
        "llm": {
            "default_provider": "anthropic",
            "routing_profile": "quality_first",
            "anthropic": {
                "default_model": "claude-3-5-sonnet-20241022",
                "temperature": 0.5
            },
            "openai": {
                "default_model": "gpt-4",
                "temperature": 0.7
            }
        }
    }
    
    with open("settings.example.json", "w") as f:
        json.dump(example, f, indent=2)
    
    return "settings.example.json"


if __name__ == "__main__":
    # Quick validation when run directly
    settings = get_settings()
    validation = settings.validate_configuration()
    
    print("🔧 EnMapper Settings Validation")
    print("=" * 40)
    print(f"Environment: {validation['environment']}")
    print(f"Active Providers: {', '.join(validation['providers']['active'])}")
    print(f"Default Provider: {validation['providers']['default']}")
    
    if validation['warnings']:
        print(f"\n⚠️  Warnings: {len(validation['warnings'])}")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    
    if validation['errors']:
        print(f"\n❌ Errors: {len(validation['errors'])}")
        for error in validation['errors']:
            print(f"  - {error}")
    else:
        print("\n✅ Configuration valid!")
    
    # Create example JSON
    example_file = create_example_settings_json()
    print(f"\n📄 Example settings file created: {example_file}")
