"""
Comprehensive Observability Stack for EnMapper

This module provides:
- LangSmith integration for LLM call tracing
- Prometheus metrics collection  
- Performance monitoring
- Cost tracking
- Error tracking integration
"""

import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager, contextmanager
import logging

try:
    from langsmith import Client as LangSmithClient
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    LangSmithClient = None

try:
    from core.prometheus_metrics import (
        prometheus_metrics, record_http_request, record_llm_call,
        record_run_start, record_run_complete, initialize_prometheus_metrics
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    prometheus_metrics = None

logger = logging.getLogger(__name__)


class ObservabilityManager:
    """Central observability manager for all monitoring and tracing."""
    
    def __init__(self, settings=None):
        self.settings = settings
        self.langsmith_client = None
        self.current_run_tree = None
        self.metrics_cache = {}
        self.cost_tracker = CostTracker()
        
    async def initialize(self):
        """Initialize all observability components."""
        await self._initialize_langsmith()
        await self._initialize_prometheus()
        logger.info("🔍 Observability stack initialized")
        
    async def _initialize_langsmith(self):
        """Initialize LangSmith client."""
        if not LANGSMITH_AVAILABLE:
            logger.warning("LangSmith not available - install with: pip install langsmith")
            return
            
        try:
            # LangSmith will auto-configure from environment variables:
            # LANGCHAIN_API_KEY, LANGCHAIN_PROJECT, LANGCHAIN_ENDPOINT
            self.langsmith_client = LangSmithClient()
            logger.info("✅ LangSmith client initialized")
        except Exception as e:
            logger.error(f"❌ LangSmith initialization failed: {e}")
    
    async def _initialize_prometheus(self):
        """Initialize Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus not available")
            return
            
        try:
            prometheus_enabled = self.settings.observability.prometheus_enabled if self.settings else True
            prometheus_port = self.settings.observability.prometheus_port if self.settings else 8001
            
            if prometheus_enabled:
                initialize_prometheus_metrics(enabled=True, port=prometheus_port)
                logger.info(f"✅ Prometheus metrics initialized on port {prometheus_port}")
            else:
                logger.info("Prometheus metrics disabled in settings")
                
        except Exception as e:
            logger.error(f"❌ Prometheus initialization failed: {e}")
    
    @contextmanager
    def trace_operation(self, operation_name: str, metadata: Dict[str, Any] = None):
        """Context manager for tracing operations."""
        start_time = time.time()
        operation_id = str(uuid.uuid4())
        
        if metadata is None:
            metadata = {}
            
        # Add to metrics cache
        self.metrics_cache[operation_id] = {
            "operation": operation_name,
            "start_time": start_time,
            "metadata": metadata
        }
        
        try:
            yield operation_id
        except Exception as e:
            self.metrics_cache[operation_id]["error"] = str(e)
            raise
        finally:
            end_time = time.time()
            self.metrics_cache[operation_id]["end_time"] = end_time
            self.metrics_cache[operation_id]["duration"] = end_time - start_time
            
            # Log metrics
            duration = end_time - start_time
            logger.info(f"⏱️ {operation_name} completed in {duration:.2f}s")
    
    async def trace_llm_call(self, 
                           provider: str,
                           model: str, 
                           prompt: str,
                           response: str,
                           metadata: Dict[str, Any] = None,
                           duration: float = 0.0) -> str:
        """Trace an LLM call with full context."""
        call_id = str(uuid.uuid4())
        success = True
        
        # Extract task type from metadata
        task_type = metadata.get("task", "general") if metadata else "general"
        
        try:
            # LangSmith tracing
            if self.langsmith_client:
                run_data = {
                    "name": f"{provider}_{model}",
                    "run_type": "llm",
                    "inputs": {"prompt": prompt, "model": model, "provider": provider},
                    "outputs": {"response": response},
                    "extra": metadata or {},
                    "id": call_id
                }
                
                # Post to LangSmith using the new client API
                self.langsmith_client.create_run(**run_data)
            
            # Track costs
            await self.cost_tracker.track_llm_usage(provider, model, prompt, response)
            
            # Prometheus metrics
            if PROMETHEUS_AVAILABLE and prometheus_metrics:
                input_tokens = self.cost_tracker._estimate_tokens(prompt)
                output_tokens = self.cost_tracker._estimate_tokens(response)
                cost = self.cost_tracker._calculate_cost(provider, model, input_tokens, output_tokens)
                
                record_llm_call(
                    provider=provider,
                    model=model, 
                    task_type=task_type,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=cost,
                    duration=duration,
                    success=success
                )
            
            logger.info(f"📊 LLM call traced: {call_id}")
            
        except Exception as e:
            success = False
            logger.error(f"❌ LLM tracing failed: {e}")
            
            # Record error in Prometheus
            if PROMETHEUS_AVAILABLE and prometheus_metrics:
                prometheus_metrics.record_llm_error(provider, model, "tracing_error")
            
        return call_id
    
    async def start_run_trace(self, run_id: str, run_config: Dict[str, Any]):
        """Start tracing a complete EnMapper run."""
        if not self.langsmith_client:
            return
            
        try:
            # Create a session for this run
            session_name = f"enmapper_session_{datetime.now().isoformat()}"
            
            # Store run metadata for later use
            self.current_run_tree = {
                "name": f"EnMapper_Run_{run_id}",
                "run_type": "chain",
                "inputs": {"config": run_config, "run_id": run_id},
                "session_name": session_name,
                "id": run_id
            }
            
            logger.info(f"🔍 Started run trace for {run_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to start run trace: {e}")
    
    async def end_run_trace(self, run_id: str, outputs: Dict[str, Any]):
        """End tracing a complete EnMapper run."""
        if not self.current_run_tree or not self.langsmith_client:
            return
            
        try:
            # Update the run with outputs
            self.current_run_tree["outputs"] = outputs
            self.current_run_tree["end_time"] = datetime.now().isoformat()
            
            # Post the complete run tree
            self.langsmith_client.create_run(**self.current_run_tree)
            
            logger.info(f"✅ Completed run trace for {run_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to end run trace: {e}")
        finally:
            self.current_run_tree = None
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        total_operations = len(self.metrics_cache)
        completed_operations = len([m for m in self.metrics_cache.values() if "end_time" in m])
        failed_operations = len([m for m in self.metrics_cache.values() if "error" in m])
        
        avg_duration = 0
        if completed_operations > 0:
            total_duration = sum(m.get("duration", 0) for m in self.metrics_cache.values() if "duration" in m)
            avg_duration = total_duration / completed_operations
        
        return {
            "total_operations": total_operations,
            "completed_operations": completed_operations,
            "failed_operations": failed_operations,
            "success_rate": completed_operations / max(total_operations, 1),
            "average_duration_seconds": avg_duration,
            "total_cost_usd": self.cost_tracker.get_total_cost(),
            "timestamp": datetime.now().isoformat()
        }


class CostTracker:
    """Track costs across all LLM providers."""
    
    def __init__(self):
        self.usage_log = []
        self.total_cost = 0.0
        
        # Cost per 1K tokens (approximate)
        self.pricing = {
            "openai": {
                "gpt-4": {"input": 0.03, "output": 0.06},
                "gpt-5-mini": {"input": 0.00015, "output": 0.0006},
                "gpt-5-nano": {"input": 0.0001, "output": 0.0004},
            },
            "anthropic": {
                "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
                "claude-3-5-haiku-20241022": {"input": 0.0008, "output": 0.004},
            },
            "groq": {
                "llama-3.1-70b-versatile": {"input": 0.00059, "output": 0.00079},
                "llama-3.1-8b-instant": {"input": 0.00005, "output": 0.00008},
            }
        }
    
    async def track_llm_usage(self, provider: str, model: str, prompt: str, response: str):
        """Track usage and calculate costs for an LLM call."""
        input_tokens = self._estimate_tokens(prompt)
        output_tokens = self._estimate_tokens(response)
        
        cost = self._calculate_cost(provider, model, input_tokens, output_tokens)
        
        usage_entry = {
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost
        }
        
        self.usage_log.append(usage_entry)
        self.total_cost += cost
        
        logger.debug(f"💰 Cost tracked: {provider}/{model} - ${cost:.4f}")
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters)."""
        return len(text) // 4
    
    def _calculate_cost(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on provider pricing."""
        if provider not in self.pricing or model not in self.pricing[provider]:
            # Default fallback pricing
            return (input_tokens + output_tokens) * 0.002 / 1000
        
        pricing = self.pricing[provider][model]
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost
    
    def get_total_cost(self) -> float:
        """Get total cost across all tracked calls."""
        return self.total_cost
    
    def get_cost_breakdown(self) -> Dict[str, Any]:
        """Get detailed cost breakdown by provider/model."""
        breakdown = {}
        
        for entry in self.usage_log:
            provider = entry["provider"]
            model = entry["model"]
            key = f"{provider}/{model}"
            
            if key not in breakdown:
                breakdown[key] = {
                    "calls": 0,
                    "total_tokens": 0,
                    "total_cost": 0.0
                }
            
            breakdown[key]["calls"] += 1
            breakdown[key]["total_tokens"] += entry["input_tokens"] + entry["output_tokens"]
            breakdown[key]["total_cost"] += entry["cost_usd"]
        
        return breakdown


# Global observability instance
observability = ObservabilityManager()


# Convenience functions
async def trace_llm_call(provider: str, model: str, prompt: str, response: str, metadata: Dict[str, Any] = None) -> str:
    """Convenience function to trace an LLM call."""
    return await observability.trace_llm_call(provider, model, prompt, response, metadata)


def trace_operation(operation_name: str, metadata: Dict[str, Any] = None):
    """Convenience function to trace an operation."""
    return observability.trace_operation(operation_name, metadata)


async def initialize_observability(settings=None):
    """Initialize the global observability stack."""
    global observability
    observability.settings = settings
    await observability.initialize()


def get_metrics_summary() -> Dict[str, Any]:
    """Get current metrics summary."""
    return observability.get_metrics_summary()
