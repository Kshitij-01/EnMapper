"""
Prometheus Metrics for EnMapper

This module provides comprehensive Prometheus metrics collection for:
- API performance and usage
- LLM calls and costs
- Business metrics (runs, artifacts, processing)
- System health and resource usage
"""

import time
from typing import Dict, Any, Optional
from datetime import datetime
import logging

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, Info,
        start_http_server, generate_latest, CONTENT_TYPE_LATEST,
        CollectorRegistry, REGISTRY
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = Histogram = Gauge = Summary = Info = None
    start_http_server = generate_latest = CONTENT_TYPE_LATEST = None
    CollectorRegistry = REGISTRY = None

logger = logging.getLogger(__name__)


class PrometheusMetrics:
    """Central Prometheus metrics collector for EnMapper."""
    
    def __init__(self, enabled: bool = True, port: int = 8001):
        self.enabled = enabled and PROMETHEUS_AVAILABLE
        self.port = port
        self.registry = CollectorRegistry() if PROMETHEUS_AVAILABLE else None
        self.server_started = False
        
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available - install with: pip install prometheus-client")
            return
            
        if not self.enabled:
            logger.info("Prometheus metrics disabled")
            return
            
        self._initialize_metrics()
        logger.info("📊 Prometheus metrics initialized")
    
    def _initialize_metrics(self):
        """Initialize all Prometheus metrics."""
        if not self.enabled:
            return
            
        # === API Metrics ===
        self.http_requests_total = Counter(
            'enmapper_http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.http_request_duration = Histogram(
            'enmapper_http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.http_request_size = Histogram(
            'enmapper_http_request_size_bytes',
            'HTTP request size in bytes',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.http_response_size = Histogram(
            'enmapper_http_response_size_bytes',
            'HTTP response size in bytes',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # === LLM Metrics ===
        self.llm_calls_total = Counter(
            'enmapper_llm_calls_total',
            'Total LLM API calls',
            ['provider', 'model', 'task_type'],
            registry=self.registry
        )
        
        self.llm_tokens_total = Counter(
            'enmapper_llm_tokens_total',
            'Total LLM tokens consumed',
            ['provider', 'model', 'token_type'],  # input/output
            registry=self.registry
        )
        
        self.llm_cost_total = Counter(
            'enmapper_llm_cost_usd_total',
            'Total LLM costs in USD',
            ['provider', 'model'],
            registry=self.registry
        )
        
        self.llm_request_duration = Histogram(
            'enmapper_llm_request_duration_seconds',
            'LLM request duration in seconds',
            ['provider', 'model'],
            registry=self.registry
        )
        
        self.llm_errors_total = Counter(
            'enmapper_llm_errors_total',
            'Total LLM API errors',
            ['provider', 'model', 'error_type'],
            registry=self.registry
        )
        
        # === Business Metrics ===
        self.runs_total = Counter(
            'enmapper_runs_total',
            'Total processing runs',
            ['mode', 'lane', 'status'],
            registry=self.registry
        )
        
        self.run_duration = Histogram(
            'enmapper_run_duration_seconds',
            'Processing run duration in seconds',
            ['mode', 'lane'],
            registry=self.registry
        )
        
        self.artifacts_total = Counter(
            'enmapper_artifacts_total',
            'Total artifacts generated',
            ['type', 'status'],
            registry=self.registry
        )
        
        self.data_processed = Histogram(
            'enmapper_data_processed_bytes',
            'Data processed in bytes',
            ['source_type', 'format'],
            registry=self.registry
        )
        
        self.rows_processed = Histogram(
            'enmapper_rows_processed_total',
            'Rows processed per run',
            ['source_type', 'mode'],
            registry=self.registry
        )
        
        # === System Metrics ===
        self.active_runs = Gauge(
            'enmapper_active_runs',
            'Number of currently active runs',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'enmapper_memory_usage_bytes',
            'Memory usage in bytes',
            ['component'],
            registry=self.registry
        )
        
        self.cache_hits = Counter(
            'enmapper_cache_hits_total',
            'Cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'enmapper_cache_misses_total',
            'Cache misses',
            ['cache_type'],
            registry=self.registry
        )
        
        # === PII & Security Metrics ===
        self.pii_detections = Counter(
            'enmapper_pii_detections_total',
            'PII detections',
            ['category', 'action'],  # redact/mask/skip
            registry=self.registry
        )
        
        self.policy_violations = Counter(
            'enmapper_policy_violations_total',
            'Policy violations',
            ['policy_type', 'severity'],
            registry=self.registry
        )
        
        # === Info Metrics ===
        self.build_info = Info(
            'enmapper_build_info',
            'Build information',
            registry=self.registry
        )
        
        # Set build info
        self.build_info.info({
            'version': '1.0.0',
            'environment': 'development',
            'build_time': datetime.now().isoformat()
        })
        
        logger.info("✅ All Prometheus metrics initialized")
    
    def start_server(self):
        """Start Prometheus metrics HTTP server."""
        if not self.enabled or self.server_started:
            return
            
        try:
            start_http_server(self.port, registry=self.registry)
            self.server_started = True
            logger.info(f"📊 Prometheus metrics server started on port {self.port}")
            logger.info(f"🔗 Metrics available at: http://localhost:{self.port}/metrics")
        except Exception as e:
            logger.error(f"❌ Failed to start Prometheus server: {e}")
    
    def get_metrics_output(self) -> str:
        """Get metrics in Prometheus format."""
        if not self.enabled:
            return "# Prometheus metrics disabled\n"
        return generate_latest(self.registry).decode('utf-8')
    
    # === API Metrics Methods ===
    
    def record_http_request(self, method: str, endpoint: str, status_code: int, 
                          duration: float, request_size: int = 0, response_size: int = 0):
        """Record HTTP request metrics."""
        if not self.enabled:
            return
            
        self.http_requests_total.labels(
            method=method, 
            endpoint=endpoint, 
            status_code=str(status_code)
        ).inc()
        
        self.http_request_duration.labels(
            method=method, 
            endpoint=endpoint
        ).observe(duration)
        
        if request_size > 0:
            self.http_request_size.labels(
                method=method, 
                endpoint=endpoint
            ).observe(request_size)
        
        if response_size > 0:
            self.http_response_size.labels(
                method=method, 
                endpoint=endpoint
            ).observe(response_size)
    
    # === LLM Metrics Methods ===
    
    def record_llm_call(self, provider: str, model: str, task_type: str,
                       input_tokens: int, output_tokens: int, cost: float, 
                       duration: float, success: bool = True):
        """Record LLM call metrics."""
        if not self.enabled:
            return
            
        # Record call
        self.llm_calls_total.labels(
            provider=provider,
            model=model,
            task_type=task_type
        ).inc()
        
        # Record tokens
        self.llm_tokens_total.labels(
            provider=provider,
            model=model,
            token_type='input'
        ).inc(input_tokens)
        
        self.llm_tokens_total.labels(
            provider=provider,
            model=model,
            token_type='output'
        ).inc(output_tokens)
        
        # Record cost
        self.llm_cost_total.labels(
            provider=provider,
            model=model
        ).inc(cost)
        
        # Record duration
        self.llm_request_duration.labels(
            provider=provider,
            model=model
        ).observe(duration)
        
        # Record errors if unsuccessful
        if not success:
            self.llm_errors_total.labels(
                provider=provider,
                model=model,
                error_type='api_error'
            ).inc()
    
    def record_llm_error(self, provider: str, model: str, error_type: str):
        """Record LLM error."""
        if not self.enabled:
            return
            
        self.llm_errors_total.labels(
            provider=provider,
            model=model,
            error_type=error_type
        ).inc()
    
    # === Business Metrics Methods ===
    
    def record_run_start(self, mode: str, lane: str):
        """Record run start."""
        if not self.enabled:
            return
            
        self.runs_total.labels(
            mode=mode,
            lane=lane,
            status='started'
        ).inc()
        
        self.active_runs.inc()
    
    def record_run_complete(self, mode: str, lane: str, duration: float, success: bool = True):
        """Record run completion."""
        if not self.enabled:
            return
            
        status = 'completed' if success else 'failed'
        
        self.runs_total.labels(
            mode=mode,
            lane=lane,
            status=status
        ).inc()
        
        self.run_duration.labels(
            mode=mode,
            lane=lane
        ).observe(duration)
        
        self.active_runs.dec()
    
    def record_artifact_generation(self, artifact_type: str, success: bool = True):
        """Record artifact generation."""
        if not self.enabled:
            return
            
        status = 'success' if success else 'error'
        self.artifacts_total.labels(
            type=artifact_type,
            status=status
        ).inc()
    
    def record_data_processing(self, source_type: str, format_type: str, 
                             bytes_processed: int, rows_processed: int, mode: str):
        """Record data processing metrics."""
        if not self.enabled:
            return
            
        self.data_processed.labels(
            source_type=source_type,
            format=format_type
        ).observe(bytes_processed)
        
        self.rows_processed.labels(
            source_type=source_type,
            mode=mode
        ).observe(rows_processed)
    
    # === Security Metrics Methods ===
    
    def record_pii_detection(self, category: str, action: str):
        """Record PII detection."""
        if not self.enabled:
            return
            
        self.pii_detections.labels(
            category=category,
            action=action
        ).inc()
    
    def record_policy_violation(self, policy_type: str, severity: str):
        """Record policy violation."""
        if not self.enabled:
            return
            
        self.policy_violations.labels(
            policy_type=policy_type,
            severity=severity
        ).inc()
    
    # === System Metrics Methods ===
    
    def record_cache_hit(self, cache_type: str):
        """Record cache hit."""
        if not self.enabled:
            return
            
        self.cache_hits.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str):
        """Record cache miss."""
        if not self.enabled:
            return
            
        self.cache_misses.labels(cache_type=cache_type).inc()
    
    def update_memory_usage(self, component: str, bytes_used: int):
        """Update memory usage gauge."""
        if not self.enabled:
            return
            
        self.memory_usage.labels(component=component).set(bytes_used)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of key metrics."""
        if not self.enabled:
            return {"prometheus_enabled": False}
        
        return {
            "prometheus_enabled": True,
            "metrics_server_port": self.port,
            "metrics_server_running": self.server_started,
            "metrics_endpoint": f"http://localhost:{self.port}/metrics",
            "total_metrics_families": len(list(self.registry._collector_to_names.keys())),
            "timestamp": datetime.now().isoformat()
        }


# Global metrics instance
prometheus_metrics = PrometheusMetrics()


# Convenience functions
def record_http_request(method: str, endpoint: str, status_code: int, 
                       duration: float, request_size: int = 0, response_size: int = 0):
    """Record HTTP request metrics."""
    prometheus_metrics.record_http_request(method, endpoint, status_code, duration, request_size, response_size)


def record_llm_call(provider: str, model: str, task_type: str,
                   input_tokens: int, output_tokens: int, cost: float, 
                   duration: float, success: bool = True):
    """Record LLM call metrics."""
    prometheus_metrics.record_llm_call(provider, model, task_type, input_tokens, output_tokens, cost, duration, success)


def record_run_start(mode: str, lane: str):
    """Record run start."""
    prometheus_metrics.record_run_start(mode, lane)


def record_run_complete(mode: str, lane: str, duration: float, success: bool = True):
    """Record run completion."""
    prometheus_metrics.record_run_complete(mode, lane, duration, success)


def get_prometheus_metrics() -> str:
    """Get metrics in Prometheus format."""
    return prometheus_metrics.get_metrics_output()


def start_prometheus_server(port: int = 8001):
    """Start Prometheus metrics server."""
    prometheus_metrics.port = port
    prometheus_metrics.start_server()


def initialize_prometheus_metrics(enabled: bool = True, port: int = 8001):
    """Initialize Prometheus metrics."""
    global prometheus_metrics
    prometheus_metrics = PrometheusMetrics(enabled=enabled, port=port)
    if enabled:
        prometheus_metrics.start_server()
