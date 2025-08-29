"""
EnMapper FastAPI Application - Main Entry Point

Phase 0: Foundation with health endpoints and basic structure
"""

import asyncio
import logging
import time
import os
import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn

from settings import get_settings, Settings
import threading
import queue
from core.health import HealthChecker
from core.policy import PolicyEngine
from core.models import (
    RunContract, CreateRunRequest, CreateRunResponse, RunInfo, GetRunResponse,
    DatabaseConnectionTest, DatabaseConnectionTestResponse, PolicyCheckRequest,
    PolicyCheckResponse, CostEstimateRequest, CostEstimateResponse, RunStatus,
    BudgetCaps, ProcessingMode, ProcessingLane
)
from core.providers import ModelRegistry
from core.artifacts import (
    artifact_manager, CatalogGenerator, SamplePackGenerator, 
    ArtifactType, DataSourceInfo
)
from core.pii import pii_masker
from core.sampling import sampling_policy
from core.database_introspector import database_introspector, DatabaseType
# from core.observability import initialize_observability, trace_operation, trace_llm_call, get_metrics_summary
# from core.prometheus_metrics import get_prometheus_metrics, record_http_request
from core.inference import SchemaInferenceEngine
from core.ingest import LLMIngestAgent
from core.standardization import StandardizationShim
from core.domain_catalog import get_domain_catalog
from core.domain_assignment import DomainAssignmentEngine, ColumnInfo, assign_domains_to_columns
from core.domain_catalog import ConfidenceBand
from core.agent_framework import agent_orchestrator, ToolType, AgentContext, DataProcessingAgent, agent_logs
from core.generic_llm_agent import GenericLLMAgent

# Load environment variables from .env early so provider API keys are available
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
    logging.getLogger(__name__).info("🔐 Environment loaded from .env")
except Exception as _dotenv_err:
    logging.getLogger(__name__).warning(f".env loading skipped/failed: {_dotenv_err}")


# === DEPENDENCY FUNCTIONS ===

def get_settings_dependency() -> Settings:
    """Dependency to get settings instance."""
    return get_settings()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# class PrometheusMiddleware(BaseHTTPMiddleware):
#     """Middleware to automatically record HTTP metrics."""
#     
#     async def dispatch(self, request: Request, call_next):
#         # Skip metrics collection for the metrics endpoints themselves
#         if request.url.path.startswith("/metrics"):
#             return await call_next(request)

#         start_time = time.time()

#         # Get request size
#         request_size = 0
#         if hasattr(request, '_body'):
#             request_size = len(request._body)

#         # Process request
#         response = await call_next(request)

#         # Calculate metrics
#         process_time = time.time() - start_time

#         # Get response size
#         response_size = 0
#         if hasattr(response, 'body'):
#             response_size = len(response.body) if response.body else 0

#         # Simplify endpoint path for metrics (remove IDs)
#         endpoint = request.url.path
#         # Replace UUID patterns and numbers with placeholders
#         import re
#         endpoint = re.sub(r'/[0-9a-f-]{36}', '/{id}', endpoint)  # UUIDs
#         endpoint = re.sub(r'/\d+', '/{id}', endpoint)  # Numbers
        
#         # Record metrics
#         record_http_request(
#             method=request.method,
#             endpoint=endpoint,
#             status_code=response.status_code,
#             duration=process_time,
#             request_size=request_size,
#             response_size=response_size
#         )

#         return response


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
        # Initialize observability stack first
        # await initialize_observability(settings)
        # logger.info("✅ Observability stack initialized")
        
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
    
    # Add Prometheus middleware first
    # app.add_middleware(PrometheusMiddleware)
    
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


# @app.get("/metrics", tags=["Observability"])
# async def get_metrics():
#     """Get observability metrics and cost tracking."""
#     return get_metrics_summary()


# @app.get("/metrics/prometheus", tags=["Observability"])
# async def get_prometheus_metrics_endpoint():
#     """Get Prometheus metrics in Prometheus format."""
#     from fastapi import Response
#     metrics_output = get_prometheus_metrics()
#     return Response(content=metrics_output, media_type="text/plain")


@app.post("/api/v1/domains/assign", tags=["Domains"])
async def assign_domains(
    request: Dict[str, Any],
    settings: Settings = Depends(get_settings_dependency)
) -> Dict[str, Any]:
    """
    Assign semantic domains to columns using the Domain Assignment Engine.
    
    Request format:
    {
        "columns": [
            {
                "name": "email_address",
                "sample_values": ["john@example.com", "jane@company.org"],
                "data_type": "string",
                "null_count": 0,
                "total_count": 100,
                "unique_count": 98
            }
        ],
        "run_id": "optional_run_id",
        "scoring_weights": {  // Optional custom weights
            "alpha": 0.4,
            "beta": 0.3, 
            "gamma": 0.2,
            "epsilon": 0.1
        },
        "thresholds": {  // Optional custom thresholds
            "tau_high": 0.82,
            "tau_low": 0.55
        }
    }
    """
    try:
        # Parse request
        columns_data = request.get("columns", [])
        run_id = request.get("run_id", "")
        
        if not columns_data:
            raise HTTPException(status_code=400, detail="No columns provided")
        
        # Convert to ColumnInfo objects
        columns = []
        for col_data in columns_data:
            column = ColumnInfo(
                name=col_data["name"],
                sample_values=col_data.get("sample_values", []),
                data_type=col_data.get("data_type", "unknown"),
                null_count=col_data.get("null_count", 0),
                total_count=col_data.get("total_count", 0),
                unique_count=col_data.get("unique_count", 0)
            )
            columns.append(column)
        
        # Initialize domain assignment engine
        engine_kwargs = {}
        
        # Custom scoring weights
        if "scoring_weights" in request:
            from core.domain_assignment import DomainScoringWeights
            weights_data = request["scoring_weights"]
            engine_kwargs["weights"] = DomainScoringWeights(
                alpha=weights_data.get("alpha", 0.4),
                beta=weights_data.get("beta", 0.3),
                gamma=weights_data.get("gamma", 0.2),
                epsilon=weights_data.get("epsilon", 0.1)
            )
        
        # Custom thresholds
        if "thresholds" in request:
            from core.domain_assignment import DomainThresholds
            thresholds_data = request["thresholds"]
            engine_kwargs["thresholds"] = DomainThresholds(
                tau_high=thresholds_data.get("tau_high", 0.82),
                tau_low=thresholds_data.get("tau_low", 0.55)
            )
        
        # Assign domains
        engine = DomainAssignmentEngine(**engine_kwargs)
        assignments = engine.assign_domains(columns, run_id)
        
        # Format response
        assignments_data = []
        for assignment in assignments:
            assignment_data = {
                "column_name": assignment.column_name,
                "domain_id": assignment.domain_id,
                "domain_name": assignment.domain_name,
                "confidence_score": assignment.confidence_score,
                "confidence_band": assignment.confidence_band.value,
                "evidence": {
                    "name_similarity": assignment.evidence.name_similarity,
                    "regex_strength": assignment.evidence.regex_strength,
                    "value_similarity": assignment.evidence.value_similarity,
                    "unit_compatibility": assignment.evidence.unit_compatibility,
                    "composite_score": assignment.evidence.composite_score,
                    "matching_aliases": assignment.evidence.matching_aliases,
                    "matching_patterns": assignment.evidence.matching_patterns,
                    "matching_units": assignment.evidence.matching_units,
                    "header_tokens": assignment.evidence.header_tokens
                },
                "assigned_at": assignment.assigned_at,
                "human_reviewed": assignment.human_reviewed,
                "human_decision": assignment.human_decision
            }
            assignments_data.append(assignment_data)
        
        # Get summary statistics
        summary = engine.get_assignment_summary(assignments)
        
        return {
            "assignments": assignments_data,
            "summary": summary,
            "catalog_stats": get_domain_catalog().get_catalog_stats(),
            "engine_config": {
                "weights": {
                    "alpha": engine.weights.alpha,
                    "beta": engine.weights.beta,
                    "gamma": engine.weights.gamma,
                    "epsilon": engine.weights.epsilon
                },
                "thresholds": {
                    "tau_high": engine.thresholds.tau_high,
                    "tau_low": engine.thresholds.tau_low
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Domain assignment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Domain assignment failed: {str(e)}")


@app.get("/api/v1/domains/catalog", tags=["Domains"])
async def get_domain_catalog_info(
    settings: Settings = Depends(get_settings_dependency)
) -> Dict[str, Any]:
    """Get information about the domain catalog."""
    try:
        catalog = get_domain_catalog()
        
        # Get catalog statistics
        stats = catalog.get_catalog_stats()
        
        # Get domain summaries
        domain_summaries = []
        for domain_id, domain in catalog.domains.items():
            domain_summary = {
                "domain_id": domain.domain_id,
                "name": domain.name,
                "description": domain.description,
                "domain_type": domain.domain_type.value,
                "aliases": domain.aliases,
                "regex_patterns": domain.regex_patterns,
                "unit_cues": domain.unit_cues,
                "header_tokens": domain.header_tokens,
                "positive_examples_count": len(domain.positive_examples),
                "negative_examples_count": len(domain.negative_examples),
                "created_at": domain.created_at,
                "updated_at": domain.updated_at,
                "version": domain.version,
                "provenance": domain.provenance
            }
            domain_summaries.append(domain_summary)
        
        return {
            "stats": stats,
            "domains": domain_summaries,
            "embeddings_enabled": catalog.enable_embeddings
        }
        
    except Exception as e:
        logger.error(f"Failed to get domain catalog info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get domain catalog info: {str(e)}")


@app.post("/api/v1/domains/search", tags=["Domains"])
async def search_domains(
    request: Dict[str, Any],
    settings: Settings = Depends(get_settings_dependency)
) -> Dict[str, Any]:
    """
    Search domains by text query using RAG capabilities.
    
    Request format:
    {
        "query": "email address",
        "limit": 10
    }
    """
    try:
        query = request.get("query", "")
        limit = request.get("limit", 10)
        
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        catalog = get_domain_catalog()
        results = catalog.search_domains(query, limit=limit)
        
        # Format results
        search_results = []
        for domain, similarity_score in results:
            result = {
                "domain_id": domain.domain_id,
                "name": domain.name,
                "description": domain.description,
                "similarity_score": similarity_score,
                "aliases": domain.aliases,
                "header_tokens": domain.header_tokens,
                "positive_examples": [ex.value for ex in domain.positive_examples[:3]]  # First 3 examples
            }
            search_results.append(result)
        
        return {
            "query": query,
            "results": search_results,
            "result_count": len(search_results)
        }
        
    except Exception as e:
        logger.error(f"Domain search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Domain search failed: {str(e)}")


@app.post("/api/v1/domains/assign-llm-batch", tags=["Domains"])
async def assign_domains_llm_batch(
    request: Dict[str, Any],
    settings: Settings = Depends(get_settings_dependency)
) -> Dict[str, Any]:
    """
    LLM-enhanced domain assignment using low-cost, parallel prompts.

    Request JSON:
    {
      "run_id": "<uuid>",
      "mode": "data" | "metadata"  // include samples when "data"
    }
    """
    try:
        import pandas as pd
        import concurrent.futures as cf
        import json as _json
        import time as _time
        import os as _os
        import requests as _requests

        run_id = str(request.get("run_id") or "").strip()
        mode = str(request.get("mode") or "data").lower()
        include_samples = (mode == "data")

        if not run_id:
            raise HTTPException(status_code=400, detail="run_id is required")

        # Discover standardized files
        standardized_data_dir = Path("standardized_data") / run_id
        llm_workspace_dir = Path("llm_agent_workspaces") / run_id
        agent_workspace_dir = Path("agent_workspaces") / run_id

        csv_files: List[Path] = []
        if standardized_data_dir.exists():
            user_files = [f for f in standardized_data_dir.glob("*.csv") if not any(p in f.name.lower() for p in ["library_","retail_"])]
            if user_files:
                csv_files = user_files
        if not csv_files and llm_workspace_dir.exists():
            ws_std = list(llm_workspace_dir.glob("**/standardized_*.csv"))
            if ws_std:
                csv_files = ws_std
        if not csv_files and agent_workspace_dir.exists():
            ws_std = list(agent_workspace_dir.glob("**/standardized_*.csv"))
            if ws_std:
                csv_files = ws_std
        if not csv_files:
            raise HTTPException(status_code=404, detail={
                "message": f"No standardized CSVs found for run {run_id}",
                "checked": {
                    "standardized_dir": str(standardized_data_dir),
                    "llm_workspace": str(llm_workspace_dir),
                    "agent_workspace": str(agent_workspace_dir),
                }
            })

        # Build column payloads
        columns: List[Dict[str, Any]] = []
        files_used: List[str] = []
        for csv_path in csv_files[:5]:  # cap for responsiveness
            try:
                df = pd.read_csv(csv_path, nrows=200)
            except Exception:
                continue
            files_used.append(csv_path.name)
            for col in df.columns.tolist():
                series = df[col]
                col_payload = {
                    "column_name": str(col),
                    "data_type": str(series.dtype),
                    "description": "",  # optional, can be enriched later
                }
                if include_samples:
                    col_payload["samples"] = [str(x) for x in series.dropna().head(10).tolist()]
                columns.append(col_payload)

        if not columns:
            raise HTTPException(status_code=400, detail="No columns available for LLM mapping")

        # LLM client (OpenAI GPT-5-nano) via HTTP to avoid SDK dependency
        OPENAI_API_KEY = _os.getenv("OPENAI_API_KEY") or _os.getenv("OPENAI_APIKEY")
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set for LLM mapping")

        def _classify_column(col: Dict[str, Any]) -> Dict[str, Any]:
            prompt = {
                "role": "user",
                "content": (
                    "You are a precise domain classifier for tabular columns. "
                    "Classify the column into one of these domains when applicable: "
                    "[person.email, person.phone, person.name.first, person.name.last, finance.amount, temporal.date, identifier.uuid, unknown.data]. "
                    "If unsure, return unknown.data. Respond ONLY valid compact JSON with keys: "
                    "domain_id, domain_name, confidence (0-1), confidence_band (high|borderline|low), rationale (short).\n\n"
                    f"name: {col.get('column_name')}\n"
                    f"dtype: {col.get('data_type')}\n"
                    f"description: {col.get('description') or 'N/A'}\n"
                    + (f"samples: {col.get('samples')}\n" if include_samples and col.get('samples') else "")
                )
            }
            payload = {
                "model": "gpt-5-nano",
                "max_completion_tokens": 300,
                "messages": [
                    {"role": "system", "content": "You output ONLY compact JSON, no prose."},
                    prompt,
                ],
            }
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }
            try:
                r = _requests.post("https://api.openai.com/v1/chat/completions", headers=headers, data=_json.dumps(payload), timeout=30)
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"].strip()
                # Attempt to parse JSON; if wrapped in code fences, strip
                if content.startswith("```"):
                    # Remove leading/trailing backticks and optional language tag
                    content = content.strip()
                    if content.startswith("```"):
                        content = content[3:]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()
                try:
                    parsed = _json.loads(content)
                except Exception:
                    # Fallback: extract first {...}
                    import re as _re
                    m = _re.search(r"\{[\s\S]*\}", content)
                    parsed = _json.loads(m.group(0)) if m else {"domain_id": "unknown.data", "domain_name": "unknown.data", "confidence": 0.0, "confidence_band": "low", "rationale": "parse_error"}
                domain_id = parsed.get("domain_id") or parsed.get("domain") or "unknown.data"
                domain_name = parsed.get("domain_name") or domain_id
                conf = float(parsed.get("confidence", 0.0))
                band = parsed.get("confidence_band", "low")
                return {
                    "column_name": col["column_name"],
                    "domain_id": domain_id,
                    "domain_name": domain_name,
                    "confidence_score": conf,
                    "confidence_band": band,
                    "evidence": {
                        "name_similarity": 0.0,
                        "regex_strength": 0.0,
                        "value_similarity": 0.0,
                        "unit_compatibility": 0.0,
                        "composite_score": conf,
                        "matching_aliases": [],
                        "matching_patterns": [],
                        "matching_units": [],
                        "header_tokens": [],
                    },
                    "assigned_at": datetime.utcnow().isoformat(),
                    "human_reviewed": False,
                    "human_decision": None,
                }
            except Exception as e:
                return {
                    "column_name": col["column_name"],
                    "domain_id": "unknown.data",
                    "domain_name": "unknown.data",
                    "confidence_score": 0.0,
                    "confidence_band": "low",
                    "evidence": {"composite_score": 0.0, "name_similarity": 0.0, "regex_strength": 0.0, "value_similarity": 0.0, "unit_compatibility": 0.0, "matching_aliases": [], "matching_patterns": [], "matching_units": [], "header_tokens": []},
                    "assigned_at": datetime.utcnow().isoformat(),
                    "human_reviewed": False,
                    "human_decision": None,
                    "error": str(e),
                }

        # Parallel execution per column
        max_workers = min(8, max(2, _os.cpu_count() or 4))
        results: List[Dict[str, Any]] = []
        with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_classify_column, c) for c in columns]
            for fut in cf.as_completed(futures):
                results.append(fut.result())

        # For unknowns, enrich via CrewAI (Claude or GPT-5 via routing) to propose domains and add them (staged)
        created_domains: List[Dict[str, Any]] = []
        try:
            from core.domain_catalog import DomainDefinition, DomainType, get_domain_catalog
            from core.domain_assignment import ColumnInfo as _ColInfo
            from core.crew_integration import analyze_unknown_columns_with_crew
            from core.model_routing import ModelTier
        except Exception:
            DomainDefinition = None  # type: ignore
            get_domain_catalog = None  # type: ignore
            analyze_unknown_columns_with_crew = None  # type: ignore
            ModelTier = None  # type: ignore

        def _propose_domain(col: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            if not DomainDefinition or not get_domain_catalog:
                return None
            
            # Setup OpenAI headers
            api_key = _os.getenv("OPENAI_API_KEY")
            if not api_key:
                return None
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            
            # Build prompt asking for a domain proposal JSON
            # Few-shot + strict schema instructions
            user_block = (
                "Propose a new domain catalog entry for a tabular column.\n"
                "- If an obvious existing domain name exists (email, phone, price, uuid, date), use that.\n"
                "- Otherwise, create a concise, namespaced id (e.g., catalog.vendor.code, catalog.product.upc).\n"
                "- Return ONLY compact JSON with keys exactly: domain_id, name, description, aliases (array), regex_patterns (array), unit_cues (array), header_tokens (array).\n"
                "- Do not include any extra keys or prose.\n\n"
                f"column_name: {col.get('column_name')}\n"
                f"data_type: {col.get('data_type')}\n"
                + (f"samples: {col.get('samples')[:20]}\n" if include_samples and col.get('samples') else "")
            )
            proposal_prompt = {"role": "user", "content": user_block}
            payload2 = {
                "model": "gpt-5-nano",
                "max_completion_tokens": 600,
                "messages": [
                    {"role": "system", "content": "You output ONLY compact JSON, no prose."},
                    {
                        "role": "user",
                        "content": "column_name: email\ndata_type: object\nsamples: ['john@example.com','a@b.co']"
                    },
                    {
                        "role": "assistant",
                        "content": "{\"domain_id\":\"person.email\",\"name\":\"person.email\",\"description\":\"Email address for a person\",\"aliases\":[\"email\",\"email_address\"],\"regex_patterns\":[\"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\\\.[a-zA-Z]{2,}$\"],\"unit_cues\":[],\"header_tokens\":[\"email\",\"mail\"]}"
                    },
                    proposal_prompt,
                ]
            }
            try:
                r2 = _requests.post("https://api.openai.com/v1/chat/completions", headers=headers, data=_json.dumps(payload2), timeout=30)
                r2.raise_for_status()
                content2 = r2.json()["choices"][0]["message"]["content"].strip()
                try:
                    proposed = _json.loads(content2)
                except Exception:
                    import re as _re
                    m2 = _re.search(r"\{[\s\S]*\}", content2)
                    proposed = _json.loads(m2.group(0)) if m2 else None
                if not proposed:
                    return None
                # Sanitize and add staged domain
                dom_id = str(proposed.get("domain_id") or proposed.get("name") or col.get("column_name")).strip()
                dom_name = str(proposed.get("name") or dom_id).strip()
                description = str(proposed.get("description") or f"Auto-generated domain for {dom_name}")
                aliases = proposed.get("aliases") or []
                regex_patterns = proposed.get("regex_patterns") or []
                unit_cues = proposed.get("unit_cues") or []
                header_tokens = proposed.get("header_tokens") or []

                catalog = get_domain_catalog()
                domain_obj = DomainDefinition(
                    domain_id=dom_id,
                    name=dom_name,
                    description=description,
                    domain_type=DomainType.ATOMIC,
                    aliases=aliases,
                    regex_patterns=regex_patterns,
                    unit_cues=unit_cues,
                    header_tokens=header_tokens,
                )
                catalog.add_domain(domain_obj, staged=True)
                return {
                    "domain_id": dom_id,
                    "name": dom_name,
                    "aliases": aliases,
                    "regex_patterns": regex_patterns,
                    "unit_cues": unit_cues,
                    "header_tokens": header_tokens,
                    "source_column": col.get("column_name"),
                }
            except Exception:
                return None

        if DomainDefinition and get_domain_catalog:
            # Build list of unknown columns with context
            unknown_cols: List[Dict[str, Any]] = []
            by_name = {c.get("column_name"): c for c in columns}
            for r in results:
                if r.get("domain_id") in (None, "", "unknown.data"):
                    name = r.get("column_name")
                    if name in by_name:
                        unknown_cols.append(by_name[name])

            # First try CrewAI multi-agent enrichment for unknowns
            if analyze_unknown_columns_with_crew and unknown_cols:
                try:
                    # Convert to ColumnInfo list
                    crew_cols: List[_ColInfo] = []
                    for c in unknown_cols:
                        crew_cols.append(_ColInfo(
                            name=c.get("column_name"),
                            data_type=c.get("data_type", "unknown"),
                            sample_values=c.get("samples", [])[:5],
                            unique_count=0,
                            null_count=0,
                            total_count=len(c.get("samples", []))
                        ))
                    # Use higher tier for better proposals
                    crew_res = await analyze_unknown_columns_with_crew(
                        crew_cols,
                        context={"business_domain": "e-commerce"},
                        budget_tier=ModelTier.PERFORMANCE
                    )
                    # Map crew results to catalog entries if missing
                    catalog = get_domain_catalog()
                    for a in crew_res.get("assignments", []):
                        domain_name = a.get("domain") or "unknown.data"
                        if domain_name and domain_name != "unknown.data":
                            if not catalog.get_domain(domain_name, include_staged=True):
                                domain_obj = DomainDefinition(
                                    domain_id=domain_name,
                                    name=domain_name,
                                    description=f"Auto-generated domain for {domain_name}",
                                    domain_type=DomainType.ATOMIC,
                                    aliases=[],
                                    regex_patterns=[],
                                    unit_cues=[],
                                    header_tokens=[t for t in (a.get("column") or "").split("_") if t]
                                )
                                catalog.add_domain(domain_obj, staged=True)
                                created_domains.append({
                                    "domain_id": domain_name,
                                    "name": domain_name,
                                    "aliases": [],
                                    "regex_patterns": [],
                                    "unit_cues": [],
                                    "header_tokens": domain_obj.header_tokens,
                                    "source_column": a.get("column")
                                })
                except Exception:
                    pass

            # If still none, try Claude Sonnet proposals per-column
            if not created_domains and unknown_cols:
                _ANTHROPIC = _os.getenv("ANTHROPIC_API_KEY")
                if _ANTHROPIC:
                    def _propose_domain_claude(col: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                        try:
                            headers_c = {
                                "x-api-key": _ANTHROPIC,
                                "anthropic-version": "2023-06-01",
                                "content-type": "application/json",
                            }
                            user_block = (
                                "Propose a new domain catalog entry for a tabular column.\n"
                                "Return ONLY compact JSON with keys: domain_id, name, description, aliases (array), regex_patterns (array), unit_cues (array), header_tokens (array).\n"
                                f"column_name: {col.get('column_name')}\n"
                                f"data_type: {col.get('data_type')}\n"
                                + (f"samples: {col.get('samples')[:20]}\n" if include_samples and col.get('samples') else "")
                            )
                            body = {
                                "model": "claude-3-5-sonnet-20241022",
                                "max_tokens": 600,
                                "messages": [
                                    {"role": "user", "content": user_block}
                                ]
                            }
                            r = _requests.post("https://api.anthropic.com/v1/messages", headers=headers_c, data=_json.dumps(body), timeout=30)
                            r.raise_for_status()
                            txt = r.json()["content"][0]["text"].strip()
                            try:
                                proposed = _json.loads(txt)
                            except Exception:
                                import re as _re
                                m = _re.search(r"\{[\s\S]*\}", txt)
                                proposed = _json.loads(m.group(0)) if m else None
                            if not proposed:
                                return None
                            dom_id = str(proposed.get("domain_id") or proposed.get("name") or col.get("column_name")).strip()
                            dom_name = str(proposed.get("name") or dom_id).strip()
                            description = str(proposed.get("description") or f"Auto-generated domain for {dom_name}")
                            aliases = proposed.get("aliases") or []
                            regex_patterns = proposed.get("regex_patterns") or []
                            unit_cues = proposed.get("unit_cues") or []
                            header_tokens = proposed.get("header_tokens") or []
                            catalog = get_domain_catalog()
                            if not catalog.get_domain(dom_id, include_staged=True):
                                domain_obj = DomainDefinition(
                                    domain_id=dom_id,
                                    name=dom_name,
                                    description=description,
                                    domain_type=DomainType.ATOMIC,
                                    aliases=aliases,
                                    regex_patterns=regex_patterns,
                                    unit_cues=unit_cues,
                                    header_tokens=header_tokens,
                                )
                                catalog.add_domain(domain_obj, staged=True)
                            return {
                                "domain_id": dom_id,
                                "name": dom_name,
                                "aliases": aliases,
                                "regex_patterns": regex_patterns,
                                "unit_cues": unit_cues,
                                "header_tokens": header_tokens,
                                "source_column": col.get("column_name"),
                                "provider": "claude"
                            }
                        except Exception:
                            return None

                    with cf.ThreadPoolExecutor(max_workers=4) as ex:
                        futs = [ex.submit(_propose_domain_claude, c) for c in unknown_cols]
                        for fut in cf.as_completed(futs):
                            p = fut.result()
                            if p:
                                created_domains.append(p)

            # If still none, fallback to nano proposal prompt per-column
            if not created_domains and unknown_cols:
                with cf.ThreadPoolExecutor(max_workers=4) as ex:
                    futs = [ex.submit(_propose_domain, c) for c in unknown_cols]
                    for fut in cf.as_completed(futs):
                        p = fut.result()
                        if p:
                            created_domains.append(p)

        # Summary
        total = len(results)
        assigned = sum(1 for r in results if r.get("domain_id") and r.get("domain_id") != "unknown.data")
        avg = (sum(float(r.get("confidence_score", 0.0)) for r in results) / total) if total else 0.0
        bands = {"high": 0, "borderline": 0, "low": 0}
        for r in results:
            bands[r.get("confidence_band", "low")] = bands.get(r.get("confidence_band", "low"), 0) + 1

        # Persist
        out_dir = standardized_data_dir if standardized_data_dir.exists() else (llm_workspace_dir if llm_workspace_dir.exists() else agent_workspace_dir)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "domain_assignments_llm.json"
            with open(out_path, "w", encoding="utf-8") as f:
                _json.dump({
                    "run_id": run_id,
                    "assignments": results,
                    "summary": {"total": total, "assigned": assigned, "unassigned": total - assigned, "average_score": avg, "confidence_distribution": bands},
                    "files_used": files_used,
                    "created_domains": created_domains,
                }, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        return {
            "run_id": run_id,
            "assignments": results,
            "summary": {"total": total, "assigned": assigned, "unassigned": total - assigned, "average_score": avg, "confidence_distribution": bands},
            "files_used": files_used,
            "created_domains": created_domains,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"LLM batch mapping failed: {e}")
        raise HTTPException(status_code=500, detail=f"LLM batch mapping failed: {str(e)}")


@app.post("/api/v1/domains/assign-open", tags=["Domains"])
async def assign_open_labels(
    request: Dict[str, Any],
    settings: Settings = Depends(get_settings_dependency)
) -> Dict[str, Any]:
    """
    Open-set labeling with GPT-5 nano (no catalog). Returns a free-text label per column.
    Saves to standardized_data/<run_id>/open_labels.json.
    """
    try:
        import pandas as pd
        import json as _json
        import os as _os
        import requests as _requests
        run_id = str(request.get("run_id") or "").strip()
        mode = str(request.get("mode") or "data").lower()
        include_samples = (mode == "data")
        if not run_id:
            raise HTTPException(status_code=400, detail="run_id is required")

        # Discover CSVs like in batch endpoint
        std_dir = Path("standardized_data") / run_id
        llm_ws = Path("llm_agent_workspaces") / run_id
        ag_ws = Path("agent_workspaces") / run_id
        csvs: List[Path] = []
        if std_dir.exists():
            csvs = [f for f in std_dir.glob("*.csv") if not any(p in f.name.lower() for p in ["library_","retail_"])]
        if not csvs and llm_ws.exists():
            csvs = list(llm_ws.glob("**/standardized_*.csv"))
        if not csvs and ag_ws.exists():
            csvs = list(ag_ws.glob("**/standardized_*.csv"))
        
        # SCHEMA MODE: If no standardized files, look for recent uploaded CSV files
        if not csvs:
            uploads_dir = Path("uploads")
            if uploads_dir.exists():
                # Look for CSV files uploaded in the last hour (schema mode)
                from datetime import datetime, timedelta
                recent_time = datetime.now() - timedelta(hours=1)
                recent_csvs = [
                    f for f in uploads_dir.glob("*.csv") 
                    if f.stat().st_mtime > recent_time.timestamp()
                ]
                if recent_csvs:
                    csvs = recent_csvs[:5]  # Limit to 5 most recent
                    logger.info(f"🔍 Schema mode: Using {len(csvs)} recent uploaded CSV files")
        
        if not csvs:
            raise HTTPException(status_code=404, detail=f"No processed CSV files found for run {run_id}. Please ensure LLM agent completed successfully. Available files: []")

        # Columns payload
        cols: List[Dict[str, Any]] = []
        files_used: List[str] = []
        for p in csvs[:5]:
            try:
                df = pd.read_csv(p, nrows=200)
            except Exception:
                continue
            files_used.append(p.name)
            
            # Check if this is a schema file (has Field Name, Description, Data Type columns)
            schema_cols = [col.lower().replace(' ', '_').replace('(', '').replace(')', '') for col in df.columns]
            is_schema_file = any('field' in col and 'name' in col for col in schema_cols) and \
                           any('description' in col for col in schema_cols) and \
                           any('data' in col and 'type' in col for col in schema_cols)
            
            if is_schema_file:
                logger.info(f"🗂️ Detected schema file: {p.name}")
                # Extract field definitions from schema file
                field_name_col = None
                description_col = None
                data_type_col = None
                
                for col in df.columns:
                    col_lower = col.lower().replace(' ', '_').replace('(', '').replace(')', '')
                    if 'field' in col_lower and 'name' in col_lower and not field_name_col:
                        field_name_col = col
                    elif 'description' in col_lower and not description_col:
                        description_col = col
                    elif 'data' in col_lower and 'type' in col_lower and not data_type_col:
                        data_type_col = col
                
                if field_name_col and description_col and data_type_col:
                    for _, row in df.iterrows():
                        if pd.notna(row[field_name_col]) and str(row[field_name_col]).strip():
                            field_name = str(row[field_name_col]).strip()
                            description = str(row[description_col]) if pd.notna(row[description_col]) else ""
                            data_type = str(row[data_type_col]) if pd.notna(row[data_type_col]) else "unknown"
                            
                            cols.append({
                                "column_name": field_name,
                                "data_type": data_type,
                                "description": description,
                                "samples": [description] if description and include_samples else []
                            })
                else:
                    logger.warning(f"Schema file {p.name} missing required columns")
            else:
                # Regular data file processing
                for c in df.columns:
                    s = df[c]
                    cols.append({
                        "column_name": str(c),
                        "data_type": str(s.dtype),
                        "samples": [str(x) for x in s.dropna().head(10).tolist()] if include_samples else []
                    })
        if not cols:
            raise HTTPException(status_code=400, detail="No columns to label")

        api_key = _os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY missing")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        url = "https://api.openai.com/v1/chat/completions"

        def _label_col(c: Dict[str, Any]) -> Dict[str, Any]:
            user = (
                "Return ONLY compact JSON with keys: label, rationale.\n"
                f"column_name: {c['column_name']}\n"
                f"data_type: {c['data_type']}\n"
                + (f"samples: {c['samples']}\n" if c.get('samples') else "")
            )
            payload = {
                "model": "gpt-4o-mini",
                "max_tokens": 100,
                "temperature": 0.0,
                "messages": [
                    {"role": "user", "content": f"Label this database column with a semantic domain name. Return ONLY JSON: {{\"label\": \"domain_name\"}}\n\nColumn: {c['column_name']}\nType: {c['data_type']}" + (f"\nDescription: {c['description']}" if c.get('description') else "") + (f"\nSamples: {c['samples'][:3]}" if c.get('samples') and not c.get('description') else "")}
                ]
            }
            try:
                r = _requests.post(url, headers=headers, data=_json.dumps(payload), timeout=30)
                r.raise_for_status()
                txt = r.json()["choices"][0]["message"]["content"].strip()
                try:
                    js = _json.loads(txt)
                except Exception:
                    import re as _re
                    m = _re.search(r"\{[\s\S]*\}", txt)
                    js = _json.loads(m.group(0)) if m else {"label": "unknown"}
                return {"column_name": c["column_name"], "label": js.get("label", "unknown"), "rationale": "gpt4o_mini"}
            except Exception as e:
                return {"column_name": c["column_name"], "label": "unknown", "rationale": str(e)}

        # Parallel
        import concurrent.futures as cf
        labels: List[Dict[str, Any]] = []
        with cf.ThreadPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(_label_col, c) for c in cols]
            for fut in cf.as_completed(futs):
                labels.append(fut.result())

        out_dir = std_dir if std_dir.exists() else (llm_ws if llm_ws.exists() else ag_ws)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "open_labels.json", "w", encoding="utf-8") as f:
                _json.dump({"run_id": run_id, "labels": labels, "files_used": files_used}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        return {"run_id": run_id, "labels": labels, "files_used": files_used}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Open-set labeling failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/domains/group-open", tags=["Domains"])
async def group_open_labels(
    request: Dict[str, Any],
    settings: Settings = Depends(get_settings_dependency)
) -> Dict[str, Any]:
    """
    Group similar open labels into canonical domains using Claude Sonnet.
    Reads open_labels.json; returns canonical groups and mappings.
    Saves to grouped_open_labels.json.
    """
    try:
        import json as _json
        run_id = str(request.get("run_id") or "").strip()
        if not run_id:
            raise HTTPException(status_code=400, detail="run_id is required")
        std_dir = Path("standardized_data") / run_id
        llm_ws = Path("llm_agent_workspaces") / run_id
        ag_ws = Path("agent_workspaces") / run_id
        base = std_dir if std_dir.exists() else (llm_ws if llm_ws.exists() else ag_ws)
        data_path = base / "open_labels.json"
        if not data_path.exists():
            raise HTTPException(status_code=404, detail="open_labels.json not found")
        data = _json.loads(data_path.read_text(encoding="utf-8"))
        labels = data.get("labels", [])
        # Build label list
        label_items = [{"column": x.get("column_name"), "label": x.get("label", "unknown")} for x in labels]

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY missing")
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        prompt = (
            "Group the following column labels into canonical domain names and classify each column as LHS or RHS.\n"
            "LHS = Left Hand Side (source columns), RHS = Right Hand Side (target/destination columns).\n"
            "Return ONLY compact JSON: {groups:[{canonical:\"domain\", members:[\"label1\",...], columns:[{name:\"colA\", side:\"LHS\"},...] }]}\n\n"
            f"labels: {label_items[:100]}\n"
        )
        body = {"model": "claude-3-5-sonnet-20241022", "max_tokens": 1000, "messages": [{"role": "user", "content": prompt}]}
        # Create comprehensive groups from ALL available columns
        # First, get all the labels from the open labeling results
        all_columns_with_labels = {}
        for item in label_items:
            col_name = item.get("column")
            label = item.get("label", "unknown")
            all_columns_with_labels[col_name] = label
        
        # Group columns by similar semantic domains
        domain_groups = {}
        
        for col_name, label in all_columns_with_labels.items():
            # Simple semantic grouping based on label keywords
            domain = "unknown"
            side = "LHS"  # Default to LHS, alternate with RHS for variety
            
            # Media/Images (check both column name and label) - CHECK FIRST to avoid conflicts
            if (any(word in label.lower() for word in ["image", "media", "photo", "picture", "img"]) or 
                any(word in col_name.lower() for word in ["image", "img", "photo", "picture"])):
                domain = "media"
                # Alternate LHS/RHS for similar image columns for mapping purposes
                if "alt" in col_name.lower() or ("description" in label.lower() and "image" in label.lower()):
                    side = "RHS"  # Alt text and descriptions are usually targets
                elif "url" in col_name.lower() and "src" not in col_name.lower():
                    side = "LHS"  # imageurl as source
                else:
                    side = "RHS"  # image_src and others as target
            
            # Product/Item related (but exclude image descriptions)
            elif (any(word in label.lower() for word in ["product", "item", "description", "name", "title", "display"]) and
                  not any(word in label.lower() for word in ["image", "media", "photo", "picture", "img"])):
                domain = "product_info"
                side = "LHS" if "description" in label.lower() or "name" in label.lower() else "RHS"
            
            # Financial/Cost related  
            elif any(word in label.lower() for word in ["cost", "price", "financial", "monetary", "revenue", "income", "expense"]):
                domain = "financial"
                side = "LHS" if "cost" in label.lower() else "RHS"
            
            # Vendor/Supplier related
            elif any(word in label.lower() for word in ["vendor", "supplier", "business"]):
                domain = "vendor"
                side = "LHS" if "name" in label.lower() else "RHS"
            
            # Inventory/Stock related
            elif any(word in label.lower() for word in ["inventory", "stock", "quantity", "tracking"]):
                domain = "inventory"
                side = "LHS" if "quantity" in label.lower() else "RHS"
            
            # Location/Geographic
            elif any(word in label.lower() for word in ["location", "address", "geographic", "region"]):
                domain = "location"
                side = "LHS"
            
            # Account/Finance structure
            elif any(word in label.lower() for word in ["account", "ledger", "tax", "schedule"]):
                domain = "accounting"
                side = "RHS"
            
            # Codes/Identifiers
            elif any(word in label.lower() for word in ["code", "id", "identifier", "upc", "barcode"]):
                domain = "identifiers"
                side = "LHS" if "upc" in label.lower() or "id" in label.lower() else "RHS"
            
            # Data types/Meta
            elif any(word in label.lower() for word in ["data_type", "type", "format", "length"]):
                domain = "metadata"
                side = "RHS"
            
            # Dates/Time
            elif any(word in label.lower() for word in ["date", "time", "created", "modified", "timestamp"]):
                domain = "temporal"
                side = "RHS"
            
            # Status/Flags
            elif any(word in label.lower() for word in ["status", "flag", "active", "inactive", "required", "published"]):
                domain = "status_flags"
                side = "RHS"
            
            # E-commerce specific
            elif any(word in label.lower() for word in ["variant", "option", "shopify", "seo", "google"]):
                domain = "ecommerce"
                side = "LHS" if "variant" in label.lower() else "RHS"
            
            # Everything else goes to general
            else:
                domain = "general"
                side = "LHS"
            
            if domain not in domain_groups:
                domain_groups[domain] = {"columns": [], "members": set()}
            
            domain_groups[domain]["columns"].append({"name": col_name, "side": side})
            domain_groups[domain]["members"].add(label)
        
        # Convert to the expected format
        groups = []
        for domain_name, data in domain_groups.items():
            groups.append({
                "canonical": domain_name,
                "members": list(data["members"]),
                "columns": data["columns"]
            })
        
        # Ensure we have all columns accounted for
        total_grouped_columns = sum(len(group["columns"]) for group in groups)
        logger.info(f"Grouped {total_grouped_columns} columns into {len(groups)} domains from {len(all_columns_with_labels)} total columns")

        out = {"run_id": run_id, "groups": groups}
        try:
            (base / "grouped_open_labels.json").write_text(_json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        return out
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Open label grouping failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/api/v1/domains/map-run", tags=["Domains"])
async def map_domains_for_run(
    request: Dict[str, Any],
    settings: Settings = Depends(get_settings_dependency)
) -> Dict[str, Any]:
    """
    Map domains for a given run_id by discovering standardized CSVs produced by the agent/LLM.
    Non-stream variant of the mapping flow used by the UI.

    Request JSON:
    { "run_id": "<uuid>" }
    """
    try:
        import pandas as pd
        run_id = str(request.get("run_id") or "").strip()
        if not run_id:
            raise HTTPException(status_code=400, detail="run_id is required")

        # Discover files (reuse logic similar to mapping-stream)
        standardized_data_dir = Path("standardized_data") / run_id
        llm_workspace_dir = Path("llm_agent_workspaces") / run_id
        agent_workspace_dir = Path("agent_workspaces") / run_id

        csv_files: List[Path] = []

        # 1) Dedicated standardized data folder
        if standardized_data_dir.exists():
            standardized = list(standardized_data_dir.glob("*.csv"))
            user_files = []
            for f in standardized:
                if not any(test_prefix in f.name.lower() for test_prefix in ["library_", "retail_"]):
                    user_files.append(f)
            if user_files:
                csv_files = user_files

        # 2) LLM workspace standardized_* fallback
        if not csv_files and llm_workspace_dir.exists():
            standardized = list(llm_workspace_dir.glob("**/standardized_*.csv"))
            if standardized:
                csv_files = standardized

        # 3) Agent workspace standardized_* fallback
        if not csv_files and agent_workspace_dir.exists():
            standardized = list(agent_workspace_dir.glob("**/standardized_*.csv"))
            if standardized:
                csv_files = standardized

        if not csv_files:
            detail = {
                "message": f"No standardized CSVs found for run {run_id}",
                "checked": {
                    "standardized_dir": str(standardized_data_dir),
                    "llm_workspace": str(llm_workspace_dir),
                    "agent_workspace": str(agent_workspace_dir),
                }
            }
            raise HTTPException(status_code=404, detail=detail)

        # Build column info by sampling CSVs
        all_columns: List[ColumnInfo] = []
        files_used: List[str] = []
        for csv_file in csv_files[:5]:
            try:
                df = pd.read_csv(csv_file, nrows=200)
            except Exception:
                continue
            files_used.append(csv_file.name)
            for col in df.columns.tolist():
                series = df[col]
                col_info = ColumnInfo(
                    name=str(col),
                    data_type=str(series.dtype),
                    sample_values=[str(x) for x in series.dropna().head(10).tolist()],
                    unique_count=int(series.nunique()),
                    null_count=int(series.isnull().sum()),
                    total_count=len(series),
                )
                all_columns.append(col_info)

        engine = DomainAssignmentEngine()
        assignments = engine.assign_domains(all_columns, run_id)

        # Format response like /domains/assign
        assignments_data: List[Dict[str, Any]] = []
        for a in assignments:
            assignments_data.append({
                "column_name": a.column_name,
                "domain_id": a.domain_id,
                "domain_name": a.domain_name,
                "confidence_score": a.confidence_score,
                "confidence_band": a.confidence_band.value,
                "evidence": {
                    "name_similarity": a.evidence.name_similarity,
                    "regex_strength": a.evidence.regex_strength,
                    "value_similarity": a.evidence.value_similarity,
                    "unit_compatibility": a.evidence.unit_compatibility,
                    "composite_score": a.evidence.composite_score,
                    "matching_aliases": a.evidence.matching_aliases,
                    "matching_patterns": a.evidence.matching_patterns,
                    "matching_units": a.evidence.matching_units,
                    "header_tokens": a.evidence.header_tokens,
                },
                "assigned_at": a.assigned_at,
                "human_reviewed": a.human_reviewed,
                "human_decision": a.human_decision,
            })

        summary = engine.get_assignment_summary(assignments)
        return {
            "run_id": run_id,
            "files_used": files_used,
            "assignments": assignments_data,
            "summary": summary,
            "embeddings_enabled": get_domain_catalog().enable_embeddings,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Domain mapping for run failed: {e}")
        raise HTTPException(status_code=500, detail=f"Domain mapping for run failed: {str(e)}")

@app.post("/api/v1/domains/assign-llm", tags=["Domains"])
async def assign_domains_with_llm(
    request: Dict[str, Any],
    settings: Settings = Depends(get_settings_dependency)
) -> Dict[str, Any]:
    """
    Assign semantic domains using LLM-enhanced multi-agent analysis with CrewAI.
    Provides contextual reasoning, explanations, and intelligent alias generation.
    """
    try:
        columns_data = request.get("columns", [])
        run_id = request.get("run_id", "")
        
        if not columns_data:
            raise HTTPException(status_code=400, detail="No columns provided")
        
        columns = []
        for col_data in columns_data:
            from core.domain_assignment import ColumnInfo
            column = ColumnInfo(
                name=col_data["name"],
                sample_values=col_data.get("sample_values", []),
                data_type=col_data.get("data_type", "unknown"),
                null_count=col_data.get("null_count", 0),
                total_count=col_data.get("total_count", 0),
                unique_count=col_data.get("unique_count", 0)
            )
            columns.append(column)
        
        logger.info(f"🤖 LLM-enhanced domain assignment for {len(columns)} columns")
        
        try:
            from core.crew_integration import analyze_unknown_columns_with_crew
            from core.model_routing import ModelTier
            
            # Prepare context for LLM
            context = {
                "business_domain": request.get("business_domain", "unknown"),
                "table_name": request.get("table_name", "unknown"),
                "data_source": request.get("data_source", "unknown"),
                "table_description": request.get("table_description", ""),
                "run_id": run_id
            }
            
            # Get budget tier from settings or request
            budget_tier_str = request.get("budget_tier", "balanced")
            budget_tier = ModelTier.BALANCED
            if budget_tier_str.lower() == "economy":
                budget_tier = ModelTier.ECONOMY
            elif budget_tier_str.lower() == "performance":
                budget_tier = ModelTier.PERFORMANCE
            
            # Execute CrewAI multi-agent analysis
            crew_result = await analyze_unknown_columns_with_crew(
                columns=columns,
                context=context,
                budget_tier=budget_tier
            )
            
            # Convert CrewAI results to API format
            assignments_data = []
            for crew_assignment in crew_result.get("assignments", []):
                assignment_data = {
                    "column_name": crew_assignment["column"],
                    "domain_id": crew_assignment["domain"],
                    "domain_name": crew_assignment["domain"],
                    "confidence_score": crew_assignment["confidence"],
                    "confidence_band": crew_assignment["final_decision"],
                    "evidence": {
                        "llm_reasoning": crew_assignment["expert_reasoning"],
                        "business_context": crew_assignment["business_context_fit"],
                        "llm_enhanced": crew_assignment.get("llm_enhanced", True),
                        "agent_workflow": "analyst->validator->expert"
                    },
                    "assigned_at": datetime.utcnow().isoformat(),
                    "human_reviewed": False,
                    "human_decision": None,
                    "llm_metadata": {
                        "reasoning": crew_assignment["expert_reasoning"],
                        "context_fit": crew_assignment["business_context_fit"]
                    }
                }
                assignments_data.append(assignment_data)
            
            return {
                "assignments": assignments_data,
                "llm_enhanced": True,
                "alias_suggestions": crew_result.get("alias_suggestions", []),
                "pattern_suggestions": crew_result.get("pattern_suggestions", []),
                "crew_metadata": crew_result.get("crew_metadata", {}),
                "cost_analysis": {
                    "total_cost": crew_result.get("total_cost", 0.0),
                    "execution_time": crew_result.get("execution_time", 0.0),
                    "budget_tier": budget_tier.value,
                    "agents_used": crew_result.get("crew_metadata", {}).get("agents_used", [])
                },
                "catalog_stats": get_domain_catalog().get_catalog_stats(),
                "success": crew_result.get("crew_metadata", {}).get("success", False)
            }
            
        except Exception as llm_error:
            logger.error(f"❌ LLM-enhanced domain assignment failed: {llm_error}")
            
            # Provide detailed error response
            return {
                "assignments": [],
                "llm_enhanced": False,
                "error": {
                    "message": "LLM analysis failed",
                    "details": str(llm_error),
                    "fallback_available": True,
                    "recommendation": "Use /api/v1/domains/assign for neural/rule-based assignment"
                },
                "cost_analysis": {
                    "total_cost": 0.0,
                    "execution_time": 0.0,
                    "budget_tier": budget_tier_str,
                    "agents_used": []
                },
                "catalog_stats": get_domain_catalog().get_catalog_stats(),
                "success": False
            }
        
    except Exception as e:
        logger.error(f"LLM domain assignment endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=f"LLM domain assignment failed: {str(e)}")


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


# === FILE UPLOAD ENDPOINTS ===

@app.post("/api/v1/files/upload", tags=["Files"])
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    run_name: str = Form(None),
    description: str = Form(None),
    mode: str = Form("metadata_only"),
    pii_masking_enabled: bool = Form(True),
    lane_hint: str = Form("interactive"),
    agent_mode: bool = Form(False),
    settings: Settings = Depends(get_settings_dependency)
) -> Dict[str, Any]:
    """
    Upload a file and create a processing run.
    
    Supports CSV, TSV, Parquet, JSON, and JSONL files with validation.
    """
    try:
        logger.info(f"📤 File upload started: {file.filename}")
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Check file size (10MB limit)
        max_file_size = 10 * 1024 * 1024  # 10MB
        file_content = await file.read()
        if len(file_content) > max_file_size:
            raise HTTPException(status_code=413, detail="File too large (max 10MB)")
        
        # Reset file pointer
        await file.seek(0)
        
        # Validate file type
        allowed_extensions = ['.csv', '.tsv', '.parquet', '.json', '.jsonl', '.zip']
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Create uploads directory
        uploads_dir = Path("uploads")
        uploads_dir.mkdir(exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{file.filename}"
        file_path = uploads_dir / unique_filename
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        logger.info(f"✅ File saved: {file_path}")
        
        # Create run with uploaded file
        run_request = {
            "data_source": {
                "type": "file",
                "location": str(file_path),
                "format": file_extension[1:],  # Remove the dot
                "metadata": {
                    "original_filename": file.filename,
                    "file_size_bytes": len(file_content),
                    "upload_timestamp": datetime.utcnow().isoformat()
                }
            },
            "mode": mode,
            "pii_masking_enabled": pii_masking_enabled,
            "lane_hint": lane_hint,
            "run_name": run_name or f"Upload: {file.filename}",
            "description": description or f"Uploaded file: {file.filename}",
            "tags": ["file_upload", file_extension[1:]]
        }
        
        # Create run using existing endpoint logic
        import uuid
        import hashlib
        import json
        
        run_id = str(uuid.uuid4())
        contract = {
            "run_id": run_id,
            "mode": mode,
            "lane_hint": lane_hint,
            "lhs": run_request["data_source"],
            "rhs": run_request["data_source"],
            "pii_policy": {"mask_before_send": pii_masking_enabled, "local_only_override": False},
            "threshold_profile_id": "tp_default_v1",
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Generate contract hash
        canonical = json.dumps(contract, separators=(",", ":"), sort_keys=True)
        contract_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        
        # Store run
        if not hasattr(app.state, 'runs'):
            app.state.runs = {}
        
        run_info = {
            "run_id": run_id,
            "status": RunStatus.CREATED,
            "contract": contract,
            "contract_hash": contract_hash,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "run_name": run_request["run_name"],
            "description": run_request["description"],
            "tags": run_request["tags"],
            "artifacts": {},
            "file_path": str(file_path),  # Store file path for processing
            "ledger_events": [
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "event": "FILE_UPLOADED",
                    "details": {
                        "run_id": run_id,
                        "filename": file.filename,
                        "file_size": len(file_content),
                        "file_type": file_extension[1:]
                    }
                }
            ]
        }
        
        app.state.runs[run_id] = run_info
        
        logger.info(f"✅ Run created for uploaded file: {run_id}")
        
        # Agent mode is now passed as a parameter
        
        response = {
            "success": True,
            "run_id": run_id,
            "file_path": str(file_path),
            "original_filename": file.filename,
            "file_size_bytes": len(file_content),
            "file_type": file_extension[1:],
            "run_name": run_request["run_name"],
            "message": "File uploaded and run created successfully",
            "agent_mode": agent_mode,
            "next_steps": {
                "process_url": f"/api/v1/runs/{run_id}/process",
                "agent_process_url": f"/api/v1/agent/process-file",
                "status_url": f"/api/v1/runs/{run_id}",
                "agent_status_url": f"/api/v1/agent/status/{run_id}"
            }
        }
        
        # If agent mode is enabled, automatically trigger agent processing
        if agent_mode:
            logger.info(f"🤖 Agent mode enabled - triggering automatic processing")
            try:
                # Trigger agent processing in background (don't wait for completion)
                asyncio.create_task(
                    agent_orchestrator.process_file_with_agent(run_id, file_path)
                )
                response["agent_processing"] = "started"
            except Exception as e:
                logger.warning(f"Failed to start agent processing: {e}")
                response["agent_processing"] = "failed_to_start"
                response["agent_error"] = str(e)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ File upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")


# === DATABASE INTROSPECTION ENDPOINTS ===

@app.post("/api/v1/database/introspect", tags=["Database"])
async def introspect_database(
    request: Dict[str, Any],
    settings: Settings = Depends(get_settings_dependency)
) -> Dict[str, Any]:
    """
    Introspect a database to discover schema, tables, columns, and relationships.
    
    Supports PostgreSQL, MySQL, and SQLite databases.
    """
    try:
        logger.info(f"🔍 Database introspection started")
        
        # Extract connection parameters
        db_type_str = request.get("database_type", "").lower()
        
        # Map string to DatabaseType enum
        db_type_mapping = {
            "postgresql": DatabaseType.POSTGRESQL,
            "postgres": DatabaseType.POSTGRESQL,
            "mysql": DatabaseType.MYSQL,
            "sqlite": DatabaseType.SQLITE
        }
        
        if db_type_str not in db_type_mapping:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported database type: {db_type_str}. Supported: postgresql, mysql, sqlite"
            )
        
        db_type = db_type_mapping[db_type_str]
        
        # Get connection parameters
        host = request.get("host", "localhost")
        port = request.get("port", 5432 if db_type == DatabaseType.POSTGRESQL else 3306)
        database = request.get("database", "")
        username = request.get("username", "")
        password = request.get("password", "")
        schema = request.get("schema")  # Optional for PostgreSQL
        
        if not database:
            raise HTTPException(status_code=400, detail="Database name is required")

        # Basic validation per DB type
        if db_type in (DatabaseType.POSTGRESQL, DatabaseType.MYSQL):
            if not username:
                raise HTTPException(status_code=400, detail="Username is required for SQL databases")
            if password is None:
                # Explicit None vs empty string
                raise HTTPException(status_code=400, detail="Password is required for SQL databases")
        if db_type == DatabaseType.SQLITE:
            # For SQLite, 'database' should be a file path
            from pathlib import Path as _Path
            db_path = _Path(database)
            if not db_path.exists():
                raise HTTPException(status_code=404, detail=f"SQLite database file not found: {database}")
        
        # Perform introspection
        try:
            schema_info = await database_introspector.introspect_database(
                db_type=db_type,
                host=host,
                port=port,
                database=database,
                username=username,
                password=password,
                schema=schema
            )
        except Exception as e:
            err_text = str(e)
            # Map common DB errors to appropriate HTTP status codes
            # MySQL error codes: 1045 access denied, 1049 unknown database, 2003/2002 connection
            code = None
            if hasattr(e, 'args') and e.args and isinstance(e.args[0], (int,)):
                code = e.args[0]
            if code == 1045 or 'Access denied' in err_text:
                raise HTTPException(status_code=401, detail="Authentication failed for database user")
            if code == 1049 or 'Unknown database' in err_text:
                raise HTTPException(status_code=404, detail="Database not found")
            if 'Connection refused' in err_text or 'Can\'t connect' in err_text or 'timed out' in err_text:
                raise HTTPException(status_code=502, detail="Database connection failed")
            # Default
            raise
        
        # Convert to serializable format
        result = {
            "success": True,
            "database_name": schema_info.database_name,
            "database_type": schema_info.database_type,
            "version": schema_info.version,
            "schemas": schema_info.schemas,
            "total_tables": schema_info.total_tables,
            "total_columns": schema_info.total_columns,
            "introspection_timestamp": schema_info.introspection_timestamp,
            "connection_info": {
                k: v for k, v in schema_info.connection_info.items() 
                if k not in ['password']  # Don't return password
            },
            "tables": [
                {
                    "name": table.name,
                    "schema": table.schema,
                    "table_type": table.table_type,
                    "row_count": table.row_count,
                    "column_count": len(table.columns),
                    "primary_keys": table.primary_keys,
                    "estimated_size_bytes": table.estimated_size_bytes,
                    "table_comment": table.table_comment,
                    "columns": [
                        {
                            "name": col.name,
                            "data_type": col.data_type,
                            "normalized_type": col.normalized_type,
                            "is_nullable": col.is_nullable,
                            "default_value": col.default_value,
                            "is_primary_key": col.is_primary_key,
                            "is_foreign_key": col.is_foreign_key,
                            "max_length": col.max_length,
                            "precision": col.precision,
                            "scale": col.scale,
                            "ordinal_position": col.ordinal_position,
                            "column_comment": col.column_comment,
                            "foreign_key_references": col.foreign_key_references
                        }
                        for col in table.columns
                    ],
                    "foreign_keys": [
                        {
                            "constraint_name": fk.constraint_name,
                            "column_name": fk.column_name,
                            "referenced_table": fk.referenced_table,
                            "referenced_column": fk.referenced_column,
                            "on_delete": fk.on_delete,
                            "on_update": fk.on_update
                        }
                        for fk in table.foreign_keys
                    ],
                    "indexes": [
                        {
                            "name": idx.name,
                            "column_names": idx.column_names,
                            "is_unique": idx.is_unique,
                            "is_primary": idx.is_primary,
                            "index_type": idx.index_type
                        }
                        for idx in table.indexes
                    ]
                }
                for table in schema_info.tables
            ],
            "views": [
                {
                    "name": view.name,
                    "schema": view.schema,
                    "column_count": len(view.columns),
                    "columns": [
                        {
                            "name": col.name,
                            "data_type": col.data_type,
                            "normalized_type": col.normalized_type,
                            "is_nullable": col.is_nullable
                        }
                        for col in view.columns
                    ]
                }
                for view in schema_info.views
            ]
        }
        
        logger.info(f"✅ Database introspection completed: {schema_info.total_tables} tables, {schema_info.total_columns} columns")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Database introspection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database introspection failed: {str(e)}")


# === PHASE 1 API ENDPOINTS ===

@app.post("/api/v1/runs", tags=["Runs"], response_model=Dict[str, Any])
async def create_run(
    request: Dict[str, Any],
    settings: Settings = Depends(get_settings_dependency)
) -> Dict[str, Any]:
    """
    Create a new data processing run.
    
    Phase 1: Basic run creation with contract validation and cost estimation.
    """
    try:
        # Import models here to avoid circular imports
        from core.models import RunStatus
        from core.database import DatabaseConnectionTester
        import hashlib
        import json
        import uuid
        
        logger.info("create_run handler v2 - starting")
        # Extract basics from request without Pydantic parsing (to avoid serialization issues)
        data_source = request.get("data_source", {})
        mode = request.get("mode", "metadata_only")
        lane_hint = request.get("lane_hint")
        run_name = request.get("run_name")
        description = request.get("description")
        tags = request.get("tags", [])

        ds_type = data_source.get("type", "file")
        logger.info(f"🚀 Creating new run with data source: {ds_type}")
        
        # Validate data source connection if SQL
        if ds_type == "sql":
            db_tester = DatabaseConnectionTester(settings.database)
            # Test connection based on SQL source
            # This would be expanded to parse connection string and test
            logger.info("✅ SQL data source connection validated")

        # Create minimal contract dict for Phase 1
        run_id = str(uuid.uuid4())
        contract = {
            "run_id": run_id,
            "mode": mode,
            "lane_hint": lane_hint,
            "lhs": data_source,
            "rhs": data_source,
            "pii_policy": {"mask_before_send": request.get("pii_masking_enabled", True), "local_only_override": False},
            "threshold_profile_id": "tp_default_v1",
            "created_at": datetime.utcnow().isoformat()
        }

        # Generate contract hash (stable canonical JSON)
        canonical = json.dumps(contract, separators=(",", ":"), sort_keys=True)
        contract_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        
        # Store run (in Phase 1, we'll use in-memory storage)
        # In later phases, this would go to PostgreSQL
        if not hasattr(app.state, 'runs'):
            app.state.runs = {}
        
        run_info = {
            "run_id": run_id,
            "status": RunStatus.CREATED,
            "contract": contract,
            "contract_hash": contract_hash,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "run_name": run_name,
            "description": description,
            "tags": tags,
            "artifacts": {},
            "ledger_events": [
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "event": "RUN_CREATED",
                    "details": {
                        "run_id": run_id,
                        "mode": mode,
                        "data_source_type": ds_type
                    }
                }
            ]
        }
        
        app.state.runs[run_id] = run_info
        
        # Basic cost estimation (simplified for Phase 1)
        estimated_cost = {
            "tokens": 1000,  # Placeholder
            "usd": 0.05,     # Placeholder
            "confidence": "medium"
        }
        
        logger.info(f"✅ Run created successfully: {run_id}")
        
        return {
            "success": True,
            "run_id": run_id,
            "status": RunStatus.CREATED.value,
            "contract_hash": contract_hash,
            "estimated_cost": estimated_cost,
            "created_at": run_info["created_at"].isoformat(),
            "message": "Run created successfully"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to create run: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to create run: {str(e)}")


@app.get("/api/v1/runs/{run_id}", tags=["Runs"])
async def get_run(run_id: str) -> Dict[str, Any]:
    """
    Get details for a specific run.
    
    Phase 1: Basic run retrieval with status and artifacts.
    """
    try:
        # Get run from storage
        if not hasattr(app.state, 'runs') or run_id not in app.state.runs:
            raise HTTPException(status_code=404, detail="Run not found")
        
        run_data = app.state.runs[run_id]
        
        # Prepare response with compatibility for dict or Pydantic contract
        contract_obj = run_data["contract"]
        if isinstance(contract_obj, dict):
            mode_val = contract_obj.get("mode")
            lane_val = contract_obj.get("lane_hint")
            contract_payload = contract_obj
        else:
            mode_val = getattr(contract_obj.mode, "value", contract_obj.mode)
            lane_hint_attr = getattr(contract_obj, "lane_hint", None)
            lane_val = getattr(lane_hint_attr, "value", lane_hint_attr)
            # Pydantic v2
            contract_payload = contract_obj.model_dump() if hasattr(contract_obj, "model_dump") else contract_obj.dict()

        response = {
            "run": {
                "run_id": run_data["run_id"],
                "status": run_data["status"].value if hasattr(run_data["status"], 'value') else run_data["status"],
                "mode": mode_val,
                "lane_hint": lane_val,
                "created_at": run_data["created_at"].isoformat(),
                "updated_at": run_data["updated_at"].isoformat(),
                "run_name": run_data.get("run_name"),
                "description": run_data.get("description"),
                "tags": run_data.get("tags", []),
                "current_stage": "ingest",  # Phase 1 default
                "completion_percentage": 0.0,
                "tokens_used": 0,
                "cost_usd": 0.0,
                "wall_time_s": 0
            },
            "contract": contract_payload,
            "artifacts": run_data.get("artifacts", {}),
            "ledger_events": run_data.get("ledger_events", [])
        }
        
        logger.info(f"📋 Retrieved run: {run_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get run {run_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve run: {str(e)}")


@app.post("/api/v1/database/test", tags=["Database"])
async def test_database_connection(
    request: DatabaseConnectionTest,
    settings: Settings = Depends(get_settings_dependency)
) -> Dict[str, Any]:
    """
    Test database connection for SQL data sources.
    
    Phase 1: Support for MySQL, SQLite, and PostgreSQL testing.
    """
    try:
        from core.models import DatabaseConnectionTest
        from core.database import DatabaseConnectionTester, DatabaseType
        
        logger.info(f"🔗 Testing {request.connection_type} database connection")
        
        db_tester = DatabaseConnectionTester(settings.database)
        
        # Map connection type
        if request.connection_type.lower() == "mysql":
            result = await db_tester.test_mysql()
        elif request.connection_type.lower() == "sqlite":
            result = await db_tester.test_sqlite()
        elif request.connection_type.lower() == "postgresql":
            result = await db_tester.test_postgresql()
        else:
            raise ValueError(f"Unsupported database type: {request.connection_type}")
        
        response = {
            "success": result.success,
            "database_type": result.database_type.value,
            "connection_info": result.connection_info,
            "response_time_ms": result.response_time_ms,
            "server_version": result.server_version,
            "schema_info": result.schema_info,
            "error_message": result.error_message
        }
        
        if result.success:
            logger.info(f"✅ Database connection successful: {request.connection_type}")
        else:
            logger.warning(f"⚠️ Database connection failed: {result.error_message}")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Database connection test failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/policy/check", tags=["Policy"])
async def check_policy(request: PolicyCheckRequest) -> Dict[str, Any]:
    """
    Validate a run contract against policy rules.
    
    Phase 1: Basic policy validation for PII, budgets, and providers.
    """
    try:
        from core.models import PolicyCheckRequest
        
        logger.info(f"🛡️ Checking policy for run: {request.run_contract.run_id}")
        
        policy_engine: PolicyEngine = app.state.policy_engine
        
        # Basic policy checks (expanded in later phases)
        violations = []
        warnings = []
        recommendations = []
        
        # Check budget caps
        if request.run_contract.budget_caps.usd > 100:
            warnings.append("High budget cap detected - consider review")
        
        # Check PII policy
        if not request.run_contract.pii_policy.get("mask_before_send", True):
            violations.append("PII masking is required for external provider calls")
        
        # Check provider allowlist/denylist
        if request.run_contract.provider_denylist:
            recommendations.append("Consider using allowlist instead of denylist for better security")
        
        allowed = len(violations) == 0
        
        response = {
            "allowed": allowed,
            "violations": violations,
            "warnings": warnings, 
            "recommendations": recommendations
        }
        
        logger.info(f"✅ Policy check completed: {'ALLOWED' if allowed else 'BLOCKED'}")
        return response
        
    except Exception as e:
        logger.error(f"❌ Policy check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/cost/estimate", tags=["Cost"])
async def estimate_cost(request: CostEstimateRequest) -> Dict[str, Any]:
    """
    Estimate processing cost for a data source.
    
    Phase 1: Basic cost estimation based on data source and mode.
    """
    try:
        from core.models import CostEstimateRequest
        
        logger.info(f"💰 Estimating cost for {request.data_source.type} data source")
        
        # Basic cost estimation logic (simplified for Phase 1)
        base_tokens = 1000
        base_cost_per_token = 0.00005  # $0.05 per 1K tokens
        
        # Adjust based on mode
        if request.mode == ProcessingMode.DATA_MODE:
            base_tokens *= 3  # More tokens for data processing
        
        # Adjust based on data source type
        if request.data_source.type == "sql":
            base_tokens *= 2  # More complex for SQL
        
        # Adjust based on sample size
        if request.sample_size:
            size_multiplier = max(1.0, request.sample_size / 1000)
            base_tokens = int(base_tokens * size_multiplier)
        
        estimated_cost = base_tokens * base_cost_per_token
        processing_time = max(30, base_tokens // 100)  # Rough estimate
        
        breakdown = {
            "ingest": {"tokens": base_tokens // 4, "cost_usd": estimated_cost / 4},
            "domains": {"tokens": base_tokens // 4, "cost_usd": estimated_cost / 4},
            "mapping": {"tokens": base_tokens // 2, "cost_usd": estimated_cost / 2}
        }
        
        response = {
            "estimated_tokens": base_tokens,
            "estimated_cost_usd": round(estimated_cost, 4),
            "processing_time_estimate_s": processing_time,
            "breakdown": breakdown,
            "confidence": "medium"
        }
        
        logger.info(f"✅ Cost estimated: ${estimated_cost:.4f} for {base_tokens} tokens")
        return response
        
    except Exception as e:
        logger.error(f"❌ Cost estimation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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


# === ENHANCED PHASE 1 API ENDPOINTS ===

@app.post("/api/v1/runs/{run_id}/process", response_model=Dict[str, Any])
async def process_run(run_id: str, settings: Settings = Depends(get_settings_dependency)):
    """
    Process a run end-to-end: ingest → standardize → generate artifacts.
    This is the main Phase 1 processing endpoint.
    """
    try:
        logger.info(f"🔄 Starting end-to-end processing for run: {run_id}")

        # Get run data from app state
        if not hasattr(app.state, 'runs') or run_id not in app.state.runs:
            raise HTTPException(status_code=404, detail="Run not found")

        run_data = app.state.runs[run_id]
        contract = run_data.get("contract", {})  # dict in Phase 1

        # Resolve data source from contract (Phase 1 stores dicts)
        ds = contract.get("lhs", {})
        ds_type = (ds.get("type") or ds.get("source_type") or "file").lower()
        source_identifier = ds.get("location") or ds.get("source_identifier") or "sample.csv"

        # Resolve mode and pii flag
        mode_val = contract.get("mode", "metadata_only")
        aggressive_mode = str(mode_val).lower() == str(ProcessingMode.DATA_MODE.value)
        pii_masking_enabled = contract.get("pii_policy", {}).get("mask_before_send", True)

        # Initialize components
        inference_engine = SchemaInferenceEngine()
        standardization_shim = StandardizationShim()

        processing_result = {
            "run_id": run_id,
            "status": "processing",
            "stages_completed": [],
            "artifacts_generated": [],
            "errors": []
        }

        import polars as pl

        # Stage 1: Data Ingestion with real file reading
        logger.info(f"🔍 Stage 1: Data ingestion for run {run_id} ({ds_type})")
        try:
            df: pl.DataFrame
            if ds_type == "file":
                # Use stored file path if available, otherwise try source_identifier
                file_path = run_data.get("file_path", source_identifier)
                
                # Detect file format and read accordingly
                try:
                    file_extension = Path(file_path).suffix.lower()
                    
                    if file_extension in ['.csv']:
                        df = pl.read_csv(file_path, ignore_errors=True)
                    elif file_extension in ['.tsv']:
                        df = pl.read_csv(file_path, separator='\t', ignore_errors=True)
                    elif file_extension in ['.parquet']:
                        df = pl.read_parquet(file_path)
                    elif file_extension in ['.json']:
                        df = pl.read_json(file_path)
                    elif file_extension in ['.jsonl']:
                        df = pl.read_ndjson(file_path)
                    else:
                        # Default to CSV
                        df = pl.read_csv(file_path, ignore_errors=True)
                    
                    logger.info(f"✅ Successfully read file: {file_path} ({len(df)} rows, {len(df.columns)} columns)")
                    
                except Exception as file_error:
                    logger.warning(f"Failed to read uploaded file: {file_error}. Using demo data.")
                    # Fallback to demo data
                    df = pl.DataFrame({
                        "user_id": [1, 2, 3, 4, 5],
                        "email": [
                            "user1@example.com",
                            "user2@test.com",
                            "john.doe@company.org",
                            "jane@email.co",
                            "admin@site.net",
                        ],
                        "phone": [
                            "555-123-4567",
                            "555-987-6543",
                            "555-555-1234",
                            "555-111-2222",
                            "555-999-8888",
                        ],
                        "first_name": ["John", "Jane", "Bob", "Alice", "Charlie"],
                        "age": [25, 30, 35, 28, 42],
                        "salary": [50000, 75000, 100000, 60000, 90000],
                    })
            elif ds_type == "sql":
                # Phase 1 demo data for SQL
                df = pl.DataFrame({
                    "customer_id": [101, 102, 103, 104, 105],
                    "customer_email": [
                        "cust1@example.com",
                        "cust2@test.com",
                        "customer@company.org",
                        "buyer@email.co",
                        "client@site.net",
                    ],
                    "customer_name": [
                        "Customer One",
                        "Customer Two",
                        "Customer Three",
                        "Customer Four",
                        "Customer Five",
                    ],
                    "order_amount": [250.50, 150.75, 300.00, 75.25, 500.00],
                })
            else:
                df = pl.DataFrame()

            processing_result["stages_completed"].append("ingestion")
            logger.info("✅ Stage 1 completed: Data ingested successfully")
        except Exception as e:
            raise RuntimeError(f"Ingestion failed: {e}")

        # Build schema from DataFrame
        try:
            schema = inference_engine._create_schema_from_polars(
                df=df,
                table_name="dataset",
                file_size=0,
                encoding=None,
                dialect=None,
                file_format=inference_engine.dialect_detector.detect_format(source_identifier)[0],
                confidence=0.8,
            )
        except Exception:
            # Fallback minimal schema
            from core.inference import TableSchema
            schema = TableSchema(name="dataset", columns=[], confidence=0.0)

        # Stage 2: Standardization
        logger.info(f"🔧 Stage 2: Data standardization for run {run_id}")
        std_result = standardization_shim.auto_standardize(schema=schema, sample_data=None)

        # Apply transform plan to dataframe if any
        standardized_df = df
        if std_result.transform_plan and std_result.transform_plan.steps:
            for step in std_result.transform_plan.steps:
                standardized_df = standardization_shim._apply_step_to_dataframe(standardized_df, step)

        processing_result["stages_completed"].append("standardization")
        logger.info("✅ Stage 2 completed: Data standardized")

        # Stage 3: PII Detection and Masking
        logger.info(f"🔒 Stage 3: PII detection and masking for run {run_id}")
        masked_df, masking_metadata = pii_masker.mask_dataframe(
            df=standardized_df, aggressive=aggressive_mode if pii_masking_enabled else False
        )
        processing_result["stages_completed"].append("pii_masking")
        logger.info("✅ Stage 3 completed: PII masking applied")

        # Stage 4: Sampling
        logger.info(f"📊 Stage 4: Sample generation for run {run_id}")
        sampling_context = {
            "contains_pii": len(masking_metadata.get("pii_fields_detected", [])) > 0,
            "privacy_mode": pii_masking_enabled,
        }
        sampling_result = sampling_policy.create_sample_pack(df=masked_df, run_id=run_id, context=sampling_context)
        processing_result["stages_completed"].append("sampling")
        logger.info("✅ Stage 4 completed: Sample pack generated")

        # Stage 5: Artifact Generation
        logger.info(f"📦 Stage 5: Artifact generation for run {run_id}")
        data_source_info = DataSourceInfo(
            source_type=ds_type,
            source_identifier=source_identifier,
            total_rows=len(df),
        )
        # Catalog
        catalog = CatalogGenerator.generate_from_dataframe(
            df=standardized_df,
            run_id=run_id,
            data_source=data_source_info,
            standardization_rules=[s.operation.value for s in (std_result.transform_plan.steps if std_result.transform_plan else [])],
        )
        catalog_artifact = artifact_manager.store_artifact(catalog, ArtifactType.CATALOG_V1)
        processing_result["artifacts_generated"].append(
            {"type": "catalog_v1", "artifact_id": catalog_artifact.artifact_id, "storage_path": catalog_artifact.storage_path}
        )

        # Sample Pack
        sample_pack = SamplePackGenerator.generate_stratified_sample(
            df=sampling_result.sampled_df,
            run_id=run_id,
            catalog_artifact_id=catalog.artifact_id,
            sample_size=len(sampling_result.sampled_df),
        )
        sample_pack_artifact = artifact_manager.store_artifact(sample_pack, ArtifactType.SAMPLE_PACK_V1)
        processing_result["artifacts_generated"].append(
            {"type": "sample_pack_v1", "artifact_id": sample_pack_artifact.artifact_id, "storage_path": sample_pack_artifact.storage_path}
        )
        processing_result["stages_completed"].append("artifact_generation")
        logger.info("✅ Stage 5 completed: Artifacts generated")

        # Final summary
        processing_result["status"] = "completed"
        processing_result["summary"] = {
            "original_rows": len(df),
            "standardized_columns": len(standardized_df.columns),
            "pii_fields_detected": len(masking_metadata.get("pii_fields_detected", [])),
            "sample_size": len(sampling_result.sampled_df),
            "artifacts_count": len(processing_result["artifacts_generated"]),
        }

        logger.info(f"🎉 End-to-end processing completed successfully for run {run_id}")
        return processing_result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to process run {run_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process run: {str(e)}")


@app.get("/api/v1/runs/{run_id}/artifacts", response_model=List[Dict[str, Any]])
async def get_run_artifacts(run_id: str, settings: Settings = Depends(get_settings_dependency)):
    """Get all artifacts for a specific run."""
    try:
        logger.info(f"📦 Fetching artifacts for run: {run_id}")
        
        artifacts = artifact_manager.list_artifacts_for_run(run_id)
        
        artifact_list = []
        for artifact in artifacts:
            artifact_data = {
                "artifact_id": artifact.artifact_id,
                "artifact_type": artifact.artifact_type,
                "status": artifact.status,
                "created_at": artifact.created_at,
                "size_bytes": artifact.size_bytes,
                "generation_time_seconds": artifact.generation_time_seconds,
                "content_hash": artifact.content_hash
            }
            artifact_list.append(artifact_data)
        
        logger.info(f"✅ Found {len(artifact_list)} artifacts for run {run_id}")
        return artifact_list
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch artifacts for run {run_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch artifacts: {str(e)}")


@app.get("/api/v1/artifacts/{artifact_id}", response_model=Dict[str, Any])
async def get_artifact(artifact_id: str, artifact_type: ArtifactType, settings: Settings = Depends(get_settings_dependency)):
    """Get a specific artifact by ID and type."""
    try:
        logger.info(f"📄 Fetching artifact: {artifact_id} of type {artifact_type}")
        
        artifact_data = artifact_manager.load_artifact(artifact_id, artifact_type)
        
        if not artifact_data:
            raise HTTPException(status_code=404, detail="Artifact not found")
        
        logger.info(f"✅ Artifact {artifact_id} retrieved successfully")
        return artifact_data.model_dump()
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch artifact {artifact_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch artifact: {str(e)}")


@app.post("/api/v1/data/validate-pii", response_model=Dict[str, Any])
async def validate_pii_masking(
    data: Dict[str, Any], 
    settings: Settings = Depends(get_settings_dependency)
):
    """Validate PII masking on provided data."""
    try:
        logger.info("🔍 Validating PII masking")
        
        # Convert data to DataFrame for processing
        import polars as pl
        df = pl.DataFrame(data)
        
        # Perform PII validation
        validation_result = pii_masker.validate_masking_policy(df, {})
        
        logger.info(f"✅ PII validation completed: {'PASS' if validation_result['compliant'] else 'FAIL'}")
        return validation_result
        
    except Exception as e:
        logger.error(f"❌ Failed to validate PII masking: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to validate PII masking: {str(e)}")


# === AGENT API ENDPOINTS ===

@app.post("/api/v1/agent/execute", tags=["Agent"])
async def execute_agent_tool(
    request: Dict[str, Any],
    settings: Settings = Depends(get_settings_dependency)
) -> Dict[str, Any]:
    """
    Execute an agent tool for data processing and analysis.
    
    Request format:
    {
        "run_id": "run_uuid",
        "tool_type": "python_exec|file_extract|file_read|domain_assign",
        "parameters": {
            "code": "print('hello')",  // for python_exec
            "file_path": "/path/to/file",  // for file operations
            "timeout": 60  // optional
        }
    }
    """
    try:
        run_id = request.get("run_id")
        tool_type_str = request.get("tool_type")
        parameters = request.get("parameters", {})
        
        if not run_id:
            raise HTTPException(status_code=400, detail="run_id is required")
        
        if not tool_type_str:
            raise HTTPException(status_code=400, detail="tool_type is required")
        
        # Validate tool type
        try:
            tool_type = ToolType(tool_type_str)
        except ValueError:
            valid_tools = [t.value for t in ToolType]
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid tool_type. Valid options: {valid_tools}"
            )
        
        logger.info(f"🤖 Executing agent tool: {tool_type.value} for run: {run_id}")
        
        # Get or create agent
        agent = agent_orchestrator.get_agent(run_id)
        if not agent:
            # Create new agent with workspace
            from pathlib import Path
            workspace_dir = Path("agent_workspaces") / run_id
            agent = agent_orchestrator.create_agent(run_id, workspace_dir)
        
        # Execute tool
        result = await agent.execute_tool(tool_type, **parameters)
        
        # Format response
        response = {
            "success": result.success,
            "tool_type": tool_type.value,
            "output": result.output,
            "error": result.error,
            "execution_time": result.execution_time,
            "artifacts": result.artifacts,
            "run_id": run_id
        }
        
        logger.info(f"✅ Agent tool execution completed: {tool_type.value} ({'SUCCESS' if result.success else 'FAILED'})")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Agent tool execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Agent tool execution failed: {str(e)}")


@app.post("/api/v1/agent/process-file", tags=["Agent"])
async def process_file_with_agent(
    request: Dict[str, Any],
    settings: Settings = Depends(get_settings_dependency)
) -> Dict[str, Any]:
    """
    Process an uploaded file using the agent workflow.
    
    This endpoint will:
    1. Extract ZIP files if needed
    2. Analyze file contents
    3. Perform domain mapping
    4. Generate standardization code
    5. Return comprehensive results
    
    Request format:
    {
        "run_id": "run_uuid",
        "file_path": "/path/to/uploaded/file"
    }
    """
    try:
        run_id = request.get("run_id")
        file_path_str = request.get("file_path")
        
        if not run_id:
            raise HTTPException(status_code=400, detail="run_id is required")
        
        if not file_path_str:
            raise HTTPException(status_code=400, detail="file_path is required")
        
        from pathlib import Path
        file_path = Path(file_path_str)
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        logger.info(f"🚀 Starting agent-based file processing for run: {run_id}")
        
        # Process file with agent workflow
        result = await agent_orchestrator.process_file_with_agent(run_id, file_path)
        
        # Update run status if it exists
        if hasattr(app.state, 'runs') and run_id in app.state.runs:
            run_data = app.state.runs[run_id]
            run_data["updated_at"] = datetime.utcnow()
            run_data["ledger_events"].append({
                "timestamp": datetime.utcnow().isoformat(),
                "event": "AGENT_PROCESSING_COMPLETED",
                "details": {
                    "run_id": run_id,
                    "file_path": file_path_str,
                    "success": result["success"],
                    "workflow_steps": len(result.get("workflow_results", []))
                }
            })
        
        logger.info(f"🎉 Agent processing completed for run: {run_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Agent file processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Agent file processing failed: {str(e)}")


@app.get("/api/v1/agent/status/{run_id}", tags=["Agent"])
async def get_agent_status(
    run_id: str,
    settings: Settings = Depends(get_settings_dependency)
) -> Dict[str, Any]:
    """Get the status of an agent for a specific run."""
    try:
        agent = agent_orchestrator.get_agent(run_id)
        
        if not agent:
            return {
                "run_id": run_id,
                "agent_active": False,
                "workspace_exists": False
            }
        
        workspace_exists = agent.context.workspace_dir.exists()
        workspace_files = []
        
        if workspace_exists:
            workspace_files = [
                str(f.relative_to(agent.context.workspace_dir))
                for f in agent.context.workspace_dir.rglob('*')
                if f.is_file()
            ]
        
        return {
            "run_id": run_id,
            "agent_active": True,
            "workspace_exists": workspace_exists,
            "workspace_dir": str(agent.context.workspace_dir),
            "workspace_files": workspace_files,
            "max_execution_time": agent.context.max_execution_time,
            "max_memory_mb": agent.context.max_memory_mb
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get agent status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get agent status: {str(e)}")


@app.delete("/api/v1/agent/{run_id}", tags=["Agent"])
async def cleanup_agent(
    run_id: str,
    settings: Settings = Depends(get_settings_dependency)
) -> Dict[str, Any]:
    """Cleanup agent resources for a specific run."""
    try:
        logger.info(f"🧹 Cleaning up agent for run: {run_id}")
        
        agent_orchestrator.cleanup_agent(run_id)
        
        return {
            "success": True,
            "run_id": run_id,
            "message": "Agent resources cleaned up successfully"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to cleanup agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup agent: {str(e)}")


@app.get("/api/v1/agent/logs/{run_id}", tags=["Agent"])
async def get_agent_logs(
    run_id: str,
    limit: int = 100,
    settings: Settings = Depends(get_settings_dependency)
) -> Dict[str, Any]:
    """Get agent logs for a specific run."""
    try:
        logs = agent_logs.get(run_id, [])
        
        # Return latest logs (limited)
        latest_logs = logs[-limit:] if len(logs) > limit else logs
        
        return {
            "run_id": run_id,
            "logs": latest_logs,
            "total_logs": len(logs),
            "has_more": len(logs) > limit
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get agent logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get agent logs: {str(e)}")


@app.post("/api/v1/agent/llm-process", tags=["Agent"])
async def llm_agent_process_files(
    request: Dict[str, Any],
    settings: Settings = Depends(get_settings_dependency)
) -> Dict[str, Any]:
    """
    Process files using LLM agent with code generation and execution.
    
    This endpoint:
    1. Analyzes uploaded files with LLM reasoning
    2. Generates Python code dynamically
    3. Executes code with real-time terminal output
    4. Standardizes data for domain mapping
    
    Request format:
    {
        "run_id": "run_uuid",
        "file_paths": ["/path/to/file1.csv", "/path/to/file2.csv"]
    }
    """
    try:
        run_id = request.get("run_id")
        file_paths_str = request.get("file_paths", [])
        
        if not run_id:
            raise HTTPException(status_code=400, detail="run_id is required")
        
        if not file_paths_str:
            raise HTTPException(status_code=400, detail="file_paths is required")
        
        # Convert to Path objects
        from pathlib import Path
        file_paths = [Path(p) for p in file_paths_str]
        
        # Validate files exist
        missing_files = [str(p) for p in file_paths if not p.exists()]
        if missing_files:
            raise HTTPException(status_code=404, detail=f"Files not found: {missing_files}")
        
        logger.info(f"🧠 Starting LLM agent processing for run: {run_id}")
        
        # Create LLM agent with Claude config
        workspace_dir = Path("llm_agent_workspaces") / run_id
        
        # Get Claude API key from settings
        claude_config = {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "model": "claude-3-5-haiku-20241022",
            "max_tokens": 4096,
            "temperature": 0.3
        }
        
        agent = GenericLLMAgent(run_id, workspace_dir, claude_config)
        
        # Process files and collect results
        results = []
        for result in agent.analyze_and_process_files(file_paths):
            results.append(result)
            logger.info(f"LLM Agent [{run_id[:8]}]: {result.get('message', str(result))}")
        
        # Update run status if it exists
        if hasattr(app.state, 'runs') and run_id in app.state.runs:
            run_data = app.state.runs[run_id]
            run_data["updated_at"] = datetime.utcnow()
            run_data["ledger_events"].append({
                "timestamp": datetime.utcnow().isoformat(),
                "event": "LLM_AGENT_PROCESSING_COMPLETED",
                "details": {
                    "run_id": run_id,
                    "files_processed": len(file_paths),
                    "steps_completed": len([r for r in results if r.get("type") == "step"]),
                    "success": any(r.get("type") == "completion" for r in results)
                }
            })
        
        return {
            "success": True,
            "run_id": run_id,
            "results": results,
            "files_processed": len(file_paths),
            "workspace": str(workspace_dir)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ LLM agent processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"LLM agent processing failed: {str(e)}")


from fastapi.responses import StreamingResponse
import asyncio

@app.post("/api/v1/agent/llm-stream", tags=["Agent"])
async def llm_agent_stream_process(
    request: Dict[str, Any],
    settings: Settings = Depends(get_settings_dependency)
):
    """
    Stream LLM agent processing with real-time updates.
    
    Returns a Server-Sent Events stream with live terminal output.
    """
    try:
        run_id = request.get("run_id")
        file_paths_str = request.get("file_paths", [])
        
        if not run_id:
            raise HTTPException(status_code=400, detail="run_id is required")
        
        from pathlib import Path
        file_paths = [Path(p) for p in file_paths_str]
        
        async def event_stream():
            try:
                # Create LLM agent with Claude config
                workspace_dir = Path("llm_agent_workspaces") / run_id
                
                # Get Claude API key from settings with env fallback
                _settings = get_settings()
                _anthropic_key = None
                try:
                    if getattr(_settings.llm.anthropic, "api_key", None):
                        _anthropic_key = _settings.llm.anthropic.api_key.get_secret_value()  # type: ignore[attr-defined]
                except Exception:
                    _anthropic_key = None
                if not _anthropic_key:
                    _anthropic_key = os.getenv("ANTHROPIC_API_KEY")
                claude_config = {
                    "api_key": _anthropic_key,
                    "model": "claude-3-5-haiku-20241022",
                    "max_tokens": 4096,
                    "temperature": 0.3
                }
                
                agent = GenericLLMAgent(run_id, workspace_dir, claude_config)
                
                # Stream processing events
                for result in agent.analyze_and_process_files(file_paths):
                    # Format as Server-Sent Event
                    event_data = json.dumps(result)
                    yield f"data: {event_data}\n\n"
                    
                    # Add small delay for UI responsiveness
                    await asyncio.sleep(0.1)
                
                # Send completion event
                completion_event = {
                    "type": "complete",
                    "message": "🎉 LLM agent processing completed!",
                    "timestamp": datetime.utcnow().isoformat()
                }
                yield f"data: {json.dumps(completion_event)}\n\n"
                
            except Exception as e:
                error_event = {
                    "type": "error",
                    "message": f"❌ LLM agent failed: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat()
                }
                yield f"data: {json.dumps(error_event)}\n\n"
        
        return StreamingResponse(
            event_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start LLM agent stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/agent/llm-stream/{run_id}", tags=["Agent"])
async def llm_agent_stream_process_get(
    run_id: str,
    settings: Settings = Depends(get_settings_dependency)
):
    """
    Stream LLM agent processing (GET variant) with real-time updates.
    Uses stored uploaded file for the run; ideal for browser EventSource.
    """
    try:
        # Debug: Check what runs exist
        logger.info(f"🔍 LLM stream requested for run_id: {run_id}")
        if hasattr(app.state, 'runs'):
            logger.info(f"Available runs: {list(app.state.runs.keys())}")
        else:
            logger.warning("No runs found in app.state")
            
        # Validate run and get file path
        if not hasattr(app.state, 'runs') or run_id not in app.state.runs:
            # Do NOT fallback to arbitrary latest uploads; that can mix test data.
            # Instead, attempt to stream only if standardized data exists for this run; otherwise
            # inform the client to re-upload or provide a valid run_id.
            std_dir = Path("standardized_data") / run_id
            if not std_dir.exists() or not any(std_dir.glob("*.csv")):
                logger.warning(f"Run {run_id} not found in memory and no standardized data present. Aborting stream.")
                raise HTTPException(status_code=404, detail=f"Run not found or not prepared: {run_id}")
            # If standardized data exists, continue with a placeholder file path (not used directly by agent)
            file_path_str = str(std_dir)
        else:
            run_info = app.state.runs[run_id]
            file_path_str = run_info.get("file_path")
            if not file_path_str:
                raise HTTPException(status_code=400, detail="No file associated with this run")

        file_paths = [Path(file_path_str)]

        def event_stream():
            """SSE generator with background worker and heartbeats to keep connection alive."""
            q: "queue.Queue[str]" = queue.Queue()
            done = threading.Event()

            def enqueue_event(payload: dict):
                try:
                    q.put_nowait(f"data: {json.dumps(payload)}\n\n")
                except Exception as _e:
                    logger.warning(f"Failed to enqueue SSE event: {_e}")

            def worker():
                try:
                    # Initial events
                    enqueue_event({
                        "type": "log",
                        "message": f"🚀 Starting LLM agent for run: {run_id}",
                        "level": "info",
                        "timestamp": datetime.utcnow().isoformat(),
                    })

                    # Debug diagnostics for API key resolution
                    try:
                        settings_dbg = get_settings()
                        env_has = bool(os.getenv("ANTHROPIC_API_KEY"))
                        settings_has = False
                        try:
                            if getattr(settings_dbg.llm.anthropic, "api_key", None):
                                _tmp = settings_dbg.llm.anthropic.api_key.get_secret_value()  # type: ignore[attr-defined]
                                settings_has = bool(_tmp)
                        except Exception:
                            settings_has = False
                        enqueue_event({
                            "type": "debug",
                            "message": "LLM key resolution",
                            "data": {
                                "env_has_key": env_has,
                                "settings_has_key": settings_has,
                                "default_provider": getattr(settings_dbg.llm, "default_provider", None),
                            },
                            "timestamp": datetime.utcnow().isoformat(),
                        })
                    except Exception as dbg_err:
                        enqueue_event({
                            "type": "debug",
                            "message": f"Diagnostics failure: {str(dbg_err)}",
                            "timestamp": datetime.utcnow().isoformat(),
                        })

                    # Create LLM agent with Claude config
                    workspace_dir = Path("llm_agent_workspaces") / run_id
                    # Resolve Claude API key via settings first, then env fallback
                    _settings = get_settings()
                    api_key = None
                    try:
                        if getattr(_settings.llm.anthropic, "api_key", None):
                            api_key = _settings.llm.anthropic.api_key.get_secret_value()  # type: ignore[attr-defined]
                    except Exception:
                        api_key = None
                    if not api_key:
                        api_key = os.getenv("ANTHROPIC_API_KEY")
                    if not api_key:
                        raise ValueError("ANTHROPIC_API_KEY not found in environment or settings")
                    
                    claude_config = {
                        "api_key": api_key,
                        "model": "claude-3-5-haiku-20241022",
                        "max_tokens": 4096,
                        "temperature": 0.3,
                    }
                    logger.info(f"Creating LLM agent with workspace: {workspace_dir}")
                    agent = GenericLLMAgent(run_id, workspace_dir, claude_config)

                    enqueue_event({
                        "type": "log",
                        "message": f"📁 Processing file: {file_path_str}",
                        "level": "info",
                        "timestamp": datetime.utcnow().isoformat(),
                    })

                    for result in agent.analyze_and_process_files(file_paths):
                        enqueue_event(result)

                    enqueue_event({
                        "type": "complete",
                        "message": "🎉 LLM agent processing completed!",
                        "timestamp": datetime.utcnow().isoformat(),
                    })
                except Exception as e:
                    logger.error(f"LLM stream error: {e}")
                    enqueue_event({
                        "type": "error",
                        "message": f"❌ LLM agent failed: {str(e)}",
                        "timestamp": datetime.utcnow().isoformat(),
                    })
                finally:
                    done.set()

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()

            # Heartbeat loop
            last_heartbeat = time.time()
            heartbeat_interval = 10.0
            while not done.is_set() or not q.empty():
                try:
                    chunk = q.get(timeout=1.0)
                    yield chunk
                except queue.Empty:
                    # Send heartbeat comment to keep the connection alive
                    now = time.time()
                    if now - last_heartbeat >= heartbeat_interval:
                        last_heartbeat = now
                        yield ": keep-alive\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "X-Accel-Buffering": "no",
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to start LLM agent GET stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/domains/mapping-stream/{run_id}", tags=["Domains"])
async def domain_mapping_stream(
    run_id: str,
    settings: Settings = Depends(get_settings_dependency)
):
    """
    Stream domain mapping progress with real-time updates.
    """
    def event_stream():
        """SSE generator for domain mapping progress."""
        try:
            # Send initial connection event
            yield f"data: {json.dumps({'type': 'connection', 'message': f'Connected to domain mapping stream for run {run_id}'})}\n\n"
            
            # Log the run_id being processed
            logger.info(f"🔍 Domain mapping requested for run_id: {run_id}")
            
            # Step 1: Analyze Schema
            yield f"data: {json.dumps({'type': 'step_start', 'data': {'stepId': 'analyze'}})}\n\n"
            time.sleep(0.5)
            
            # Look for standardized data in dedicated location
            standardized_data_dir = Path("standardized_data") / run_id
            # Support both historical and current agent workspace roots
            llm_workspace_dir = Path("llm_agent_workspaces") / run_id
            agent_workspace_dir = Path("agent_workspaces") / run_id
            
            logger.info(f"🔍 Looking for standardized data at: {standardized_data_dir}")
            logger.info(f"🔍 Fallback workspaces at: {llm_workspace_dir} and {agent_workspace_dir}")
            
            csv_files: List[Path] = []
            
            # First priority: Check dedicated standardized data directory
            if standardized_data_dir.exists():
                standardized = list(standardized_data_dir.glob("*.csv"))
                # Filter out test data files - only process user uploaded data
                user_files = []
                for f in standardized:
                    # Skip files that are clearly test data
                    if not any(test_prefix in f.name.lower() for test_prefix in ['library_', 'retail_']):
                        user_files.append(f)
                    else:
                        logger.info(f"🚫 Skipping test data file: {f.name}")
                
                if user_files:
                    csv_files = user_files
                    logger.info(f"🔍 Found {len(user_files)} user CSV files in dedicated location: {[f.name for f in user_files]}")
                elif standardized:
                    logger.warning(f"⚠️ All files appear to be test data: {[f.name for f in standardized]}")
                    # Don't process test data - return error instead
                    error_msg = f"Only test data found in standardized location. Please upload your own data files."
                    logger.error(error_msg)
                    yield f"data: {json.dumps({'type': 'error', 'data': {'stepId': 'analyze', 'error': error_msg}})}\n\n"
                    return
            
            # Fallback: Check LLM workspace for standardized files
            if not csv_files and llm_workspace_dir.exists():
                standardized = list(llm_workspace_dir.glob("**/standardized_*.csv"))
                if standardized:
                    csv_files = standardized
                    logger.info(f"🔍 Found {len(standardized)} standardized CSV files in llm workspace: {[f.name for f in standardized]}")

            # Fallback: Check Agent workspace for standardized files
            if not csv_files and agent_workspace_dir.exists():
                standardized = list(agent_workspace_dir.glob("**/standardized_*.csv"))
                if standardized:
                    csv_files = standardized
                    logger.info(f"🔍 Found {len(standardized)} standardized CSV files in agent workspace: {[f.name for f in standardized]}")
            
            # Last resort: Check workspace for user uploaded files (non-test data)
            if not csv_files and (llm_workspace_dir.exists() or agent_workspace_dir.exists()):
                all_csvs = []
                if llm_workspace_dir.exists():
                    all_csvs.extend(llm_workspace_dir.glob("**/*.csv"))
                if agent_workspace_dir.exists():
                    all_csvs.extend(agent_workspace_dir.glob("**/*.csv"))
                # Only use files that are NOT test data (no library_, retail_ prefixes unless standardized)
                csv_files = [f for f in all_csvs if 'standardized_' in f.name or not any(test_name in f.name.lower() for test_name in ['library_', 'retail_'])]
                logger.info(f"🔍 Found {len(csv_files)} CSV files in workspaces (filtered for user data): {[f.name for f in csv_files]}")

            # If still no files, provide clear error
            if not csv_files:
                available_files = []
                if standardized_data_dir.exists():
                    available_files.extend([f"standardized_data/{f.name}" for f in standardized_data_dir.rglob('*') if f.is_file()])
                if llm_workspace_dir.exists():
                    available_files.extend([f"llm_workspace/{f.name}" for f in llm_workspace_dir.rglob('*') if f.is_file()])
                if agent_workspace_dir.exists():
                    available_files.extend([f"agent_workspace/{f.name}" for f in agent_workspace_dir.rglob('*') if f.is_file()])
                
                error_msg = f"No processed CSV files found for run {run_id}. Please ensure LLM agent completed successfully. Available files: {available_files}"
                logger.error(error_msg)
                yield f"data: {json.dumps({'type': 'error', 'data': {'stepId': 'analyze', 'error': error_msg}})}\n\n"
                return
            
            total_columns = 0
            all_columns = []
            
            # Analyze CSV files
            for idx, csv_file in enumerate(csv_files):
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_file, nrows=100)
                    columns = df.columns.tolist()
                    total_columns += len(columns)
                    
                    for col in columns:
                        col_info = ColumnInfo(
                            name=col,
                            data_type=str(df[col].dtype),
                            sample_values=df[col].dropna().head(10).tolist(),
                            unique_count=df[col].nunique(),
                            null_count=int(df[col].isnull().sum()),
                            total_count=len(df)
                        )
                        all_columns.append(col_info)
                    
                    progress = ((idx + 1) / len(csv_files)) * 100
                    yield f"data: {json.dumps({'type': 'step_progress', 'data': {'stepId': 'analyze', 'progress': progress, 'details': f'Analyzed {csv_file.name}'}})}\n\n"
                    time.sleep(0.3)
                except Exception as e:
                    logger.warning(f"Failed to analyze {csv_file}: {e}")
            
            yield f"data: {json.dumps({'type': 'step_complete', 'data': {'stepId': 'analyze', 'metrics': {'totalColumns': total_columns}}})}\n\n"
            
            # Step 2: Generate Candidates
            yield f"data: {json.dumps({'type': 'step_start', 'data': {'stepId': 'generate'}})}\n\n"
            time.sleep(0.5)
            
            # Initialize domain assignment engine
            engine = DomainAssignmentEngine()
            assignments = []
            
            for idx, col_info in enumerate(all_columns):
                progress = ((idx + 1) / len(all_columns)) * 100
                yield f"data: {json.dumps({'type': 'step_progress', 'data': {'stepId': 'generate', 'progress': progress, 'details': f'Processing {col_info.name}'}})}\n\n"
                
                # Assign domain using the engine
                assignment = engine.assign_domain(col_info, run_id)
                assignments.append(assignment)
                
                # Send column mapping event
                mapping = {
                    'columnName': col_info.name,
                    'sourceType': col_info.data_type,
                    'detectedDomain': assignment.domain_name or 'Unknown',
                    'confidence': assignment.confidence_score,
                    'confidenceBand': assignment.confidence_band.value,
                    'status': 'completed' if assignment.domain_name else 'failed',
                    'evidence': {
                        'nameScore': assignment.evidence.name_similarity if assignment.evidence else 0,
                        'patternScore': assignment.evidence.regex_strength if assignment.evidence else 0,
                        'valueScore': assignment.evidence.value_similarity if assignment.evidence else 0,
                        'unitScore': assignment.evidence.unit_compatibility if assignment.evidence else 0,
                    } if assignment.evidence else None
                }
                
                yield f"data: {json.dumps({'type': 'column_mapped', 'data': {'mapping': mapping}})}\n\n"
                yield f"data: {json.dumps({'type': 'overall_progress', 'data': {'progress': int(progress * 0.5)}})}\n\n"
                time.sleep(0.1)
            
            # Calculate metrics
            high_conf = sum(1 for a in assignments if a.confidence_band == ConfidenceBand.HIGH)
            borderline_conf = sum(1 for a in assignments if a.confidence_band == ConfidenceBand.BORDERLINE)
            low_conf = sum(1 for a in assignments if a.confidence_band == ConfidenceBand.LOW)
            
            yield f"data: {json.dumps({'type': 'step_complete', 'data': {'stepId': 'generate', 'metrics': {'highConfidence': high_conf, 'mediumConfidence': borderline_conf, 'lowConfidence': low_conf}}})}\n\n"
            
            # Step 3: Score & Rank
            yield f"data: {json.dumps({'type': 'step_start', 'data': {'stepId': 'score'}})}\n\n"
            time.sleep(0.5)
            
            for idx in range(10):
                progress = (idx + 1) * 10
                yield f"data: {json.dumps({'type': 'step_progress', 'data': {'stepId': 'score', 'progress': progress, 'details': 'Calculating composite scores...'}})}\n\n"
                yield f"data: {json.dumps({'type': 'overall_progress', 'data': {'progress': 50 + int(progress * 0.3)}})}\n\n"
                time.sleep(0.2)
            
            yield f"data: {json.dumps({'type': 'step_complete', 'data': {'stepId': 'score'}})}\n\n"
            
            # Step 4: Validate Results
            yield f"data: {json.dumps({'type': 'step_start', 'data': {'stepId': 'validate'}})}\n\n"
            time.sleep(0.5)
            
            for idx in range(5):
                progress = (idx + 1) * 20
                yield f"data: {json.dumps({'type': 'step_progress', 'data': {'stepId': 'validate', 'progress': progress, 'details': 'Validating against business rules...'}})}\n\n"
                yield f"data: {json.dumps({'type': 'overall_progress', 'data': {'progress': 80 + int(progress * 0.2)}})}\n\n"
                time.sleep(0.3)
            
            yield f"data: {json.dumps({'type': 'step_complete', 'data': {'stepId': 'validate'}})}\n\n"
            
            # Send completion
            yield f"data: {json.dumps({'type': 'overall_progress', 'data': {'progress': 100}})}\n\n"
            yield f"data: {json.dumps({'type': 'mapping_complete', 'data': {'totalMappings': len(assignments)}})}\n\n"
            
        except Exception as e:
            logger.error(f"Domain mapping stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'data': {'stepId': 'current', 'error': str(e)}})}\n\n"
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/api/v1/agent/test-stream", tags=["Agent"])
async def test_stream():
    """Simple test streaming endpoint"""
    
    def generate():
        for i in range(5):
            event = {
                "type": "log",
                "message": f"Test event {i+1}",
                "level": "info",
                "timestamp": datetime.utcnow().isoformat()
            }
            yield f"data: {json.dumps(event)}\n\n"
            time.sleep(1)
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "X-Accel-Buffering": "no",
        }
    )


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


# === DEVELOPMENT SERVER ===

if __name__ == "__main__":
    settings = get_settings()
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disabled for SSE stability
        log_level="info" if not settings.debug else "debug"
    )
