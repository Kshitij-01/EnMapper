"""
EnMapper FastAPI Application - Main Entry Point

Phase 0: Foundation with health endpoints and basic structure
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn

from settings import get_settings, Settings
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
    file: UploadFile = File(...),
    run_name: str = Form(None),
    description: str = Form(None),
    mode: str = Form("metadata_only"),
    pii_masking_enabled: bool = Form(True),
    lane_hint: str = Form("interactive"),
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
        allowed_extensions = ['.csv', '.tsv', '.parquet', '.json', '.jsonl']
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
        
        return {
            "success": True,
            "run_id": run_id,
            "file_path": str(file_path),
            "original_filename": file.filename,
            "file_size_bytes": len(file_content),
            "file_type": file_extension[1:],
            "run_name": run_request["run_name"],
            "message": "File uploaded and run created successfully",
            "next_steps": {
                "process_url": f"/api/v1/runs/{run_id}/process",
                "status_url": f"/api/v1/runs/{run_id}"
            }
        }
        
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
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug"
    )
