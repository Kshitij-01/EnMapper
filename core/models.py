"""
Core Models for EnMapper

Phase 0: Basic models for Run Contract, artifacts, and data structures.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
import ulid

from settings import ProcessingMode, ProcessingLane, LLMProvider, RoutingProfile


class RunStatus(str, Enum):
    """Run execution status."""
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StageStatus(str, Enum):
    """Stage execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class SupervisorDecision(str, Enum):
    """Supervisor gate decisions."""
    ACCEPT = "accept"
    HOLD = "hold"
    FIX = "fix"
    REJECT = "reject"
    RETRY = "retry"
    ESCALATE = "escalate"


# === DATA SOURCE MODELS ===

class DataSourceType(str, Enum):
    """Supported data source types."""
    FILE = "file"
    SQL = "sql"


class FileSource(BaseModel):
    """File-based data source."""
    type: str = "file"
    location: str  # File path or URL
    format: str    # csv, tsv, parquet, json, etc.
    encoding: Optional[str] = "utf-8"
    dialect: Optional[Dict[str, Any]] = None
    schema_ref: Optional[str] = None


class SQLSource(BaseModel):
    """SQL database source."""
    type: str = "sql" 
    connection_string: str
    schema: Optional[str] = None
    tables: List[str] = Field(default_factory=list)
    query: Optional[str] = None
    schema_ref: Optional[str] = None


DataSource = Union[FileSource, SQLSource]


# === BUDGET AND CAPS ===

class BudgetCaps(BaseModel):
    """Budget limits for a run."""
    tokens: int = Field(default=100000, description="Maximum tokens allowed")
    usd: float = Field(default=10.0, description="Maximum USD cost allowed") 
    wall_time_s: int = Field(default=3600, description="Maximum wall time in seconds")


class StageBudgets(BaseModel):
    """Per-stage budget allocation."""
    ingest: BudgetCaps
    domains: BudgetCaps 
    mapping: BudgetCaps
    analysis: BudgetCaps
    migration: BudgetCaps


# === SAMPLING CONFIGURATION ===

class SampleCaps(BaseModel):
    """Data sampling limits."""
    rows_max: int = Field(default=1000, description="Maximum rows to sample")
    bytes_max: int = Field(default=10_000_000, description="Maximum bytes to sample")
    freshness_window_hours: int = Field(default=24, description="Sample cache freshness")


# === RUN CONTRACT (PHASE 0) ===

class RunContract(BaseModel):
    """
    Run Contract v1 - Normative contract for job execution.
    
    Phase 0: Basic structure with required fields.
    Later phases will add validation and enforcement logic.
    """
    
    # Core identification
    run_id: str = Field(default_factory=lambda: str(ulid.ULID()))
    roadmap_version: str = Field(default="1.1.0", description="Roadmap version used")
    
    # Processing configuration
    mode: ProcessingMode = Field(description="Processing mode (immutable once started)")
    lane_hint: Optional[ProcessingLane] = Field(None, description="Preferred processing lane")
    
    # Data sources
    lhs: DataSource = Field(description="Left-hand side (source) data")
    rhs: DataSource = Field(description="Right-hand side (target) data")
    
    # Privacy and security
    pii_policy: Dict[str, Any] = Field(
        default_factory=lambda: {
            "mask_before_send": True,
            "local_only_override": False
        }
    )
    
    # Budget and cost controls
    budget_caps: BudgetCaps = Field(default_factory=BudgetCaps)
    stage_budgets: Optional[StageBudgets] = None
    
    # Provider configuration  
    routing_profile: RoutingProfile = Field(default=RoutingProfile.QUALITY_FIRST)
    provider_allowlist: List[str] = Field(default_factory=list)
    provider_denylist: List[str] = Field(default_factory=list)
    
    # Processing parameters
    threshold_profile_id: str = Field(default="tp_default_v1")
    dsl_version: str = Field(default="1.0.0", description="Transform DSL version")
    sample_caps: Optional[SampleCaps] = None
    
    # UI and caching
    cache_scope: Dict[str, Any] = Field(
        default_factory=lambda: {"run": True, "tab": True, "filters": True}
    )
    
    # Workflow controls
    human_confirm_required: bool = Field(default=False)
    
    # RBAC and audit
    rbac_actor: Dict[str, str] = Field(
        default_factory=lambda: {"user_id": "system", "role": "operator"}
    )
    audit_tags: List[str] = Field(default_factory=list)
    
    # Timestamps
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Contract integrity
    contract_hash: Optional[str] = None
    
    def generate_contract_hash(self) -> str:
        """Generate SHA-256 hash of canonical contract JSON."""
        import hashlib
        import json
        
        # Create canonical representation
        contract_dict = self.dict(exclude={"contract_hash", "created_at", "updated_at"})
        canonical_json = json.dumps(contract_dict, sort_keys=True, separators=(',', ':'))
        
        # Generate hash
        hash_obj = hashlib.sha256(canonical_json.encode('utf-8'))
        return hash_obj.hexdigest()
    
    def finalize(self) -> "RunContract":
        """Finalize contract by generating hash and updating timestamp."""
        self.contract_hash = self.generate_contract_hash()
        self.updated_at = datetime.utcnow().isoformat()
        return self


# === RUN EXECUTION TRACKING ===

class StageExecution(BaseModel):
    """Execution tracking for a pipeline stage."""
    stage: str  # "ingest", "domains", "mapping", "analysis", "migration"
    status: StageStatus = StageStatus.PENDING
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    # Resource usage
    tokens_used: int = 0
    cost_usd: float = 0.0
    provider_used: Optional[str] = None
    model_used: Optional[str] = None
    
    # Outputs
    artifacts_produced: List[str] = Field(default_factory=list)
    supervisor_decision: Optional[SupervisorDecision] = None
    
    # Error handling
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0


class RunExecution(BaseModel):
    """Execution state for a run."""
    run_id: str
    contract_hash: str
    status: RunStatus = RunStatus.CREATED
    
    # Lane assignment (may differ from hint)
    assigned_lane: Optional[ProcessingLane] = None
    lane_assignment_reason: Optional[str] = None
    
    # Stage tracking
    stages: Dict[str, StageExecution] = Field(default_factory=dict)
    current_stage: Optional[str] = None
    
    # Overall resource usage
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0
    
    # Timestamps
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    # Final outputs
    artifacts_generated: List[str] = Field(default_factory=list)
    
    def get_stage(self, stage_name: str) -> StageExecution:
        """Get or create stage execution."""
        if stage_name not in self.stages:
            self.stages[stage_name] = StageExecution(stage=stage_name)
        return self.stages[stage_name]
    
    def start_stage(self, stage_name: str) -> StageExecution:
        """Start execution of a stage."""
        stage = self.get_stage(stage_name)
        stage.status = StageStatus.RUNNING
        stage.started_at = datetime.utcnow().isoformat()
        self.current_stage = stage_name
        return stage
    
    def complete_stage(self, stage_name: str, decision: SupervisorDecision = SupervisorDecision.ACCEPT) -> StageExecution:
        """Complete execution of a stage."""
        stage = self.get_stage(stage_name)
        stage.status = StageStatus.COMPLETED
        stage.completed_at = datetime.utcnow().isoformat()
        stage.supervisor_decision = decision
        
        # Update totals
        self.total_tokens_used += stage.tokens_used
        self.total_cost_usd += stage.cost_usd
        
        return stage


# === THRESHOLD PROFILES ===

class ThresholdProfile(BaseModel):
    """Quality thresholds for pipeline decisions."""
    profile_id: str = "tp_default_v1"
    
    # Domaining thresholds
    domaining_tau_high: float = Field(default=0.82, description="Auto-assign threshold")
    domaining_tau_low: float = Field(default=0.55, description="Unknown threshold")
    
    # Mapping thresholds  
    mapping_accept_rate_min: float = Field(default=0.95, description="Sample success rate to accept")
    mapping_near_match_window: float = Field(default=0.03, description="Tolerance for near matches")
    
    # Analysis thresholds
    analysis_null_rate_max: float = Field(default=0.02, description="Maximum null rate")
    analysis_dup_key_rate_max: float = Field(default=0.00, description="Maximum duplicate key rate")
    
    # Migration settings
    migration_batch_size: int = Field(default=50000, description="Chunk size for processing")
    migration_validation_checks: List[str] = Field(
        default_factory=lambda: ["row_count", "checksum", "fk_check"]
    )
    
    # Budget allocations (as fractions of total)
    budget_ingest_pct: float = Field(default=0.15, description="Ingest budget percentage")
    budget_domains_pct: float = Field(default=0.25, description="Domains budget percentage") 
    budget_mapping_pct: float = Field(default=0.35, description="Mapping budget percentage")
    budget_analysis_pct: float = Field(default=0.15, description="Analysis budget percentage")
    budget_migration_pct: float = Field(default=0.10, description="Migration budget percentage")


# === SUPERVISOR LEDGER EVENT ===

class SupervisorEvent(BaseModel):
    """Event record for Supervisor Ledger."""
    event_id: str = Field(default_factory=lambda: str(ulid.ULID()))
    run_id: str
    timestamp_utc: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    
    stage: str  # INGEST, DOMAINS, MAPPING, ANALYSIS, MIGRATION, SUPERVISOR
    decision: SupervisorDecision
    
    # Input/Output tracking
    inputs_hash: str
    outputs_hash: Optional[str] = None
    
    # Provider and model info
    provider: Optional[str] = None
    model: Optional[str] = None
    params_hash: Optional[str] = None
    
    # Cost tracking
    token_estimate: int = 0
    token_actual: int = 0
    usd_estimate: float = 0.0
    usd_actual: float = 0.0
    
    # Retry and fallback
    retries: int = 0
    fallback_chain: List[str] = Field(default_factory=list)
    
    # Privacy and policy
    pii_mask_applied: bool = False
    local_only_override: bool = False
    policy_version: str = "0.1.0"
    
    # Configuration versions
    threshold_profile_id: str = "tp_default_v1"
    dsl_version: str = "1.0.0"
    
    # Error handling
    error_code: Optional[str] = None
    error_detail_id: Optional[str] = None  # Link to trace
    
    # Human oversight
    human_confirmation: Optional[Dict[str, Any]] = None
