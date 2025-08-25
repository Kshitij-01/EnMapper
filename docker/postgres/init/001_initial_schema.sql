-- EnMapper Initial Database Schema
-- Phase 0: Basic tables for core functionality

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable vector extension for PGVector
CREATE EXTENSION IF NOT EXISTS vector;

-- =============================================================================
-- CORE TABLES
-- =============================================================================

-- Runs table: Track all data integration runs
CREATE TABLE runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id VARCHAR(26) UNIQUE NOT NULL, -- ULID format
    contract_hash VARCHAR(64) NOT NULL,
    
    -- Basic run info
    status VARCHAR(20) NOT NULL DEFAULT 'created',
    mode VARCHAR(20) NOT NULL, -- metadata_only, data_mode
    assigned_lane VARCHAR(20), -- interactive, flex, batch
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Resource usage
    total_tokens_used INTEGER DEFAULT 0,
    total_cost_usd DECIMAL(10, 4) DEFAULT 0.0,
    
    -- Contract and configuration
    contract_data JSONB NOT NULL,
    roadmap_version VARCHAR(10) NOT NULL,
    threshold_profile_id VARCHAR(50) NOT NULL,
    
    -- Indexes
    CONSTRAINT chk_status CHECK (status IN ('created', 'queued', 'running', 'completed', 'failed', 'cancelled')),
    CONSTRAINT chk_mode CHECK (mode IN ('metadata_only', 'data_mode')),
    CONSTRAINT chk_lane CHECK (assigned_lane IS NULL OR assigned_lane IN ('interactive', 'flex', 'batch'))
);

-- Stage executions: Track individual pipeline stages
CREATE TABLE stage_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id VARCHAR(26) NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    
    -- Stage info
    stage_name VARCHAR(20) NOT NULL, -- ingest, domains, mapping, analysis, migration
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    
    -- Timestamps
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Resource usage
    tokens_used INTEGER DEFAULT 0,
    cost_usd DECIMAL(10, 4) DEFAULT 0.0,
    provider_used VARCHAR(20),
    model_used VARCHAR(100),
    
    -- Outputs and decisions
    artifacts_produced TEXT[],
    supervisor_decision VARCHAR(20),
    
    -- Error handling
    error_code VARCHAR(50),
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    
    -- Constraints
    CONSTRAINT chk_stage_name CHECK (stage_name IN ('ingest', 'domains', 'mapping', 'analysis', 'migration')),
    CONSTRAINT chk_stage_status CHECK (status IN ('pending', 'running', 'completed', 'failed', 'skipped')),
    CONSTRAINT chk_supervisor_decision CHECK (supervisor_decision IS NULL OR supervisor_decision IN ('accept', 'hold', 'fix', 'reject', 'retry', 'escalate')),
    
    -- Unique constraint
    UNIQUE(run_id, stage_name)
);

-- =============================================================================
-- DOMAIN CATALOG TABLES
-- =============================================================================

-- Live domains: Production domain definitions
CREATE TABLE domains_live (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    domain_id VARCHAR(100) UNIQUE NOT NULL,
    
    -- Domain definition
    name VARCHAR(200) NOT NULL,
    category VARCHAR(100) NOT NULL,
    description TEXT,
    
    -- Matching patterns
    aliases TEXT[],
    regex_patterns TEXT[],
    units TEXT[],
    
    -- Vector representations for similarity matching
    name_embedding vector(1536), -- OpenAI embedding dimension
    value_embedding vector(1536),
    
    -- Metadata
    version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(100) NOT NULL DEFAULT 'system',
    
    -- Examples
    examples JSONB NOT NULL DEFAULT '[]'
);

-- Staged domains: Domains being authored/tested
CREATE TABLE domains_staged (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    domain_id VARCHAR(100) NOT NULL,
    
    -- Domain definition (same structure as live)
    name VARCHAR(200) NOT NULL,
    category VARCHAR(100) NOT NULL,
    description TEXT,
    
    aliases TEXT[],
    regex_patterns TEXT[],
    units TEXT[],
    
    name_embedding vector(1536),
    value_embedding vector(1536),
    
    -- Staging metadata
    version INTEGER NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'draft', -- draft, testing, approved, rejected
    based_on_live_id UUID REFERENCES domains_live(id),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(100) NOT NULL,
    reviewed_by VARCHAR(100),
    
    examples JSONB NOT NULL DEFAULT '[]',
    
    -- Constraints
    CONSTRAINT chk_staged_status CHECK (status IN ('draft', 'testing', 'approved', 'rejected'))
);

-- Domain examples: Curated examples for domain matching
CREATE TABLE domain_examples (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    domain_id VARCHAR(100) NOT NULL,
    
    -- Example data
    example_type VARCHAR(20) NOT NULL, -- positive, negative
    name_example VARCHAR(500),
    value_example TEXT,
    context TEXT,
    
    -- Quality metrics
    confidence DECIMAL(3, 2) DEFAULT 1.0,
    source VARCHAR(100) NOT NULL DEFAULT 'manual',
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(100) NOT NULL DEFAULT 'system',
    
    -- Constraints
    CONSTRAINT chk_example_type CHECK (example_type IN ('positive', 'negative')),
    CONSTRAINT chk_confidence CHECK (confidence >= 0.0 AND confidence <= 1.0)
);

-- =============================================================================
-- POLICY AND CONFIGURATION TABLES
-- =============================================================================

-- Threshold profiles: Quality gates and decision thresholds
CREATE TABLE threshold_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    profile_id VARCHAR(50) UNIQUE NOT NULL,
    
    -- Profile metadata
    name VARCHAR(200) NOT NULL,
    description TEXT,
    version INTEGER NOT NULL DEFAULT 1,
    
    -- Threshold values (stored as JSONB for flexibility)
    thresholds JSONB NOT NULL,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(100) NOT NULL DEFAULT 'system',
    is_active BOOLEAN DEFAULT true
);

-- Policy manifests: Security and governance policies
CREATE TABLE policy_manifests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    version VARCHAR(20) UNIQUE NOT NULL,
    
    -- Policy content
    manifest JSONB NOT NULL,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(100) NOT NULL DEFAULT 'system',
    is_active BOOLEAN DEFAULT false,
    activated_at TIMESTAMP WITH TIME ZONE
);

-- =============================================================================
-- SUPERVISOR LEDGER
-- =============================================================================

-- Supervisor events: Complete audit trail of all decisions
CREATE TABLE supervisor_ledger (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_id VARCHAR(26) UNIQUE NOT NULL, -- ULID format
    run_id VARCHAR(26) NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    
    -- Event details
    timestamp_utc TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    stage VARCHAR(20) NOT NULL,
    decision VARCHAR(20) NOT NULL,
    
    -- Input/Output tracking
    inputs_hash VARCHAR(64) NOT NULL,
    outputs_hash VARCHAR(64),
    
    -- Provider and model info
    provider VARCHAR(20),
    model VARCHAR(100),
    params_hash VARCHAR(64),
    
    -- Cost tracking
    token_estimate INTEGER DEFAULT 0,
    token_actual INTEGER DEFAULT 0,
    usd_estimate DECIMAL(10, 4) DEFAULT 0.0,
    usd_actual DECIMAL(10, 4) DEFAULT 0.0,
    
    -- Retry and fallback
    retries INTEGER DEFAULT 0,
    fallback_chain TEXT[],
    
    -- Privacy and policy
    pii_mask_applied BOOLEAN DEFAULT false,
    local_only_override BOOLEAN DEFAULT false,
    policy_version VARCHAR(20) NOT NULL,
    
    -- Configuration versions
    threshold_profile_id VARCHAR(50) NOT NULL,
    dsl_version VARCHAR(20) NOT NULL,
    
    -- Error handling
    error_code VARCHAR(50),
    error_detail_id VARCHAR(100), -- Link to trace
    
    -- Human oversight
    human_confirmation JSONB,
    
    -- Constraints
    CONSTRAINT chk_ledger_stage CHECK (stage IN ('INGEST', 'DOMAINS', 'MAPPING', 'ANALYSIS', 'MIGRATION', 'SUPERVISOR')),
    CONSTRAINT chk_ledger_decision CHECK (decision IN ('accept', 'hold', 'fix', 'reject', 'retry', 'escalate'))
);

-- =============================================================================
-- INDEXES FOR PERFORMANCE
-- =============================================================================

-- Runs indexes
CREATE INDEX idx_runs_status ON runs(status);
CREATE INDEX idx_runs_mode ON runs(mode);
CREATE INDEX idx_runs_created_at ON runs(created_at);
CREATE INDEX idx_runs_contract_hash ON runs(contract_hash);

-- Stage executions indexes
CREATE INDEX idx_stage_executions_run_id ON stage_executions(run_id);
CREATE INDEX idx_stage_executions_status ON stage_executions(status);
CREATE INDEX idx_stage_executions_stage_name ON stage_executions(stage_name);

-- Domain catalog indexes
CREATE INDEX idx_domains_live_domain_id ON domains_live(domain_id);
CREATE INDEX idx_domains_live_category ON domains_live(category);
CREATE INDEX idx_domains_staged_domain_id ON domains_staged(domain_id);
CREATE INDEX idx_domains_staged_status ON domains_staged(status);

-- Supervisor ledger indexes  
CREATE INDEX idx_supervisor_ledger_run_id ON supervisor_ledger(run_id);
CREATE INDEX idx_supervisor_ledger_timestamp ON supervisor_ledger(timestamp_utc);
CREATE INDEX idx_supervisor_ledger_stage ON supervisor_ledger(stage);
CREATE INDEX idx_supervisor_ledger_decision ON supervisor_ledger(decision);

-- Vector similarity indexes (using HNSW for PGVector)
CREATE INDEX idx_domains_live_name_embedding ON domains_live USING hnsw (name_embedding vector_cosine_ops);
CREATE INDEX idx_domains_live_value_embedding ON domains_live USING hnsw (value_embedding vector_cosine_ops);
CREATE INDEX idx_domains_staged_name_embedding ON domains_staged USING hnsw (name_embedding vector_cosine_ops);

-- =============================================================================
-- INITIAL DATA
-- =============================================================================

-- Insert default threshold profile
INSERT INTO threshold_profiles (profile_id, name, description, thresholds) VALUES 
(
    'tp_default_v1',
    'Default Threshold Profile v1',
    'Default quality gates and confidence thresholds for Phase 0',
    '{
        "domaining": {
            "tau_high": 0.82,
            "tau_low": 0.55
        },
        "mapping": {
            "accept_rate_min": 0.95,
            "near_match_window": 0.03
        },
        "analysis": {
            "null_rate_max": 0.02,
            "dup_key_rate_max": 0.00
        },
        "migration": {
            "batch_size": 50000,
            "validation_checks": ["row_count", "checksum", "fk_check"]
        },
        "budgets": {
            "ingest_pct": 0.15,
            "domains_pct": 0.25,
            "mapping_pct": 0.35,
            "analysis_pct": 0.15,
            "migration_pct": 0.10
        }
    }'
);

-- Insert default policy manifest
INSERT INTO policy_manifests (version, manifest, is_active, activated_at) VALUES 
(
    '0.1.0',
    '{
        "pii_masking_enabled": true,
        "pii_local_only_override": false,
        "pii_detection_threshold": 0.8,
        "max_daily_cost_usd": 100.0,
        "max_tokens_per_operation": 50000,
        "cost_alert_threshold": 0.8,
        "allowed_providers": ["anthropic", "openai", "groq", "ollama"],
        "denied_providers": [],
        "require_local_for_pii": true,
        "allow_data_mode": true,
        "require_approval_for_data_mode": false,
        "viewer_can_see_pii": false,
        "analyst_can_approve_operations": true,
        "operator_can_override_local_only": false
    }',
    true,
    NOW()
);

-- Insert some basic domain definitions for Phase 0
INSERT INTO domains_live (domain_id, name, category, description, aliases, regex_patterns, examples) VALUES 
(
    'person.first_name',
    'First Name',
    'person',
    'Person''s given/first name',
    ARRAY['first_name', 'fname', 'given_name', 'forename'],
    ARRAY['^(first|given|fore)_?name$'],
    '[
        {"type": "positive", "name": "first_name", "value": "John"},
        {"type": "positive", "name": "fname", "value": "Sarah"},
        {"type": "positive", "name": "given_name", "value": "Michael"}
    ]'
),
(
    'person.last_name', 
    'Last Name',
    'person',
    'Person''s family/surname',
    ARRAY['last_name', 'lname', 'surname', 'family_name'],
    ARRAY['^(last|sur|family)_?name$'],
    '[
        {"type": "positive", "name": "last_name", "value": "Smith"},
        {"type": "positive", "name": "surname", "value": "Johnson"},
        {"type": "positive", "name": "family_name", "value": "Brown"}
    ]'
),
(
    'contact.email',
    'Email Address', 
    'contact',
    'Electronic mail address',
    ARRAY['email', 'email_address', 'mail'],
    ARRAY['^.*mail.*$', '^.*@.*$'],
    '[
        {"type": "positive", "name": "email", "value": "user@example.com"},
        {"type": "positive", "name": "email_address", "value": "test@domain.org"},
        {"type": "negative", "name": "mail_count", "value": "5"}
    ]'
);

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'EnMapper database schema initialized successfully';
    RAISE NOTICE 'Created % tables with initial data', (
        SELECT count(*) 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name IN ('runs', 'stage_executions', 'domains_live', 'domains_staged', 'domain_examples', 'threshold_profiles', 'policy_manifests', 'supervisor_ledger')
    );
END $$;
