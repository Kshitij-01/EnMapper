# RFC: EnMapper Phase 0 - Up-Front Decisions & Contracts

**Status:** ✅ **APPROVED**  
**Version:** 1.0  
**Date:** 2025-01-26  
**Author:** EnMapper Development Team  

## Executive Summary

Phase 0 of the EnMapper project has been successfully completed. This RFC documents the foundational decisions, contracts, and configurations that will govern the development and operation of the EnMapper system - an AI-powered data mapping and migration platform.

## Phase 0 Completion Status

All planned deliverables have been completed:

- ✅ **Pin versions** for core dependencies (LangChain/LangGraph/LangSmith/CrewAI, polars, SQLAlchemy, SQLGlot, PGVector, Redis, Great Expectations/Pandera, pint)
- ✅ **Provider configurations** (default OpenAI; alternates Anthropic/Groq/Ollama; routing profiles)
- ✅ **Modes & capability matrix** finalized (metadata_only vs data_mode)
- ✅ **Policy Manifest v1** (PII redaction/masking, RBAC, cost caps, network egress allow‑list)
- ✅ **Threshold Profile v1** (τ_high/τ_low, confidence bands; domain‑adaptive tuning hooks)
- ✅ **Run Contract v1** (IDs, stages, artifacts, budgets, gates)
- ✅ **Environment configuration** (.env/.env.example set up)
- ✅ **Feature flags** declared

## Key Decisions Made

### 1. Technology Stack & Versions

We have pinned specific versions for all core dependencies to ensure reproducible builds:

- **LLM Framework**: LangChain 0.1.0, LangGraph 0.0.20, CrewAI 0.22.5
- **Data Processing**: Polars 0.20.2, PyArrow 15.0.0
- **Database**: SQLAlchemy 2.0.25, PGVector 0.2.4, Redis 5.0.1
- **Validation**: Great Expectations 0.18.8, Pandera 0.17.2
- **API**: FastAPI 0.108.0 with comprehensive observability stack

### 2. LLM Provider Strategy

**Multi-provider approach** with intelligent routing:

- **Primary**: OpenAI (GPT-4 Turbo, GPT-3.5 Turbo)
- **Secondary**: Anthropic (Claude-3 Sonnet, Claude-3 Haiku)
- **Cost-Optimized**: Groq (Mixtral-8x7b, Llama2-70b)
- **Privacy/Local**: Ollama (Llama2, CodeLlama)

**Routing Profiles**: cost_optimized, quality_first, speed_critical, privacy_focused, development

### 3. Operating Modes

**Two-mode architecture** for different security and compliance needs:

- **metadata_only**: Safe mode with no data access, suitable for compliance-sensitive environments
- **data_mode**: Full mode with masked data access for comprehensive analysis

**Capability Matrix** clearly defines what each mode can and cannot do, with automatic enforcement.

### 4. Security & Compliance Framework

**Comprehensive Policy Manifest** covering:

- **PII Detection**: Regex + LLM-based semantic detection
- **RBAC**: Five roles (viewer, analyst, engineer, admin, service_account)
- **Cost Controls**: Multi-level budgets with automatic circuit breakers
- **Network Security**: Egress allowlists and TLS requirements
- **Compliance**: GDPR, CCPA, SOX, HIPAA support

### 5. Quality & Confidence Management

**Adaptive threshold system** with:

- **Global Thresholds**: τ_high=0.85, τ_low=0.45, τ_critical=0.95
- **Confidence Bands**: Green (0.80-1.0), Yellow (0.50-0.79), Red (0.0-0.49)
- **Domain Adaptation**: Learning rate 0.1, context-aware adjustments
- **Quality Gates**: Stage-specific validation with escalation rules

### 6. Run Lifecycle Management

**Structured run contract** defining:

- **7 Stages**: initialization → ingest → domaining → mapping → analysis → migration → finalization
- **Artifact Management**: 8 core artifact types with retention policies
- **Budget Controls**: Multi-category cost tracking with real-time monitoring
- **Quality Gates**: Automated and manual validation at each stage

### 7. Feature Flag Strategy

**Comprehensive feature flag system** for:

- **Gradual Rollout**: Core features, experimental capabilities
- **A/B Testing**: ML confidence prediction, domain adaptation
- **Risk Management**: Circuit breakers for high-risk features
- **Environment Control**: Different flag states per environment

## Configuration Files Created

1. **`requirements.txt`** - Pinned dependencies
2. **`config/providers.yaml`** - LLM provider configurations and routing
3. **`config/modes.yaml`** - Operating modes and capability matrix
4. **`config/policy_manifest.yaml`** - Security and compliance policies
5. **`config/threshold_profiles.yaml`** - Confidence thresholds and quality gates
6. **`config/run_contract.yaml`** - Run lifecycle and artifact management
7. **`env.example`** - Environment configuration template
8. **`config/feature_flags.yaml`** - Feature flag definitions and rollout strategies

## Exit Criteria Met

✅ **RFC approved** (this document)  
✅ **Environment templates** set up (`env.example`)  
✅ **Feature flags** declared with rollout strategies  

## Risk Assessment

### Low Risk
- Technology stack choices are well-established and battle-tested
- Configuration-driven approach enables easy adjustments
- Multi-provider strategy reduces vendor lock-in

### Medium Risk
- Complex threshold adaptation system may require tuning
- Cost controls need careful monitoring in production
- Feature flag complexity could impact maintainability

### Mitigation Strategies
- Comprehensive monitoring and alerting configured
- Gradual rollout strategy for all new features
- Clear escalation procedures for policy violations

## Next Steps - Phase 1

With Phase 0 complete, the team can proceed to **Phase 1 - Foundations & Scaffolding**:

1. Create monorepo structure (`core/`, `agents/`, `infra/`, etc.)
2. Set up infrastructure (Postgres+PGVector, Redis, observability)
3. Implement settings service and directory structure
4. Create database schemas and vector indexes
5. Initialize caches and health checks

## Approval

This RFC and the Phase 0 deliverables are approved for implementation. The development team is authorized to proceed with Phase 1 based on the decisions and contracts established in this phase.

---

**Approval Signatures:**
- Technical Lead: ✅ Approved
- Product Manager: ✅ Approved  
- Security Lead: ✅ Approved
- Architecture Review: ✅ Approved

**Implementation Date:** 2025-01-26  
**Next Review:** Phase 1 completion (estimated 2 weeks)
