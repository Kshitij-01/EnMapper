# ROADMAP.md — Unified Architecture & Core Logic (LLM Agent, Provider-Agnostic)

**Version:** v1.1.0  
**Last updated:** 2025-08-25

This is the single source of truth. Implement this architecture exactly. The **LLM Agent** (and any tooling) must not change the structure, modes, gating, roles, policies, or DSL without a versioned change to this file.

---

## 0) Design Principles
- Deterministic, explainable pipelines governed by a **Run Contract**, **Policy Manifest**, and **Threshold Profiles**.
- Two first-class modes: **metadata_only** and **data_mode**, enforced end-to-end by a **capability matrix**.
- **Supervisor-led orchestration** over a directed graph of specialist nodes with stage gates and explicit human confirmations.
- **Provider-agnostic inference**: local (Ollama) and cloud (OpenAI, Anthropic, Groq, etc.) via one uniform adapter and router.
- **Declarative, whitelisted Transform DSL** compiled to SQL (ELT) or Polars (ETL). No arbitrary runtime code.
- **Privacy by default**: PII redaction/masking before any external model call; local-only override is policy-gated and audit-logged.
- **Observability everywhere**: traces, metrics, cost, policy decisions, and a **Supervisor Ledger** are durable artifacts.
- **Cost awareness & routing**: quality-first default with budgets, early exits, caching, and fallback chains.
- **Idempotent, checkpointed migrations** with quarantine and atomic promotion.
- **Single Policy Engine** used by Core, API, and GUI to prevent drift.

## 1) Non-Goals
- No arbitrary Python/SQL beyond the approved Transform DSL.
- No silent policy bypass; all overrides are RBAC-gated and logged.
- No provider-specific logic in business code; differences live only in adapters and routing.

## 2) Glossary
- **Run**: One end-to-end job tracked by a Run Contract.
- **Lane**: Processing class selected by Supervisor: Interactive, Flex, Batch.
- **Supervisor**: Orchestrator that validates contracts, routes lanes, enforces gates, records decisions.
- **Specialists**: Ingest, Domains, Mapper, Analyst, Migrator nodes.
- **Domain**: Semantic type (e.g., person.first_name, contact.email, money.amount).
- **Transform DSL**: Declarative list of whitelisted operations for value shaping.
- **Artifacts**: Versioned JSON objects: Catalog, Domain Assignments, Mapping Plan, Analysis Report, Migration Run, Threshold Profiles, Policy Manifest, Supervisor Ledger.
- **Context Capsule**: GUI scope (run, tab, mode, PII flag, filters, artifact versions) that also informs cache keys.

## 3) Modes & Capability Matrix

| Capability | metadata_only | data_mode |
|---|---|---|
| Catalog inference | Yes | Yes |
| Row samples | No | Yes (masked, capped) |
| Domaining evidence | Headers/regex/units | Headers/regex/units + tiny value sketches |
| Transform previews | Symbolic/dry-run | Sample execution |
| Analysis checks | Logical/feasibility | Coverage/nulls/dups + stats |
| Migration | Dry-run planning | Full execution with checkpoints |
| GUI previews | Masked schema/IDs | Masked slices + diffs |

**Mode is immutable during a run.** Switching modes requires a new Run Contract version.

## 4) Processing Lanes & Supervisor Routing
- **Interactive**: GUI prompt, small catalog ops. Hard latency SLO. No queueing beyond debounce.
- **Flex**: Non-urgent interactive jobs (large previews, ad-hoc analysis). Variable latency. Cost-optimized.
- **Batch**: Large offline workloads (migration chunks, dataset analysis, golden-set evals). Deterministic, checkpointed, auditable.

**Routing**: Supervisor chooses a lane from job type, size estimate, data sensitivity, user role, cost caps, provider health, queue depth. Lane recorded in the Ledger, included in cache keys, and visible in the Context Capsule. Lane changes invalidate caches and re-evaluate policy.

## 5) Orchestration Graph (Core Logic)
**Supervisor responsibilities**
- Validate Run Contract, choose lane, allocate budgets.
- Route to specialists and enforce stage gates with accept/hold/fix decisions.
- Manage retries, provider fallback, escalation to human confirmation.
- Persist decisions in a **Supervisor Ledger**.

**Specialist nodes**
- **LLM-INGEST (Files)**: inventory, dialect sniff, schema inference; emits **Standard Schema Catalog**.
- **LLM-INGEST (SQL)**: SQLAlchemy + SQLGlot read-only introspection; data-mode sampling under caps.
- **LLM-D (Domains)**: domaining via RAG + vector similarity; scoring, confidence banding, arbitration; emits **Domain Assignments**.
- **LLM-M (Mapper)**: proposes Transform DSL pipelines per target; policy validation; optional sample execution; emits **Mapping Plan**.
- **LLM-A (Analyst)**: mode-aware checks; emits **Analysis Report** with conclusions and recommended fixes.
- **LLM-G (Migrator)**: ELT vs ETL selection; batch execution with checkpoints, idempotency, quarantine; emits **Migration Run**.

**Stage order (required)**: Ingest → Domains → Mapping → Analysis → (if accepted & data_mode) Migration.  
No external provider call occurs before **Policy Engine** approval and confirmed PII masking.

## 6) Policy Engine
- Enforces mode, PII masking/redaction, RBAC, cost caps, provider allow/deny lists.
- Evaluates requests pre-send and annotates results for audits.
- Versioned Policy Manifest; artifacts record `policy_version`.
- Audit fields include: `pii_mask_applied`, `local_only_override`, `provider`, `model`, `prompt_hash`, `catalog_version`.
- Network egress allow-list to provider endpoints; deny by default.

## 7) Storage Layer
**Postgres (+PGVector)**
- Tables: `runs`, `prompts`, `domains_live`, `domains_staged`, `domain_examples`, `mapping_plans`, `analysis_reports`, `migration_runs`, `quarantine_rows`, `threshold_profiles`, `supervisor_ledger`.
- PGVector stores centroids for name/value descriptors (domaining).

**Redis**
- TTL caches for completions, embeddings, retrieval sets; idempotency keys; checkpoint markers.

**Artifacts**
- Stored with content hashes; DB rows reference hashes for integrity and reproducibility.

## 8) Domain Catalog
- Atomic and composite domains with aliases, negatives, regex, units; 10–20 curated examples each.
- Vector centroids for name and value descriptors.
- **Promotion workflow**: staged authoring → auto-tests → version bump with provenance → promote to live → invalidate domain-dependent caches.
- **Feedback loop**: accepted/overridden mappings propose additions to `domains_staged` (aliases/regex).

## 9) Ingest → Standard Schema Catalog
- Files: inventory, encoding/dialect sniff, schema inference (Polars/PyArrow).
- SQL: SQLAlchemy + SQLGlot; read-only introspection; data-mode sampling under strict caps; dialect-portable queries.
- Catalog always includes tables, columns, dtypes, semantics. In data_mode, a masked and capped **Sample Pack** is attached for previews.
- **Sampling policy (deterministic)**: stratified by key/date where available; fixed seed; caps by `{rows_max, bytes_max}`; freshness window for sample caches.

## 10) Domaining Logic
- **Evidence extraction**: header tokens, regex hits, unit cues; limited value sketches in data_mode when allowed.
- **Candidate generation**: RAG over Domain Catalog plus vector similarity on names and values.
- **Scoring**: `alpha·name_sim + beta·regex_strength + gamma·value_sim + epsilon·unit_compat`.
- **Confidence bands via Threshold Profile**: high ≥ τ_high; low < τ_low; otherwise borderline.

**Arbitration**
- High → assign with rationale.
- Borderline → discovery prompts and staged catalog deltas.
- Low → unknown; escalate selectively.

All assignments persist evidence and version references.

## 11) Transform DSL (Whitelisted)
**Ops (ordered, declarative)**: `trim`, `case`, `collapse_whitespace`, `regex_extract`, `regex_split`, `concat`, `map_dict`, `unit_convert`, `date_parse`, `date_format`, `type_cast`, `safe_math`

**Compilation targets**: SQL (ELT) via SQLGlot; Polars expressions (ETL).

**Determinism**
- Regex flavor fixed and documented; timezone and locale fixed; NaN/NaT semantics stable across backends; type coercions explicitly defined.
- Any out-of-policy op is rejected with a Supervisor-logged decision.

## 12) Mapping Logic
- **Strategy enumeration per target**: atomic→atomic; compose (e.g., first+last → full_name); decompose (e.g., full_name → first,last); unit normalization; type alignment.

**Flow**
- LLM-M proposes one or more Transform DSL pipelines with confidence and rationale.
- Supervisor validates against DSL and policy.
- In data_mode, sample execution computes success/near-match rates and constraint checks.
- Decision per mapping: accept/hold/fix using thresholds; Mapping Plan persists pipelines, confidences, examples, decisions.

## 13) Analysis Logic
- **metadata_only**: logical consistency, domain coverage, transform feasibility matrix.
- **data_mode**: coverage, nulls, duplicates; unit/date sanity; optional PSI/KS; cross-field rules.
- Output: anomalies, recommended fixes, suggested threshold adjustments (reliability curves).

## 14) Migration Logic
- **Pre-flight**: accepted mappings only; key strategy (natural, composite, surrogate).
- **Strategy**: ELT (in-DB SQL) when expressible; ETL (Polars batch) otherwise.

**Batching & idempotency**
- Deterministic chunking by key or hash.
- Content hashes per chunk; idempotent writes.
- Checkpoints after each batch with validation outcomes.
- Quarantine failed records with masked previews and diffs.
- Atomic promotion from staging to final when validations pass.

## 15) Multi-Provider Inference Layer (Ollama + Cloud)
- Uniform adapter interface: `infer`, `embed`, `token_estimate`, `health`.
- **Model Registry**: tags for quality, latency, cost, context length, modality.
- **Routing profiles**: `quality_first` (default), `cost_first`, `latency_first`, `offline_first`.
- **Fallbacks**: retry/backoff; fail-open only within policy caps and lane budgets.
- Cache keys include provider, model, parameters, policy version, artifact hashes, lane, mode, PII flag, context scope.
- **PII handling**: redaction/masking before any external call. Local-only override (e.g., Ollama) is policy-gated and logged.
- **Provider conformance**: normalization matrix for temperature, top_p, stop tokens, truncation; identical error taxonomy promised by adapters.

**Ollama compatibility**
- Local HTTP streaming via adapter; configurable concurrency; GPU/CPU selection by Ollama config.
- Flex/Batch realized by our scheduler queues (Ollama is single-tenant; no native flex/batch).
- Prefer Ollama for sensitive workloads; fall back to cloud if capacity constrained.

**Cloud mapping**
- Interactive → on-demand APIs.
- Flex → cloud “flex” scheduling when available; otherwise background worker pools.
- Batch → cloud batch APIs (offline) or local batch workers for Ollama.

**Contract**: Swapping providers/models must not change output schema or error taxonomy. Pre-send cost estimates are visible in GUI.

## 16) Queue & Worker Topology
- **Queues (Redis)**: `interactive_fast`, `flex_bg`, `batch_jobs`.
- **Workers**: per-lane pools with max concurrency and per-provider quotas.
- **Scheduler**: prioritizes queues, respects cost caps and provider health, executes checkpoint logic.

**Checkpointing & idempotency**
- Batch workers create checkpoints per chunk; safe resume on failure.
- Content hashes and Run IDs prevent duplicate materialization.
- Quarantine lane stores failed items and preview diffs.
- Provider outage triggers automatic fallback without violating policy or budgets. Interactive SLO holds under spike.

## 17) API Surface (Conceptual)
- Endpoints (illustrative): `Create Run`, `Get Run`, `Stop Run`, `Get Artifacts`, `Create Supervisor Ticket`, `Policy Check (dry-run)`, `Threshold Profiles (get/tune)`, `Cost Estimate (pre-send)`.
- Uniform error taxonomy and rate limits; every response includes artifact and policy version hashes used and strong ETag.

## 18) GUI Prompt & Actions
- **Prompt panel**: floating dock / sticky footer with command palette (always available to the user).
- **Context Capsule chips**: Run, Tab, Mode, PII, Lane, Threshold Profile (editable scope).
- **Input**: templates with variables; slash commands like `/explain`, `/gen-rule`, `/preview`, `/cost`, `/thresholds`; attach masked snapshots.
- **Output**: streaming with live token/cost; confidence bands; diffs; actions (Preview on Sample, Apply Transform, Create Staged Alias, Open Trace).
- **Confirm-to-Apply**: two-step apply → Supervisor gate → staged delta → ledger log. Undo for non-destructive ops; destructive ops require Admin approval (RBAC).
- Accessibility: keyboard shortcuts; WCAG AA compliance.

## 19) Caching Policy
- **What**: prompt→completion outputs, embeddings, retrieval results, schema inference for files.
- **Key**: provider, model, parameters, system/user prompt hashes, lane, mode, PII flag, context scope (run/tab/filters), artifact version hashes (catalog/mappings/thresholds/policy), locale.
- **Invalidation**: mode/PII toggle; artifact or policy version change; provider/model/param change; explicit `fresh=true`.
- **Storage**: Redis TTL for hot entries; optional on-disk for deterministic preprocessors.
- **Mitigations**: negative-cache TTLs; per-key and per-scope size limits; no-cache contexts for PII-heavy prompts.

## 20) Security & Compliance
- Read-only SQL; schema allow-lists; sample caps.
- Secrets via secret manager or Docker/K8s secrets; outbound egress allow-list to LLM endpoints.
- PII decision logs as structured fields (`masked_before_send`, `local_only_override`).
- Every artifact includes provider, model, prompt hash, policy version, catalog version.

## 21) Observability & Cost
- Every LLM call traced (LangSmith or equivalent) with run linkage.
- OpenTelemetry metrics to Prometheus/Grafana for latency, error, and cost per stage.
- Sentry (or equivalent) for exceptions with run_id correlation.
- Golden-set evals for domaining/mapping; nightly canary across providers.
- Reliability curves per domain inform threshold calibration (domain-adaptive tuning).

## 22) Acceptance & Test Matrix
- **Functional**: E2E run in both modes on one LHS/RHS table; all artifacts produced; gates enforced. Supervisor routing deterministically assigns and records lanes. GUI actions stage/apply safely; undo works for safe ops.
- **Cross-provider**: Interactive, Flex, Batch exercised across OpenAI, Anthropic, Groq, and Ollama. Redaction on/off yields expected masking; no PII egress unless policy allows. Failover drills for provider outage, quota exhaustion, cost cap breaches.
- **Performance**: Interactive SLO holds under spike. Batch completes with checkpoints; restart recovers mid-run without duplication.
- **Cost**: Token budgets enforced per stage; early exit on overages; caching effective.
- **Conformance**: Adapters pass a suite ensuring identical error taxonomy and parameter normalization across providers.

## 23) Runbooks
- **How to run metadata_only**: required inputs/outputs, artifacts, expected stage gates.
- **How to run data_mode**: sampling policy, thresholds, previews, validations.
- **Remediation cookbook**: name splits, date formats, unit conflicts, dialect quirks.
- **Promotion guide**: staged→live domain promotions with auto-tests and manual approval.

## 24) Packaging & Runtime
- Docker images for API/Worker (CPU) and optional GPU image for local models.
- Compose (dev): Postgres (+PGVector), Redis, tracing stack, core service.
- Health/readiness endpoints; startup warmers (load catalog, prime caches).
- Kubernetes plan: HPA on queue depth/CPU, NetworkPolicies, PodSecurityContext.

## 25) RACI
- **Core Orchestration (LangGraph/Supervisor)**: Backend (R), Core Lead (A)
- **Domain Catalog & Evals**: Data/ML (R), ML Lead (A)
- **Policy Engine & Security**: Platform (R), Security Lead (A)
- **API Contracts**: Backend (R), Platform Lead (A)
- **GUI Prompt & Actions**: Frontend (R), Product Lead (A)
- **Observability/Cost Dashboards**: Platform (R), All Teams (C)

## 26) Now / Next / Later
- **Now (Weeks 1–2)**: Decisions, scaffolding, DB/caches, Policy Engine skeleton, Run Contract v1; minimal API for `create_run` and `policy_check`.
- **Next (Weeks 3–6)**: Domain Catalog v1; Ingest paths; Graph Supervisor + Domains/Mapping; Mapping Plan v1; GUI dock + context chips + templates; confirm-to-apply (staged).
- **Later (Weeks 7–10)**: Analysis checks; Migration with checkpoints/quarantine; cost dashboards; golden-set evals; GUI confidence bands, diffs, trace links.

**v1 Exit**: E2E in both modes for one LHS/RHS table; Supervisor gates enforced; GUI actions stage/apply safely; traces visible; domain promotion works; migration completes with idempotency, quarantine, and atomic promotion.

## 27) Versioning & Compatibility (Normative)
- Roadmap versioning: **SemVer** MAJOR.MINOR.PATCH in the header. Any change to roles, lanes, DSL ops, stage order, or policy checkpoints = **MAJOR**.
- Artifact compatibility: `policy_version`, `dsl_version`, `catalog_version`, and `threshold_profile_id` are immutable within a run. Cross-version use requires an explicit promotion step.
- Deprecation: MINOR adds allowed with a 2-version deprecation window and a migration guide.
- Hashing: All artifacts stored with content SHA-256; references always use the hash + version tuple.

## 28) Run Contract (Normative Keys)
| Key | Type | Required | Notes |
|---|---|---|---|
| run_id | string (ULID) | Yes | Deterministic across retries |
| roadmap_version | semver | Yes | Locks structure |
| mode | enum{metadata_only,data_mode} | Yes | Immutable once run starts |
| lane_hint | enum{interactive,flex,batch} | No | Supervisor may override; override recorded |
| lhs | object | Yes | `{source:{type:file/sql}, location, schema_ref}` |
| rhs | object | Yes | Same shape as lhs |
| pii_policy | object | Yes | `{mask_before_send:true, local_only_override:false}` |
| budget_caps | object | Yes | `{tokens, usd, wall_time_s}` per stage |
| routing_profile | enum{quality_first,cost_first,latency_first,offline_first} | Yes | Default quality_first |
| provider_allowlist | [string] | No | Model/provider tags |
| provider_denylist | [string] | No | Model/provider tags |
| threshold_profile_id | string | Yes | Links to Threshold Profiles |
| dsl_version | semver | Yes | Transform DSL version |
| sample_caps | object | Yes (data_mode) | `{rows_max, bytes_max}` |
| cache_scope | object | Yes | `{run, tab, filters}` |
| human_confirm_required | bool | Yes | Gates destructive/apply steps |
| rbac_actor | object | Yes | `{user_id, role}` |
| audit_tags | [string] | No | Freeform labels |

**Contract hash**: SHA-256 over canonical JSON with sorted keys; stored on the Ledger and echoed in all artifacts.

## 29) Threshold Profiles (Normative)
| Field | Type | Example | Purpose |
|---|---|---|---|
| profile_id | string | tp_default_v1 | Stable identifier |
| domaining.tau_high | float | 0.82 | ≥ assigns automatically |
| domaining.tau_low | float | 0.55 | < becomes unknown |
| mapping.accept_rate_min | float | 0.95 | Sample success rate to accept |
| mapping.near_match_window | float | 0.03 | Tolerance band for “near” |
| analysis.null_rate_max | float | 0.02 | Fails if exceeded |
| analysis.dup_key_rate_max | float | 0.00 | No dup keys |
| migration.batch_size | int | 50000 | Deterministic chunk size |
| migration.validation_checks | [string] | row_count, checksum, fk_check | Required validations |
| budgets.domaining.tokens | int | 150000 | Hard cap per stage |
| budgets.mapping.tokens | int | 250000 | Hard cap per stage |

## 30) Supervisor Ledger: Event Schema
- `event_id` (ULID), `run_id` (ULID), `timestamp_utc`, `stage` {INGEST,DOMAINS,MAPPING,ANALYSIS,MIGRATION,SUPERVISOR}, `decision` {accept,hold,fix,reject,retry,escalate}, `inputs_hash` (sha256), `outputs_hash` (sha256), `provider`, `model`, `params_hash`, `token_estimate`, `token_actual`, `usd_estimate`, `usd_actual`, `retries`, `fallback_chain` ([provider:model]), `pii_mask_applied` (bool), `local_only_override` (bool), `policy_version`, `threshold_profile_id`, `dsl_version`, `error_code`, `error_detail_id` (trace link), `human_confirmation` {required, actor, timestamp}.
- Events are append-only. A per-run **Decision Summary** is materialized for fast GUI reads.

## 31) Error Taxonomy & Retry Classes
| Code | Class | Retry | Fallback | Notes |
|---|---|---|---|---|
| POLICY_BLOCK | deterministic | No | No | Violates policy/PII/RBAC |
| DSL_REJECT | deterministic | No | No | Out-of-policy op |
| PROVIDER_UNHEALTHY | transient | Yes | Yes | Health probe failed |
| RATE_LIMIT | transient | Yes (backoff) | Yes | Respect headers |
| TIMEOUT | transient | Yes | Yes | Lane-specific deadlines |
| QUOTA_EXHAUSTED | transient | Yes (delay) | Yes | May switch lane/provider |
| TRANSIENT_ERROR | transient | Yes | Yes | Generic 5xx |
| DETERMINISTIC_FAILURE | deterministic | No | No | Bad input/contract |
| QUARANTINE_REQUIRED | deterministic | No | N/A | Migration validation fail |

## 32) RBAC Roles & Permissions Matrix
| Action | Viewer | Analyst | Operator | Maintainer | Admin |
|---|:---:|:---:|:---:|:---:|:---:|
| View artifacts (masked) | ✓ | ✓ | ✓ | ✓ | ✓ |
| View PII (unmasked) |  | ✓ | ✓ | ✓ | ✓ |
| Create run |  | ✓ | ✓ | ✓ | ✓ |
| Approve destructive ops |  |  | ✓ | ✓ | ✓ |
| Toggle local_only_override |  |  |  | ✓ | ✓ |
| Edit Policy Manifest |  |  |  | ✓ | ✓ |
| Domain promotion (staged→live) |  | ✓ | ✓ | ✓ | ✓ |
| Release quarantine rows |  |  | ✓ | ✓ | ✓ |
| Manage provider registry |  |  |  | ✓ | ✓ |

## 33) Data Residency, Retention, and Encryption
- **Residency**: Region-pinned Postgres volumes; no cross-region replication by default.
- **Encryption**: TLS 1.3 in flight; AES-256 at rest; key rotation every 90 days; secrets via Docker/K8s secrets.
- **Retention (defaults)**: artifacts: 365d, supervisor_ledger: 365d, quarantine_rows: 30d, traces: 90d, redis caches: ≤ 7d.
- **PITR**: Postgres WAL archived for 7 days; documented restore procedure.
- **Egress allow-list**: Only LLM endpoints in Provider Registry; all calls recorded with prompt hash and masking flags.

## 34) Mapping Upsert & Referential Policy
- **Keys**: Natural/composite preferred; surrogate keys only when target requires them (recorded in Mapping Plan).
- **UPSERT**: ELT uses `ON CONFLICT DO UPDATE`; ETL emulates via hash-partitioned merge.
- **Referential checks**: Validate FK integrity in staging before promotion; violations go to quarantine with masked diffs.
- **SCD**: Optional SCD-2 supported via DSL primitives and a `valid_from`/`valid_to` convention.

## 35) Transform DSL Extensions & Guards
- **Frozen v1 ops**: Exactly as in §11. No user-defined functions.
- **Guardrail (planned v1.1 behind feature flag)**: `select_when(condition=regex/unit/domain_hit)` as a non-Turing conditional.
- **Serialization**: DSL expressions are content-hashed; Mapping Plans must include the declarative plan and compiled SQL/Polars previews.

## 36) Cache Defaults & Poisoning Mitigation
- **Defaults**: TTL = 24h (Interactive), 72h (Flex), 7d (Batch); max entry 2 MB; LFU eviction.
- **Key salting**: Include `policy_version` and `artifact_hashes` to avoid cross-context reuse.
- **Bypass**: `fresh=true` forces provider call and re-materializes cache.
- **Negative caching**: bounded TTL for `404/empty` results to avoid thundering herds.

## 37) Checkpoints, Content Hashes & Quarantine Records
- **Chunk content hash**: SHA-256 over ordered row keys + selected columns post-transform.
- **Checkpoint record**: `{run_id, stage, chunk_id, content_hash, started_at, finished_at, row_count_in, row_count_out, validations, status, next_action}`.
- **Quarantine record**: `{run_id, chunk_id, key, reason_codes, masked_preview_before, masked_preview_after, diff_ptr, created_at, rbac_required_role}`.

## 38) Health/Readiness & SLOs (Signals)
- Health: liveness/readiness endpoints expose DB, Redis, provider health, queue depth.
- Degradation policy: Automatic provider fallback when health < threshold; preserve Interactive SLO while Flex/Batch drain later.

---

# Addenda (v1.1)

## 39) Threat Model & Controls
- **Scope**: Data ingestion, storage, inference egress, artifacts, GUI.
- **Risks (STRIDE)**: spoofed providers, tampered artifacts, data exfiltration, PII leakage, denial of service, elevation of privilege.
- **Controls**: network egress allow-list; MTLS to internal services; secrets rotation; SBOM + dependency scan gate; signed images and attestations (Sigstore); supply-chain scan in CI; optional WORM storage for Ledger; k-anonymity/ℓ-diversity targets for previews; PII detector recall target ≥ 0.98 on golden set.

## 40) Tenancy & Isolation
- **Boundary**: tenant_id at DB schema level by default; optional DB-per-tenant or cluster-per-tenant for high isolation.
- **Per-tenant**: keys, policies, artifacts, quotas, cost caps, and rate limits.
- **Cross-tenant**: hard deny; no cross-tenant joins; metrics and traces partitioned by tenant.

## 41) Lineage & Provenance
- **Dataset IDs** with content + schema hashes.
- **Snapshots**: source→snapshot→staging→final; lineage links between versions.
- **Column-level lineage** recorded from Mapping Plan; GUI provenance drawer for hover-to-explain.
- **Deterministic sampling**: fixed seed and seed logged in artifacts.

## 42) Dataset & Schema Versioning
- **Schema diffs**: additive vs breaking; mapping re-validation on breaking changes.
- **Snapshot naming**: timestamped + ULID; retention configurable; reproducible sample packs stored with seed and caps.

## 43) Evaluation & Golden Sets
- **Metrics**: precision/recall/F1 for domaining/mapping; Brier/calibration; acceptance thresholds per domain.
- **Golden sets**: curated examples per domain and mapping; nightly evals across providers; dashboards compare drift.

## 44) Concurrency & Locks
- Advisory locks/leases per table/key-space; chunk ownership tokens; idempotency keys scope = `{run_id, chunk_id}`.
- Exactly-once semantics via checkpoints + content hashes; duplicate suppression across worker restarts.

## 45) Rollback & Promotions
- **Rollback matrix**: catalog, mappings, policy, thresholds—what can be reverted and under which conditions.
- **Pre-promote checks**: validations, canary run, error budget headroom.
- **Time-boxed rollbacks** with automated artifact re-linking and cache invalidation.

## 46) SLOs & Alerting
- **Interactive**: p95 ≤ 2.0s small prompts; p99 ≤ 3.5s; error rate ≤ 0.5%. Alert on 5-min breach.
- **Flex**: queue wait p95 ≤ 3m; job success ≥ 99%; auto-throttle on provider health.
- **Batch**: ≥ 1M rows/hour/node sustained; checkpoint MTTR ≤ 10m; recovery success ≥ 99.5%.
- **PII masking**: detector recall ≥ 0.98 on golden set; precision ≥ 0.97.
- **Alerting**: on-call rotation; paging via error budget burn rate; runbooks linked from alerts.

## 47) Data Subject Requests (DSR)
- **Erase**: tombstone + redaction markers preserving audit integrity; propagate to artifacts and invalidate caches.
- **Export**: per-tenant data bundle with lineage and hashes; SLA documented.

## 48) Licensing & Compliance
- **Provider/model/data license tables** and export-control flags.
- **Jurisdiction toggles** in Policy Engine for data residency and retention.
- **Framework mapping**: SOC2/ISO27001 control checklist with ownership and audit pointers.

---

## Directive for the LLM Agent and all contributors
Do **not** change roles, lanes, Transform DSL, stage order, or policy checkpoints. Do **not** insert provider-specific logic outside adapters. All deviations require a versioned update to this file. The jump from v1.0.0 → **v1.1.0** is a MINOR bump adding Addenda (§§39–48) and clarifications without breaking normative contracts.



---

# Phased Implementation Plan (LLM Agent)

> UI builds in **parallel** with every phase **except Phase 0 (setup-only)**. Each phase ends with a runnable demo, clear artifacts, and acceptance tests. The first UI screen includes **CSV upload** or **SQL credential connect**; once submitted, the **LLM Agent** takes over to spin up code-execution steps and standardize data **before** the next stage.

## Phase 0 — Foundation & Setup (Setup-only)
**Objectives**
- Stand up core infra, repos, CI/CD, secrets, and observability. No end-user UI beyond a minimal shell.

**Core Build**
- Repos & codeowners; commit hooks; SemVer release workflow.
- Docker images (API/Worker CPU; optional GPU for local models); Compose stack (Postgres+PGVector, Redis, tracing, API/Worker).
- Policy Engine skeleton + Run Contract v1; Model Registry + adapters skeleton (Ollama + one cloud provider).
- Logging/tracing (OTel) + Sentry; Prometheus/Grafana baseline dashboards.
- Postgres schemas/tables + migrations; Redis namespaces and key-conventions.

**Artifacts**: empty Domain Catalog (staged), Policy Manifest v0, Threshold Profile `tp_default_v1`, health/readiness endpoints.

**DoD**: `docker compose up` yields healthy services; `/healthz` green; seed data loads; basic traces visible.

---

## Phase 1 — Intake & Standardization Shim (First-Run Wizard & Catalog)
**Objectives**
- Deliver the **first page experience**: upload CSVs or connect SQL; create a Run; standardize raw inputs to a **Standard Schema Catalog** plus a deterministic **Sample Pack** (masked).

**UI Track (parallel)**
- **Start a Run** screen: two cards → **Upload Files (CSV/TSV/Parquet)** drag-drop with validation; **Connect SQL** form (host, db, RO user, auth, SSL, allow-list note).  
- Mode toggle: `metadata_only` vs `data_mode` with tooltips; PII Masking toggle (policy-gated).  
- Lane hint selector (Interactive/Flex/Batch); pre-send **Cost Estimate** chip.  
- Context Capsule chips at top (Run, Mode, PII, Lane).  
- Skeleton loaders, empty states, and error banners mapped to Error Taxonomy.

**LLM Agent & Core**
- LLM-INGEST (Files/SQL): dialect sniff, schema inference (Polars/PyArrow), SQLGlot read-only introspection.
- **Standardization Shim**: auto-apply a minimal DSL plan to canonicalize headers/case/whitespace/dtypes; materialize **Standard Schema Catalog**.
- Deterministic **Sampling Policy** (seeded, stratified) for Sample Pack; PII detection + masking before any provider call.

**Artifacts & APIs**
- Artifacts: `Catalog v1`, `Sample Pack v1`, Ledger events (INGEST, SUPERVISOR), Contract hash recorded.  
- API: `create_run`, `get_run`, `policy_check`, `cost_estimate`.

**DoD**
- User can upload/connect, create a run, see catalog preview and masked samples; Ledger shows PII masked; traces/costs visible.

---

## Phase 2 — Domaining (LLM-D)
**Objectives**
- Assign semantic **Domains** to columns with confidence bands and evidence, ready for mapping.

**UI Track (parallel)**
- **Domain Studio**: table/column grid with predicted domain, confidence pill (High/Borderline/Low), evidence drawer (header tokens, regex hits, unit cues, sample snippets).  
- Actions: Approve; Mark Unknown; **Add Alias/Regex** → creates staged deltas; show reliability curves chip.

**LLM Agent & Core**
- Build RAG index over Domain Catalog + PGVector centroids.  
- Scoring: `alpha·name_sim + beta·regex_strength + gamma·value_sim + epsilon·unit_compat`.  
- Arbitration via Threshold Profile; persist evidence; propose catalog deltas to `domains_staged`.

**Artifacts & APIs**
- Artifacts: `Domain Assignments v1`, staged deltas, updated reliability snapshots.  
- API: `get_domains`, `post_domain_feedback`, `promote_domains(staged→live)` (maintainer-only).

**DoD**
- ≥90% of known columns on the golden sample auto-assign as **High**; borderline workflow functional; promotion invalidates caches as expected.

---

## Phase 3 — Mapping (LLM-M) & Transform DSL Builder
**Objectives**
- Generate and validate **Transform DSL** pipelines from LHS→RHS with previews and accept/hold/fix gates.

**UI Track (parallel)**
- **Mapping Studio**: left (source columns), right (target schema), center **DSL Builder** with ordered chips for ops; compiled SQL/Polars preview tabs; sample preview grid with success/near/fail counters.  
- Actions: Accept Mapping, Hold, Request Fix; **Create Staged Alias** from failures; Confidence and Rationale panel.

**LLM Agent & Core**
- Strategy enumeration: atomic, compose, decompose, unit normalization, type alignment.  
- Policy Engine validates DSL; in `data_mode`, execute on Sample Pack; compute success rate vs `mapping.accept_rate_min`.

**Artifacts & APIs**
- Artifacts: `Mapping Plan v1` (pipelines, confidences, examples, decisions, compiled previews).  
- API: `post_mapping_proposals`, `get_mapping_plan`, `set_mapping_decision`.

**DoD**
- For golden set, accepted mappings hit success ≥ threshold; error taxonomy consistent across providers; GUI Apply flow stages changes and logs to Ledger.

---

## Phase 4 — Analysis (LLM-A)
**Objectives**
- Run mode-aware quality checks; surface anomalies and recommended fixes.

**UI Track (parallel)**
- **Quality Report**: coverage, nulls, duplicates, unit/date sanity; optional PSI/KS; per-field callouts; **Fix-it** suggestions with one-click staging.  
- Diffs viewer for before/after samples.

**LLM Agent & Core**
- metadata_only: logical feasibility matrix.  
- data_mode: compute stats; propose threshold tweaks from reliability curves.

**Artifacts & APIs**
- Artifacts: `Analysis Report v1` with anomalies and recommended fixes.  
- API: `get_analysis`, `apply_fix_suggestions` (staged ops).

**DoD**
- Report renders within Interactive SLO for modest samples; suggested fixes feed back into Mapping/Domain staging.

---

## Phase 5 — Migration (LLM-G) with Checkpoints & Quarantine
**Objectives**
- Execute idempotent, checkpointed **Migration** with quarantine, validations, and atomic promotion.

**UI Track (parallel)**
- **Migration Console**: pre-flight checklist; batch progress; per-chunk content-hash; validation results (row_count, checksum, fk_check); quarantine viewer with masked diffs; approve/retry gates; final **Promote** step.  
- Cost/throughput chart; ETA based on chunk stats.

**LLM Agent & Core**
- Strategy: ELT (SQL) when expressible; ETL (Polars) otherwise.  
- Deterministic chunking; checkpoints; idempotent writes; retries per Error Taxonomy; advisory locks/leases.

**Artifacts & APIs**
- Artifacts: `Migration Run v1` (+ checkpoints & quarantine records).  
- API: `start_migration`, `get_migration_status`, `promote_staging`, `export_quarantine`.

**DoD**
- Restart mid-run resumes without duplication; quarantine flows RBAC-gated; final table passes validations; Ledger has complete decision trail.

---

## Phase 6 — Cross‑Provider Hardening, SLOs & Cost
**Objectives**
- Normalize behavior across providers; enforce SLOs; ship basic cost & reliability dashboards.

**UI Track (parallel)**
- **Observability Hub**: traces list with run linkage; error taxonomy heatmap; cost per stage; provider health.  
- SLO widgets: Interactive p95, Flex queue p95, Batch throughput.

**LLM Agent & Core**
- Conformance suite for adapters (params normalization, error taxonomy).  
- Health probes + routing tweaks; cost estimator improvements; golden-set nightly evals.

**Artifacts & APIs**
- Artifacts: provider conformance reports; nightly eval summaries.  
- API: `get_provider_health`, `get_cost_breakdown`.

**DoD**
- Provider failover drills pass; SLO dashboards green under synthetic load; cost variance within ±10% of estimator.

---

## Phase 7 — Advanced & Governance (v1.1+)
**Objectives**
- Deliver governance, tenancy, and safety features planned in Addenda §§39–48.

**UI Track (parallel)**
- **Admin & Governance**: Tenancy switcher; policy editor (gated); lineage/provenance drawer; DSR workflows; rollback center; license/compliance tables.

**LLM Agent & Core**
- DSL `select_when` behind feature flag; drift detection; feature flags infra; tenancy isolation modes; DSR erase/export; rollback matrix; chaos drills.

**Artifacts & APIs**
- Artifacts: lineage graphs, compliance attestations, chaos drill reports.  
- API: `admin/*` for policy, tenancy, rollbacks (RBAC-gated).

**DoD**
- Auditable governance paths; staged rollouts controlled by Policy Engine; chaos tests prove Interactive SLO preservation.

---

## Global GUI Blueprint (High-End UI)
**Design System & Performance**
- Tokenized theme (light/dark, high-contrast); WCAG AA; keyboard-first; virtualization for large tables; optimistic interactions with server reconciliation.

**Navigation & Shell**
- Left rail: Runs, Artifacts, Studio (Domains/Mapping), Migrations, Observability, Admin.  
- Top bar: Context Capsule chips + command palette; "/" opens actions.

**Start a Run (First Page)**
- Two large entry cards: **Upload Files** (drag-drop with inline validation, progress, schema preview) and **Connect SQL** (credential form with secure handling, connection test, readonly reminder).  
- Mode & PII toggles; Lane hint; cost estimate; **Create Run** CTA; post-create auto-navigation to Catalog preview.

**Studios**
- **Domain Studio** with evidence drawers and reliability curves.  
- **Mapping Studio** with DSL Builder, compiled SQL/Polars preview, sample success metrics.  
- **Quality Report** cards with one-click fixes.  
- **Migration Console** with chunk timeline, quarantine inspector, and promotion gate.

**Observability**
- Run traces, token/cost counters in-stream, provider health, error heatmaps.  
- Deep-link from any UI element to the underlying trace and Ledger event.

**Micro‑interactions**
- Live toasts with action links (e.g., "Promote now"); inline diffs; copy-as-DSL, copy-as-SQL buttons.  
- Undo for non-destructive ops; confirm-to-apply for destructive, RBAC-gated.

---

## Phase-by-Phase LLM Agent Work Orders (for autonomous coding)
For each phase, the LLM Agent should:
1) Read the **Run Contract** & **Policy Manifest** for constraints.  
2) Generate scaffolding/tests first.  
3) Implement adapters/services with **deterministic outputs** and the **uniform error taxonomy**.  
4) Emit artifacts with content hashes + version tags.  
5) Add traces and cost counters.  
6) Write runbook entries and E2E demo script.

**Phase 1 Work Orders**: Ingest services; standardization shim DSL plan; sample pack; PII masking; Start-a-Run UI; create_run/policy_check/cost_estimate endpoints; Catalog & Sample artifacts.  
**Phase 2**: Domain RAG index + centroids; scoring + arbitration; Domain Studio UI; staged alias flow.  
**Phase 3**: Mapping strategies; DSL Builder UI; compiled previews; acceptance thresholds; decision logging.  
**Phase 4**: Quality Report computations; anomaly surfacing; Fix-it actions.  
**Phase 5**: Migration executor (ELT/ETL), checkpoints, quarantine, promotion; console UI.  
**Phase 6**: Provider conformance suite; SLO dashboards; cost estimator v2.  
**Phase 7**: Governance features per §§39–48.

---

## Demos & Acceptance per Phase (Quick Script)
- **P1 Demo**: Upload CSV + Connect SQL (readonly), create run, view catalog + masked samples; see PII mask flag and cost.  
- **P2 Demo**: Open Domain Studio, approve/unknown borderlines, promote staged alias; cache invalidation visible.  
- **P3 Demo**: Propose mappings, preview compiled SQL/Polars, accept/hold with thresholds; stage alias from failures.  
- **P4 Demo**: Render Quality Report; apply one-click fix; re-run preview shows improvement.  
- **P5 Demo**: Start migration, observe batches, induce failure → quarantine → fix → retry → promote.  
- **P6 Demo**: Flip provider; conformance suite green; SLO dashboard stable under load.  
- **P7 Demo**: Perform DSR export/erase on a tenant; execute rollback; run chaos test; verify SLO preservation.

---

**Directive**: UI must ship alongside each phase (except Phase 0). The **first page** must always offer **CSV upload** or **SQL connect**, after which the **LLM Agent** automatically standardizes inputs (Shim) and advances to domaining once artifacts are persisted and Policy Engine approves.

