# Unified ROADMAP.md — Core + API + Web GUI (with Always‑On Prompt)
_Last updated: 2025-08-25_

This single roadmap merges Core Architecture and Product/Web/API/GUI plans, including added improvements:
- Single Policy Engine shared by Core, API, GUI
- Explicit mode separation (metadata_only vs data_mode) with capability matrix
- Domain‑adaptive thresholds + reliability curves
- Supervisor tickets for all GUI actions, logged to same ledger
- Artifact Preview Protocol (masked preview slices) for UI diffs/confirmations
- Migration checkpoint‑restart; idempotency and quarantine lanes
- Cost observability (per‑stage budgets, live UI cost, caching & early‑exit)
- Confidence bands (green/yellow/red) tied to Threshold Profiles
- Feedback→Catalog loop to auto‑seed domains_staged
- Structured PII decision logs for audits

---

## Phase 0 — Up‑Front Decisions & Contracts
- [ ] Pin versions (LangChain/LangGraph/LangSmith/CrewAI, polars, SQLAlchemy, SQLGlot, PGVector, Redis, Great Expectations/Pandera, pint)
- [ ] Providers: default OpenAI; alternates Anthropic/Groq/Ollama; routing profile definitions
- [ ] Modes & capability matrix finalized: metadata_only vs data_mode
- [ ] Policy Manifest v1 (PII redaction/masking, RBAC, cost caps, network egress allow‑list)
- [ ] Threshold Profile v1 (τ_high/τ_low, confidence bands; domain‑adaptive tuning hooks)
- [ ] Run Contract v1 (IDs, stages, artifacts, budgets, gates)

**Exit:** RFC approved; .env/.env.example set; feature flags declared.

---

## Phase 1 — Foundations & Scaffolding
- [ ] Monorepo: `core/`, `agents/`, `infra/`, `catalog/`, `storage/`, `evals/`, `api/`, `web/`, `docs/`
- [ ] Infra: Postgres(+PGVector), Redis, OTel→Prometheus/Grafana, Sentry
- [ ] Settings service (flags, provider routing, caps, canary %); directories `runs/`, `artifacts/`, `quarantine/`
- [ ] DB schemas: `runs`, `prompts`, `domains_live/staged`, `domain_examples`, `mapping_plans`, `analysis_reports`, `migration_runs`, `quarantine_rows`, `threshold_profiles`
- [ ] Vector indexes for domain descriptors; Redis TTL caches (embeddings, completion, idempotency)

**Exit:** Services healthy; migrations applied; caches warm.

---

## Phase 2 — Domain Catalog & Shared Policy Engine
- [ ] Domain taxonomy (atomic + composites), aliases/keywords/negatives/regex (10–20 examples each)
- [ ] Ingest to `domains_live` with name/value centroids; staged→live promotion workflow with auto‑tests + provenance
- [ ] Single Policy Engine module (used by Core, API, GUI for PII/RBAC/cost/mode gates)

**Artifacts:** Domain Catalog v1, Promotion Guide v1  
**Exit:** Promotion works; policy checks callable from API & GUI.

---

## Phase 3 — Ingest → Standard Schema Catalog
- [ ] Files: inventory, encoding/dialect sniff, schema inference (polars/pyarrow)
- [ ] SQL: SQLAlchemy + SQLGlot read‑only introspection; capped sampling; dialect portability
- [ ] Emit Standard Schema Catalog (tables/columns/dtypes/semantics) for both modes
- [ ] Emit Row Sample Pack only in data_mode (respect masks, caps)

**Exit:** Deterministic catalogs; gated samples; unit tests pass.

---

## Phase 4 — LangGraph Supervisor + LLM Roles (CrewAI inside)
- [ ] Graph: Supervisor node (run contract, thresholds, budgets, retries, escalation), Orchestrator node (path/mode selection)
- [ ] Specialists: LLM‑INGEST(files/sql), LLM‑D(Domains), LLM‑M(Mapper), LLM‑A(Analyst), LLM‑G(Migrator)
- [ ] CrewAI sub‑roles for propose/validate/repair in D/M/A/G
- [ ] Supervisor Ledger for all stage tickets

**Exit:** End‑to‑end dry run in metadata_only with ledger & retries logged.

---

## Phase 5 — Domaining → Mapping (Declarative, Whitelisted Pipelines)
**Domaining**
- [ ] Evidence extraction: headers, regex hits, unit cues; small value samples if allowed
- [ ] Candidate generation via RAG + vector similarity
- [ ] Conditional score: α·name_sim + β·regex_strength + γ·value_sim + ε·unit_compat
- [ ] Confidence banding; arbitration prompts for borderline/low

**Mapping**
- [ ] Group by domain; enumerate atomic/compose/decompose candidates
- [ ] Emit transform pipelines from whitelisted ops (trim, case, collapse whitespace, regex split/extract, concat, map dict, unit convert, date parse/format, type cast, safe math)
- [ ] Sample execution in data_mode; accept/hold/fix per thresholds with rationale

**Exit:** Mapping Plan v1 with per‑mapping confidence, rationale, and preview examples (masked if needed).

---

## Phase 6 — Analysis & Migration
**Analysis**
- [ ] data_mode: coverage/nulls/dups, unit/date sanity, PSI/KS (optional), cross‑field rules
- [ ] metadata_only: logical consistency, domain coverage, transform feasibility matrix

**Migration**
- [ ] Pre‑flight: accepted mappings only; key strategy (natural/composite/surrogate)
- [ ] ELT vs ETL auto‑selection (in‑DB SQL vs polars batch)
- [ ] Batching + idempotency (content hashes/checkpoints); quarantine lane
- [ ] Checkpoint‑restart; per‑batch validations; atomic promotion staging→final

**Exit:** Migration Run JSON + DataFrames (batches, validations, quarantine, summary).

---

## Phase 7 — Observability, Evals, Cost Controls
- [ ] LangSmith traces linked to Run IDs for all LLM calls
- [ ] System dashboards: latency, error, **cost per stage**; alerts
- [ ] Golden‑set evals (domaining & mapping); nightly canary across providers
- [ ] Provider router (quality‑first; cost/latency fallbacks); token/cost budgets; early‑exit; Redis/LangChain caching
- [ ] Reliability curves by domain; threshold tuning loop (domain‑adaptive)

**Exit:** Thresholds calibrated; cost visibility and guardrails active.

---

## Phase 8 — API Surface (Contracts for GUI & Automation)
- [ ] Endpoints: `/runs`, `/artifacts/*` (catalog, domains, mappings, analysis, migration, ledger), `/supervisor/*` (tickets, actions), `/policy/check`, `/thresholds/*`, `/cost/estimate`
- [ ] Policy Engine enforced at all endpoints; rate limits; error taxonomy
- [ ] Artifact **Preview Protocol**: masked preview slices for GUI diffs/confirmations
- [ ] OpenAPI spec + contract tests

**Exit:** API stable; GUI consumes preview slices; CI contract tests pass.

---

## Phase 9 — Web GUI with Always‑On LLM Prompt
**Prompt UX**
- [ ] Floating Dock + Command Palette (`Ctrl/Cmd+/`, `Ctrl/Cmd+K`); sticky footer (compact)
- [ ] Context Capsule chips (Run, Tab, Mode, PII); editable scope
- [ ] Templates with variable injection; slash commands: `/explain`, `/gen-rule`, `/preview`, `/cost`, `/thresholds`
- [ ] Attach artifact snapshots (client‑trimmed; masked by policy)

**Output UX**
- [ ] Streaming + live cost; **confidence bands** and diffs
- [ ] Two‑step Confirm‑to‑Apply → Supervisor gate → staged delta → ledger log
- [ ] Undo for safe ops; destructive ops require Admin approval per RBAC

**Safety & A11y**
- [ ] PII/cost/role guards; Redis‑backed prompt cache
- [ ] Keyboardable; mobile bottom sheet with large CTAs

**Exit:** v1 MAC: Dock + Palette + Context + Templates + Actions + Policy guards + Trace link.

---

## Phase 10 — Security & Compliance
- [ ] Read‑only SQL; schema allow‑list; sample caps
- [ ] Secrets via secret manager / Docker secrets; egress allow‑list for LLM endpoints
- [ ] Audit fields in every artifact (provider/model/prompt hash/catalog version)
- [ ] Structured logs for PII decisions (“masked before send”)

**Exit:** Pen‑test checklist complete; audit queries ready; red‑team scenarios documented.

---

## Phase 11 — Packaging & Runtime
- [ ] Dockerfiles (multi‑stage): API/Worker (CPU) + optional GPU image for local models
- [ ] Docker Compose (dev): Postgres(+PGVector), Redis, tracing stack, core service
- [ ] Health/readiness endpoints; warmers (load catalog, prime caches)
- [ ] Kubernetes plan: HPA on queue depth/CPU, NetworkPolicies, PodSecurityContext

**Exit:** One‑command local dev; CI images; smoke tests green.

---

## Phase 12 — Docs, Runbooks, Acceptance
- [ ] How‑to: metadata‑only and data‑mode runs (inputs, outputs, artifacts, thresholds)
- [ ] Remediation cookbook: name splits, date formats, unit conflicts
- [ ] Staged→live promotion guide with auto‑tests + manual approval

**Exit:** Ops can run E2E; on‑call playbook ready.

---

## Now / Next / Later
**Now (Weeks 1–2)**
- Phase 0–1: decisions, scaffolding, DB/caches, Policy Engine skeleton, Run Contract v1
- Minimal API stub: `/runs`, `/policy/check`

**Next (Weeks 3–6)**
- Phase 2–5: Domain Catalog v1; Ingest paths; Graph Supervisor + D/M; Mapping Plan v1
- GUI: Dock + Context chips + Templates; Confirm‑to‑Apply (staged only)

**Later (Weeks 7–10)**
- Phase 6–7: Analysis; Migration with checkpoints/quarantine; cost dashboards; golden‑set evals
- GUI: confidence bands, diffs, full trace links

**v1 Exit**
- E2E in both modes for 1 LHS/RHS table; all artifacts produced
- Supervisor gates enforced; GUI actions stage and apply safely
- Traces visible; domain promotion and migration complete with idempotency/quarantine/promotion

---

## RACI (Lightweight)
- Core Orchestration (LangGraph/Supervisor) — R: Backend, A: Core Lead
- Domain Catalog & Evals — R: Data/ML, A: ML Lead
- Policy Engine & Security — R: Platform, A: Security Lead
- API Contracts — R: Backend, A: Platform Lead
- GUI Prompt & Actions — R: Frontend, A: Product Lead
- Observability/Cost Dashboards — R: Platform, C: All

---

## Capability Matrix (Modes)
| Capability | metadata_only | data_mode |
|---|---|---|
| Catalog inference | ✓ | ✓ |
| Row samples | ✗ | ✓ (masked, capped) |
| Domaining evidence | headers/regex/units | headers/regex/units + small values |
| Transform previews | symbolic/dry‑run | sample execution |
| Analysis checks | logical/feasibility | coverage/nulls/dups + stats |
| Migration | dry‑run planning | full execution with checkpoints |
| GUI previews | masked schema/IDs | masked slices + diffs |

---

## Backlog (v1.1+)
- Inline coach tips and prompt suggestions
- Explanation vs Action toggle in Prompt output
- Multi‑step approvals beyond confirm‑to‑apply
- Cross‑run semantic search over prompt history
