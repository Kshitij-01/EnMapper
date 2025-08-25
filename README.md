# EnMapper 🗺️

> **AI-Powered Data Mapping and Migration Platform**

EnMapper is an advanced LLM-powered system for automated data discovery, semantic mapping, and safe migration between different data sources and schemas.

## 🚀 Project Status

**✅ Phase 0 COMPLETE** - Up-Front Decisions & Contracts
- All core configurations implemented
- Multi-provider LLM setup (OpenAI ✅, Anthropic ✅, LangSmith ✅)
- Security policies and RBAC configured
- Quality thresholds and feature flags ready

## 🏗️ Architecture Overview

### Core Components
- **🤖 Multi-Provider LLM Engine** - OpenAI, Anthropic, Groq, Ollama
- **🔒 Dual-Mode Operation** - `metadata_only` (safe) vs `data_mode` (full access)
- **📊 Quality Gates** - Confidence bands with adaptive thresholds
- **🛡️ Security-First** - PII detection, RBAC, audit trails
- **💰 Cost Controls** - Multi-level budgets with circuit breakers

### Key Features
- **Semantic Domain Classification** - Automatic data categorization
- **Transform Generation** - AI-powered mapping logic
- **Migration Safety** - Checkpoints, quarantine, rollback
- **Real-time Observability** - LangSmith integration
- **Always-On LLM Interface** - Interactive web GUI

## 🚦 Operating Modes

| Feature | metadata_only | data_mode |
|---------|---------------|-----------|
| Schema Analysis | ✅ | ✅ |
| Value Sampling | ❌ | ✅ (masked) |
| Transform Preview | Symbolic | Sample execution |
| Migration | Dry-run only | Full execution |

## 🔧 Quick Start

### Prerequisites
- Python 3.12+
- API Keys: OpenAI, Anthropic, LangSmith
- PostgreSQL + PGVector (for production)
- Redis (for caching)

### Setup
1. **Clone and setup environment:**
   ```bash
   git clone <repository-url>
   cd EnMapper
   python -m venv venv1
   source venv1/bin/activate  # or venv1\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

3. **Validate setup:**
   ```bash
   python scripts/test_phase0_setup.py
   python scripts/test_api_connectivity.py
   ```

## 📋 Development Roadmap

- **✅ Phase 0** - Foundations and contracts (COMPLETE)
- **🚧 Phase 1** - Infrastructure scaffolding (2 weeks)
- **📅 Phase 2** - Domain catalog and policy engine (2 weeks)
- **📅 Phase 3** - Ingest and schema discovery (2 weeks)
- **📅 Phase 4** - LangGraph supervisor and LLM roles (2 weeks)

[View Full Roadmap →](ROADMAP.md)

## 🔐 Security & Compliance

- **PII Protection** - Automatic detection and masking
- **RBAC** - Role-based access control (5 role types)
- **Audit Logging** - Complete decision trails
- **Cost Controls** - Budget limits and circuit breakers
- **Network Security** - Egress allowlists and TLS

## 🧪 Testing & Validation

EnMapper includes comprehensive testing:

```bash
# Validate all configurations
python scripts/test_phase0_setup.py

# Test API connectivity  
python scripts/test_api_connectivity.py

# Setup environment
python scripts/setup_env.py
```

## 📊 Configuration

EnMapper uses a comprehensive configuration system:

- **`config/providers.yaml`** - LLM provider settings and routing
- **`config/modes.yaml`** - Operating mode capabilities
- **`config/policy_manifest.yaml`** - Security and compliance
- **`config/threshold_profiles.yaml`** - Quality gates and confidence
- **`config/run_contract.yaml`** - Lifecycle and artifact management
- **`config/feature_flags.yaml`** - Feature rollout and A/B testing

## 💡 Key Innovation

**Adaptive Confidence System**: EnMapper uses domain-specific confidence thresholds that learn and adapt based on:
- Historical accuracy by domain type
- Data sensitivity levels
- User feedback loops
- Statistical validation

## 🤝 Contributing

1. Follow the phase-based development approach
2. All changes must pass validation tests
3. Security configurations are mandatory
4. Cost controls must be respected

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 📞 Support

- **Email**: kshitijpatilhere@gmail.com
- **Issues**: GitHub Issues
- **Documentation**: [docs/](docs/)

---

**Built with ❤️ using LangChain, FastAPI, and modern AI technologies**
