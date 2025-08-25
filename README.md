# Encora Data Integration Platform

An LLM-powered data integration platform that automatically understands, maps, and migrates data between different systems while maintaining strict governance, privacy, and cost controls.

## 🎯 Overview

This platform leverages multiple LLM agents working in supervised orchestration to automate complex data integration tasks:

- **Automated Data Understanding**: LLM agents analyze schemas and data to infer semantic meaning
- **Smart Mapping**: Automatically generates transformation pipelines between source and target systems  
- **Privacy-First**: Built-in PII detection and masking before any external LLM calls
- **Provider-Agnostic**: Works with multiple LLM providers (OpenAI, Anthropic, Groq, Ollama)
- **Deterministic & Auditable**: Every decision is logged, versioned, and reproducible

## 🏗️ Architecture

The system operates through **5 specialist LLM agents** in a supervised workflow:

1. **LLM-INGEST**: Analyzes files/databases to create standardized schema catalogs
2. **LLM-D (Domains)**: Assigns semantic types to columns (e.g., "person.first_name", "contact.email")  
3. **LLM-M (Mapper)**: Generates transformation pipelines using a whitelisted DSL
4. **LLM-A (Analyst)**: Performs quality checks and suggests improvements
5. **LLM-G (Migrator)**: Executes migrations with checkpoints and quarantine handling

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL with PGVector extension
- Redis
- API keys for LLM providers

### Setup

1. **Clone and setup environment:**
   ```bash
   git clone <your-repo>
   cd Encora
   python -m venv venv1
   venv1\\Scripts\\activate  # Windows
   # source venv1/bin/activate  # macOS/Linux
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   ```bash
   python setup_env.py
   ```
   This will create a `.env` file with your API keys and default configuration.

4. **Setup databases:**
   - PostgreSQL with PGVector extension
   - Redis server
   - Update database URLs in `.env`

5. **Initialize the system:**
   ```bash
   # Database migrations (when implemented)
   # alembic upgrade head
   
   # Start the API server (when implemented)
   # uvicorn main:app --reload
   ```

## 📋 API Keys Required

The platform supports multiple LLM providers:

- **Anthropic**: For Claude models (recommended default)
- **OpenAI**: For GPT models  
- **Groq**: For fast inference
- **LangChain**: For observability and tracing

## 🎮 Operating Modes

- **metadata_only**: Schema analysis without touching actual data
- **data_mode**: Full analysis including masked data samples and execution

## 🔒 Security & Privacy

- **PII Detection**: Automatic detection and masking before external API calls
- **Policy Engine**: Enforces governance, RBAC, and compliance rules
- **Audit Trail**: Complete decision logging in the Supervisor Ledger
- **Local-First Option**: Ollama support for sensitive workloads

## 📊 Processing Lanes

- **Interactive**: Real-time GUI operations with strict latency SLOs
- **Flex**: Background processing for larger workloads
- **Batch**: Offline processing with checkpoints and recovery

## 🛠️ Development Status

This project is currently in the **specification/planning phase**. The comprehensive roadmap in `roadmap.md` serves as the single source of truth for implementation.

### Implementation Phases

- **Phase 0-1**: Foundation and data intake ⏳
- **Phase 2-3**: Domain assignment and mapping capabilities 
- **Phase 4-5**: Quality analysis and migration execution
- **Phase 6-7**: Cross-provider hardening and governance features

## 📚 Documentation

- See `roadmap.md` for complete technical specifications
- Architecture follows deterministic, explainable pipelines
- All artifacts are versioned and content-hashed for reproducibility

## 🤝 Contributing

This project follows the specifications in `roadmap.md`. Any changes to core architecture, DSL, or policy frameworks require versioned updates to the roadmap.

## 📄 License

[Add your license here]

## 🔗 References

- Inspired by data mapping concepts from [ReBy298/Data-Mapper](https://github.com/ReBy298/Data-Mapper/blob/changes/Data_Mapper.ipynb)
- Built with modern Python stack: FastAPI, SQLAlchemy, Polars, LangChain
