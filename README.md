# 🎯 EnMapper - AI-Powered Data Mapping Platform

A sophisticated data integration platform that leverages multiple LLM agents to automatically understand, map, and migrate data between different systems while maintaining strict governance, privacy, and cost controls.

![EnMapper UI](https://img.shields.io/badge/UI-React%20TypeScript-blue) ![Backend](https://img.shields.io/badge/Backend-Python%20FastAPI-green) ![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen) ![Version](https://img.shields.io/badge/Version-v2.1.0-purple)

## 🆕 **Current Version: v2.1.0** - Schema-Only Mode + LLM Integration

### **New in v2.1.0:**
- ✅ **Schema-Only Mode**: Process metadata/schema files without requiring sample data
- ✅ **Full LLM Agent Integration**: Claude Sonnet 4 & GPT-4o-mini for intelligent domain mapping  
- ✅ **Real-time Processing**: Live streaming of LLM agent processing with Server-Sent Events
- ✅ **Dual Mode Support**: Data mode (with samples) and Schema mode (metadata only)
- ✅ **Advanced Domain Grouping**: LHS/RHS classification with semantic domain organization
- ✅ **85+ Column Support**: Handle large schema files with intelligent field extraction

## 🌟 What We've Built

### 🎨 **Beautiful Modern UI**
- **React TypeScript Frontend** with Material-UI components
- **Glass Morphism Design** with gradient backgrounds and blur effects
- **Responsive Layout** with professional branding
- **Real-time Configuration** display (Metadata Only, Interactive, PII Masking, Budget controls)
- **Dual Interface**: Start Run screen and Domain Studio workspace

### 🚀 **Robust Backend API**
- **FastAPI Python Server** with async capabilities
- **Uvicorn ASGI Server** for high performance
- **Policy Engine** with governance and compliance
- **Model Registry** supporting multiple LLM providers
- **Comprehensive Health Monitoring** and logging

### 🔧 **Key Features Implemented**
- **File Upload System** with drag-and-drop interface (supports CSV, ZIP archives)
- **LLM Agent Terminal** with real-time processing streams
- **Schema-Only Mode** for metadata files without sample data
- **GPT-4o-mini Domain Labeling** with intelligent field recognition  
- **Claude Sonnet 4 Grouping** with LHS/RHS classification
- **Domain Assignment Engine** for semantic data classification
- **Database Connection Manager** for multiple database types
- **Configuration Management** with real-time updates
- **Error Handling & Notifications** system
- **Development & Production** environment support

## 🏗️ Architecture Overview

### **Frontend (React + TypeScript)**
```
ui/
├── src/
│   ├── components/
│   │   ├── DomainStudio.tsx         # Main workspace interface with LLM integration
│   │   ├── StartRunScreen.tsx       # Initial configuration screen  
│   │   ├── LLMAgentTerminal.tsx     # Real-time LLM processing terminal
│   │   ├── FileUploadCard.tsx       # File upload component
│   │   ├── SQLConnectionCard.tsx    # Database connection UI
│   │   └── RunConfigurationPanel.tsx # Configuration management
│   ├── services/
│   │   └── api.ts                   # API client with request/response logging
│   ├── App.tsx                      # Main application with theme & state management
│   ├── App.css                      # Custom styling and glass effects
│   └── index.tsx                    # Application entry point
└── package.json                     # Dependencies and scripts
```

### **Backend (Python + FastAPI)**
```
main.py                       # FastAPI application entry point with LLM endpoints
core/
├── agent_framework.py        # LLM agent orchestration framework
├── llm_domain_agent.py       # Claude agent for data processing
├── model_routing.py          # Multi-provider LLM routing (OpenAI, Anthropic)
├── domain_assignment.py     # Semantic classification engine
├── standardization.py       # Data standardization pipeline
├── ingest.py                # Data ingestion pipeline  
├── database.py              # Database connectivity
├── policy.py                # Governance and compliance engine
├── models.py                # Data models and schemas
├── health.py                # Health monitoring
└── observability.py         # Logging and monitoring
```

## 🤖 LLM Agent Architecture

### **Current Implementation (v2.1.0):**

**🔍 Claude Sonnet 4 Agent** - Data Processing & Standardization
- Extracts and analyzes uploaded files (CSV, ZIP archives)  
- Performs PII detection and masking
- Creates standardized schema catalogs
- Real-time streaming output via Server-Sent Events

**🏷️ GPT-4o-mini** - Open Domain Labeling  
- Assigns semantic labels to columns using field names, types, and descriptions
- Works in both data mode (with samples) and schema mode (metadata only)
- Processes 85+ columns efficiently with cost optimization

**🧠 Claude Sonnet 4** - Semantic Grouping & Classification
- Groups related columns into canonical semantic domains
- Assigns LHS (source) vs RHS (target) classification  
- Organizes fields into 12+ semantic categories (product_info, financial, media, etc.)

### **Planned Expansion:**
- **🔄 LLM-M (Mapper)**: Transformation pipeline generation
- **📊 LLM-A (Analyst)**: Quality checks and validation  
- **🚀 LLM-G (Migrator)**: Migration execution with checkpoints

## 🚀 Quick Start

### **Prerequisites**
- **Node.js 18+** for frontend development
- **Python 3.12+** for backend services
- **API Keys** for LLM providers:
  - `ANTHROPIC_API_KEY` for Claude Sonnet 4 (required)
  - `OPENAI_API_KEY` for GPT-4o-mini (required)
- **Virtual Environment** recommended for Python dependencies

### **Installation & Setup**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/enmapper.git
   cd enmapper
   ```

2. **Setup Python Backend:**
   ```bash
   # Create and activate virtual environment
   python -m venv venv1
   venv1\Scripts\activate  # Windows
   # source venv1/bin/activate  # macOS/Linux
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Setup React Frontend:**
   ```bash
   cd ui
   npm install
   ```

4. **Environment Configuration:**
   ```bash
   # Create .env file with your API keys
   echo "ANTHROPIC_API_KEY=your_claude_api_key_here" > .env
   echo "OPENAI_API_KEY=your_openai_api_key_here" >> .env
   # Note: Keep .env file secure and never commit to version control
   ```

## 🖥️ **Starting the Servers**

### **Method 1: Manual Start (Recommended for Development)**

**Terminal 1 - Backend Server:**
```bash
# In project root directory
# Activate virtual environment  
venv1\Scripts\activate          # Windows
# source venv1/bin/activate     # macOS/Linux

# Start FastAPI server
python main.py
```
✅ *Backend runs on: `http://localhost:8000`*  
📚 *API docs available at: `http://localhost:8000/docs`*

**Terminal 2 - Frontend Server:**
```bash
# In ui/ directory (separate terminal)
cd ui
npm start
```
✅ *Frontend runs on: `http://localhost:3000`*

### **Method 2: Production Start**
```bash
# Backend with proper ASGI server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Frontend production build  
cd ui && npm run build && npm run start
```

### **Verification Steps:**
1. ✅ Backend health check: `http://localhost:8000/health`
2. ✅ Frontend loads: `http://localhost:3000`  
3. ✅ Upload a CSV file and see LLM Agent Terminal activate
4. ✅ Try "Open Labels (GPT-4o-mini)" button in Domain Studio

## 📁 **Using Schema-Only Mode (v2.1.0)**

**Perfect for when you only have metadata/schema files, not actual data:**

### **Supported Schema File Formats:**
```csv
Field Name (Technical),Data Element,Description,Data Type,Length,Source Infotype/Table,Notes
EMPLOYEE_ID,Employee_ID,Employee ID,CHAR,5,lhs1.csv,
FIRST_NAME,First_Name,First Name,CHAR,5,lhs1.csv,
LAST_NAME,Last_Name,Last Name,CHAR,6,lhs1.csv,
```

### **How Schema Mode Works:**
1. **📁 Upload** schema/metadata CSV files (no sample data required)
2. **🔍 Auto-Detection** - System recognizes schema format automatically  
3. **🏷️ GPT-4o-mini** labels each field using name, type, and description
4. **🧠 Claude Sonnet 4** groups fields into semantic domains with LHS/RHS classification

### **Example Results:**
- `EMPLOYEE_ID` → `employee_identifier` (LHS)
- `FIRST_NAME` → `person_first_name` (LHS)  
- `DEPARTMENT_ID` → `organizational_unit` (LHS)
- `HIRE_DATE` → `employment_start_date` (RHS)

**✨ Perfect for compliance scenarios where actual data cannot be shared with external LLMs!**

## 🎨 UI Features

### **🎯 Header & Navigation**
- **EnMapper Branding** with professional styling
- **Real-time Configuration Display**: Metadata Only, Interactive mode, PII Masking status, Budget tracking
- **Tabbed Interface**: Start Run and Domain Studio

### **🚀 Start Run Screen**
- **File Upload Interface** with drag-and-drop functionality
- **Database Connection Manager** supporting multiple database types
- **Configuration Panel** with real-time updates
- **Run Status Monitoring** with progress indicators

### **🎨 Domain Studio**
- **Advanced Workspace** for data mapping and domain assignment
- **LLM Integration Buttons**: "Open Labels (GPT-4o-mini)" and "Grouped (Claude Sonnet 4)"
- **Dual View Modes**: Column View and Group View with LHS/RHS classification
- **Interactive Tables** with modern styling and React keys
- **Real-time Updates** and notifications  
- **Professional Data Grid** components

### **✨ Design System**
- **Glass Morphism Effects** with backdrop blur
- **Gradient Backgrounds** with purple/blue color scheme
- **Modern Typography** using Inter font family
- **Responsive Design** for all screen sizes
- **Dark/Light Theme** support ready

## 🔧 Technical Stack

### **Frontend Technologies**
- **React 18** with TypeScript
- **Material-UI (MUI)** for components
- **Modern CSS** with glass morphism effects
- **Responsive Design** principles
- **Real-time State Management**

### **Backend Technologies**
- **FastAPI** for high-performance APIs
- **Uvicorn** ASGI server
- **SQLAlchemy** for database ORM
- **Pydantic** for data validation
- **LangChain** for LLM integration
- **Polars** for data processing

### **Infrastructure & Tools**
- **Docker** support with compose files
- **Prometheus** metrics collection
- **Grafana** dashboards
- **PostgreSQL** with PGVector extension
- **Redis** for caching and sessions

## 🔒 Security & Privacy

- **🛡️ PII Detection**: Automatic detection and masking before external LLM calls
- **📋 Policy Engine**: Enforces governance, RBAC, and compliance rules
- **📝 Audit Trail**: Complete decision logging and versioning
- **🏠 Local-First Option**: Ollama support for sensitive workloads
- **🔐 API Key Management**: Secure credential handling

## 🎮 Operating Modes

- **📊 Metadata Only**: Schema analysis without touching actual data
- **⚡ Interactive**: Real-time GUI operations with strict latency SLOs
- **🔄 Flex**: Background processing for larger workloads
- **📦 Batch**: Offline processing with checkpoints and recovery

## 📊 Current Development Status

### ✅ **Completed Features (v2.1.0)**
- [x] Modern React UI with glass morphism design
- [x] FastAPI backend with health monitoring  
- [x] **LLM Agent Integration** - Claude Sonnet 4 + GPT-4o-mini
- [x] **Schema-Only Mode** - Process metadata without sample data
- [x] **Real-time LLM Terminal** - Server-Sent Events streaming
- [x] **Domain Mapping Engine** - Open labeling + semantic grouping
- [x] **LHS/RHS Classification** - Source vs target field identification
- [x] File upload system with drag-and-drop (CSV, ZIP support)
- [x] Database connection management
- [x] Configuration management system
- [x] Policy engine foundation
- [x] Error handling and notifications
- [x] Development environment setup

### 🔄 **In Progress (v2.2.0)**
- [ ] Advanced transformation DSL implementation
- [ ] Cross-provider migration workflows  
- [ ] Enhanced analytics dashboard
- [ ] User authentication system

### 🆕 **Current Capabilities**
- ✅ **85+ Column Processing** in schema mode
- ✅ **Dual Mode Support** (data + schema)  
- ✅ **Cost-Optimized LLM Usage** with smart model routing
- ✅ **Professional UI/UX** with state persistence

### 📋 **Planned Features**
- [ ] Advanced data transformation DSL
- [ ] Cross-provider migration tools
- [ ] Real-time collaboration features
- [ ] Advanced governance controls
- [ ] Enterprise SSO integration

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### **Development Guidelines**
- Follow TypeScript best practices for frontend
- Use Python type hints and docstrings for backend
- Maintain responsive design principles
- Add comprehensive error handling
- Include unit tests for new features

## 📚 Documentation

- **`roadmap.md`**: Complete technical specifications and implementation roadmap
- **`PHASE0_CHECKLIST.md`**: Development milestones and progress tracking
- **API Documentation**: Available at `http://localhost:8000/docs` when running
- **Component Storybook**: In development for UI components

## 🎯 Key Achievements

### **🤖 LLM Agent Integration (v2.1.0)**  
- **Claude Sonnet 4** for intelligent data processing and standardization
- **GPT-4o-mini** for cost-effective open domain labeling (85+ columns)
- **Real-time streaming** with Server-Sent Events for live processing updates
- **Schema-only mode** supporting metadata files without sample data

### **🎨 Advanced User Interface**
- **Dual-view modes**: Column View and Group View with seamless switching
- **LHS/RHS classification** display with color-coded chips
- **Professional React components** with Material-UI and glass morphism
- **State persistence** across tab navigation with localStorage

### **🚀 Production-Ready Backend**
- **Multi-provider LLM routing** (OpenAI, Anthropic) with intelligent fallbacks
- **Schema detection engine** for automatic metadata vs data file classification  
- **Batch processing optimization** for cost-effective LLM operations
- **Comprehensive error handling** with detailed logging and user feedback

### **🔧 Enterprise Developer Experience**
- **Hot-reload development** environment with proper virtual environment setup
- **API-first design** with OpenAPI documentation at `/docs`
- **Modular architecture** supporting future agent expansions
- **Type-safe implementations** across React TypeScript and Python

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by modern data integration platforms
- Built with love using React, TypeScript, and Python
- Special thanks to the open-source community

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/your-username/enmapper/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/enmapper/discussions)
- **Documentation**: [Project Wiki](https://github.com/your-username/enmapper/wiki)

---

**🎯 EnMapper - Making Data Integration Beautiful and Intelligent**