# 🎯 EnMapper - AI-Powered Data Mapping Platform

A sophisticated data integration platform that leverages multiple LLM agents to automatically understand, map, and migrate data between different systems while maintaining strict governance, privacy, and cost controls.

![EnMapper UI](https://img.shields.io/badge/UI-React%20TypeScript-blue) ![Backend](https://img.shields.io/badge/Backend-Python%20FastAPI-green) ![Status](https://img.shields.io/badge/Status-Active%20Development-orange)

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
- **File Upload System** with drag-and-drop interface
- **Database Connection Manager** for multiple database types
- **Domain Assignment Engine** for semantic data classification
- **Configuration Management** with real-time updates
- **Error Handling & Notifications** system
- **Development & Production** environment support

## 🏗️ Architecture Overview

### **Frontend (React + TypeScript)**
```
ui/
├── src/
│   ├── components/
│   │   ├── DomainStudio.tsx       # Main workspace interface
│   │   ├── StartRunScreen.tsx     # Initial configuration screen
│   │   ├── FileUploadCard.tsx     # File upload component
│   │   ├── SQLConnectionCard.tsx  # Database connection UI
│   │   └── RunConfigurationPanel.tsx # Configuration management
│   ├── App.tsx                    # Main application with theme
│   ├── App.css                    # Custom styling and glass effects
│   └── index.tsx                  # Application entry point
└── package.json                   # Dependencies and scripts
```

### **Backend (Python + FastAPI)**
```
core/
├── main.py                 # FastAPI application entry point
├── policy.py              # Governance and compliance engine
├── providers.py           # LLM provider management
├── models.py              # Data models and schemas
├── health.py              # Health monitoring
├── ingest.py              # Data ingestion pipeline
├── domain_assignment.py   # Semantic classification
├── database.py            # Database connectivity
└── observability.py       # Logging and monitoring
```

## 🎯 LLM Agent Architecture

The system operates through **5 specialist LLM agents** in a supervised workflow:

1. **🔍 LLM-INGEST**: Analyzes files/databases to create standardized schema catalogs
2. **🏷️ LLM-D (Domains)**: Assigns semantic types to columns (e.g., "person.first_name", "contact.email")
3. **🔄 LLM-M (Mapper)**: Generates transformation pipelines using a whitelisted DSL
4. **📊 LLM-A (Analyst)**: Performs quality checks and suggests improvements
5. **🚀 LLM-G (Migrator)**: Executes migrations with checkpoints and quarantine handling

## 🚀 Quick Start

### **Prerequisites**
- **Node.js 18+** for frontend development
- **Python 3.12+** for backend services
- **API Keys** for LLM providers (OpenAI, Anthropic, etc.)

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
   # Copy and configure settings
   cp settings.example.json settings.json
   # Add your API keys and configuration
   ```

### **Running the Application**

**Start Backend Server:**
```bash
# In project root with venv activated
python main.py
```
*Backend runs on: `http://localhost:8000`*

**Start Frontend Server:**
```bash
# In ui/ directory (separate terminal)
cd ui
npm start
```
*Frontend runs on: `http://localhost:3000`*

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
- **Interactive Tables** with modern styling
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

### ✅ **Completed Features**
- [x] Modern React UI with glass morphism design
- [x] FastAPI backend with health monitoring
- [x] File upload system with drag-and-drop
- [x] Database connection management
- [x] Configuration management system
- [x] Policy engine foundation
- [x] LLM provider registry
- [x] Error handling and notifications
- [x] Development environment setup

### 🔄 **In Progress**
- [ ] Domain assignment implementation
- [ ] Data mapping pipeline
- [ ] Advanced analytics dashboard
- [ ] User authentication system

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

### **🎨 Modern User Interface**
- Built a professional, responsive UI with React and TypeScript
- Implemented glass morphism design with beautiful gradients
- Created reusable component library with Material-UI

### **🚀 Robust Backend**
- Developed scalable FastAPI application with async support
- Implemented comprehensive health monitoring and logging
- Created modular architecture for LLM provider management

### **🔧 Developer Experience**
- Set up hot-reload development environment
- Configured linting and type checking
- Implemented comprehensive error handling

### **📊 Data Management**
- Built file upload system with validation
- Created database connection management
- Implemented configuration state management

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