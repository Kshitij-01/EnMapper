#!/usr/bin/env python3
"""
Comprehensive Phase 0 Status Check for EnMapper

Validates all components, configurations, and services are ready.
"""

import sys
import asyncio
import json
from datetime import datetime
from pathlib import Path


def check_files_structure():
    """Check that all required files are present."""
    required_files = [
        "settings.py",
        "main.py", 
        "worker.py",
        "docker-compose.yml",
        "requirements.txt",
        "README.md",
        "roadmap.md",
        "PHASE0_CHECKLIST.md",
        ".gitignore",
        "core/__init__.py",
        "core/health.py",
        "core/policy.py",
        "core/providers.py", 
        "core/models.py",
        "docker/api/Dockerfile",
        "docker/worker/Dockerfile",
        "docker/postgres/init/001_initial_schema.sql",
        "docker/prometheus/prometheus.yml"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    return {
        "status": "pass" if not missing_files else "fail",
        "required": len(required_files),
        "present": len(required_files) - len(missing_files),
        "missing": missing_files
    }


def check_python_imports():
    """Check that all Python modules can be imported."""
    import_checks = {
        "settings": "from settings import get_settings",
        "fastapi": "import fastapi",
        "uvicorn": "import uvicorn", 
        "httpx": "import httpx",
        "redis": "import redis",
        "sqlalchemy": "import sqlalchemy",
        "asyncpg": "import asyncpg",
        "ulid": "import ulid",
        "pydantic": "import pydantic",
        "pydantic_settings": "import pydantic_settings"
    }
    
    results = {}
    for name, import_stmt in import_checks.items():
        try:
            exec(import_stmt)
            results[name] = {"status": "pass", "error": None}
        except Exception as e:
            results[name] = {"status": "fail", "error": str(e)}
    
    passed = sum(1 for r in results.values() if r["status"] == "pass")
    total = len(results)
    
    return {
        "status": "pass" if passed == total else "fail",
        "passed": passed,
        "total": total,
        "details": results
    }


def check_core_modules():
    """Check that core EnMapper modules can be imported."""
    core_modules = [
        "settings",
        "core.health",
        "core.policy", 
        "core.providers",
        "core.models"
    ]
    
    results = {}
    for module in core_modules:
        try:
            exec(f"import {module}")
            results[module] = {"status": "pass", "error": None}
        except Exception as e:
            results[module] = {"status": "fail", "error": str(e)}
    
    passed = sum(1 for r in results.values() if r["status"] == "pass")
    total = len(results)
    
    return {
        "status": "pass" if passed == total else "fail",
        "passed": passed,
        "total": total,
        "details": results
    }


def check_settings_configuration():
    """Check settings system configuration."""
    try:
        from settings import get_settings
        
        settings = get_settings()
        validation = settings.validate_configuration()
        
        return {
            "status": "pass",
            "environment": settings.environment.value,
            "debug": settings.debug,
            "providers": validation.get("providers", {}),
            "warnings": len(validation.get("warnings", [])),
            "errors": len(validation.get("errors", []))
        }
    
    except Exception as e:
        return {
            "status": "fail",
            "error": str(e)
        }


def check_docker_files():
    """Check Docker configuration files."""
    docker_files = {
        "docker-compose.yml": "Docker compose configuration",
        "docker/api/Dockerfile": "API service Dockerfile",
        "docker/worker/Dockerfile": "Worker service Dockerfile"
    }
    
    results = {}
    for file_path, description in docker_files.items():
        path = Path(file_path)
        if path.exists():
            try:
                content = path.read_text()
                results[file_path] = {
                    "status": "pass",
                    "size": len(content),
                    "description": description
                }
            except Exception as e:
                results[file_path] = {
                    "status": "fail",
                    "error": str(e),
                    "description": description
                }
        else:
            results[file_path] = {
                "status": "fail", 
                "error": "File not found",
                "description": description
            }
    
    passed = sum(1 for r in results.values() if r["status"] == "pass")
    total = len(results)
    
    return {
        "status": "pass" if passed == total else "fail",
        "passed": passed,
        "total": total,
        "details": results
    }


async def check_application_startup():
    """Check that the FastAPI application can start up."""
    try:
        from main import create_app
        
        app = create_app()
        
        return {
            "status": "pass",
            "app_title": app.title,
            "debug": app.debug,
            "routes": len(app.routes)
        }
    
    except Exception as e:
        return {
            "status": "fail",
            "error": str(e)
        }


def check_api_keys():
    """Check API key configuration."""
    try:
        from settings import get_settings
        
        settings = get_settings()
        
        keys_status = {
            "anthropic": bool(settings.llm.anthropic.api_key),
            "openai": bool(settings.llm.openai.api_key), 
            "groq": bool(settings.llm.groq.api_key),
            "langchain": bool(settings.observability.langchain_api_key) if hasattr(settings, "observability") else False
        }
        
        configured_count = sum(keys_status.values())
        
        return {
            "status": "pass" if configured_count > 0 else "warn",
            "configured": configured_count,
            "total": len(keys_status),
            "details": keys_status
        }
    
    except Exception as e:
        return {
            "status": "fail",
            "error": str(e)
        }


async def run_all_checks():
    """Run all Phase 0 checks and return comprehensive status."""
    
    print("🔍 EnMapper Phase 0 Comprehensive Status Check")
    print("=" * 60)
    print(f"Timestamp: {datetime.utcnow().isoformat()}")
    print()
    
    checks = {
        "📁 File Structure": check_files_structure(),
        "🐍 Python Dependencies": check_python_imports(),
        "🧩 Core Modules": check_core_modules(),
        "⚙️  Settings Configuration": check_settings_configuration(),
        "🐳 Docker Files": check_docker_files(),
        "🚀 Application Startup": await check_application_startup(),
        "🔑 API Keys": check_api_keys()
    }
    
    # Print results
    total_checks = len(checks)
    passed_checks = 0
    
    for check_name, result in checks.items():
        status = result.get("status", "unknown")
        
        if status == "pass":
            print(f"✅ {check_name}: PASS")
            passed_checks += 1
        elif status == "warn":
            print(f"⚠️  {check_name}: WARNING") 
            passed_checks += 0.5  # Partial credit for warnings
        else:
            print(f"❌ {check_name}: FAIL")
            if "error" in result:
                print(f"   Error: {result['error']}")
        
        # Print additional details for some checks
        if check_name == "🐍 Python Dependencies" and "details" in result:
            failed_imports = [name for name, details in result["details"].items() if details["status"] == "fail"]
            if failed_imports:
                print(f"   Missing: {', '.join(failed_imports)}")
        
        elif check_name == "⚙️  Settings Configuration" and result.get("status") == "pass":
            print(f"   Environment: {result.get('environment', 'unknown')}")
            providers = result.get('providers', {}).get('active', [])
            print(f"   Active Providers: {', '.join(providers) if providers else 'None'}")
        
        elif check_name == "🔑 API Keys" and result.get("status") in ["pass", "warn"]:
            configured = result.get("configured", 0)
            total = result.get("total", 0)
            print(f"   Configured: {configured}/{total} providers")
        
        print()
    
    # Overall summary
    completion_rate = (passed_checks / total_checks) * 100
    
    print("=" * 60)
    print(f"📊 PHASE 0 STATUS SUMMARY")
    print(f"Completion Rate: {completion_rate:.1f}% ({passed_checks:.1f}/{total_checks})")
    
    if completion_rate >= 90:
        print("🎉 Phase 0: EXCELLENT - Ready for Phase 1!")
    elif completion_rate >= 75:
        print("✅ Phase 0: GOOD - Minor issues to resolve")
    elif completion_rate >= 50:
        print("⚠️  Phase 0: PARTIAL - Significant work needed")
    else:
        print("❌ Phase 0: INCOMPLETE - Major issues to resolve")
    
    print()
    print("🚀 Next Steps:")
    if completion_rate >= 90:
        print("   - Start Phase 1 implementation")
        print("   - Test Docker compose stack")
        print("   - Begin data ingestion development")
    else:
        print("   - Fix failed dependency imports")
        print("   - Ensure all core modules work")
        print("   - Validate Docker configuration")
    
    return {
        "completion_rate": completion_rate,
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "details": checks,
        "timestamp": datetime.utcnow().isoformat()
    }


async def main():
    """Main entry point."""
    try:
        result = await run_all_checks()
        return 0 if result["completion_rate"] >= 75 else 1
    except Exception as e:
        print(f"❌ Check failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
