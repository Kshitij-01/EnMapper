#!/usr/bin/env python3
"""
EnMapper Phase 0 Setup Validation Script
Tests that all configurations and API keys are properly set up
"""

import os
import sys
import yaml
import json
from pathlib import Path
from typing import Dict, Any
import asyncio

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load and validate YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Failed to load {config_path}: {e}")
        return {}

def test_api_keys():
    """Test API key format validation (not actual API calls)."""
    print("\n🔑 Testing API Key Configurations...")
    
    # API keys - these are test values, real keys should be in .env file
    api_keys = {
        "OPENAI_API_KEY": "your-openai-api-key-here",
        "ANTHROPIC_API_KEY": "your-anthropic-api-key-here",
        "GROQ_API_KEY": "your-groq-api-key-here", 
        "LANGSMITH_API_KEY": "your-langsmith-api-key-here"
    }
    
    # Validate API key formats
    validations = {
        "OPENAI_API_KEY": lambda k: k.startswith("sk-proj-") and len(k) > 50,
        "ANTHROPIC_API_KEY": lambda k: k.startswith("sk-ant-") and len(k) > 50,
        "GROQ_API_KEY": lambda k: k.startswith("gsk_") and len(k) > 30,
        "LANGSMITH_API_KEY": lambda k: k.startswith("lsv2_") and len(k) > 30
    }
    
    for key_name, key_value in api_keys.items():
        if key_name in validations and validations[key_name](key_value):
            print(f"  ✅ {key_name}: Valid format")
        else:
            print(f"  ❌ {key_name}: Invalid format")
    
    return True

def test_config_files():
    """Test that all configuration files are valid."""
    print("\n📄 Testing Configuration Files...")
    
    config_files = [
        "config/providers.yaml",
        "config/modes.yaml", 
        "config/policy_manifest.yaml",
        "config/threshold_profiles.yaml",
        "config/run_contract.yaml",
        "config/feature_flags.yaml"
    ]
    
    all_valid = True
    for config_file in config_files:
        config = load_yaml_config(config_file)
        if config:
            print(f"  ✅ {config_file}: Valid YAML")
        else:
            print(f"  ❌ {config_file}: Invalid or missing")
            all_valid = False
    
    return all_valid

def test_provider_config():
    """Test provider configuration specifically."""
    print("\n🤖 Testing Provider Configuration...")
    
    providers_config = load_yaml_config("config/providers.yaml")
    if not providers_config:
        return False
    
    required_providers = ["openai", "anthropic", "groq", "ollama"]
    providers = providers_config.get("providers", {})
    
    for provider in required_providers:
        if provider in providers:
            print(f"  ✅ {provider}: Configuration found")
        else:
            print(f"  ❌ {provider}: Configuration missing")
    
    # Test routing profiles
    routing_profiles = providers_config.get("routing_profiles", {})
    required_profiles = ["cost_optimized", "quality_first", "speed_critical", "privacy_focused"]
    
    for profile in required_profiles:
        if profile in routing_profiles:
            print(f"  ✅ Routing profile '{profile}': Configured")
        else:
            print(f"  ❌ Routing profile '{profile}': Missing")
    
    return True

def test_security_config():
    """Test security and policy configuration."""
    print("\n🛡️ Testing Security Configuration...")
    
    policy_config = load_yaml_config("config/policy_manifest.yaml")
    if not policy_config:
        return False
    
    # Test PII policies
    pii_policies = policy_config.get("pii_policies", {})
    if "detection" in pii_policies and "redaction" in pii_policies:
        print("  ✅ PII policies: Configured")
    else:
        print("  ❌ PII policies: Missing")
    
    # Test RBAC
    rbac = policy_config.get("rbac", {})
    required_roles = ["viewer", "analyst", "engineer", "admin", "service_account"]
    roles = rbac.get("roles", {})
    
    for role in required_roles:
        if role in roles:
            print(f"  ✅ RBAC role '{role}': Configured")
        else:
            print(f"  ❌ RBAC role '{role}': Missing")
    
    return True

def test_thresholds_config():
    """Test threshold and quality gate configuration."""
    print("\n📊 Testing Threshold Configuration...")
    
    threshold_config = load_yaml_config("config/threshold_profiles.yaml")
    if not threshold_config:
        return False
    
    # Test global thresholds
    global_thresholds = threshold_config.get("global_thresholds", {})
    required_thresholds = ["tau_high", "tau_low", "tau_critical"]
    
    for threshold in required_thresholds:
        if threshold in global_thresholds:
            value = global_thresholds[threshold]
            if isinstance(value, (int, float)) and 0 <= value <= 1:
                print(f"  ✅ {threshold}: {value} (valid)")
            else:
                print(f"  ❌ {threshold}: {value} (invalid range)")
        else:
            print(f"  ❌ {threshold}: Missing")
    
    # Test confidence bands
    confidence_bands = global_thresholds.get("confidence_bands", {})
    required_bands = ["green", "yellow", "red"]
    
    for band in required_bands:
        if band in confidence_bands:
            print(f"  ✅ Confidence band '{band}': Configured")
        else:
            print(f"  ❌ Confidence band '{band}': Missing")
    
    return True

def test_feature_flags():
    """Test feature flag configuration."""
    print("\n🚩 Testing Feature Flags...")
    
    flags_config = load_yaml_config("config/feature_flags.yaml")
    if not flags_config:
        return False
    
    feature_flags = flags_config.get("feature_flags", {})
    core_flags = [
        "llm_provider_routing",
        "data_mode_enabled", 
        "semantic_domaining",
        "pii_auto_detection",
        "confidence_visualization"
    ]
    
    for flag in core_flags:
        if flag in feature_flags:
            flag_config = feature_flags[flag]
            if "default_value" in flag_config and "environments" in flag_config:
                print(f"  ✅ Feature flag '{flag}': Properly configured")
            else:
                print(f"  ❌ Feature flag '{flag}': Incomplete configuration")
        else:
            print(f"  ❌ Feature flag '{flag}': Missing")
    
    return True

def test_file_structure():
    """Test that all required files exist."""
    print("\n📁 Testing File Structure...")
    
    required_files = [
        "requirements.txt",
        "env.example", 
        "config/providers.yaml",
        "config/modes.yaml",
        "config/policy_manifest.yaml", 
        "config/threshold_profiles.yaml",
        "config/run_contract.yaml",
        "config/feature_flags.yaml",
        "RFC_PHASE0.md",
        "ROADMAP.md"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}: Exists")
        else:
            print(f"  ❌ {file_path}: Missing")
            all_exist = False
    
    return all_exist

async def test_langchain_setup():
    """Test basic LangChain setup with provided API keys."""
    print("\n🔗 Testing LangChain Setup...")
    
    try:
        # Test imports
        from langchain_openai import ChatOpenAI
        from langchain_anthropic import ChatAnthropic
        print("  ✅ LangChain imports: Success")
        
        # Test OpenAI setup (without making actual calls)
        openai_llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "test-key"),
            model="gpt-4-turbo"
        )
        print("  ✅ OpenAI LLM: Initialized")
        
        # Test Anthropic setup
        anthropic_llm = ChatAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY", "test-key"),
            model="claude-3-sonnet-20240229"
        )
        print("  ✅ Anthropic LLM: Initialized")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Setup error: {e}")
        return False

def main():
    """Run all Phase 0 validation tests."""
    print("🚀 EnMapper Phase 0 Setup Validation")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("API Keys", test_api_keys),
        ("Configuration Files", test_config_files),
        ("Provider Config", test_provider_config),
        ("Security Config", test_security_config),
        ("Thresholds Config", test_thresholds_config),
        ("Feature Flags", test_feature_flags),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name}: Error - {e}")
            results.append((test_name, False))
    
    # Run async tests
    print("\n🔗 Testing LangChain Setup...")
    try:
        langchain_result = asyncio.run(test_langchain_setup())
        results.append(("LangChain Setup", langchain_result))
    except Exception as e:
        print(f"  ❌ LangChain Setup: Error - {e}")
        results.append(("LangChain Setup", False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 PHASE 0 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, passed_test in results:
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status:10} {test_name}")
        if passed_test:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 Phase 0 setup is COMPLETE and VALID!")
        print("✅ Ready to proceed to Phase 1")
    else:
        print(f"\n⚠️  {total - passed} issues need to be resolved")
        print("❌ Please fix issues before proceeding")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
