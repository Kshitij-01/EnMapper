#!/usr/bin/env python3
"""
EnMapper API Connectivity Test
Test actual API connectivity with provided keys
"""

import os
import sys
import asyncio
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

async def test_openai_connection():
    """Test OpenAI API connectivity."""
    try:
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-3.5-turbo",  # Use cheaper model for testing
            max_tokens=10
        )
        
        # Simple test prompt
        response = await llm.ainvoke("Say 'Hello from OpenAI!'")
        print(f"  ✅ OpenAI: {response.content[:50]}...")
        return True
        
    except Exception as e:
        print(f"  ❌ OpenAI: {str(e)[:100]}...")
        return False

async def test_anthropic_connection():
    """Test Anthropic API connectivity."""
    try:
        from langchain_anthropic import ChatAnthropic
        
        llm = ChatAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-3-haiku-20240307",  # Use cheaper model for testing
            max_tokens=10
        )
        
        # Simple test prompt
        response = await llm.ainvoke("Say 'Hello from Anthropic!'")
        print(f"  ✅ Anthropic: {response.content[:50]}...")
        return True
        
    except Exception as e:
        print(f"  ❌ Anthropic: {str(e)[:100]}...")
        return False

async def test_langsmith_connection():
    """Test LangSmith connectivity."""
    try:
        import langsmith
        
        # Just test that we can create a client
        client = langsmith.Client(
            api_key=os.getenv("LANGSMITH_API_KEY")
        )
        
        # Test basic connectivity (this will try to reach the API)
        try:
            # This will fail gracefully if no projects exist
            projects = list(client.list_runs(project_name="non-existent-project", limit=1))
            print(f"  ✅ LangSmith: Connected successfully")
            return True
        except Exception as inner_e:
            # If we get here, it means the API call was made but project doesn't exist
            # This is actually good - it means connectivity works
            if "not found" in str(inner_e).lower() or "does not exist" in str(inner_e).lower():
                print(f"  ✅ LangSmith: Connected successfully (no projects yet)")
                return True
            else:
                raise inner_e
                
    except Exception as e:
        print(f"  ❌ LangSmith: {str(e)[:100]}...")
        return False

def test_environment_loading():
    """Test that environment variables are loaded correctly."""
    print("\n🌍 Testing Environment Loading...")
    
    required_vars = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY", 
        "GROQ_API_KEY",
        "LANGSMITH_API_KEY"
    ]
    
    all_loaded = True
    for var in required_vars:
        value = os.getenv(var)
        if value and len(value) > 10:  # Basic validation
            print(f"  ✅ {var}: Loaded (***{value[-8:]})")
        else:
            print(f"  ❌ {var}: Missing or invalid")
            all_loaded = False
    
    return all_loaded

async def main():
    """Run all connectivity tests."""
    print("🧪 EnMapper API Connectivity Test")
    print("=" * 50)
    
    # Test environment loading first
    env_ok = test_environment_loading()
    if not env_ok:
        print("\n❌ Environment variables not loaded properly!")
        return False
    
    print("\n🔌 Testing API Connectivity...")
    print("⚠️  This will make actual API calls with small costs")
    
    # Run API tests
    tests = [
        ("OpenAI", test_openai_connection()),
        ("Anthropic", test_anthropic_connection()),
        ("LangSmith", test_langsmith_connection())
    ]
    
    results = []
    for test_name, test_coro in tests:
        try:
            result = await test_coro
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name}: Unexpected error - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 API CONNECTIVITY SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, passed_test in results:
        status = "✅ CONNECTED" if passed_test else "❌ FAILED"
        print(f"{status:15} {test_name}")
        if passed_test:
            passed += 1
    
    print(f"\nResults: {passed}/{total} APIs connected successfully")
    
    if passed >= 2:  # At least 2 out of 3 should work for a good setup
        print("\n🎉 API connectivity is GOOD!")
        print("✅ EnMapper can connect to LLM providers")
    else:
        print(f"\n⚠️  Only {passed} API connections working")
        print("⚠️  You may want to check your API keys")
    
    return passed >= 2

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        sys.exit(1)
