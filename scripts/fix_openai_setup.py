#!/usr/bin/env python3
"""
Fix OpenAI Setup - Remove organization requirement
"""

import os
from pathlib import Path
from dotenv import load_dotenv

def fix_openai_config():
    """Fix OpenAI configuration by removing organization requirement."""
    
    # Load current environment
    load_dotenv()
    
    print("🔧 Fixing OpenAI Configuration...")
    
    # Test OpenAI without organization header
    try:
        from langchain_openai import ChatOpenAI
        
        # Create client without organization
        llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-3.5-turbo",
            max_tokens=10
        )
        
        # Test with a simple prompt
        response = llm.invoke("Say 'Hello from OpenAI!'")
        print(f"✅ OpenAI Fixed: {response.content}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI Error: {e}")
        
        # If still failing, try removing organization from environment
        env_path = Path(".env")
        if env_path.exists():
            with open(env_path, 'r') as f:
                content = f.read()
            
            # Comment out organization line if it exists
            if "OPENAI_ORG_ID=" in content and not "OPENAI_ORG_ID=#" in content:
                content = content.replace("OPENAI_ORG_ID=", "# OPENAI_ORG_ID=")
                
                with open(env_path, 'w') as f:
                    f.write(content)
                
                print("🔧 Commented out OPENAI_ORG_ID in .env")
                print("🔄 Try running the test again...")
        
        return False

if __name__ == "__main__":
    success = fix_openai_config()
    if success:
        print("\n🎉 OpenAI configuration fixed!")
    else:
        print("\n⚠️ OpenAI may need manual configuration")
