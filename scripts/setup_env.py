#!/usr/bin/env python3
"""
EnMapper Environment Setup Script
Creates .env file with provided API keys
"""

import os
from pathlib import Path

def create_env_file():
    """Create .env file with the provided API keys."""
    
    # API keys - these will be read from user input or environment
    # NOTE: Replace these with your actual API keys when running locally
    api_keys = {
        "OPENAI_API_KEY": "your-openai-api-key-here",
        "ANTHROPIC_API_KEY": "your-anthropic-api-key-here", 
        "GROQ_API_KEY": "your-groq-api-key-here",
        "LANGSMITH_API_KEY": "your-langsmith-api-key-here"
    }
    
    # Read the template
    template_path = Path("env.example")
    if not template_path.exists():
        print("❌ env.example not found!")
        return False
    
    with open(template_path, 'r') as f:
        content = f.read()
    
    # Replace placeholder values with actual API keys
    for key, value in api_keys.items():
        # Find the template line and replace it
        placeholder_patterns = [
            f"{key}=sk-your-openai-api-key-here",
            f"{key}=sk-ant-your-anthropic-api-key-here", 
            f"{key}=gsk_your-groq-api-key-here",
            f"{key}=lsv2_your-langsmith-api-key-here"
        ]
        
        for pattern in placeholder_patterns:
            if pattern in content:
                content = content.replace(pattern, f"{key}={value}")
    
    # Write the .env file
    env_path = Path(".env")
    try:
        with open(env_path, 'w') as f:
            f.write(content)
        print(f"✅ Created .env file with your API keys")
        print(f"⚠️  WARNING: .env contains sensitive data - never commit to git!")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

if __name__ == "__main__":
    print("🔧 EnMapper Environment Setup")
    print("=" * 40)
    
    if create_env_file():
        print("\n🎉 Environment setup complete!")
        print("✅ You can now run EnMapper with your API keys")
    else:
        print("\n❌ Environment setup failed!")
        print("Please manually copy env.example to .env and add your API keys")
