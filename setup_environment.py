#!/usr/bin/env python3
"""
Environment setup script for EnMapper.
Creates .env file from user input or existing environment variables.
"""

import os
from pathlib import Path


def get_api_key_from_user(provider: str, current_value: str = None) -> str:
    """Get API key from user input."""
    if current_value:
        response = input(f"{provider} API key (press Enter to keep current): ").strip()
        return response if response else current_value
    else:
        return input(f"Enter your {provider} API key: ").strip()


def create_env_file_interactive():
    """Create .env file interactively."""
    print("🔧 EnMapper Environment Setup")
    print("=" * 40)
    
    # Check if .env already exists
    env_path = Path(".env")
    if env_path.exists():
        response = input("⚠️  .env file already exists. Overwrite? (y/N): ").strip().lower()
        if response != 'y':
            print("❌ Setup cancelled.")
            return False
    
    # Get API keys
    print("\n📋 API Key Configuration:")
    print("(You can leave any field empty to skip)")
    
    anthropic_key = get_api_key_from_user("Anthropic")
    openai_key = get_api_key_from_user("OpenAI")
    groq_key = get_api_key_from_user("Groq")
    langchain_key = get_api_key_from_user("LangChain")
    
    # Create env content
    env_content = f'''# LLM Provider API Keys
ANTHROPIC_API_KEY={anthropic_key}
OPENAI_API_KEY={openai_key}
GROQ_API_KEY={groq_key}
LANGCHAIN_API_KEY={langchain_key}

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/enmapper_dev
REDIS_URL=redis://localhost:6379/0

# Application Settings
APP_NAME=EnMapper
ENVIRONMENT=development
DEBUG=true

# Security
SECRET_KEY=dev-secret-key-change-in-production
PII_DETECTION_ENABLED=true

# LLM Configuration
DEFAULT_LLM_PROVIDER=anthropic
ROUTING_PROFILE=quality_first

# Processing Settings
DEFAULT_PROCESSING_MODE=metadata_only
MAX_SAMPLE_ROWS=1000
MAX_SAMPLE_BYTES=10000000

# Budget Controls
DEFAULT_TOKEN_BUDGET=100000
DEFAULT_USD_BUDGET=10.0
COST_TRACKING_ENABLED=true

# Feature Flags
ENABLE_DATA_MODE=true
ENABLE_MIGRATION=true
ENABLE_QUARANTINE=true
ENABLE_ADAPTIVE_THRESHOLDS=false

# Observability
LANGCHAIN_TRACING_ENABLED=true
LANGCHAIN_PROJECT=enmapper
LOG_LEVEL=INFO
METRICS_ENABLED=true

# Quality Thresholds
DOMAIN_CONFIDENCE_HIGH=0.82
DOMAIN_CONFIDENCE_LOW=0.55
MAPPING_SUCCESS_RATE_MIN=0.95

# Optional: JSON Settings Override
# SETTINGS_JSON_PATH=settings.json
'''
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print("\n✅ .env file created successfully!")
        print("🔒 Your API keys are configured for local development")
        print("   (This file is in .gitignore and won't be pushed to GitHub)")
        
        # Test the configuration
        print("\n🧪 Testing configuration...")
        try:
            from settings import get_settings
            settings = get_settings()
            validation = settings.validate_configuration()
            
            active_providers = validation.get('providers', {}).get('active', [])
            print(f"   Active providers: {', '.join(active_providers) if active_providers else 'None'}")
            
            if validation.get('errors'):
                print(f"   ❌ Errors: {len(validation['errors'])}")
                for error in validation['errors']:
                    print(f"      - {error}")
            else:
                print("   ✅ Configuration valid!")
                
        except ImportError:
            print("   ⚠️  Install dependencies first: pip install -r requirements.txt")
        except Exception as e:
            print(f"   ⚠️  Configuration test failed: {e}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False


def create_env_from_existing():
    """Create .env file using existing environment variables."""
    print("🔧 Creating .env from existing environment variables...")
    
    # Get existing API keys from environment
    api_keys = {
        'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY', ''),
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY', ''),
        'GROQ_API_KEY': os.getenv('GROQ_API_KEY', ''),
        'LANGCHAIN_API_KEY': os.getenv('LANGCHAIN_API_KEY', '')
    }
    
    found_keys = [k for k, v in api_keys.items() if v]
    if found_keys:
        print(f"   Found {len(found_keys)} API keys in environment")
        return create_env_file_with_keys(**api_keys)
    else:
        print("   No API keys found in environment")
        return create_env_file_interactive()


def create_env_file_with_keys(**keys):
    """Create .env file with provided keys."""
    env_content = f'''# LLM Provider API Keys
ANTHROPIC_API_KEY={keys.get('ANTHROPIC_API_KEY', '')}
OPENAI_API_KEY={keys.get('OPENAI_API_KEY', '')}
GROQ_API_KEY={keys.get('GROQ_API_KEY', '')}
LANGCHAIN_API_KEY={keys.get('LANGCHAIN_API_KEY', '')}

# Database Configuration  
DATABASE_URL=postgresql://user:password@localhost:5432/enmapper_dev
REDIS_URL=redis://localhost:6379/0

# Application Settings
APP_NAME=EnMapper
ENVIRONMENT=development
DEBUG=true

# Security
SECRET_KEY=dev-secret-key-change-in-production
PII_DETECTION_ENABLED=true

# LLM Configuration
DEFAULT_LLM_PROVIDER=anthropic
ROUTING_PROFILE=quality_first

# Processing Settings
DEFAULT_PROCESSING_MODE=metadata_only
MAX_SAMPLE_ROWS=1000
MAX_SAMPLE_BYTES=10000000

# Budget Controls
DEFAULT_TOKEN_BUDGET=100000
DEFAULT_USD_BUDGET=10.0
COST_TRACKING_ENABLED=true

# Feature Flags
ENABLE_DATA_MODE=true
ENABLE_MIGRATION=true
ENABLE_QUARANTINE=true
ENABLE_ADAPTIVE_THRESHOLDS=false

# Observability
LANGCHAIN_TRACING_ENABLED=true
LANGCHAIN_PROJECT=enmapper
LOG_LEVEL=INFO
METRICS_ENABLED=true

# Quality Thresholds
DOMAIN_CONFIDENCE_HIGH=0.82
DOMAIN_CONFIDENCE_LOW=0.55
MAPPING_SUCCESS_RATE_MIN=0.95

# Optional: JSON Settings Override
# SETTINGS_JSON_PATH=settings.json
'''
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ .env file created successfully!")
        return True
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 EnMapper Environment Setup")
    print("Choose an option:")
    print("1. Interactive setup (enter API keys manually)")
    print("2. Use existing environment variables")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        create_env_file_interactive()
    elif choice == '2':
        create_env_from_existing()
    elif choice == '3':
        print("Setup cancelled.")
    else:
        print("Invalid choice. Please run again.")


if __name__ == "__main__":
    main()
