#!/usr/bin/env python3
"""
Create New GitHub Repository for EnMapper
Step-by-step instructions for creating and pushing to GitHub
"""

def create_github_repository_guide():
    """Guide for creating new GitHub repository and pushing code."""
    
    print("🚀 Creating New GitHub Repository for EnMapper")
    print("=" * 60)
    
    print("\n📝 STEP 1: Create Repository on GitHub")
    print("-" * 40)
    print("1. Go to: https://github.com/new")
    print("2. Login with: kshitijpatilhere@gmail.com")
    print("3. Repository settings:")
    print("   • Repository name: EnMapper")
    print("   • Description: AI-Powered Data Mapping and Migration Platform")
    print("   • Visibility: Public (recommended) or Private")
    print("   • ❌ Do NOT check 'Add a README file'")
    print("   • ❌ Do NOT add .gitignore")
    print("   • ❌ Do NOT choose a license")
    print("4. Click 'Create repository'")
    
    print("\n🔗 STEP 2: Connect Local Repository")
    print("-" * 40)
    print("After creating the repo, GitHub will show you commands.")
    print("Copy and run THESE EXACT COMMANDS in order:")
    print()
    print("git remote add origin https://github.com/kshitijpatilhere/EnMapper.git")
    print("git branch -M main")
    print("git push -u origin main")
    print()
    
    print("\n🔐 STEP 3: Authentication")
    print("-" * 40)
    print("When prompted for credentials:")
    print("• Username: kshitijpatilhere")
    print("• Password: [Use Personal Access Token, NOT your GitHub password]")
    print()
    print("To get a Personal Access Token:")
    print("1. Go to: https://github.com/settings/tokens")
    print("2. Click 'Generate new token (classic)'")
    print("3. Give it a name: 'EnMapper Development'")
    print("4. Select scopes: ✅ repo, ✅ workflow")
    print("5. Click 'Generate token'")
    print("6. COPY THE TOKEN (you won't see it again!)")
    print("7. Use this token as your password when pushing")
    
    print("\n✅ STEP 4: Verify Success")
    print("-" * 40)
    print("After successful push, run these commands:")
    print("git remote -v")
    print("git log --oneline")
    print()
    print("Your repository will be live at:")
    print("🌐 https://github.com/kshitijpatilhere/EnMapper")
    
    print("\n📦 STEP 5: What Gets Pushed")
    print("-" * 40)
    print("✅ Complete Phase 0 configuration")
    print("✅ All YAML configuration files")
    print("✅ Validation and setup scripts")
    print("✅ Professional README and documentation")
    print("✅ Requirements and environment template")
    print("❌ .env file (safely protected by .gitignore)")
    print("❌ venv1/ virtual environment (ignored)")
    
    print("\n⚠️  IMPORTANT NOTES")
    print("-" * 40)
    print("• Your API keys are SAFE - they're in .env which is git-ignored")
    print("• The virtual environment won't be uploaded (it's huge and unnecessary)")
    print("• Others can recreate the environment using requirements.txt")
    print("• Your project will be professional and ready for collaboration")

if __name__ == "__main__":
    create_github_repository_guide()
