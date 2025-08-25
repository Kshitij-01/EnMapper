#!/usr/bin/env python3
"""
GitHub Setup Helper
Instructions for creating and connecting to GitHub repository
"""

def print_github_instructions():
    """Print step-by-step GitHub setup instructions."""
    
    print("🐙 GitHub Repository Setup Instructions")
    print("=" * 50)
    
    print("\n📝 Step 1: Create Repository on GitHub")
    print("   1. Go to https://github.com")
    print("   2. Log in with: kshitijpatilhere@gmail.com")
    print("   3. Click the '+' icon and select 'New repository'")
    print("   4. Repository name: EnMapper")
    print("   5. Description: AI-Powered Data Mapping and Migration Platform")
    print("   6. Set to Public or Private (your choice)")
    print("   7. Do NOT initialize with README (we already have one)")
    print("   8. Click 'Create repository'")
    
    print("\n🔗 Step 2: Connect Local Repository")
    print("   Copy and paste these commands one by one:")
    print()
    print("   git remote add origin https://github.com/kshitijpatilhere/EnMapper.git")
    print("   git branch -M main")
    print("   git push -u origin main")
    print()
    
    print("\n⚠️  Step 3: Authentication")
    print("   When prompted for credentials:")
    print("   Username: kshitijpatilhere (or your GitHub username)")
    print("   Password: Use your GitHub Personal Access Token")
    print("   (GitHub no longer accepts passwords - you need a token)")
    print()
    print("   To create a token:")
    print("   1. Go to GitHub → Settings → Developer settings → Personal access tokens")
    print("   2. Generate new token (classic)")
    print("   3. Select scopes: repo, workflow")
    print("   4. Copy the token and use it as password")
    
    print("\n✅ Step 4: Verify Setup")
    print("   After successful push, run:")
    print("   git remote -v")
    print("   git status")
    
    print("\n🎉 Your repository will be available at:")
    print("   https://github.com/kshitijpatilhere/EnMapper")
    
    print("\n📋 What's included in your repository:")
    print("   ✅ Complete Phase 0 configuration")
    print("   ✅ Multi-provider LLM setup")
    print("   ✅ Security policies and RBAC")
    print("   ✅ Comprehensive documentation")
    print("   ✅ Validation and setup scripts")
    print("   ✅ Professional README")
    print("   ❌ .env file (protected by .gitignore)")

if __name__ == "__main__":
    print_github_instructions()
