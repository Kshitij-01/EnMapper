#!/usr/bin/env python3
"""
Quick Push to GitHub - Ready-to-Copy Commands
"""

def show_quick_commands():
    """Show the exact commands to push to GitHub."""
    
    print("⚡ QUICK PUSH TO GITHUB")
    print("=" * 40)
    
    print("\n🌐 1. First, create repo at: https://github.com/new")
    print("   Name: EnMapper")
    print("   Don't initialize with anything!")
    
    print("\n📋 2. Copy and paste these commands ONE BY ONE:")
    print("-" * 40)
    print("git remote add origin https://github.com/kshitijpatilhere/EnMapper.git")
    print("git branch -M main") 
    print("git push -u origin main")
    
    print("\n🔑 3. When prompted for password, use GitHub Personal Access Token")
    print("   Get token at: https://github.com/settings/tokens")
    
    print("\n✅ 4. After push, verify with:")
    print("git remote -v")
    
    print("\n🎉 Your repository will be live at:")
    print("https://github.com/kshitijpatilhere/EnMapper")

if __name__ == "__main__":
    show_quick_commands()
