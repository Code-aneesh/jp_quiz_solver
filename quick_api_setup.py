#!/usr/bin/env python3
"""
Quick API Key Setup for Ultimate Japanese Quiz Solver
Helps you set up API keys easily
"""

import os
import sys

def setup_api_keys():
    """Interactive API key setup"""
    print("🔑 Ultimate Japanese Quiz Solver - API Key Setup")
    print("=" * 60)
    
    print("\n📋 API Provider Options:")
    print("1. Gemini (Google) - Free tier: 50 requests/day")
    print("   Get key from: https://aistudio.google.com/app/apikey")
    print("2. OpenAI (GPT-4) - Paid service: $0.03/1K tokens") 
    print("   Get key from: https://platform.openai.com/api-keys")
    print("3. Both (recommended for reliability)")
    
    choice = input("\nWhich provider would you like to setup? (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        print("\n🔑 Setting up Gemini API...")
        gemini_key = input("Enter your Gemini API key (starts with AIzaSy...): ").strip()
        
        if gemini_key and gemini_key.startswith('AIzaSy'):
            # Method 1: Set environment variable (Windows)
            try:
                import subprocess
                subprocess.run(['setx', 'GEMINI_API_KEY', gemini_key], check=True, shell=True)
                print("✅ Gemini API key set as environment variable")
            except:
                print("⚠️  Could not set environment variable automatically")
                
            # Method 2: Update config.py
            try:
                with open('config.py', 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Replace the placeholder
                if 'YOUR_GEMINI_KEY_HERE' in content:
                    content = content.replace('YOUR_GEMINI_KEY_HERE', gemini_key)
                    with open('config.py', 'w', encoding='utf-8') as f:
                        f.write(content)
                    print("✅ Gemini API key added to config.py")
                else:
                    print("ℹ️  config.py already has API key configured")
            except Exception as e:
                print(f"⚠️  Could not update config.py: {e}")
        else:
            print("❌ Invalid Gemini API key format")
    
    if choice in ['2', '3']:
        print("\n🔑 Setting up OpenAI API...")
        openai_key = input("Enter your OpenAI API key (starts with sk-...): ").strip()
        
        if openai_key and openai_key.startswith('sk-'):
            try:
                import subprocess
                subprocess.run(['setx', 'OPENAI_API_KEY', openai_key], check=True, shell=True)
                print("✅ OpenAI API key set as environment variable")
            except:
                print("⚠️  Could not set environment variable automatically")
        else:
            print("❌ Invalid OpenAI API key format")
    
    print("\n🔄 Testing API Key Setup...")
    
    # Test current environment
    gemini_set = bool(os.getenv('GEMINI_API_KEY'))
    openai_set = bool(os.getenv('OPENAI_API_KEY'))
    
    # Check config.py
    try:
        import config
        config_gemini = config.GEMINI_API_KEY != "YOUR_GEMINI_KEY_HERE"
    except:
        config_gemini = False
    
    print(f"Environment - Gemini: {'✅' if gemini_set else '❌'}")
    print(f"Environment - OpenAI: {'✅' if openai_set else '❌'}")  
    print(f"Config.py - Gemini: {'✅' if config_gemini else '❌'}")
    
    print("\n📝 Next Steps:")
    print("1. Restart your terminal/PowerShell")
    print("2. Run: python main_phase2a.py --mode gui")
    print("3. Or run: python ultimate_main.py")
    
    if not (gemini_set or config_gemini or openai_set):
        print("\n⚠️  Manual Setup Required:")
        print("If automatic setup failed, you can manually:")
        print("1. Edit config.py and replace 'YOUR_GEMINI_KEY_HERE' with your key")
        print("2. Or set environment variables through Windows System Properties")
    
    return gemini_set or config_gemini or openai_set

if __name__ == "__main__":
    try:
        success = setup_api_keys()
        if success:
            print("\n🎉 Setup complete! Your system is ready to use.")
        else:
            print("\n⚠️  Setup incomplete. Please check the instructions above.")
        
        input("\nPress Enter to exit...")
    except KeyboardInterrupt:
        print("\n❌ Setup cancelled by user.")
    except Exception as e:
        print(f"\n❌ Setup error: {e}")
        input("Press Enter to exit...")
