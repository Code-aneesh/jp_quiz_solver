#!/usr/bin/env python3
"""
Direct OpenAI Launcher - Forces OpenAI usage
"""

import os
import sys
from pathlib import Path

# Force environment variables
os.environ['AI_PROVIDER'] = 'openai'
os.environ['OPENAI_API_KEY'] = 'sk-qrst5678qrst5678qrst5678qrst5678qrst5678'

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

print("🚀 DIRECT OPENAI LAUNCHER")
print("=" * 50)
print(f"✅ OpenAI API Key: {os.environ['OPENAI_API_KEY'][:8]}...{os.environ['OPENAI_API_KEY'][-8:]}")
print(f"✅ AI Provider: {os.environ['AI_PROVIDER']}")

# Import and override config
import config
config.AI_PROVIDER = "openai"
config.OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

print(f"✅ Config overridden - Provider: {config.AI_PROVIDER}")

# Test OpenAI connection
try:
    from openai import OpenAI
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    print("✅ OpenAI client initialized successfully")
except Exception as e:
    print(f"❌ OpenAI client failed: {e}")
    input("Press Enter to continue anyway...")

# Launch the main system
print("\n🎯 Launching Ultimate Japanese Quiz Solver with OpenAI...")

try:
    from main_phase2a import run_gui_mode
    run_gui_mode()
except Exception as e:
    print(f"❌ Error launching GUI: {e}")
    
    # Fallback: try the offline solver
    print("\n🏮 Launching offline solver as fallback...")
    try:
        from offline_jlpt_solver import test_with_sample_text
        test_with_sample_text()
    except Exception as e2:
        print(f"❌ Offline solver also failed: {e2}")
    
input("\nPress Enter to exit...")
