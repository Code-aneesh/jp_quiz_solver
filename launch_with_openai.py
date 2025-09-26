#!/usr/bin/env python3
"""
Launch Ultimate Japanese Quiz Solver with OpenAI
Forces OpenAI usage and bypasses quota issues
"""

import os
import sys
from pathlib import Path

# Force OpenAI as primary provider
os.environ['AI_PROVIDER'] = 'openai'

# Ensure OpenAI key is available (from setup)
openai_key = os.getenv('OPENAI_API_KEY')
if not openai_key:
    print("❌ OpenAI API key not found in environment variables")
    print("💡 Please restart your PowerShell terminal to load the new environment variable")
    print("   Or manually set it:")
    print("   $env:OPENAI_API_KEY=\"your_openai_key_here\"")
    input("Press Enter to exit...")
    sys.exit(1)

print("🔑 OpenAI API Key loaded successfully")
print(f"🤖 Using OpenAI with key: {openai_key[:8]}...{openai_key[-4:]}")

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def main():
    """Launch with OpenAI configuration"""
    print("🏮 Ultimate Japanese Quiz Solver - OpenAI Mode")
    print("=" * 60)
    
    try:
        # Import and override config
        import config
        config.AI_PROVIDER = "openai"
        
        print(f"✅ AI Provider set to: {config.AI_PROVIDER}")
        
        # Launch the main system
        from main_phase2a import UltimateJapaneseQuizSolver
        
        solver = UltimateJapaneseQuizSolver(
            enable_all_features=True,
            performance_mode="balanced",
            max_workers=4
        )
        
        print("✅ Ultimate solver initialized with OpenAI")
        
        # Option 1: GUI Mode
        print("\n🚀 Choose your mode:")
        print("1. GUI Mode (recommended)")
        print("2. CLI Mode - test single question")  
        print("3. Test offline solver")
        
        choice = input("Enter choice (1/2/3): ").strip()
        
        if choice == "1":
            print("🎮 Starting GUI mode...")
            from main_phase2a import run_gui_mode
            run_gui_mode()
            
        elif choice == "2":
            print("⌨️ CLI mode - testing with sample")
            # Test with sample image
            sample_image_path = "debug_screenshot.png"
            if os.path.exists(sample_image_path):
                result = solver.solve_quiz(sample_image_path)
                print(solver.format_analysis_report(result))
            else:
                print("No sample image found. Please place a quiz image and try again.")
                
        elif choice == "3":
            print("🏮 Testing offline solver...")
            from offline_jlpt_solver import test_with_sample_text
            test_with_sample_text()
            
        else:
            print("❌ Invalid choice")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
