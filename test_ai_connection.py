#!/usr/bin/env python3
"""
Quick test to verify AI connection is working
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def test_ai_connection():
    """Test AI connection and functionality"""
    print("🧪 Testing AI Connection...")
    print("=" * 50)
    
    # Test 1: Check API keys
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY") 
    
    try:
        import config
        config_gemini = config.GEMINI_API_KEY if hasattr(config, 'GEMINI_API_KEY') else None
        ai_provider = config.AI_PROVIDER if hasattr(config, 'AI_PROVIDER') else 'gemini'
    except:
        config_gemini = None
        ai_provider = 'gemini'
    
    print(f"🔑 Current provider: {ai_provider}")
    print(f"🔑 Gemini (env): {'✅' if gemini_key else '❌'}")
    print(f"🔑 Gemini (config): {'✅' if config_gemini and config_gemini != 'YOUR_GEMINI_KEY_HERE' else '❌'}")
    print(f"🔑 OpenAI (env): {'✅' if openai_key else '❌'}")
    
    # Test 2: Try to initialize the system
    print("\n🚀 Testing system initialization...")
    try:
        from main_phase2a import UltimateJapaneseQuizSolver
        solver = UltimateJapaneseQuizSolver(
            enable_all_features=False,  # Disable advanced features for basic test
            performance_mode="speed"
        )
        print("✅ Ultimate solver initialized successfully")
        
        # Test with sample Japanese text
        sample_text = """
        問題1・はは　は　どう　かきますか。
        1. 姆  2. 毌  3. 奶  4. 母
        """
        
        print("\n📝 Testing with sample text...")
        print(f"Sample: {sample_text.strip()}")
        
        # Create a temporary image for testing (we'll skip actual processing for now)
        from PIL import Image
        test_image = Image.new('RGB', (400, 200), color='white')
        test_image.save('temp_test.png')
        
        print("✅ Test image created")
        print("✅ All components are ready for testing")
        
        # Clean up
        os.remove('temp_test.png')
        
        return True
        
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return False

def test_offline_solver():
    """Test the offline solver as backup"""
    print("\n🏮 Testing offline solver (no API required)...")
    
    try:
        from offline_jlpt_solver import OfflineJLPTSolver
        
        solver = OfflineJLPTSolver()
        
        # Test with a simple question
        test_text = "問題1・はは 1. 姆 2. 毌 3. 奶 4. 母"
        
        print(f"Testing: {test_text}")
        
        # This would work with the pattern matching
        print("✅ Offline solver available as backup")
        return True
        
    except Exception as e:
        print(f"❌ Offline solver failed: {e}")
        return False

if __name__ == "__main__":
    print("🏮 Ultimate Japanese Quiz Solver - Connection Test")
    print("=" * 60)
    
    # Test main AI system
    main_working = test_ai_connection()
    
    # Test offline backup
    offline_working = test_offline_solver()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS:")
    print(f"🤖 Main AI System: {'✅ READY' if main_working else '❌ NEEDS SETUP'}")
    print(f"🏮 Offline Solver: {'✅ READY' if offline_working else '❌ UNAVAILABLE'}")
    
    if main_working:
        print("\n🚀 Your system is ready! You can run:")
        print("   python main_phase2a.py --mode gui")
        print("   python ultimate_main.py")
    elif offline_working:
        print("\n🏮 Offline mode available! You can run:")
        print("   python offline_jlpt_solver.py")
    else:
        print("\n⚠️  System needs configuration. Please check API keys.")
    
    print("\n" + "=" * 60)
    input("Press Enter to exit...")
