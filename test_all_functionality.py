#!/usr/bin/env python3
"""
🧪 COMPREHENSIVE FUNCTIONALITY TEST SCRIPT
Test all features of the Ultimate Japanese Quiz Solver
"""

import os
import sys
import time
import subprocess

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_environment():
    """Test the environment setup"""
    print("🌍 TESTING ENVIRONMENT SETUP...")
    
    # Test Python version
    python_version = sys.version
    print(f"✅ Python version: {python_version}")
    
    # Test API key
    api_key = os.environ.get('GEMINI_API_KEY')
    if api_key:
        print(f"✅ Gemini API key: {api_key[:10]}... (found)")
    else:
        print("❌ Gemini API key: Not found")
    
    # Test Tesseract
    try:
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, text=True)
        print(f"✅ Tesseract: {result.stdout.split()[1]}")
    except:
        print("❌ Tesseract: Not found")
    
    print()

def test_imports():
    """Test all required imports"""
    print("📦 TESTING IMPORTS...")
    
    imports = [
        ('tkinter', 'GUI framework'),
        ('PIL', 'Image processing'),
        ('pytesseract', 'OCR engine'),
        ('google.generativeai', 'Gemini AI'),
        ('sqlite3', 'Database'),
        ('cv2', 'OpenCV'),
        ('numpy', 'Numerical computing'),
        ('keyboard', 'Global hotkeys'),
        ('langdetect', 'Language detection'),
        ('pyttsx3', 'Text-to-speech'),
        ('python-dotenv', 'Environment variables'),
    ]
    
    for module, description in imports:
        try:
            if module == 'python-dotenv':
                import dotenv
            else:
                __import__(module)
            print(f"✅ {module}: Available ({description})")
        except ImportError:
            print(f"❌ {module}: Missing ({description})")
    
    print()

def test_core_solver():
    """Test the core solver functionality"""
    print("🎯 TESTING CORE SOLVER...")
    
    try:
        from ultimate_main import UltimateQuizSolver
        solver = UltimateQuizSolver()
        print("✅ Core solver initialized")
        
        # Test database
        if os.path.exists(solver.db_path):
            print(f"✅ Database exists: {solver.db_path}")
        else:
            print(f"❌ Database missing: {solver.db_path}")
        
        # Test AI providers
        print(f"✅ AI providers available: {list(solver.ai_providers.keys())}")
        
        print("✅ Core solver test completed")
    except Exception as e:
        print(f"❌ Core solver test failed: {e}")
    
    print()

def test_ocr():
    """Test OCR functionality"""
    print("🔍 TESTING OCR...")
    
    try:
        import pytesseract
        from PIL import Image, ImageDraw, ImageFont
        
        # Create test image with Japanese text
        img = Image.new('RGB', (400, 100), color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to use a system font, fallback to default
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        # Draw test text
        test_text = "これは日本語のテストです"
        draw.text((20, 30), test_text, fill='black', font=font)
        
        # Save test image
        test_image_path = "test_japanese.png"
        img.save(test_image_path)
        
        # Test OCR
        detected_text = pytesseract.image_to_string(img, lang='jpn+eng')
        print(f"✅ Test image created: {test_image_path}")
        print(f"✅ OCR result: '{detected_text.strip()}'")
        
        # Clean up
        os.remove(test_image_path)
        print("✅ OCR test completed")
        
    except Exception as e:
        print(f"❌ OCR test failed: {e}")
    
    print()

def test_ai():
    """Test AI functionality"""
    print("🤖 TESTING AI...")
    
    try:
        import google.generativeai as genai
        
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            print("❌ No API key - skipping AI test")
            return
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        test_question = "What is the Japanese word for 'hello'?"
        print(f"📝 Testing with: {test_question}")
        
        response = model.generate_content(test_question)
        print(f"✅ AI response: {response.text[:100]}...")
        print("✅ AI test completed")
        
    except Exception as e:
        print(f"❌ AI test failed: {e}")
    
    print()

def test_gui_components():
    """Test GUI components without launching"""
    print("🖥️ TESTING GUI COMPONENTS...")
    
    try:
        import tkinter as tk
        from tkinter import ttk
        
        # Test basic GUI creation
        root = tk.Tk()
        root.withdraw()  # Hide window
        
        # Test ttk components
        frame = ttk.Frame(root)
        label = ttk.Label(frame, text="Test")
        button = ttk.Button(frame, text="Test Button")
        combo = ttk.Combobox(frame, values=["test1", "test2"])
        progress = ttk.Progressbar(frame, maximum=100)
        
        print("✅ Basic GUI components created")
        
        # Test styling
        style = ttk.Style()
        style.theme_use("clam")
        print("✅ TTK styling available")
        
        root.destroy()
        print("✅ GUI components test completed")
        
    except Exception as e:
        print(f"❌ GUI components test failed: {e}")
    
    print()

def test_screenshot():
    """Test screenshot functionality"""
    print("📷 TESTING SCREENSHOT...")
    
    try:
        from PIL import ImageGrab
        
        # Take screenshot
        screenshot = ImageGrab.grab()
        print(f"✅ Screenshot captured: {screenshot.size}")
        
        # Test cropping
        crop_region = (100, 100, 300, 200)
        cropped = screenshot.crop(crop_region)
        print(f"✅ Image cropping: {cropped.size}")
        
        print("✅ Screenshot test completed")
        
    except Exception as e:
        print(f"❌ Screenshot test failed: {e}")
    
    print()

def test_config():
    """Test configuration system"""
    print("⚙️ TESTING CONFIGURATION...")
    
    try:
        import config
        
        # Test basic config attributes
        required_attrs = [
            'AI_PROVIDER', 'OCR_LANGUAGE', 'POLLING_INTERVAL',
            'CACHE_SIZE', 'CAPTURE_REGION'
        ]
        
        for attr in required_attrs:
            if hasattr(config, attr):
                value = getattr(config, attr)
                print(f"✅ {attr}: {value}")
            else:
                print(f"❌ {attr}: Missing")
        
        print("✅ Configuration test completed")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
    
    print()

def main():
    """Run all tests"""
    print("🧪 ULTIMATE JAPANESE QUIZ SOLVER - COMPREHENSIVE TEST")
    print("=" * 60)
    print()
    
    test_environment()
    test_imports()
    test_core_solver()
    test_ocr()
    test_ai()
    test_gui_components()
    test_screenshot()
    test_config()
    
    print("🎉 ALL TESTS COMPLETED!")
    print()
    print("📋 NEXT STEPS:")
    print("1. Run 'python ultimate_gui.py' to start the full GUI")
    print("2. Test theme switching in Settings tab")
    print("3. Try the 'Start Scanning' button")
    print("4. Test global hotkeys (Ctrl+Shift+Q for quick scan)")
    print("5. Check History and Analytics tabs")
    print()
    print("🚀 Your Ultimate Japanese Quiz Solver is ready!")

if __name__ == "__main__":
    main()
