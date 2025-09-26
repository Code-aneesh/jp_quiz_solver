#!/usr/bin/env python3
"""
Test script to verify Japanese Quiz Solver setup
"""

import sys
import os
import traceback

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    try:
        import mss
        print("✓ mss (screen capture)")
    except ImportError as e:
        print(f"✗ mss failed: {e}")
        return False

    try:
        import pytesseract
        print("✓ pytesseract")
    except ImportError as e:
        print(f"✗ pytesseract failed: {e}")
        return False

    try:
        from PIL import Image, ImageOps, ImageFilter
        print("✓ PIL (Pillow)")
    except ImportError as e:
        print(f"✗ PIL failed: {e}")
        return False

    try:
        import tkinter as tk
        print("✓ tkinter")
    except ImportError as e:
        print(f"✗ tkinter failed: {e}")
        return False

    try:
        import google.generativeai as genai
        print("✓ google-generativeai")
    except ImportError as e:
        print(f"✗ google-generativeai failed: {e}")
        return False

    try:
        from openai import OpenAI
        print("✓ openai (optional)")
    except ImportError:
        print("⚠ openai not installed (optional for OpenAI provider)")

    return True

def test_tesseract():
    """Test Tesseract installation"""
    print("\nTesting Tesseract...")
    try:
        import pytesseract
        # Test basic Tesseract functionality
        version = pytesseract.get_tesseract_version()
        print(f"✓ Tesseract version: {version}")
        
        # Test Japanese language support
        languages = pytesseract.get_languages()
        if 'jpn' in languages:
            print("✓ Japanese language support confirmed")
        else:
            print("⚠ Japanese language not found in:", languages)
            return False
            
        return True
    except Exception as e:
        print(f"✗ Tesseract test failed: {e}")
        print("Make sure Tesseract is installed and in PATH")
        return False

def test_screen_capture():
    """Test screen capture functionality"""
    print("\nTesting screen capture...")
    try:
        import mss
        with mss.mss() as sct:
            # Get screen info
            monitors = sct.monitors
            print(f"✓ Found {len(monitors)} monitors")
            
            # Test capture
            test_region = {"left": 0, "top": 0, "width": 100, "height": 100}
            img = sct.grab(test_region)
            print("✓ Screen capture working")
            
            # Test PIL conversion
            from PIL import Image
            pil_img = Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
            print("✓ PIL conversion working")
            
        return True
    except Exception as e:
        print(f"✗ Screen capture test failed: {e}")
        return False

def test_config():
    """Test config file"""
    print("\nTesting config...")
    try:
        import config
        print(f"✓ Config loaded")
        print(f"  Tesseract path: {config.TESSERACT_PATH}")
        print(f"  AI Provider: {config.AI_PROVIDER}")
        print(f"  Capture region: {config.CAPTURE_REGION}")
        
        # Check if API key is set
        if hasattr(config, 'GEMINI_API_KEY') and config.GEMINI_API_KEY != "YOUR_GEMINI_KEY_HERE":
            print("✓ Gemini API key appears to be set")
        else:
            print("⚠ Gemini API key not set - edit config.py or set GEMINI_API_KEY environment variable")
            
        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Japanese Quiz Solver - Setup Test")
    print("=" * 40)
    
    all_passed = True
    
    if not test_imports():
        all_passed = False
    
    if not test_tesseract():
        all_passed = False
    
    if not test_screen_capture():
        all_passed = False
    
    if not test_config():
        all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("✓ All tests passed! Setup looks good.")
        print("\nTo run the application:")
        print("python main.py")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
