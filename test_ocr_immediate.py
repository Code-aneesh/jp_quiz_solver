#!/usr/bin/env python3
"""
🔍 IMMEDIATE OCR TEST - Debug Japanese Detection
Test OCR and detection capabilities immediately
"""

import sys
import os
import time
import mss
import pytesseract
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import cv2
import numpy as np

# Configuration
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def contains_japanese(text):
    """Check if text contains Japanese characters"""
    if not text:
        return False
    
    # Unicode ranges for Japanese characters
    japanese_ranges = [
        (0x3040, 0x309F),  # Hiragana
        (0x30A0, 0x30FF),  # Katakana
        (0x4E00, 0x9FAF),  # Kanji (CJK Unified Ideographs)
        (0x3400, 0x4DBF),  # CJK Extension A
        (0xFF65, 0xFF9F),  # Half-width Katakana
    ]
    
    japanese_chars = 0
    for char in text:
        char_code = ord(char)
        for start, end in japanese_ranges:
            if start <= char_code <= end:
                japanese_chars += 1
                break
    
    return japanese_chars > 0

def preprocess_image_advanced(image):
    """Advanced image preprocessing for better OCR"""
    # Convert to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Apply noise reduction
    gray = cv2.fastNlMeansDenoising(gray)
    
    # Scale up 3x for better OCR
    height, width = gray.shape
    gray = cv2.resize(gray, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
    
    # Convert back to PIL
    processed = Image.fromarray(gray)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(processed)
    processed = enhancer.enhance(2.0)
    
    # Sharpen
    processed = processed.filter(ImageFilter.SHARPEN)
    
    return processed

def test_ocr_methods(image):
    """Test multiple OCR methods"""
    results = []
    
    # OCR configurations to try
    configs = [
        ("Default", "--psm 6 --oem 3"),
        ("Japanese Optimized", "--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZあいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンー々〜？！（）「」『』。、①②③④⑤"),
        ("Multiline", "--psm 4 --oem 3"),
        ("Single Block", "--psm 6 --oem 3"),
        ("Sparse Text", "--psm 11 --oem 3")
    ]
    
    # Test original image
    for config_name, config_str in configs:
        try:
            text = pytesseract.image_to_string(image, lang="jpn+eng", config=config_str).strip()
            if text:
                results.append((f"Original - {config_name}", text))
        except Exception as e:
            print(f"Failed {config_name}: {e}")
    
    # Test preprocessed image
    try:
        processed = preprocess_image_advanced(image)
        for config_name, config_str in configs:
            try:
                text = pytesseract.image_to_string(processed, lang="jpn+eng", config=config_str).strip()
                if text:
                    results.append((f"Processed - {config_name}", text))
            except Exception as e:
                print(f"Failed Processed {config_name}: {e}")
    except Exception as e:
        print(f"Preprocessing failed: {e}")
    
    return results

def main():
    print("🔍 IMMEDIATE OCR TEST STARTING")
    print("=" * 50)
    
    # Test Tesseract
    try:
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract version: {version}")
        
        languages = pytesseract.get_languages()
        if 'jpn' in languages:
            print("✅ Japanese language pack available")
        else:
            print("❌ Japanese language pack NOT found!")
            return
    except Exception as e:
        print(f"❌ Tesseract test failed: {e}")
        return
    
    print("\n🖥️ Capturing screen in 3 seconds...")
    print("Position your Japanese text on screen!")
    
    for i in range(3, 0, -1):
        print(f"⏰ {i}...")
        time.sleep(1)
    
    try:
        # Capture screen
        with mss.mss() as sct:
            # Get primary monitor
            monitor = sct.monitors[1]
            print(f"📺 Monitor: {monitor}")
            
            # Take screenshot
            screenshot = sct.grab(monitor)
            image = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            
            # Save screenshot for debugging
            debug_path = "debug_screenshot.png"
            image.save(debug_path)
            print(f"💾 Screenshot saved: {debug_path}")
            
            print("\n🔍 RUNNING OCR TESTS...")
            print("-" * 50)
            
            # Test OCR methods
            results = test_ocr_methods(image)
            
            if results:
                print(f"\n✅ FOUND {len(results)} OCR RESULTS:")
                for i, (method, text) in enumerate(results, 1):
                    has_japanese = contains_japanese(text)
                    has_numbers = any(c.isdigit() for c in text)
                    has_options = any(c in text for c in ['①', '②', '③', '④', '⑤', 'A', 'B', 'C', 'D', '1', '2', '3', '4'])
                    
                    print(f"\n{i}. {method}")
                    print(f"Text ({len(text)} chars): {text[:200]}...")
                    print(f"Japanese: {has_japanese} | Numbers: {has_numbers} | Options: {has_options}")
                    
                    if has_japanese:
                        print("🎯 THIS CONTAINS JAPANESE!")
                
            else:
                print("❌ NO TEXT DETECTED")
                print("\nTroubleshooting:")
                print("- Make sure Japanese text is clearly visible")
                print("- Try larger font sizes")
                print("- Ensure good contrast (dark text on light background)")
                print("- Check if Tesseract Japanese language pack is installed")
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
