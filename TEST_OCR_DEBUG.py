#!/usr/bin/env python3
"""
OCR TEST AND DEBUG TOOL
========================
Simple test to see what OCR is detecting from your screen
"""

import mss
import pytesseract
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import time
import os
import re

# Configuration
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def is_japanese_text(text):
    """Check if text contains Japanese characters"""
    if not text:
        return False
    
    japanese_patterns = [
        r'[あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん]',  # Hiragana
        r'[アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン]',  # Katakana
        r'[一-龯]',  # Kanji
        r'[。、]',   # Japanese punctuation
    ]
    
    japanese_count = 0
    for pattern in japanese_patterns:
        japanese_count += len(re.findall(pattern, text))
    
    return japanese_count >= 1

def enhance_image(pil_img):
    """Enhanced image processing"""
    img = pil_img.convert('L')
    width, height = img.size
    img = img.resize((width * 3, height * 3), Image.LANCZOS)
    
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(3.0)
    
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)
    img = img.filter(ImageFilter.SHARPEN)
    
    return img

def test_ocr_detection():
    """Test OCR detection across the screen"""
    print("🔍 OCR TEST MODE - Press Ctrl+C to exit")
    print("=" * 50)
    
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Primary monitor
        
        # Test different areas of the screen
        grid_size = 3
        grid_width = monitor['width'] // grid_size
        grid_height = monitor['height'] // grid_size
        
        region_counter = 0
        
        while True:
            try:
                for row in range(grid_size):
                    for col in range(grid_size):
                        left = monitor['left'] + col * grid_width
                        top = monitor['top'] + row * grid_height
                        width = grid_width
                        height = grid_height
                        
                        region = {
                            'left': left,
                            'top': top,
                            'width': width,
                            'height': height
                        }
                        
                        # Capture region
                        img = sct.grab(region)
                        pil_img = Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
                        
                        # Try simple OCR first
                        simple_text = pytesseract.image_to_string(pil_img, lang="jpn+eng").strip()
                        
                        # Try enhanced OCR
                        enhanced = enhance_image(pil_img)
                        enhanced_text = pytesseract.image_to_string(enhanced, lang="jpn+eng").strip()
                        
                        # Show results if any text found
                        if simple_text or enhanced_text:
                            region_counter += 1
                            print(f"\n📍 REGION {region_counter}: ({left}, {top}) - {width}x{height}")
                            
                            if simple_text:
                                print(f"📄 Simple OCR: {simple_text[:100]}...")
                                print(f"🗾 Contains Japanese: {is_japanese_text(simple_text)}")
                            
                            if enhanced_text and enhanced_text != simple_text:
                                print(f"✨ Enhanced OCR: {enhanced_text[:100]}...")
                                print(f"🗾 Contains Japanese: {is_japanese_text(enhanced_text)}")
                                
                            print("-" * 30)
                
                # Wait before next scan
                time.sleep(2)
                
            except KeyboardInterrupt:
                print("\n🛑 OCR test stopped by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(1)

if __name__ == "__main__":
    # Check if Tesseract exists
    if not os.path.exists(TESSERACT_PATH):
        print(f"❌ Tesseract not found at: {TESSERACT_PATH}")
        print("Please install Tesseract OCR with Japanese language support")
        exit(1)
    
    print("✅ Tesseract found!")
    test_ocr_detection()
