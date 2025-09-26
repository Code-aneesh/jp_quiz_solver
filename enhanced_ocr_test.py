#!/usr/bin/env python3
"""
Enhanced OCR Test for JLPT Questions
This script tests different OCR configurations to improve text recognition
"""

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import mss
import cv2
import numpy as np

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def enhanced_preprocess(image):
    """Enhanced preprocessing for JLPT text recognition"""
    
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_cv = img_array
    
    # Convert to grayscale
    if len(img_cv.shape) == 3:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_cv
    
    # Method 1: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Method 2: Gaussian blur + threshold
    blurred = cv2.GaussianBlur(enhanced, (1, 1), 0)
    
    # Method 3: Morphological operations to clean up
    kernel = np.ones((1,1), np.uint8)
    cleaned = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
    
    # Method 4: Adaptive threshold
    binary = cv2.adaptiveThreshold(cleaned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Method 5: Resize for better OCR (3x scaling)
    height, width = binary.shape
    resized = cv2.resize(binary, (width*3, height*3), interpolation=cv2.INTER_CUBIC)
    
    # Convert back to PIL
    return Image.fromarray(resized)

def test_multiple_ocr_configs(image):
    """Test multiple OCR configurations"""
    
    configs = [
        # PSM 6: Uniform block of text
        {'psm': 6, 'oem': 3, 'name': 'PSM 6 (Uniform block)'},
        
        # PSM 7: Single text line
        {'psm': 7, 'oem': 3, 'name': 'PSM 7 (Single line)'},
        
        # PSM 8: Single word
        {'psm': 8, 'oem': 3, 'name': 'PSM 8 (Single word)'},
        
        # PSM 11: Sparse text
        {'psm': 11, 'oem': 3, 'name': 'PSM 11 (Sparse text)'},
        
        # PSM 12: Sparse text with OSD
        {'psm': 12, 'oem': 3, 'name': 'PSM 12 (Sparse + OSD)'},
        
        # PSM 13: Raw line (no layout analysis)
        {'psm': 13, 'oem': 3, 'name': 'PSM 13 (Raw line)'}
    ]
    
    results = []
    
    for config in configs:
        try:
            # Build configuration string
            config_str = f"--psm {config['psm']} --oem {config['oem']} -l jpn+eng"
            config_str += " -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZあいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンー々〜？！（）「」『』。、"
            
            # Get text
            text = pytesseract.image_to_string(image, config=config_str)
            confidence = pytesseract.image_to_data(image, config=config_str, output_type=pytesseract.Output.DICT)
            
            # Calculate average confidence
            confidences = [int(conf) for conf in confidence['conf'] if int(conf) > 0]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0
            
            results.append({
                'config': config['name'],
                'text': text.strip(),
                'confidence': avg_conf,
                'length': len(text.strip())
            })
            
        except Exception as e:
            results.append({
                'config': config['name'],
                'text': f"Error: {e}",
                'confidence': 0,
                'length': 0
            })
    
    return results

def capture_and_test():
    """Capture current screen and test OCR"""
    
    print("🔍 Enhanced JLPT OCR Test")
    print("=" * 50)
    
    # Capture screen
    with mss.mss() as sct:
        # Use the configured region or prompt for manual selection
        region = {"left": 300, "top": 200, "width": 800, "height": 400}
        
        print(f"📸 Capturing region: {region}")
        screenshot = sct.grab(region)
        
        # Convert to PIL Image
        pil_image = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        
        # Save original for reference
        pil_image.save("original_capture.png")
        print("💾 Saved original_capture.png")
        
        # Apply enhanced preprocessing
        processed_image = enhanced_preprocess(pil_image)
        processed_image.save("processed_capture.png")
        print("💾 Saved processed_capture.png")
        
        # Test multiple OCR configurations
        results = test_multiple_ocr_configs(processed_image)
        
        print("\n📊 OCR Results:")
        print("-" * 80)
        
        for result in results:
            print(f"🔧 {result['config']}")
            print(f"   Confidence: {result['confidence']:.1f}%")
            print(f"   Length: {result['length']} chars")
            print(f"   Text: {result['text'][:100]}{'...' if len(result['text']) > 100 else ''}")
            print("-" * 80)
        
        # Find best result
        best_result = max(results, key=lambda x: x['confidence'])
        print(f"\n🏆 Best Result: {best_result['config']}")
        print(f"📝 Full Text:\n{best_result['text']}")
        
        return best_result

if __name__ == "__main__":
    try:
        result = capture_and_test()
        input("\nPress Enter to exit...")
    except Exception as e:
        print(f"❌ Error: {e}")
        input("Press Enter to exit...")
