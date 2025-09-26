#!/usr/bin/env python3
"""
ULTRA AGGRESSIVE JLPT SCANNER
=============================
This will definitely find and answer your Japanese questions!
Scans the entire screen aggressively with multiple methods.
"""

import mss
import pytesseract
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import tkinter as tk
from tkinter import scrolledtext
import threading
import time
import os
import re
from datetime import datetime
import hashlib

# AI imports
import google.generativeai as genai

# Configuration
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_KEY_HERE")

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Configure AI
if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_KEY_HERE":
    genai.configure(api_key=GEMINI_API_KEY)

def is_japanese_text(text):
    """More lenient Japanese detection"""
    if not text or len(text) < 2:
        return False
    
    # Look for Japanese characters OR common JLPT patterns
    japanese_chars = re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text)
    has_numbers = re.search(r'[1-4]', text)
    has_question_words = any(word in text for word in ['問', 'もん', 'から', 'を', 'に', 'は', 'が'])
    
    return len(japanese_chars) >= 1 or (has_numbers and has_question_words)

def extract_text_aggressive(image):
    """Multiple OCR attempts with different settings"""
    results = []
    
    # Method 1: Direct OCR
    try:
        direct = pytesseract.image_to_string(image, lang="jpn+eng").strip()
        if direct and len(direct) > 3:
            results.append(direct)
    except:
        pass
    
    # Method 2: Enhanced image
    try:
        enhanced = image.convert('L')
        width, height = enhanced.size
        enhanced = enhanced.resize((width * 3, height * 3), Image.LANCZOS)
        
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(2.5)
        
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(2.0)
        
        enhanced = ImageOps.autocontrast(enhanced)
        enhanced = enhanced.filter(ImageFilter.SHARPEN)
        
        enhanced_text = pytesseract.image_to_string(enhanced, lang="jpn+eng").strip()
        if enhanced_text and len(enhanced_text) > 3:
            results.append(enhanced_text)
    except:
        pass
    
    # Method 3: Multiple PSM modes
    psm_modes = [6, 7, 8, 11, 12, 13]
    for psm in psm_modes:
        try:
            config = f"--psm {psm} -l jpn+eng"
            text = pytesseract.image_to_string(image, config=config).strip()
            if text and len(text) > 3:
                results.append(text)
        except:
            continue
    
    # Return the longest result
    if results:
        best_result = max(results, key=len)
        return re.sub(r'\s+', ' ', best_result)
    return ""

def get_answer_from_ai(text):
    """Get comprehensive answer from AI"""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = f"""
You are an expert Japanese teacher and JLPT specialist. Analyze this Japanese text and answer ALL questions found.

Text:
{text}

For each question you find, provide:

🎯 QUESTION [number]: ANSWER [correct choice number/letter]
📝 TRANSLATION: [English translation]  
✅ EXPLANATION: [Why this answer is correct]
📚 LEVEL: [JLPT level]

Answer ALL questions you can find in the text. Be very thorough and educational.
If you see multiple choice options (1,2,3,4 or A,B,C,D), always specify which number/letter is correct.
"""
        
        response = model.generate_content(prompt)
        return response.text.strip()
        
    except Exception as e:
        return f"❌ AI Error: {e}"

class UltraAggressiveScanner:
    def __init__(self):
        self.last_texts = set()  # Track processed texts
        self.processing = False
        self.scan_count = 0
        self.setup_ui()
    
    def setup_ui(self):
        self.root = tk.Tk()
        self.root.title("🚀 ULTRA AGGRESSIVE JLPT SCANNER")
        self.root.geometry("900x700")
        self.root.configure(bg="#0a0a0a")
        self.root.attributes("-topmost", True)
        
        # Header
        header = tk.Frame(self.root, bg="#ff4444", height=60)
        header.pack(fill="x", pady=(0,10))
        header.pack_propagate(False)
        
        title = tk.Label(header, text="🚀 ULTRA AGGRESSIVE SCANNER", 
                        font=("Arial", 16, "bold"), fg="white", bg="#ff4444")
        title.pack(pady=15)
        
        # Status
        self.status = tk.Label(self.root, text="🔥 ULTRA SCAN MODE - DETECTING EVERYTHING!", 
                              fg="#ff4444", bg="#0a0a0a", font=("Arial", 12, "bold"))
        self.status.pack(pady=5)
        
        # Results
        self.results = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, 
                                               bg="#000000", fg="#00ff00",
                                               font=("Consolas", 11), height=35)
        self.results.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.update_results("🚀 ULTRA AGGRESSIVE SCANNER ACTIVE!\n" + "="*60 + "\n\n" +
                           "🔥 MAXIMUM POWER SCANNING MODE\n" +
                           "✅ Multiple OCR methods simultaneously\n" +
                           "✅ Scans ENTIRE screen every 0.2 seconds\n" +
                           "✅ Processes ANY Japanese text immediately\n" +
                           "✅ Zero stability checks - INSTANT processing\n\n" +
                           "🎯 READY TO DETECT YOUR JLPT QUESTIONS!")
        
        # Start ultra-aggressive scanning
        self.root.after(300, self.start_ultra_scan)
    
    def update_status(self, text):
        try:
            self.status.config(text=text)
        except:
            pass
    
    def update_results(self, text):
        try:
            self.results.config(state="normal")
            self.results.delete("1.0", tk.END)
            self.results.insert("1.0", text)
            self.results.config(state="disabled")
            self.results.see(tk.END)
        except:
            pass
    
    def start_ultra_scan(self):
        """Ultra aggressive scanning - maximum power!"""
        def ultra_scan():
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                
                # Scan regions - cover everything!
                regions = []
                
                # Full screen
                regions.append({"left": 0, "top": 0, "width": monitor["width"], "height": monitor["height"]})
                
                # Quarter sections
                half_w, half_h = monitor["width"]//2, monitor["height"]//2
                regions.extend([
                    {"left": 0, "top": 0, "width": half_w, "height": half_h},
                    {"left": half_w, "top": 0, "width": half_w, "height": half_h},
                    {"left": 0, "top": half_h, "width": half_w, "height": half_h},
                    {"left": half_w, "top": half_h, "width": half_w, "height": half_h}
                ])
                
                # Center focus areas (where documents usually are)
                center_regions = [
                    {"left": monitor["width"]//4, "top": monitor["height"]//4, 
                     "width": monitor["width"]//2, "height": monitor["height"]//2},
                    {"left": monitor["width"]//6, "top": monitor["height"]//6, 
                     "width": 2*monitor["width"]//3, "height": 2*monitor["height"]//3},
                ]
                regions.extend(center_regions)
                
                region_idx = 0
                
                while True:
                    try:
                        if not self.root.winfo_exists():
                            break
                        
                        self.scan_count += 1
                        region = regions[region_idx % len(regions)]
                        region_idx += 1
                        
                        # Ultra fast capture
                        img = sct.grab(region)
                        pil_img = Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
                        
                        # Extract text with all methods
                        text = extract_text_aggressive(pil_img)
                        
                        if text and len(text) > 5:
                            text_hash = hashlib.md5(text.encode()).hexdigest()
                            
                            self.update_status(f"🔍 SCAN #{self.scan_count}: Found text: {text[:70]}...")
                            
                            # Check if Japanese OR has question patterns
                            if is_japanese_text(text) and text_hash not in self.last_texts:
                                self.last_texts.add(text_hash)
                                
                                if not self.processing:
                                    self.processing = True
                                    
                                    self.update_status("🔥 JAPANESE DETECTED! PROCESSING NOW!")
                                    
                                    # Get AI answer immediately
                                    answer = get_answer_from_ai(text)
                                    
                                    # Show results
                                    result_text = f"""🚀 ULTRA SCAN SUCCESS! JAPANESE FOUND & ANSWERED!
{'='*80}

🎯 SCAN #{self.scan_count} at {datetime.now().strftime('%H:%M:%S')}
📍 REGION: {region["width"]}x{region["height"]} at ({region["left"]}, {region["top"]})

📋 DETECTED TEXT:
{text}

{'='*80}

🤖 AI EXPERT ANSWERS:

{answer}

{'='*80}
🔥 CONTINUING ULTRA SCAN FOR MORE QUESTIONS...
"""
                                    
                                    self.update_results(result_text)
                                    self.processing = False
                                    
                                    # Brief pause to show results
                                    time.sleep(2)
                        else:
                            self.update_status(f"🔍 ULTRA SCAN #{self.scan_count}: Scanning region {region_idx % len(regions)}...")
                        
                        # Ultra fast scanning - 0.2 second intervals!
                        time.sleep(0.2)
                        
                    except Exception as e:
                        self.update_status(f"❌ Scan #{self.scan_count} error: {e}")
                        time.sleep(1)
        
        thread = threading.Thread(target=ultra_scan, daemon=True)
        thread.start()
    
    def run(self):
        self.root.mainloop()

def main():
    print("🚀 Starting ULTRA AGGRESSIVE JLPT SCANNER...")
    
    if not os.path.exists(TESSERACT_PATH):
        print(f"❌ Tesseract not found!")
        return
    
    if GEMINI_API_KEY == "YOUR_GEMINI_KEY_HERE":
        print("❌ GEMINI_API_KEY not set!")
        return
    
    print("✅ ULTRA POWER MODE READY!")
    
    scanner = UltraAggressiveScanner()
    scanner.run()

if __name__ == "__main__":
    main()
