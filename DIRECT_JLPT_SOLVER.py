#!/usr/bin/env python3
"""
DIRECT JLPT SOLVER
==================
Immediately processes any Japanese text found on screen and provides answers
No stability checks, no waiting - instant answers!
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

# AI imports
import google.generativeai as genai

# Configuration
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_KEY_HERE")

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Configure AI
if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_KEY_HERE":
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except:
        pass

def is_japanese_text(text):
    """Check if text contains Japanese characters"""
    if not text or len(text) < 3:
        return False
    
    # Look for any Japanese characters
    japanese_chars = re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text)
    return len(japanese_chars) >= 2

def enhance_image_for_ocr(pil_img):
    """Enhance image for better OCR"""
    img = pil_img.convert('L')
    width, height = img.size
    img = img.resize((width * 4, height * 4), Image.LANCZOS)  # 4x scaling
    
    # Enhance contrast and sharpness
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(3.0)
    
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(2.5)
    
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)
    
    return img

def extract_japanese_text(image):
    """Extract Japanese text with multiple attempts"""
    enhanced = enhance_image_for_ocr(image)
    
    # Try multiple configurations for better results
    configs = [
        "--psm 6 -l jpn+eng",
        "--psm 7 -l jpn+eng", 
        "--psm 8 -l jpn+eng",
        "--psm 11 -l jpn+eng",
        "--psm 12 -l jpn+eng",
        "--psm 13 -l jpn+eng"
    ]
    
    best_text = ""
    for config in configs:
        try:
            text = pytesseract.image_to_string(enhanced, config=config).strip()
            if len(text) > len(best_text) and is_japanese_text(text):
                best_text = text
        except:
            continue
    
    # Also try simple extraction
    try:
        simple_text = pytesseract.image_to_string(image, lang="jpn+eng").strip()
        if len(simple_text) > len(best_text) and is_japanese_text(simple_text):
            best_text = simple_text
    except:
        pass
    
    return re.sub(r'\s+', ' ', best_text) if best_text else ""

def get_jlpt_answer(text):
    """Get answer from Gemini AI"""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = f"""
You are an expert Japanese language teacher and JLPT specialist. Analyze this Japanese text and provide complete answers.

Text to analyze:
{text}

For each question you find, provide:

🎯 ANSWER: [The correct choice number/letter or complete answer]
📝 TRANSLATION: [English translation of the question]
✅ EXPLANATION: [Why this answer is correct, including grammar/vocabulary explanations]
📚 LEVEL: [JLPT level this question tests]

If there are multiple questions, answer ALL of them separately.
Be very detailed and educational in your explanations.
Always provide the correct answer with confidence.
"""
        
        response = model.generate_content(prompt)
        return response.text.strip()
        
    except Exception as e:
        return f"❌ AI Error: {e}\n\nCheck your GEMINI_API_KEY is set correctly."

class DirectJLPTSolver:
    def __init__(self):
        self.processing = False
        self.last_text = ""
        self.setup_ui()
    
    def setup_ui(self):
        """Create simple, effective UI"""
        self.root = tk.Tk()
        self.root.title("🎯 DIRECT JLPT SOLVER - Instant Answers")
        self.root.geometry("800x600")
        self.root.configure(bg="#1a1a1a")
        self.root.attributes("-topmost", True)
        
        # Header
        header = tk.Frame(self.root, bg="#2d2d2d", height=50)
        header.pack(fill="x", pady=(0,10))
        header.pack_propagate(False)
        
        title = tk.Label(header, text="🎯 DIRECT JLPT SOLVER", 
                        font=("Arial", 14, "bold"), fg="#00ff00", bg="#2d2d2d")
        title.pack(pady=10)
        
        # Status
        self.status = tk.Label(self.root, text="🟡 Scanning for Japanese questions...", 
                              fg="#ffff00", bg="#1a1a1a", font=("Arial", 10))
        self.status.pack(pady=5)
        
        # Results area
        self.results = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, 
                                               bg="#000000", fg="#ffffff",
                                               font=("Consolas", 10), height=30)
        self.results.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.update_results("🎯 DIRECT JLPT SOLVER READY!\n" + "="*50 + "\n\n" +
                           "✅ Scanning entire screen for Japanese questions\n" +
                           "✅ Instant AI-powered answers\n" +
                           "✅ No manual setup required\n\n" +
                           "Waiting for Japanese text... 📖")
        
        # Start scanning immediately
        self.root.after(500, self.start_scanning)
    
    def update_status(self, text):
        """Update status safely"""
        try:
            self.status.config(text=text)
        except:
            pass
    
    def update_results(self, text):
        """Update results safely"""
        try:
            self.results.config(state="normal")
            self.results.delete("1.0", tk.END)
            self.results.insert("1.0", text)
            self.results.config(state="disabled")
            self.results.see(tk.END)
        except:
            pass
    
    def start_scanning(self):
        """Start aggressive screen scanning"""
        def scan():
            with mss.mss() as sct:
                # Get screen dimensions
                monitor = sct.monitors[1]
                screen_width = monitor['width']
                screen_height = monitor['height']
                
                # Create scanning regions - focus on center area where documents usually are
                regions = [
                    # Center area - where most documents are displayed
                    {"left": screen_width//4, "top": screen_height//6, 
                     "width": screen_width//2, "height": screen_height//2},
                    # Wider center area
                    {"left": screen_width//6, "top": screen_height//8, 
                     "width": 2*screen_width//3, "height": 3*screen_height//4},
                    # Full screen as fallback
                    {"left": 0, "top": 0, "width": screen_width, "height": screen_height}
                ]
                
                region_index = 0
                
                while True:
                    try:
                        if not self.root.winfo_exists():
                            break
                        
                        # Cycle through regions
                        region = regions[region_index % len(regions)]
                        region_index += 1
                        
                        # Capture region
                        img = sct.grab(region)
                        pil_img = Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
                        
                        # Extract text
                        text = extract_japanese_text(pil_img)
                        
                        if text:
                            self.update_status(f"📝 Found text: {text[:60]}...")
                            
                            # Check if it's Japanese
                            if is_japanese_text(text):
                                self.update_status("🔥 Processing Japanese question...")
                                
                                # Process immediately if new text
                                if text != self.last_text and not self.processing:
                                    self.processing = True
                                    self.last_text = text
                                    
                                    # Get AI answer
                                    answer = get_jlpt_answer(text)
                                    
                                    # Display results
                                    result_text = f"""🏆 JAPANESE QUESTION FOUND & ANSWERED
{'='*70}

📍 REGION: {region['width']}x{region['height']} at ({region['left']}, {region['top']})
⏰ TIME: {datetime.now().strftime('%H:%M:%S')}

📋 DETECTED TEXT:
{text}

{'='*70}

{answer}

{'='*70}
🔍 Continuing scan for more questions...
"""
                                    
                                    self.update_results(result_text)
                                    self.update_status("✅ Answer provided! Scanning continues...")
                                    
                                    self.processing = False
                                    
                                    # Brief pause before continuing
                                    time.sleep(3)
                        else:
                            self.update_status("🔍 Scanning for Japanese text...")
                        
                        # Quick scan interval
                        time.sleep(0.5)
                        
                    except Exception as e:
                        self.update_status(f"❌ Scan error: {e}")
                        time.sleep(2)
        
        # Start in background thread
        thread = threading.Thread(target=scan, daemon=True)
        thread.start()
    
    def run(self):
        self.root.mainloop()

def main():
    print("🎯 Starting Direct JLPT Solver...")
    
    # Check Tesseract
    if not os.path.exists(TESSERACT_PATH):
        print(f"❌ Tesseract not found at: {TESSERACT_PATH}")
        return
    
    # Check API key
    if GEMINI_API_KEY == "YOUR_GEMINI_KEY_HERE":
        print("❌ GEMINI_API_KEY not set. Set it with:")
        print('setx GEMINI_API_KEY "your_api_key_here"')
        return
    
    print("✅ All systems ready!")
    
    solver = DirectJLPTSolver()
    solver.run()

if __name__ == "__main__":
    main()
