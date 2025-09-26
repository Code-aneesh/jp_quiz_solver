#!/usr/bin/env python3
"""
TARGETED JLPT SCANNER
====================
Focuses specifically on document areas where JLPT questions are displayed
Ignores application windows and interfaces
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

def is_jlpt_question_text(text):
    """Detect specifically JLPT question patterns"""
    if not text or len(text) < 10:
        return False
    
    # Look for JLPT question indicators
    jlpt_indicators = [
        '問題',  # Question
        'ちいさい', 'ひがし', 'そら', 'かれんだー',  # Words from your questions
        'はは', 'やま', 'こんしゅう', 'てんき',  # More words from your questions
        'むいか', 'ここに',  # More words
    ]
    
    # Look for multiple choice patterns with Japanese context
    has_choices = bool(re.search(r'\([0-9]+\).*[1-4]\.', text))
    has_japanese = bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text))
    has_jlpt_words = any(word in text for word in jlpt_indicators)
    
    # Check for the specific question format from your image
    has_question_format = bool(re.search(r'問[0-9]+', text)) or '___のことば' in text
    
    return (has_japanese and has_choices) or has_jlpt_words or has_question_format

def extract_clean_text(image):
    """Extract and clean text for JLPT questions"""
    # Enhance image specifically for Japanese text
    enhanced = image.convert('L')
    width, height = enhanced.size
    
    # Scale up significantly for better OCR
    enhanced = enhanced.resize((width * 4, height * 4), Image.LANCZOS)
    
    # High contrast for clean text
    enhancer = ImageEnhance.Contrast(enhanced)
    enhanced = enhancer.enhance(3.0)
    
    # Sharp text
    enhancer = ImageEnhance.Sharpness(enhanced)
    enhanced = enhancer.enhance(2.0)
    
    enhanced = ImageOps.autocontrast(enhanced)
    enhanced = enhanced.filter(ImageFilter.SHARPEN)
    
    # Try multiple OCR configurations optimized for Japanese
    configs = [
        "--psm 6 -l jpn+eng",
        "--psm 4 -l jpn+eng",
        "--psm 3 -l jpn+eng",
        "-l jpn+eng"
    ]
    
    best_text = ""
    for config in configs:
        try:
            text = pytesseract.image_to_string(enhanced, config=config).strip()
            if len(text) > len(best_text):
                best_text = text
        except:
            continue
    
    # Clean up the text
    if best_text:
        # Remove excessive whitespace
        best_text = re.sub(r'\s+', ' ', best_text)
        # Remove obviously incorrect OCR artifacts
        best_text = re.sub(r'[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF0-9A-Za-z\s\.\(\)\-・]', '', best_text)
    
    return best_text

def answer_jlpt_questions(text):
    """Get expert JLPT answers"""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = f"""
You are an expert JLPT teacher. The following text contains Japanese language questions. Please analyze and answer them.

Japanese Text:
{text}

For each question you identify, provide:

🎯 QUESTION [number]: ANSWER [correct choice number]
📝 READING: [How to read the Japanese]  
📖 TRANSLATION: [English meaning]
✅ EXPLANATION: [Why this answer is correct - detailed grammar/vocabulary explanation]
📚 LEVEL: [JLPT level N1-N5]

Focus on questions that ask you to choose the correct kanji/writing for hiragana words.
Answer ALL questions you can find with specific choice numbers (1, 2, 3, or 4).
"""
        
        response = model.generate_content(prompt)
        return response.text.strip()
        
    except Exception as e:
        return f"❌ AI Error: {e}\nCheck your GEMINI_API_KEY"

class TargetedJLPTScanner:
    def __init__(self):
        self.processed_texts = set()
        self.processing = False
        self.scan_count = 0
        self.setup_ui()
    
    def setup_ui(self):
        self.root = tk.Tk()
        self.root.title("🎯 TARGETED JLPT SCANNER")
        self.root.geometry("800x600")
        self.root.configure(bg="#1a1a2e")
        
        # Make window small and position in corner to avoid interference
        self.root.geometry("400x300+50+50")
        
        # Header
        header = tk.Frame(self.root, bg="#16213e", height=50)
        header.pack(fill="x")
        header.pack_propagate(False)
        
        title = tk.Label(header, text="🎯 JLPT SCANNER", 
                        font=("Arial", 12, "bold"), fg="#00ff88", bg="#16213e")
        title.pack(pady=10)
        
        # Status
        self.status = tk.Label(self.root, text="🔍 Scanning for JLPT questions...", 
                              fg="#00ff88", bg="#1a1a2e", font=("Arial", 9))
        self.status.pack(pady=5)
        
        # Compact results area
        self.results = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, 
                                               bg="#0f0f23", fg="#00ff88",
                                               font=("Consolas", 8), height=15)
        self.results.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.update_results("🎯 TARGETED JLPT SCANNER READY\n" + "="*40 + "\n\n" +
                           "✅ Focused on document areas\n" +
                           "✅ Ignores application windows\n" +
                           "✅ Optimized for JLPT questions\n\n" +
                           "Scanning for your JLPT questions...")
        
        # Start scanning after short delay
        self.root.after(1000, self.start_targeted_scanning)
    
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
    
    def start_targeted_scanning(self):
        """Scan specific areas where documents are typically displayed"""
        def targeted_scan():
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                screen_w, screen_h = monitor['width'], monitor['height']
                
                # Define document-focused regions (avoid taskbars and title bars)
                document_regions = [
                    # Main document area (center-right where PDFs usually open)
                    {"left": screen_w//3, "top": screen_h//10, 
                     "width": 2*screen_w//3, "height": 4*screen_h//5},
                    
                    # Center area (where most documents are displayed)
                    {"left": screen_w//4, "top": screen_h//8, 
                     "width": screen_w//2, "height": 3*screen_h//4},
                    
                    # Right half (common for split-screen document viewing)
                    {"left": screen_w//2, "top": screen_h//10, 
                     "width": screen_w//2, "height": 4*screen_h//5},
                    
                    # Full center (avoiding edges where UI elements are)
                    {"left": screen_w//6, "top": screen_h//6, 
                     "width": 2*screen_w//3, "height": 2*screen_h//3},
                ]
                
                region_index = 0
                
                while True:
                    try:
                        if not self.root.winfo_exists():
                            break
                        
                        self.scan_count += 1
                        region = document_regions[region_index % len(document_regions)]
                        region_index += 1
                        
                        # Capture document region
                        img = sct.grab(region)
                        pil_img = Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
                        
                        # Extract text with Japanese optimization
                        text = extract_clean_text(pil_img)
                        
                        if text and len(text) > 20:  # Substantial text found
                            text_hash = hashlib.md5(text.encode()).hexdigest()
                            
                            self.update_status(f"📖 Scan #{self.scan_count}: {text[:50]}...")
                            
                            # Check if this looks like JLPT questions
                            if is_jlpt_question_text(text) and text_hash not in self.processed_texts:
                                self.processed_texts.add(text_hash)
                                
                                if not self.processing:
                                    self.processing = True
                                    
                                    self.update_status("🎯 JLPT Questions Found! Processing...")
                                    
                                    # Get expert answers
                                    answers = answer_jlpt_questions(text)
                                    
                                    # Display results
                                    result_text = f"""🎯 JLPT QUESTIONS DETECTED & ANSWERED!
{"="*60}

📍 REGION: {region['width']}x{region['height']} at ({region['left']}, {region['top']})
⏰ TIME: {datetime.now().strftime('%H:%M:%S')}

📋 DETECTED QUESTIONS:
{text}

{"="*60}

🤖 EXPERT ANSWERS:

{answers}

{"="*60}
🔍 Continuing scan for more questions...
"""
                                    
                                    self.update_results(result_text)
                                    self.processing = False
                                    
                                    # Pause to show results
                                    time.sleep(5)
                        else:
                            self.update_status(f"🔍 Scan #{self.scan_count}: Searching document areas...")
                        
                        # Moderate scanning speed to avoid interference
                        time.sleep(1.0)
                        
                    except Exception as e:
                        self.update_status(f"❌ Error: {str(e)[:50]}")
                        time.sleep(2)
        
        thread = threading.Thread(target=targeted_scan, daemon=True)
        thread.start()
    
    def run(self):
        self.root.mainloop()

def main():
    print("🎯 Starting Targeted JLPT Scanner...")
    
    if not os.path.exists(TESSERACT_PATH):
        print("❌ Tesseract not found!")
        return
    
    if GEMINI_API_KEY == "YOUR_GEMINI_KEY_HERE":
        print("❌ GEMINI_API_KEY not set!")
        return
    
    print("✅ Ready to scan for JLPT questions!")
    
    scanner = TargetedJLPTScanner()
    scanner.run()

if __name__ == "__main__":
    main()
