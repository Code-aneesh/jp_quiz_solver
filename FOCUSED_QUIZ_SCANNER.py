#!/usr/bin/env python3
"""
FOCUSED JAPANESE QUIZ SCANNER
============================
Specifically designed to detect and answer Japanese quiz questions
Filters out interface noise and focuses on actual quiz content
"""

import mss
import pytesseract
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import tkinter as tk
from tkinter import scrolledtext, messagebox
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

def is_quiz_content(text):
    """Detect if text contains actual quiz questions"""
    if not text or len(text) < 10:
        return False
    
    # Look for quiz patterns
    quiz_patterns = [
        r'問題\s*[IVX\d]+',  # 問題I, 問題1, etc.
        r'問\s*[1-9]',       # 問1, 問2, etc.
        r'\(\d+\)',          # (16), (17), etc.
        r'[1-4]\.\s*[ぁ-ゟ]+', # Multiple choice options
    ]
    
    japanese_chars = re.findall(r'[ひらがな\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text)
    
    # Must have Japanese characters AND quiz patterns
    has_quiz_pattern = any(re.search(pattern, text) for pattern in quiz_patterns)
    has_enough_japanese = len(japanese_chars) > 20
    
    # Filter out interface text
    interface_keywords = ['SCANNER', 'ULTRA', 'SCAN', 'python', 'OneDrive', 'Desktop', 'cd "', '.py']
    has_interface_noise = any(keyword in text for keyword in interface_keywords)
    
    return has_quiz_pattern and has_enough_japanese and not has_interface_noise

def clean_quiz_text(text):
    """Clean and format quiz text for better processing"""
    if not text:
        return ""
    
    # Remove common OCR artifacts
    text = re.sub(r'[|{}@#$%^&*]', '', text)
    text = re.sub(r'[A-Za-z]{5,}', '', text)  # Remove long English words
    text = re.sub(r'\\[a-zA-Z]+', '', text)    # Remove path fragments
    text = re.sub(r'\s+', ' ', text)           # Normalize whitespace
    
    # Keep only lines that look like quiz content
    lines = text.split('\n')
    clean_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Keep lines with Japanese characters and quiz patterns
        has_japanese = bool(re.search(r'[ひらがな\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', line))
        has_quiz_pattern = bool(re.search(r'問題|問\s*\d|\(\d+\)|[1-4]\.\s*[ぁ-ゟ]', line))
        
        if has_japanese or has_quiz_pattern:
            clean_lines.append(line)
    
    return '\n'.join(clean_lines)

def extract_quiz_text(image):
    """Extract text optimized for Japanese quiz content"""
    try:
        # Enhance image for better OCR
        enhanced = image.convert('L')
        
        # Resize for better OCR
        width, height = enhanced.size
        enhanced = enhanced.resize((width * 2, height * 2), Image.LANCZOS)
        
        # Enhance contrast and sharpness
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(2.0)
        
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(1.5)
        
        # Apply filters
        enhanced = ImageOps.autocontrast(enhanced)
        enhanced = enhanced.filter(ImageFilter.SHARPEN)
        
        # Try multiple OCR approaches
        results = []
        
        # Method 1: Japanese only
        try:
            config = '--psm 6 -l jpn'
            text1 = pytesseract.image_to_string(enhanced, config=config, timeout=10)
            if text1:
                results.append(text1)
        except Exception as e1:
            print(f"Method 1 error: {e1}")
            
        # Method 2: Japanese + English
        try:
            config = '--psm 6 -l jpn+eng'
            text2 = pytesseract.image_to_string(enhanced, config=config, timeout=10)
            if text2:
                results.append(text2)
        except Exception as e2:
            print(f"Method 2 error: {e2}")
            
        # Method 3: Default
        try:
            text3 = pytesseract.image_to_string(enhanced, timeout=10)
            if text3:
                results.append(text3)
        except Exception as e3:
            print(f"Method 3 error: {e3}")
        
        # Return best result
        if results:
            best_text = max(results, key=len)
            return clean_quiz_text(best_text)
        else:
            return ""
        
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""

def solve_japanese_quiz(text):
    """Get answers for Japanese quiz questions"""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = f"""
You are an expert Japanese language teacher and JLPT specialist. Analyze this Japanese quiz and provide answers.

Quiz Text:
{text}

For each question you find, provide:

🎯 QUESTION [number]: [Brief description]
✅ ANSWER: [Correct choice number]
📝 EXPLANATION: [Why this answer is correct in English]
📚 READING: [How to read/pronounce key terms]

Be very specific about which numbered choice is correct (1, 2, 3, or 4).
Focus on grammar, vocabulary, and reading comprehension.
"""
        
        response = model.generate_content(prompt)
        return response.text.strip()
        
    except Exception as e:
        return f"❌ AI Error: {e}"

class FocusedQuizScanner:
    def __init__(self):
        self.processed_texts = set()
        self.scanning = False
        self.setup_ui()
    
    def setup_ui(self):
        self.root = tk.Tk()
        self.root.title("🎯 Focused Japanese Quiz Scanner")
        self.root.geometry("1000x700")
        self.root.configure(bg="#1a1a1a")
        
        # Header
        header = tk.Frame(self.root, bg="#2d5aa0", height=50)
        header.pack(fill="x", pady=(0,10))
        header.pack_propagate(False)
        
        title = tk.Label(header, text="🎯 Japanese Quiz Scanner", 
                        font=("Arial", 14, "bold"), fg="white", bg="#2d5aa0")
        title.pack(pady=12)
        
        # Controls
        controls = tk.Frame(self.root, bg="#1a1a1a")
        controls.pack(fill="x", padx=10, pady=5)
        
        self.scan_btn = tk.Button(controls, text="📷 Scan for Quiz", 
                                 command=self.manual_scan, font=("Arial", 10, "bold"),
                                 bg="#4CAF50", fg="white", padx=20)
        self.scan_btn.pack(side="left", padx=(0,10))
        
        self.auto_btn = tk.Button(controls, text="🔄 Auto Scan", 
                                 command=self.toggle_auto_scan, font=("Arial", 10),
                                 bg="#ff9800", fg="white", padx=20)
        self.auto_btn.pack(side="left", padx=(0,10))
        
        self.clear_btn = tk.Button(controls, text="🗑️ Clear", 
                                  command=self.clear_results, font=("Arial", 10),
                                  bg="#f44336", fg="white", padx=20)
        self.clear_btn.pack(side="left")
        
        # Status
        self.status = tk.Label(self.root, text="Ready to scan for Japanese quiz questions", 
                              fg="#4CAF50", bg="#1a1a1a", font=("Arial", 11))
        self.status.pack(pady=5)
        
        # Results
        self.results = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, 
                                               bg="#000000", fg="#00ff00",
                                               font=("Consolas", 10), height=35)
        self.results.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.update_results("🎯 Japanese Quiz Scanner Ready!\n" + "="*50 + "\n\n" +
                           "Instructions:\n" +
                           "1. Display your Japanese quiz on screen\n" +
                           "2. Click 'Scan for Quiz' to analyze\n" +
                           "3. Get instant answers and explanations\n\n" +
                           "Features:\n" +
                           "✅ Detects Japanese quiz questions\n" +
                           "✅ Filters out interface noise\n" +
                           "✅ Provides detailed explanations\n" +
                           "✅ JLPT-focused analysis")
    
    def update_status(self, text):
        try:
            self.status.config(text=text)
            self.root.update()
        except:
            pass
    
    def update_results(self, text):
        try:
            self.results.config(state="normal")
            self.results.insert(tk.END, "\n" + text + "\n")
            self.results.config(state="disabled")
            self.results.see(tk.END)
        except:
            pass
    
    def clear_results(self):
        try:
            self.results.config(state="normal")
            self.results.delete("1.0", tk.END)
            self.results.config(state="disabled")
        except:
            pass
    
    def manual_scan(self):
        """Single scan for quiz content"""
        self.update_status("🔍 Scanning screen for quiz content...")
        
        try:
            with mss.mss() as sct:
                # Capture full screen
                monitor = sct.monitors[1]
                screenshot = sct.grab(monitor)
                image = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                
                # Extract text
                text = extract_quiz_text(image)
                
                if is_quiz_content(text):
                    text_hash = hashlib.md5(text.encode()).hexdigest()
                    
                    if text_hash not in self.processed_texts:
                        self.processed_texts.add(text_hash)
                        
                        self.update_status("✅ Quiz detected! Getting answers...")
                        
                        # Get AI solution
                        solution = solve_japanese_quiz(text)
                        
                        result = f"""🎯 QUIZ DETECTED & SOLVED!
{datetime.now().strftime('%H:%M:%S')}
{'='*60}

📋 DETECTED QUIZ:
{text}

{'='*60}

🤖 EXPERT SOLUTIONS:

{solution}

{'='*60}"""
                        
                        self.update_results(result)
                        self.update_status("✅ Quiz solved successfully!")
                    else:
                        self.update_status("⚠️ Same quiz already processed")
                else:
                    self.update_status("❌ No quiz content detected")
                    
        except Exception as e:
            self.update_status(f"❌ Scan error: {e}")
    
    def toggle_auto_scan(self):
        """Toggle automatic scanning"""
        if not self.scanning:
            self.scanning = True
            self.auto_btn.config(text="⏸️ Stop Auto", bg="#f44336")
            self.start_auto_scan()
        else:
            self.scanning = False
            self.auto_btn.config(text="🔄 Auto Scan", bg="#ff9800")
            self.update_status("Auto scan stopped")
    
    def start_auto_scan(self):
        """Start automatic scanning in background"""
        def auto_scan_loop():
            while self.scanning:
                try:
                    if not self.root.winfo_exists():
                        break
                    
                    self.root.after(0, self.manual_scan)
                    time.sleep(5)  # Scan every 5 seconds
                    
                except:
                    break
        
        thread = threading.Thread(target=auto_scan_loop, daemon=True)
        thread.start()
    
    def run(self):
        self.root.mainloop()

def main():
    print("🎯 Starting Focused Japanese Quiz Scanner...")
    
    if not os.path.exists(TESSERACT_PATH):
        print(f"❌ Tesseract not found at {TESSERACT_PATH}")
        return
    
    if GEMINI_API_KEY == "YOUR_GEMINI_KEY_HERE":
        print("❌ GEMINI_API_KEY not set! Please set your API key.")
        return
    
    print("✅ Scanner ready!")
    
    scanner = FocusedQuizScanner()
    scanner.run()

if __name__ == "__main__":
    main()
