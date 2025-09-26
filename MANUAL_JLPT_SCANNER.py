#!/usr/bin/env python3
"""
MANUAL JLPT SCANNER
===================
Let's you manually select exactly where your JLPT questions are
Then continuously monitors that area for answers
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

# AI imports
import google.generativeai as genai

# Configuration
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_KEY_HERE")

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Configure AI
if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_KEY_HERE":
    genai.configure(api_key=GEMINI_API_KEY)

def extract_japanese_text_optimized(image):
    """Optimized extraction for JLPT questions"""
    # Convert and enhance
    enhanced = image.convert('L')
    width, height = enhanced.size
    enhanced = enhanced.resize((width * 3, height * 3), Image.LANCZOS)
    
    # High contrast and sharpness
    enhancer = ImageEnhance.Contrast(enhanced)
    enhanced = enhancer.enhance(2.5)
    
    enhancer = ImageEnhance.Sharpness(enhanced)
    enhanced = enhancer.enhance(2.0)
    
    enhanced = ImageOps.autocontrast(enhanced)
    enhanced = enhanced.filter(ImageFilter.SHARPEN)
    
    # Multiple OCR attempts
    texts = []
    
    # Standard OCR
    try:
        text1 = pytesseract.image_to_string(enhanced, lang="jpn+eng").strip()
        if text1:
            texts.append(text1)
    except:
        pass
    
    # Different PSM modes
    for psm in [3, 4, 6, 7, 8]:
        try:
            config = f"--psm {psm} -l jpn+eng"
            text = pytesseract.image_to_string(enhanced, config=config).strip()
            if text:
                texts.append(text)
        except:
            continue
    
    # Return the best (longest) result
    if texts:
        best = max(texts, key=len)
        return re.sub(r'\s+', ' ', best)
    return ""

def get_jlpt_answers(text):
    """Get comprehensive JLPT answers"""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = f"""
You are an expert JLPT teacher. I have Japanese questions that need to be answered.

Text with questions:
{text}

Please provide detailed answers for each question you find:

🎯 QUESTION [number]: ANSWER [choice number 1,2,3,or 4]
📝 JAPANESE READING: [how to pronounce the Japanese]
📖 ENGLISH TRANSLATION: [what the question/sentence means in English]
✅ DETAILED EXPLANATION: [why this answer is correct - explain the grammar, vocabulary, kanji usage]
📚 JLPT LEVEL: [N1, N2, N3, N4, or N5]

Answer ALL questions you can identify. Be very specific about choice numbers.
These are multiple choice questions where you need to select the correct kanji/writing.
"""
        
        response = model.generate_content(prompt)
        return response.text.strip()
        
    except Exception as e:
        return f"❌ AI Error: {e}\nMake sure GEMINI_API_KEY is set correctly."

class ManualJLPTScanner:
    def __init__(self):
        self.selected_region = None
        self.monitoring = False
        self.last_text = ""
        self.setup_ui()
    
    def setup_ui(self):
        self.root = tk.Tk()
        self.root.title("📍 MANUAL JLPT SCANNER")
        self.root.geometry("700x500")
        self.root.configure(bg="#2c2c2c")
        
        # Header
        header = tk.Frame(self.root, bg="#4a4a4a", height=80)
        header.pack(fill="x", pady=(0,10))
        header.pack_propagate(False)
        
        title = tk.Label(header, text="📍 MANUAL JLPT SCANNER", 
                        font=("Arial", 16, "bold"), fg="#ffffff", bg="#4a4a4a")
        title.pack(pady=10)
        
        subtitle = tk.Label(header, text="Select exactly where your JLPT questions are", 
                           font=("Arial", 10), fg="#cccccc", bg="#4a4a4a")
        subtitle.pack()
        
        # Controls
        controls = tk.Frame(self.root, bg="#2c2c2c")
        controls.pack(pady=10)
        
        self.select_btn = tk.Button(controls, text="📍 SELECT QUESTION AREA", 
                                   command=self.select_region,
                                   bg="#4CAF50", fg="white", font=("Arial", 12, "bold"),
                                   padx=20, pady=10)
        self.select_btn.pack(side="left", padx=10)
        
        self.clear_btn = tk.Button(controls, text="🔄 RESET", 
                                  command=self.reset,
                                  bg="#ff5722", fg="white", font=("Arial", 12, "bold"),
                                  padx=20, pady=10)
        self.clear_btn.pack(side="left", padx=10)
        
        # Status
        self.status = tk.Label(self.root, text="Click 'SELECT QUESTION AREA' to choose where your JLPT questions are", 
                              fg="#ffffff", bg="#2c2c2c", font=("Arial", 11))
        self.status.pack(pady=10)
        
        # Results area
        self.results = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, 
                                               bg="#1a1a1a", fg="#ffffff",
                                               font=("Consolas", 10), height=25)
        self.results.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.update_results("📍 MANUAL JLPT SCANNER\n" + "="*50 + "\n\n" +
                           "INSTRUCTIONS:\n" +
                           "1. Click 'SELECT QUESTION AREA' button\n" +
                           "2. Drag to select exactly around your JLPT questions\n" +
                           "3. Scanner will monitor that area and provide answers\n" +
                           "4. Use 'RESET' to select a different area\n\n" +
                           "Ready to scan your JLPT questions!")
    
    def update_status(self, text):
        self.status.config(text=text)
    
    def update_results(self, text):
        self.results.config(state="normal")
        self.results.delete("1.0", tk.END)
        self.results.insert("1.0", text)
        self.results.config(state="disabled")
        self.results.see(tk.END)
    
    def select_region(self):
        """Interactive region selection"""
        self.root.withdraw()  # Hide main window
        
        try:
            self.update_status("Drag to select your JLPT question area...")
            region = self.interactive_region_select()
            
            if region and region['width'] > 50 and region['height'] > 50:
                self.selected_region = region
                self.update_status(f"✅ Region selected: {region['width']}×{region['height']} - Monitoring for questions...")
                self.start_monitoring()
            else:
                self.update_status("❌ Selection cancelled or too small. Try again.")
                
        except Exception as e:
            self.update_status(f"❌ Selection error: {e}")
        finally:
            self.root.deiconify()  # Show main window
    
    def interactive_region_select(self):
        """Full-screen selection overlay"""
        coords = {}
        
        selector = tk.Toplevel(self.root)
        selector.attributes("-fullscreen", True)
        selector.attributes("-alpha", 0.3)
        selector.configure(bg="black")
        selector.attributes("-topmost", True)
        
        canvas = tk.Canvas(selector, cursor="cross", bg="black", highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        
        # Instructions
        instruction = tk.Label(selector, 
                             text="📍 DRAG TO SELECT YOUR JLPT QUESTIONS AREA\nPress ESC to cancel", 
                             fg="yellow", bg="black", font=("Arial", 20, "bold"))
        instruction.place(relx=0.5, rely=0.1, anchor="center")
        
        start_pos = [0, 0]
        rect_id = None
        
        def start_drag(event):
            nonlocal start_pos, rect_id
            start_pos[0] = event.x_root
            start_pos[1] = event.y_root
            if rect_id:
                canvas.delete(rect_id)
            rect_id = canvas.create_rectangle(event.x, event.y, event.x, event.y, 
                                            outline="red", width=4)
        
        def update_drag(event):
            if rect_id:
                x1 = start_pos[0] - selector.winfo_rootx()
                y1 = start_pos[1] - selector.winfo_rooty()
                x2 = event.x_root - selector.winfo_rootx()
                y2 = event.y_root - selector.winfo_rooty()
                canvas.coords(rect_id, x1, y1, x2, y2)
        
        def end_drag(event):
            x1 = min(start_pos[0], event.x_root)
            y1 = min(start_pos[1], event.y_root)
            x2 = max(start_pos[0], event.x_root)
            y2 = max(start_pos[1], event.y_root)
            
            coords['left'] = int(x1)
            coords['top'] = int(y1)
            coords['width'] = int(x2 - x1)
            coords['height'] = int(y2 - y1)
            selector.destroy()
        
        def cancel_selection(event):
            selector.destroy()
        
        canvas.bind("<ButtonPress-1>", start_drag)
        canvas.bind("<B1-Motion>", update_drag)
        canvas.bind("<ButtonRelease-1>", end_drag)
        selector.bind("<Escape>", cancel_selection)
        selector.focus_set()
        
        selector.mainloop()
        return coords if coords else None
    
    def start_monitoring(self):
        """Monitor the selected region continuously"""
        if not self.selected_region:
            return
        
        self.monitoring = True
        
        def monitor():
            scan_count = 0
            with mss.mss() as sct:
                while self.monitoring:
                    try:
                        scan_count += 1
                        
                        # Capture the selected region
                        img = sct.grab(self.selected_region)
                        pil_img = Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
                        
                        # Extract text
                        text = extract_japanese_text_optimized(pil_img)
                        
                        if text and len(text) > 10 and text != self.last_text:
                            # Check if it contains Japanese
                            if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text):
                                self.last_text = text
                                
                                self.update_status("🔥 Japanese text found! Getting AI answers...")
                                
                                # Get AI answers
                                answers = get_jlpt_answers(text)
                                
                                # Display results
                                result_text = f"""🎯 JLPT QUESTIONS FOUND & ANSWERED!
{"="*60}

📍 SELECTED REGION: {self.selected_region['width']}×{self.selected_region['height']}
⏰ SCAN TIME: {datetime.now().strftime('%H:%M:%S')}
🔍 SCAN COUNT: #{scan_count}

📋 DETECTED TEXT:
{text}

{"="*60}

🤖 AI EXPERT ANSWERS:

{answers}

{"="*60}
✅ Monitoring continues... Change questions to get new answers!
"""
                                
                                self.update_results(result_text)
                                self.update_status("✅ Answers provided! Monitoring for changes...")
                        else:
                            self.update_status(f"🔍 Monitoring scan #{scan_count}... Waiting for Japanese text...")
                        
                        time.sleep(2)  # Check every 2 seconds
                        
                    except Exception as e:
                        self.update_status(f"❌ Monitor error: {e}")
                        time.sleep(3)
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
    
    def reset(self):
        """Reset the scanner"""
        self.monitoring = False
        self.selected_region = None
        self.last_text = ""
        self.update_status("Click 'SELECT QUESTION AREA' to choose where your JLPT questions are")
        self.update_results("📍 SCANNER RESET\n\nReady to select a new area for JLPT questions!")
    
    def run(self):
        self.root.mainloop()

def main():
    print("📍 Starting Manual JLPT Scanner...")
    
    if not os.path.exists(TESSERACT_PATH):
        print("❌ Tesseract not found!")
        return
    
    if GEMINI_API_KEY == "YOUR_GEMINI_KEY_HERE":
        print("❌ GEMINI_API_KEY not set!")
        print("Set it with: setx GEMINI_API_KEY \"your_api_key\"")
        return
    
    print("✅ Ready to manually select JLPT questions!")
    
    scanner = ManualJLPTScanner()
    scanner.run()

if __name__ == "__main__":
    main()
