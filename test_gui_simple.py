#!/usr/bin/env python3
"""
🎯 SIMPLE TEST GUI FOR ULTIMATE JAPANESE QUIZ SOLVER 🎯
Test all functionality including theme switching, scanning, etc.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from ultimate_main import UltimateQuizSolver
    solver_available = True
except Exception as e:
    print(f"Solver not available: {e}")
    solver_available = False

class SimpleTestGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🧪 Test GUI - Ultimate Japanese Quiz Solver")
        self.root.geometry("600x500")
        
        self.current_theme = "dark"
        self.themes = {
            "dark": {
                "bg": "#2b2b2b",
                "fg": "white",
                "accent": "#4CAF50",
                "button": "#555555"
            },
            "light": {
                "bg": "#ffffff",
                "fg": "#000000", 
                "accent": "#2196F3",
                "button": "#e0e0e0"
            }
        }
        
        if solver_available:
            self.solver = UltimateQuizSolver()
        
        self.setup_ui()
        self.apply_theme()
        
    def setup_ui(self):
        # Main frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        self.title_label = tk.Label(self.main_frame, 
                                   text="🧪 FUNCTIONALITY TEST",
                                   font=("Arial", 16, "bold"))
        self.title_label.pack(pady=(0, 20))
        
        # Theme controls
        theme_frame = tk.Frame(self.main_frame)
        theme_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(theme_frame, text="🎨 Theme:", font=("Arial", 10)).pack(side=tk.LEFT)
        
        self.theme_var = tk.StringVar(value=self.current_theme)
        theme_dropdown = ttk.Combobox(theme_frame, textvariable=self.theme_var,
                                     values=["dark", "light"], state="readonly", width=10)
        theme_dropdown.pack(side=tk.LEFT, padx=(5, 0))
        theme_dropdown.bind("<<ComboboxSelected>>", self.change_theme)
        
        # API Key test
        api_frame = tk.Frame(self.main_frame)
        api_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(api_frame, text="🔑 API Key:", font=("Arial", 10)).pack(side=tk.LEFT)
        
        api_key = os.environ.get('GEMINI_API_KEY', 'Not Set')
        api_status = "✅ Set" if api_key != 'Not Set' else "❌ Missing"
        self.api_label = tk.Label(api_frame, text=api_status, font=("Arial", 10))
        self.api_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Test buttons
        buttons_frame = tk.Frame(self.main_frame)
        buttons_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.test_ocr_btn = tk.Button(buttons_frame, text="🔍 Test OCR", 
                                     command=self.test_ocr, font=("Arial", 10))
        self.test_ocr_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.test_ai_btn = tk.Button(buttons_frame, text="🤖 Test AI", 
                                    command=self.test_ai, font=("Arial", 10))
        self.test_ai_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.test_scan_btn = tk.Button(buttons_frame, text="📷 Test Quick Scan", 
                                      command=self.test_scan, font=("Arial", 10))
        self.test_scan_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Status display
        self.status_text = tk.Text(self.main_frame, height=15, width=70, 
                                  font=("Consolas", 10))
        self.status_text.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Add scrollbar
        scrollbar = tk.Scrollbar(self.main_frame, command=self.status_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text.config(yscrollcommand=scrollbar.set)
        
        # Initial status
        self.log("🎯 ULTIMATE JAPANESE QUIZ SOLVER - TEST MODE")
        self.log("=" * 50)
        self.log(f"✅ GUI initialized successfully")
        self.log(f"🔑 API Key: {'Available' if api_key != 'Not Set' else 'Missing'}")
        self.log(f"🤖 Solver: {'Available' if solver_available else 'Not Available'}")
        self.log(f"🎨 Current Theme: {self.current_theme}")
        self.log("")
        self.log("📋 INSTRUCTIONS:")
        self.log("1. Try switching themes using the dropdown")
        self.log("2. Test OCR functionality")
        self.log("3. Test AI responses")
        self.log("4. Test screen scanning")
        self.log("")
        
    def apply_theme(self):
        """Apply the current theme to all elements"""
        theme = self.themes[self.current_theme]
        
        # Root window
        self.root.configure(bg=theme["bg"])
        
        # Main frame
        self.main_frame.configure(bg=theme["bg"])
        
        # Labels
        for widget in self.main_frame.winfo_children():
            if isinstance(widget, tk.Label):
                widget.configure(bg=theme["bg"], fg=theme["fg"])
            elif isinstance(widget, tk.Frame):
                widget.configure(bg=theme["bg"])
                for child in widget.winfo_children():
                    if isinstance(child, tk.Label):
                        child.configure(bg=theme["bg"], fg=theme["fg"])
                    elif isinstance(child, tk.Button):
                        child.configure(bg=theme["button"], fg=theme["fg"],
                                      activebackground=theme["accent"])
        
        # Text widget
        self.status_text.configure(bg=theme["bg"], fg=theme["fg"], 
                                  insertbackground=theme["accent"])
        
        # Buttons
        for btn in [self.test_ocr_btn, self.test_ai_btn, self.test_scan_btn]:
            btn.configure(bg=theme["button"], fg=theme["fg"],
                         activebackground=theme["accent"])
        
        self.log(f"🎨 Theme changed to: {self.current_theme}")
        
    def change_theme(self, event=None):
        """Handle theme change"""
        new_theme = self.theme_var.get()
        if new_theme != self.current_theme:
            self.current_theme = new_theme
            self.apply_theme()
            
    def log(self, message):
        """Add message to status log"""
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        
    def test_ocr(self):
        """Test OCR functionality"""
        self.log("\n🔍 TESTING OCR...")
        try:
            if not solver_available:
                self.log("❌ Solver not available - cannot test OCR")
                return
                
            # Test basic OCR setup
            import pytesseract
            version = pytesseract.get_tesseract_version()
            self.log(f"✅ Tesseract version: {version}")
            
            # Test Japanese language support
            try:
                langs = pytesseract.get_languages()
                has_japanese = any('jpn' in lang for lang in langs)
                self.log(f"✅ Japanese language pack: {'Available' if has_japanese else 'Missing'}")
            except:
                self.log("⚠️ Could not check language packs")
                
            self.log("✅ OCR test completed")
            
        except Exception as e:
            self.log(f"❌ OCR test failed: {e}")
            
    def test_ai(self):
        """Test AI functionality"""
        self.log("\n🤖 TESTING AI...")
        try:
            if not solver_available:
                self.log("❌ Solver not available - cannot test AI")
                return
                
            # Test API key
            api_key = os.environ.get('GEMINI_API_KEY')
            if not api_key:
                self.log("❌ No API key found in environment")
                return
                
            self.log(f"✅ API key found: {api_key[:10]}...")
            
            # Test simple AI call
            test_question = "この文章を日本語で説明してください。"
            self.log(f"📝 Testing with question: {test_question}")
            
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-pro')
                
                response = model.generate_content(f"Please answer this Japanese question briefly: {test_question}")
                self.log(f"✅ AI Response: {response.text[:100]}...")
                self.log("✅ AI test completed successfully")
                
            except Exception as ai_error:
                self.log(f"❌ AI call failed: {ai_error}")
                
        except Exception as e:
            self.log(f"❌ AI test failed: {e}")
            
    def test_scan(self):
        """Test screen scanning functionality"""
        self.log("\n📷 TESTING SCREEN SCAN...")
        try:
            if not solver_available:
                self.log("❌ Solver not available - cannot test scanning")
                return
                
            # Test screenshot capability
            import PIL.ImageGrab as ImageGrab
            screenshot = ImageGrab.grab()
            self.log(f"✅ Screenshot captured: {screenshot.size}")
            
            # Test OCR on screenshot (small region)
            import pytesseract
            # Test on a small region to avoid processing the entire screen
            test_region = screenshot.crop((100, 100, 400, 200))
            text = pytesseract.image_to_string(test_region, lang='eng')
            
            if text.strip():
                self.log(f"✅ OCR detected text: '{text.strip()[:50]}...'")
            else:
                self.log("ℹ️ No text detected in test region (this is normal)")
                
            self.log("✅ Scan test completed")
            
        except Exception as e:
            self.log(f"❌ Scan test failed: {e}")
            
    def run(self):
        """Start the test GUI"""
        self.root.mainloop()

if __name__ == "__main__":
    print("🧪 Starting Test GUI...")
    app = SimpleTestGUI()
    app.run()
