#!/usr/bin/env python3
"""
Japanese Screen Solver - Windows edition (single file)

How it works (summary):
 - Select a screen region (--select-region) or pass --region LEFT TOP WIDTH HEIGHT
 - The app captures that region repeatedly, OCRs Japanese text with Tesseract,
   sends the text to the configured model provider (Gemini or OpenAI),
   and shows the AI response in an always-on-top overlay window.

Usage:
 1) Select region interactively:
    python jp_screen_solver_windows.py --select-region
    -> Draw rectangle; it prints LEFT TOP WIDTH HEIGHT

 2) Run watcher:
    python jp_screen_solver_windows.py --region 100 200 1000 500 --provider gemini --poll 1.5

Notes:
 - This script uses OpenAI if you pass --provider openai and set OPENAI_API_KEY.
 - For Gemini: set GEMINI_API_KEY (or pass --gemini-key). 
 - Keep this for practice only — do not use to bypass any monitored exam.

"""

import os, sys, time, argparse, threading, hashlib
from PIL import Image, ImageOps, ImageFilter
import mss
import pytesseract

# LLM clients (optional)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# For Gemini: try import (may not be installed)
try:
    import google.generativeai as genai
except Exception:
    genai = None

# Optional cache
try:
    from cachetools import TTLCache
except Exception:
    TTLCache = None

# GUI
import tkinter as tk
from tkinter.scrolledtext import ScrolledText

# -------------------------
# Utilities
# -------------------------
def text_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def preprocess_for_ocr(pil_img: Image.Image, ups=2) -> Image.Image:
    img = pil_img.convert("L")
    w, h = img.size
    img = img.resize((int(w * ups), int(h * ups)), Image.LANCZOS)
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)
    return img

def ocr_image(img: Image.Image, lang="jpn+eng", tesseract_cmd=None) -> str:
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    try:
        txt = pytesseract.image_to_string(img, lang=lang, config="--psm 6")
    except Exception:
        txt = pytesseract.image_to_string(img, lang=lang)
    return txt.strip()

# -------------------------
# Model adapters
# -------------------------
def ask_openai(question_text: str, api_key: str = None, model: str = "gpt-4o") -> str:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        return "[NO OPENAI API KEY set - set OPENAI_API_KEY or pass --openai-key]"
    if OpenAI is None:
        return "[openai package not installed: pip install openai]"

    system_prompt = (
        "You are an expert JLPT (Japanese Language Proficiency Test) instructor with perfect accuracy. "
        "Your job is to provide CORRECT answers for JLPT test questions with absolute precision. "
        "Analyze Japanese text completely, identify multiple choice options, and provide the definitively CORRECT answer. "
        "Be concise but thorough. Focus on test success."
    )
    
    user_prompt = f"""
JLPT QUESTION:
{question_text}

Provide response in this format:

🎯 CORRECT ANSWER: [Choice number/letter]
📝 QUESTION TRANSLATION: [Full English translation]
✅ EXPLANATION: [Why this answer is correct]
📚 GRAMMAR POINT: [Key pattern or vocabulary]
⚡ QUICK TIP: [Memory aid for similar questions]

Be 100% accurate for JLPT success.
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,  # Maximum accuracy
            max_tokens=800
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[OpenAI error] {e}"

def ask_gemini(question_text: str, api_key: str = None, model: str = "gemini-1.5-flash") -> str:
    """
    Gemini adapter optimized for JLPT questions
    """
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        return "[NO GEMINI API KEY set - set GEMINI_API_KEY or pass --gemini-key]"

    if genai is None:
        return ("[Gemini client not installed]\n"
                "Install the Google Generative AI client and set GEMINI_API_KEY:\n"
                "pip install google-generativeai\n"
                "Then re-run.")
    try:
        genai.configure(api_key=key)
        model_instance = genai.GenerativeModel(model)
        
        prompt = f"""
You are an expert JLPT (Japanese Language Proficiency Test) tutor with perfect accuracy.

JLPT QUESTION DETECTED:
{question_text}

INSTRUCTIONS:
- This is a JLPT test question requiring a PERFECT answer
- Analyze all Japanese text with complete accuracy
- If multiple choice options are present (1,2,3,4 or A,B,C,D or ア,イ,ウ,エ), identify them clearly
- Provide the definitively CORRECT answer choice
- Be 100% accurate - this is for serious test preparation

RESPOND IN THIS EXACT FORMAT:

🎯 CORRECT ANSWER: [Choice number/letter]

📝 QUESTION TRANSLATION:
[Complete English translation of the question]

✅ EXPLANATION:
[Clear, concise explanation of why this answer is correct]

📚 GRAMMAR POINT:
[Key grammar pattern, vocabulary, or concept being tested]

⚡ QUICK TIP:
[Memory tip or pattern recognition for similar JLPT questions]

Be precise, accurate, and focused on JLPT success.
        """

        response = model_instance.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[Gemini client error] {e}. Make sure your API key is valid."

# Top-level router
def ask_model(question_text: str, provider: str, api_key: str = None, model: str = None) -> str:
    if provider.lower() in ("openai", "oai"):
        model_name = model or "gpt-4o"
        return ask_openai(question_text, api_key=api_key, model=model_name)
    elif provider.lower() in ("gemini", "google"):
        model_name = model or "gemini-1.5-flash"
        return ask_gemini(question_text, api_key=api_key, model=model_name)
    else:
        return "[Unknown provider: set --provider gemini|openai]"

# -------------------------
# Tk overlay and region selector
# -------------------------
class OverlayWindow:
    def __init__(self, width=520, height=300, x=50, y=50, title="JP Solver Overlay"):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.attributes("-topmost", True)
        try:
            self.root.attributes("-alpha", 0.95)
        except Exception:
            pass
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        
        # Add provider status
        status_frame = tk.Frame(self.root, bg="black")
        status_frame.pack(fill="x", padx=5, pady=2)
        
        self.status_label = tk.Label(status_frame, text="Ready", fg="green", bg="black", font=("Arial", 8))
        self.status_label.pack(side="right")
        
        self.text = ScrolledText(self.root, wrap="word", font=("Helvetica", 11), bg="black", fg="white")
        self.text.pack(fill="both", expand=True)
        self.text.insert("1.0", "[Idle]\nSelect region and start the watcher.")
        self.text.config(state="disabled")
        
        def on_close():
            try: 
                self.root.quit()
            except: 
                pass
            try: 
                self.root.destroy()
            except: 
                pass
            os._exit(0)
        self.root.protocol("WM_DELETE_WINDOW", on_close)

    def set_text(self, s: str):
        self.root.after(0, self._set_text_sync, s)

    def _set_text_sync(self, s: str):
        self.text.config(state="normal")
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", s)
        self.text.config(state="disabled")
        
    def set_status(self, status: str, color: str = "green"):
        self.root.after(0, lambda: self.status_label.config(text=status, fg=color))

    def start(self):
        self.root.mainloop()

def select_region_interactive() -> tuple:
    coords = {}
    root = tk.Tk()
    root.attributes("-fullscreen", True)
    root.attributes("-alpha", 0.25)
    root.title("Drag to select region (ESC to cancel)")
    root.configure(bg="black")
    canvas = tk.Canvas(root, cursor="cross", bg="black")
    canvas.pack(fill="both", expand=True)
    
    # Add instruction
    instruction = tk.Label(root, text="Drag to select the quiz area. Press ESC to cancel.", 
                          fg="white", bg="black", font=("Arial", 20))
    instruction.place(relx=0.5, rely=0.1, anchor="center")
    
    start = [0,0]
    rect = None

    def on_button_press(event):
        start[0] = event.x_root
        start[1] = event.y_root
        nonlocal rect
        rect = canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="red", width=3)

    def on_move(event):
        nonlocal rect
        if rect is not None:
            x1 = start[0] - root.winfo_rootx()
            y1 = start[1] - root.winfo_rooty()
            x2 = event.x_root - root.winfo_rootx()
            y2 = event.y_root - root.winfo_rooty()
            canvas.coords(rect, x1, y1, x2, y2)

    def on_release(event):
        x1 = min(start[0], event.x_root)
        y1 = min(start[1], event.y_root)
        x2 = max(start[0], event.x_root)
        y2 = max(start[1], event.y_root)
        coords['left'] = int(x1)
        coords['top'] = int(y1)
        coords['width'] = int(x2 - x1)
        coords['height'] = int(y2 - y1)
        root.destroy()

    def on_escape(event):
        root.destroy()
        coords.clear()

    canvas.bind("<ButtonPress-1>", on_button_press)
    canvas.bind("<B1-Motion>", on_move)
    canvas.bind("<ButtonRelease-1>", on_release)
    root.bind("<Escape>", on_escape)

    root.mainloop()
    if not coords:
        raise SystemExit("Selection cancelled.")
    return coords['left'], coords['top'], coords['width'], coords['height']

# -------------------------
# Watcher
# -------------------------
def start_watcher(region, tesseract_cmd, poll_sec, overlay_obj, provider, api_key=None, model=None, lang="jpn+eng", cache_ttl=300):
    if TTLCache is not None:
        cache = TTLCache(maxsize=256, ttl=cache_ttl)
    else:
        cache = {}
    
    sct = mss.mss()
    left, top, width, height = region
    monitor = {"left": left, "top": top, "width": width, "height": height}

    overlay_obj.set_text(f"[Watcher] Starting...\nProvider: {provider.upper()}\nPolling every {poll_sec:.1f}s\nRegion: {left},{top} {width}x{height}")
    overlay_obj.set_status(f"Monitoring - {provider.upper()}", "yellow")
    
    consecutive_empty = 0
    last_text = ""
    
    while True:
        try:
            img_raw = sct.grab(monitor)
            img = Image.frombytes("RGB", img_raw.size, img_raw.bgra, "raw", "BGRX")
            proc = preprocess_for_ocr(img, ups=2)
            text = ocr_image(proc, lang=lang, tesseract_cmd=tesseract_cmd)
            
            if not text:
                consecutive_empty += 1
                if consecutive_empty == 1:  # First time empty
                    overlay_obj.set_text("[No Japanese text detected in region]\n\nTips:\n• Make sure quiz text is visible in the selected area\n• Ensure good contrast and readable text size\n• Try repositioning the quiz window")
                    overlay_obj.set_status("Waiting for text...", "orange")
                time.sleep(poll_sec)
                continue
            
            consecutive_empty = 0
            
            # Only process if text changed significantly
            if text != last_text:
                last_text = text
                h = text_hash(text)
                cached = cache.get(h) if isinstance(cache, dict) else cache.get(h)
                
                if cached:
                    overlay_obj.set_text(f"[QUESTION - From Cache]\n{text}\n\n[ANSWER]\n{cached}")
                    overlay_obj.set_status("Cached result", "blue")
                else:
                    overlay_obj.set_text(f"[DETECTED QUESTION]\n{text}\n\n[Querying {provider.upper()} AI...]")
                    overlay_obj.set_status(f"Processing with {provider.upper()}...", "yellow")
                    
                    ans = ask_model(text, provider=provider, api_key=api_key, model=model)
                    
                    if isinstance(cache, dict):
                        cache[h] = ans
                    else:
                        cache[h] = ans
                    
                    overlay_obj.set_text(f"[QUESTION]\n{text}\n\n[ANSWER]\n{ans}")
                    overlay_obj.set_status("Answer ready", "green")
            
            time.sleep(poll_sec)
            
        except KeyboardInterrupt:
            print("Interrupted.")
            return
        except Exception as e:
            overlay_obj.set_text(f"[ERROR]\n{str(e)}\n\nCheck:\n• Tesseract installed and in PATH\n• API key is valid\n• Network connection")
            overlay_obj.set_status("Error occurred", "red")
            time.sleep(poll_sec)

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Japanese Quiz Solver - Screen OCR + AI Assistant")
    p.add_argument("--select-region", action="store_true", help="Interactive region selection")
    p.add_argument("--region", nargs=4, type=int, metavar=("LEFT","TOP","WIDTH","HEIGHT"), help="Screen region coordinates")
    p.add_argument("--poll", type=float, default=1.5, help="Polling interval in seconds")
    p.add_argument("--tesseract-cmd", type=str, default=r"C:\Program Files\Tesseract-OCR\tesseract.exe", help="Path to tesseract executable")
    p.add_argument("--provider", type=str, default="gemini", choices=["gemini", "openai"], help="AI provider")
    p.add_argument("--gemini-key", type=str, default=None, help="Gemini API key (or set GEMINI_API_KEY env var)")
    p.add_argument("--openai-key", type=str, default=None, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    p.add_argument("--model", type=str, default=None, help="Model name for provider")
    p.add_argument("--lang", type=str, default="jpn+eng", help="OCR language codes")
    p.add_argument("--cache-ttl", type=int, default=300, help="Cache TTL in seconds")
    return p.parse_args()

def main():
    args = parse_args()
    
    print("Japanese Quiz Solver - Advanced Version")
    print("=" * 50)
    
    if args.select_region:
        print("Interactive region selection starting...")
        print("Instructions: Drag to select the quiz area, ESC to cancel")
        try:
            left, top, width, height = select_region_interactive()
            print(f"\nRegion selected successfully!")
            print(f"LEFT TOP WIDTH HEIGHT")
            print(f"{left} {top} {width} {height}")
            print(f"\nTo run the solver with this region:")
            print(f"python {sys.argv[0]} --region {left} {top} {width} {height} --provider {args.provider}")
        except SystemExit:
            print("Selection cancelled.")
        return

    if not args.region:
        print("Error: Must specify region coordinates!")
        print("Options:")
        print("  1) Interactive selection: python jp_screen_solver_windows.py --select-region")
        print("  2) Manual coordinates: python jp_screen_solver_windows.py --region LEFT TOP WIDTH HEIGHT")
        sys.exit(1)

    region = tuple(args.region)
    print(f"Region: {region[0]},{region[1]} {region[2]}x{region[3]}")
    
    # Set tesseract path
    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd
        print(f"Tesseract: {args.tesseract_cmd}")

    # Determine API key
    provider = args.provider.lower()
    api_key = None
    if provider == "gemini":
        api_key = args.gemini_key or os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "YOUR_GEMINI_KEY_HERE":
            print("ERROR: Gemini API key not found!")
            print("Set it with: setx GEMINI_API_KEY \"your_api_key_here\"")
            print("Or pass: --gemini-key your_api_key_here")
            sys.exit(1)
    elif provider == "openai":
        api_key = args.openai_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OpenAI API key not found!")
            print("Set it with: setx OPENAI_API_KEY \"sk-your_api_key_here\"")
            print("Or pass: --openai-key sk-your_api_key_here")
            sys.exit(1)

    print(f"AI Provider: {provider.upper()}")
    print(f"Model: {args.model or 'default'}")
    print(f"OCR Language: {args.lang}")
    print(f"Poll Interval: {args.poll}s")
    print("\nStarting overlay window...")

    overlay = OverlayWindow(title=f"JP Quiz Solver - {provider.upper()}")
    watcher_thread = threading.Thread(
        target=start_watcher,
        args=(region, args.tesseract_cmd, args.poll, overlay, provider, api_key, args.model, args.lang, args.cache_ttl),
        daemon=True
    )
    watcher_thread.start()
    
    try:
        overlay.start()
    except KeyboardInterrupt:
        print("\nShutting down...")

if __name__ == "__main__":
    main()
