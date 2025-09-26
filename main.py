import mss
import pytesseract
from PIL import Image, ImageOps, ImageFilter
import tkinter as tk
from tkinter import messagebox
import threading
import time
import config
import sys
import os
import re

# AI imports
import google.generativeai as genai
try:
    from openai import OpenAI  # optional; only used if AI_PROVIDER == "openai"
except Exception:
    OpenAI = None

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_PATH

# Configure Gemini if key present
if getattr(config, "GEMINI_API_KEY", None):
    try:
        genai.configure(api_key=config.GEMINI_API_KEY)
    except Exception:
        pass

def get_answer_from_gemini(text):
    model = genai.GenerativeModel(config.GEMINI_MODEL)
    prompt = f"""
You are an expert Japanese language tutor and quiz solver with perfect accuracy.

CONTENT DETECTED:
{text}

INSTRUCTIONS:
- Analyze all Japanese text carefully (hiragana, katakana, kanji)
- If this is a quiz/test question with multiple choice options, identify them clearly
- Multiple choice formats may include: 1,2,3,4 or A,B,C,D or ア,イ,ウ,エ or ①②③④
- Provide the CORRECT answer choice if it's a quiz
- If it's just Japanese text without questions, provide translation and explanation
- Be 100% accurate

RESPOND IN THIS EXACT FORMAT:

🎯 CORRECT ANSWER: [Choice number/letter, or "N/A" if not a quiz]

📝 JAPANESE TEXT:
[Show the original Japanese text clearly]

🔤 TRANSLATION:
[Full English translation]

✅ EXPLANATION:
[Explanation of answer choice OR explanation of the text content]

📚 KEY POINTS:
[Important vocabulary, grammar, or cultural notes]

⚡ NOTES:
[Additional helpful information]

Be precise, accurate, and comprehensive.
    """
    response = model.generate_content(prompt)
    return response.text.strip()

def get_answer_from_openai(text):
    if OpenAI is None:
        return "OpenAI client not installed. Run: pip install openai"
    try:
        client = OpenAI()
    except Exception as e:
        return f"OpenAI init error: {e} (Ensure OPENAI_API_KEY is set)"

    system_prompt = """
You are an expert Japanese language tutor and quiz solver with perfect accuracy.
Your job is to analyze Japanese text and provide correct answers with absolute precision.

Rules:
- Analyze all Japanese text completely and accurately (hiragana, katakana, kanji)
- Identify multiple choice options in any format (numbers, letters, katakana, circled numbers)
- Provide the definitively CORRECT answer if it's a quiz
- Translate and explain all Japanese content
- Be comprehensive and accurate
"""

    user_prompt = f"""
JAPANESE CONTENT:
{text}

Provide response in this format:

🎯 CORRECT ANSWER: [Choice, or "N/A" if not a quiz]
📝 JAPANESE TEXT: [Original Japanese clearly shown]
🔤 TRANSLATION: [Complete English translation]
✅ EXPLANATION: [Explanation of answer or content]
📚 KEY POINTS: [Important vocabulary/grammar]
⚡ NOTES: [Additional helpful information]

Be 100% accurate and comprehensive.
    """
    
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0  # Maximum accuracy
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI error: {e}"

def contains_japanese(text):
    """Check if text contains Japanese characters (hiragana, katakana, kanji)"""
    # Unicode ranges for Japanese characters
    hiragana = r'[\u3040-\u309F]'  # Hiragana
    katakana = r'[\u30A0-\u30FF]'  # Katakana
    kanji = r'[\u4E00-\u9FAF]'     # Kanji (CJK Unified Ideographs)
    
    japanese_pattern = f'({hiragana}|{katakana}|{kanji})'
    return bool(re.search(japanese_pattern, text))

def get_answer(text):
    provider = getattr(config, "AI_PROVIDER", "gemini").lower()
    if provider == "openai":
        return get_answer_from_openai(text)
    return get_answer_from_gemini(text)

def preprocess_image(pil_img):
    """Enhance image for better OCR accuracy"""
    # Convert to grayscale
    img = pil_img.convert("L")
    
    # Scale up 2x for better OCR
    width, height = img.size
    img = img.resize((width * 2, height * 2), Image.LANCZOS)
    
    # Enhance contrast
    img = ImageOps.autocontrast(img)
    
    # Sharpen
    img = img.filter(ImageFilter.SHARPEN)
    
    return img

def select_region():
    """Interactive region selector"""
    selection = {}
    
    def on_region_selected(region):
        selection['region'] = region
        selector_root.destroy()
    
    selector_root = tk.Toplevel()
    selector_root.withdraw()
    selector_root.attributes("-fullscreen", True)
    selector_root.attributes("-alpha", 0.3)
    selector_root.configure(bg="black")
    selector_root.attributes("-topmost", True)
    
    canvas = tk.Canvas(selector_root, cursor="cross", bg="black")
    canvas.pack(fill="both", expand=True)
    
    start_x = start_y = 0
    rect_id = None
    
    def start_selection(event):
        nonlocal start_x, start_y, rect_id
        start_x = event.x_root
        start_y = event.y_root
        rect_id = canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="red", width=2)
    
    def update_selection(event):
        if rect_id:
            canvas.coords(rect_id, start_x - selector_root.winfo_rootx(), start_y - selector_root.winfo_rooty(), 
                         event.x_root - selector_root.winfo_rootx(), event.y_root - selector_root.winfo_rooty())
    
    def end_selection(event):
        if rect_id:
            x1 = min(start_x, event.x_root)
            y1 = min(start_y, event.y_root)
            x2 = max(start_x, event.x_root)
            y2 = max(start_y, event.y_root)
            
            region = {
                "left": int(x1),
                "top": int(y1),
                "width": int(x2 - x1),
                "height": int(y2 - y1)
            }
            on_region_selected(region)
    
    def cancel_selection(event):
        selector_root.destroy()
    
    canvas.bind("<ButtonPress-1>", start_selection)
    canvas.bind("<B1-Motion>", update_selection)
    canvas.bind("<ButtonRelease-1>", end_selection)
    selector_root.bind("<Escape>", cancel_selection)
    
    # Add instruction label
    instruction = tk.Label(selector_root, text="Drag to select quiz area. Press ESC to cancel.", 
                          fg="white", bg="black", font=("Arial", 16))
    instruction.place(relx=0.5, rely=0.05, anchor="center")
    
    selector_root.deiconify()
    selector_root.wait_window()
    
    return selection.get('region')

# Tkinter UI
root = tk.Tk()
root.title("JP Quiz Solver")
root.geometry("500x350")
root.attributes("-topmost", True)
root.configure(bg="black")

# Add menu frame
menu_frame = tk.Frame(root, bg="black")
menu_frame.pack(pady=5)

select_region_btn = tk.Button(menu_frame, text="Select Region", command=lambda: select_new_region(), bg="gray", fg="white")
select_region_btn.pack(side="left", padx=5)

status_label = tk.Label(menu_frame, text=f"Provider: {getattr(config, 'AI_PROVIDER', 'gemini').upper()}", 
                       fg="yellow", bg="black", font=("Arial", 8))
status_label.pack(side="right", padx=5)

answer_label = tk.Label(root, text="Waiting for text...", fg="white", bg="black", wraplength=480, justify="left")
answer_label.pack(pady=10, padx=10, fill="both", expand=True)

def set_answer_text_safe(text):
    try:
        root.after(0, lambda: answer_label.config(text=text))
    except Exception:
        pass

def select_new_region():
    """Handle region selection from UI"""
    root.withdraw()  # Hide main window
    new_region = select_region()
    root.deiconify()  # Show main window
    
    if new_region:
        # Update config dynamically
        config.CAPTURE_REGION = new_region
        messagebox.showinfo("Region Selected", 
                           f"New region: {new_region['left']}, {new_region['top']}, {new_region['width']}, {new_region['height']}")

def update_loop():
    with mss.mss() as sct:
        last_text = ""
        last_hash = ""
        stable_count = 0
        processing = False
        
        while True:
            try:
                img = sct.grab(config.CAPTURE_REGION)
                pil_img = Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")

                # Enhanced image preprocessing for JLPT accuracy
                processed_img = preprocess_image(pil_img)

                # OCR with Japanese + English for mixed content
                ocr_lang = getattr(config, "OCR_LANGUAGE", "jpn+eng")
                ocr_config = getattr(config, "OCR_CONFIG", "--psm 6")
                text = pytesseract.image_to_string(processed_img, lang=ocr_lang, config=ocr_config).strip()
                
                # Clean up OCR text
                text = ' '.join(text.split())  # Remove extra whitespace
                text_hash = hash(text)
                
                # Check for stable text (same for multiple captures)
                if text and text_hash == last_hash:
                    stable_count += 1
                else:
                    stable_count = 0
                    last_hash = text_hash

                # Only process if text is stable, changed, and contains Japanese
                if text and text != last_text and stable_count >= 2 and not processing:
                    # Check if text contains Japanese characters
                    if contains_japanese(text) or len(text) > 5:  # Process if Japanese or substantial text
                        processing = True
                        last_text = text
                        
                        # Show processing with preview
                        preview = text[:100] + "..." if len(text) > 100 else text
                        japanese_status = "🇯🇵 JAPANESE DETECTED" if contains_japanese(text) else "📝 TEXT DETECTED"
                        set_answer_text_safe(f"🔍 ANALYZING CONTENT...\n\n{japanese_status}\n\n📝 Detected Text:\n{preview}\n\n⏳ Getting AI analysis...")
                        
                        try:
                            answer = get_answer(text)
                            # Better formatted output
                            set_answer_text_safe(f"📋 DETECTED CONTENT:\n{text}\n\n{'='*50}\n\n{answer}")
                        except Exception as e:
                            set_answer_text_safe(f"❌ Error getting answer: {e}\n\n📝 Raw detected text:\n{text}")
                        
                        processing = False
                    else:
                        # Text detected but no Japanese - show briefly then continue
                        if len(text) > 2:  # Only show for substantial text
                            set_answer_text_safe(f"📄 Text detected (no Japanese):\n{text[:150]}{'...' if len(text) > 150 else ''}\n\n👀 Looking for Japanese content...")
                    
                elif not text:
                    if last_text:  # Only update if we had text before
                        set_answer_text_safe("👀 Scanning for Japanese content...\n\nPosition Japanese text/quiz in the selected region.\n\n✅ Tesseract OCR ready (Japanese + English)\n✅ AI models ready\n✅ Japanese text detection active")
                        last_text = ""
                        stable_count = 0
                        
            except Exception as e:
                set_answer_text_safe(f"❌ System error: {e}")
                processing = False

            time.sleep(1.5)  # Faster polling for JLPT

threading.Thread(target=update_loop, daemon=True).start()
root.mainloop()
