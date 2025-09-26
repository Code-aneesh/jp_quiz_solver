import os

# Path to Tesseract OCR executable
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Screen capture region (left, top, width, height) - adjust to where your quiz appears
# Default covers center area - use "Select Region" button to set precisely
CAPTURE_REGION = {"left": 300, "top": 200, "width": 800, "height": 400}

# AI Provider: "gemini" or "openai"
AI_PROVIDER = "openai"

# Gemini Settings (get API key from https://aistudio.google.com/app/apikey)
# Replace YOUR_GEMINI_KEY_HERE with your actual API key, or set GEMINI_API_KEY environment variable
# PASTE YOUR GEMINI API KEY HERE (replace the text below):
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_KEY_HERE")

# Example: GEMINI_API_KEY = "AIzaSyDxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# Or better: set environment variable and leave this as is
GEMINI_MODEL = "gemini-1.5-flash"  # Fast and accurate for JLPT

# OpenAI Settings (set OPENAI_API_KEY environment variable)
# PASTE YOUR OPENAI API KEY HERE (replace the text below):
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-ijklmnop1234qrstijklmnop1234qrstijklmnop")
OPENAI_MODEL = "gpt-4o"  # High accuracy for JLPT questions

# JLPT-Optimized OCR Settings
OCR_LANGUAGE = "jpn+eng"  # Japanese + English for mixed content
OCR_CONFIG = "--psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZあいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンー々〜？！（）「」『』。、"

# JLPT Test Settings
POLLING_INTERVAL = 1.0  # Faster polling for real-time JLPT tests
STABLE_TEXT_THRESHOLD = 2  # Require text to be stable for 2 captures before processing

