# 🎯 Ultimate Japanese Quiz Solver - Advanced ML Pipeline

A comprehensive Japanese quiz detection and solving system using OCR → deterministic rules → morphological analysis → retrieval + LLM → ensemble decision with continuous human-in-the-loop learning.

## 🏗️ Architecture Overview

```
Screen Capture → OCR Pipeline → Rule Engines → Morphological Analysis → RAG/LLM → Ensemble Decision → GUI → Human Labeling
```

### Key Components:
- **Robust OCR Pipeline**: Multi-PSM preprocessing with OpenCV enhancement
- **Deterministic Rule Engines**: Date/reading mappings, katakana fuzzy matching
- **Morphological Analysis**: MeCab/Fugashi integration for linguistic understanding
- **RAG + LLM**: Retrieval-augmented generation with structured JSON responses
- **Ensemble Decision**: Priority-based voting system
- **Human-in-the-Loop**: Continuous labeling and improvement

## 🚀 Features

1. **Advanced OCR Processing**
   - Multi-PSM (Page Segmentation Mode) testing
   - OpenCV preprocessing (upscale, denoise, threshold)
   - Japanese character ratio optimization

2. **Intelligent Rule Systems**
   - Date/reading exact mappings (むいか → 六日)
   - Katakana fuzzy matching with similarity scoring
   - JLPT vocabulary verification

3. **Morphological Intelligence**
   - MeCab/Fugashi parsing for token analysis
   - Reading verification and kanji mapping
   - POS tagging for semantic understanding

4. **RAG-Enhanced LLM**
   - Vector database for JLPT knowledge
   - Context-aware prompt injection
   - Structured JSON response parsing

5. **Ensemble Decision Engine**
   - Priority-based rule aggregation
   - Confidence-weighted voting
   - Automatic conflict resolution

6. **Professional GUI**
   - Real-time scanning with visual feedback
   - Manual correction and labeling interface
   - Analytics dashboard and history tracking

## 🚀 Quick Start

### Method 1: Automated Setup (Recommended)
1. **Run the setup script:**
   ```powershell
   # PowerShell (recommended)
   ./setup.ps1
   
   # OR Command Prompt
   setup.bat
   ```

2. **Follow the prompts** to install dependencies and configure API keys

3. **Run the application:**
   ```
   python main.py
   ```

### Method 2: Manual Setup

#### Step 1: Install Python
- Download Python 3.10+ from [python.org](https://python.org)
- ✅ Check "Add Python to PATH" during installation

#### Step 2: Install Tesseract OCR
1. **Download Tesseract** from [UB Mannheim builds](https://github.com/UB-Mannheim/tesseract/wiki)
2. **During installation:**
   - ✅ Include Japanese language pack (jpn.traineddata)
   - Note the installation path (usually `C:\Program Files\Tesseract-OCR`)
3. **Verify installation:**
   ```cmd
   tesseract --version
   tesseract --list-langs
   ```
   You should see `jpn` in the language list.

#### Step 3: Install Python Dependencies
```powershell
cd jp_quiz_solver
pip install -r requirements.txt
```

#### Step 4: Configure API Keys

**Option A: Gemini (Google)**
1. Get API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Set environment variable:
   ```powershell
   # Temporary (current session)
   $env:GEMINI_API_KEY = "your-api-key-here"
   
   # Permanent
   setx GEMINI_API_KEY "your-api-key-here"
   ```

**Option B: OpenAI**
1. Install OpenAI client: `pip install openai`
2. Set environment variable:
   ```powershell
   setx OPENAI_API_KEY "sk-your-api-key-here"
   ```
3. Change `AI_PROVIDER = "openai"` in `config.py`

#### Step 5: Test Setup
```
python test_setup.py
```

## 🎮 Usage

1. **Launch the application:**
   ```
   python main.py
   ```

2. **Select quiz region:**
   - Click "Select Region" button
   - Drag to select the area where quiz questions appear
   - Press ESC to cancel selection

3. **Start practicing:**
   - The app monitors the selected region automatically
   - When Japanese text appears, it shows "Processing..."
   - AI answer appears with question, translation, and explanation

## 🔧 Configuration

Edit `config.py` to customize:

```python
# AI Provider: "gemini" or "openai"
AI_PROVIDER = "gemini"

# Tesseract path (if different from default)
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Default capture region (can be changed via UI)
CAPTURE_REGION = {"left": 300, "top": 200, "width": 800, "height": 400}

# OCR settings
OCR_LANGUAGE = "jpn+eng"  # Japanese + English
OCR_CONFIG = "--psm 6"    # Page segmentation mode
```

## 🔄 Switching AI Providers

**To Gemini:**
- Set `AI_PROVIDER = "gemini"` in config.py
- Ensure `GEMINI_API_KEY` environment variable is set

**To OpenAI:**
- Install: `pip install openai`
- Set `AI_PROVIDER = "openai"` in config.py  
- Ensure `OPENAI_API_KEY` environment variable is set

*No code changes needed - the app automatically routes to your selected provider!*

## 🐛 Troubleshooting

**"Tesseract not found" error:**
- Ensure Tesseract is installed with Japanese language support
- Check that the path in config.py matches your installation
- Try adding Tesseract to your system PATH

**"No text detected" message:**
- Use "Select Region" to choose the correct quiz area
- Ensure the text is clearly visible and not too small
- Check that Japanese text is actually in the selected region

**API errors:**
- Verify your API key is correctly set as environment variable
- Check your API quota/billing status
- For Gemini: ensure you're using the correct model name

**UI freezing:**
- The app uses thread-safe updates - this shouldn't happen
- If it does, please report as a bug with reproduction steps

## 📁 Project Structure
```
jp_quiz_solver/
├── main.py           # Main application
├── config.py         # Configuration settings
├── requirements.txt  # Python dependencies
├── setup.bat        # Windows batch setup script
├── setup.ps1        # PowerShell setup script
├── test_setup.py    # Installation verification
└── README.md        # This file
```

## 🔒 Privacy & Ethics
- **Use only for practice and studying**
- **Do not use during monitored exams or assessments**
- All processing happens locally on your machine
- Only the OCR text is sent to AI providers (not screenshots)
- No data is stored or transmitted beyond API calls

## 🎯 System Requirements
- Windows 10/11
- Python 3.10+
- 4GB RAM minimum
- Internet connection for AI API calls
- Tesseract OCR with Japanese language support

## 💡 Tips for Best Results
- **High contrast text** works best for OCR
- **Larger text** is more accurately recognized
- **Select tight regions** around question text only
- **Stable positioning** - avoid moving windows during capture
- **Good lighting** if using physical materials

Enjoy learning Japanese! 🇯🇵

