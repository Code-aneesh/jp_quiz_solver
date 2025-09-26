# Japanese Quiz Solver - Advanced Single File Version

A comprehensive, production-ready Japanese text recognition and AI-powered answer system. This single-file solution captures quiz questions from your screen and provides detailed answers with explanations.

## 🚀 Quick Start (10 minutes)

### Prerequisites
1. **Python 3.10+** - Download from [python.org](https://python.org) ✅
2. **Tesseract OCR with Japanese** - Download from [UB Mannheim builds](https://github.com/UB-Mannheim/tesseract/wiki) ✅

### Installation Steps

#### 1) Install Tesseract OCR (Windows)
```powershell
# Download installer from UB Mannheim builds
# During installation: ✅ Include Japanese language pack (jpn.traineddata)
# Default path: C:\Program Files\Tesseract-OCR\tesseract.exe

# Verify installation
tesseract --version
tesseract --list-langs    # Should show 'jpn'
```

#### 2) Install Python Dependencies
```powershell
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\Activate.ps1    # PowerShell
# venv\Scripts\activate.bat  # Command Prompt

# Install packages
pip install mss pillow pytesseract google-generativeai openai cachetools
```

#### 3) Set API Key
**For Gemini (recommended):**
```powershell
# Get API key from: https://aistudio.google.com/app/apikey
setx GEMINI_API_KEY "AIzaSyCVjmHF0MOFf3goIuKli5ifc-xndDpEWT0"
```

**For OpenAI:**
```powershell
setx OPENAI_API_KEY "sk-your_api_key_here"
```

### Usage

#### Step 1: Select Quiz Region
```powershell
python jp_screen_solver_windows.py --select-region
```
- Drag to select the area where quiz questions appear
- Note the coordinates printed (LEFT TOP WIDTH HEIGHT)

#### Step 2: Run the Solver
```powershell
# Using Gemini (default)
python jp_screen_solver_windows.py --region 300 200 800 400 --provider gemini

# Using OpenAI
python jp_screen_solver_windows.py --region 300 200 800 400 --provider openai

# Custom polling interval and model
python jp_screen_solver_windows.py --region 300 200 800 400 --provider gemini --poll 1.0 --model gemini-1.5-flash
```

## 📋 Command Line Options

```
python jp_screen_solver_windows.py [OPTIONS]

Required (choose one):
  --select-region              Interactive region selection
  --region LEFT TOP WIDTH HEIGHT    Screen coordinates to monitor

Optional:
  --provider {gemini,openai}   AI provider (default: gemini)
  --poll SECONDS              Polling interval (default: 1.5)
  --tesseract-cmd PATH        Tesseract executable path
  --gemini-key KEY            Gemini API key (overrides env var)
  --openai-key KEY            OpenAI API key (overrides env var)
  --model MODEL_NAME          Specific model name
  --lang LANG_CODES           OCR languages (default: jpn+eng)
  --cache-ttl SECONDS         Cache duration (default: 300)
```

## ✨ Features

### Core Functionality
- **Real-time screen monitoring** with customizable polling
- **Advanced OCR preprocessing** (2x upscaling, contrast enhancement, sharpening)
- **Dual AI provider support** (Gemini and OpenAI)
- **Interactive region selection** with visual feedback
- **Smart caching** to avoid redundant API calls
- **Always-on-top overlay** with status indicators

### AI Provider Support
- **Gemini 1.5 Flash** (default, fast and accurate)
- **OpenAI GPT-4o** (premium option)
- Easy switching between providers
- Comprehensive error handling

### OCR Optimizations
- Image preprocessing for better accuracy
- Japanese + English language support
- Configurable page segmentation modes
- Automatic text change detection

### User Interface
- Clean overlay window with status indicators
- Scrollable text area for long answers
- Color-coded status (green=ready, yellow=processing, red=error, blue=cached)
- Graceful error handling and user feedback

## 🎯 Answer Quality

The system provides comprehensive responses including:

### Gemini Responses
- **Correct Answer** in Japanese
- **English Translation** of question and answer
- **Grammar Explanation** with detailed breakdown
- **Vocabulary** with key terms and meanings
- **Furigana** for kanji characters

### OpenAI Responses
- **Short Answer** (one line, Japanese)
- **Translation** (English of the question)
- **Explanation** (grammar/vocab, bullet points)
- **Furigana** (question text with reading aids)
- **Extra Practice** (2 short examples)

## 🔧 Advanced Configuration

### Environment Variables
```powershell
# API Keys
setx GEMINI_API_KEY "your_gemini_key"
setx OPENAI_API_KEY "sk-your_openai_key"

# Custom Tesseract Path
setx TESSERACT_CMD "C:\Custom\Path\tesseract.exe"
```

### Model Selection
```powershell
# Gemini models
--model gemini-1.5-flash      # Fast, recommended
--model gemini-1.5-pro        # More detailed responses

# OpenAI models  
--model gpt-4o               # Default, balanced
--model gpt-4o-mini          # Faster, lower cost
```

### OCR Tuning
```powershell
# Language combinations
--lang jpn+eng               # Japanese + English (default)
--lang jpn                   # Japanese only
--lang jpn+eng+chi_sim       # Japanese + English + Chinese

# Polling intervals
--poll 1.0                   # Fast (1 second)
--poll 2.0                   # Balanced (2 seconds)  
--poll 0.5                   # Very fast (0.5 seconds)
```

## 🎮 Usage Examples

### Basic Quiz Monitoring
```powershell
# Select region first
python jp_screen_solver_windows.py --select-region

# Start monitoring with Gemini
python jp_screen_solver_windows.py --region 400 300 800 500 --provider gemini
```

### High-Frequency Monitoring
```powershell
# Fast polling for rapid-fire questions
python jp_screen_solver_windows.py --region 400 300 800 500 --provider gemini --poll 0.8
```

### OpenAI with Custom Model
```powershell
# Use OpenAI with specific model
python jp_screen_solver_windows.py --region 400 300 800 500 --provider openai --model gpt-4o-mini
```

### Development/Testing
```powershell
# Short cache for testing
python jp_screen_solver_windows.py --region 400 300 800 500 --provider gemini --cache-ttl 60
```

## 🐛 Troubleshooting

### Common Issues

**"Tesseract not found" error:**
```
Solution: Ensure Tesseract is installed with Japanese language pack
Check: tesseract --list-langs (should show 'jpn')
Fix: Download from UB Mannheim builds, include Japanese during install
```

**"No Japanese text detected":**
```
Solution: Adjust region selection and text visibility
- Use --select-region to reselect area
- Ensure good contrast and readable text size
- Make sure quiz text is within selected region
```

**"API key not found" error:**
```
Solution: Set environment variables correctly
Gemini: setx GEMINI_API_KEY "your_key"
OpenAI: setx OPENAI_API_KEY "sk-your_key"
Restart terminal after setx commands
```

**"Client error" or network issues:**
```
Solution: Check API key validity and network connection
- Verify API key is correct and active
- Check internet connection
- Try different model if available
```

### Performance Optimization

**For better OCR accuracy:**
- Select tight regions around question text only
- Ensure high contrast between text and background
- Use larger text sizes when possible
- Position quiz windows on primary monitor

**For faster response:**
- Use shorter polling intervals (--poll 1.0)
- Use Gemini Flash model (default)
- Enable caching for repeated questions

**For lower API costs:**
- Use longer polling intervals (--poll 2.0)
- Use caching (default enabled)
- Select minimal screen regions

## 📊 Performance Metrics

### Typical Performance
- **OCR Processing**: 0.2-0.5 seconds
- **Gemini API**: 1-3 seconds  
- **OpenAI API**: 1-2 seconds
- **Total Response Time**: 2-4 seconds
- **Cache Hit Response**: <0.1 seconds

### Resource Usage
- **RAM**: ~50-100MB during operation
- **CPU**: Low (OCR preprocessing only)
- **Network**: Minimal (text-only API calls)
- **Storage**: Negligible (in-memory cache)

## 🔒 Privacy & Security

### Data Handling
- **Local Processing**: All OCR and image processing happens locally
- **API Transmission**: Only extracted text sent to AI providers
- **No Screenshot Storage**: Images not saved or transmitted
- **Memory Cache Only**: No persistent data storage

### Ethical Usage
- ✅ **Practice and study sessions**
- ✅ **Language learning assistance**  
- ✅ **Self-paced quiz review**
- ❌ **Monitored exams or assessments**
- ❌ **Academic dishonesty**
- ❌ **Bypassing proctoring systems**

## 🏆 Best Practices

### For Optimal Results
1. **Select tight regions** around question text only
2. **Use consistent window positioning** for stable capture
3. **Ensure good text contrast** for better OCR
4. **Test region selection** before important sessions
5. **Keep quiz windows steady** during capture

### Study Workflow
1. **Practice Mode**: Use for self-study and review
2. **Region Setup**: Configure once per quiz format
3. **Review Answers**: Study explanations and grammar notes
4. **Vocabulary Building**: Note new terms from responses
5. **Progress Tracking**: Use different regions for different topics

## 📈 Advanced Tips

### Power User Features
- **Multiple Regions**: Run multiple instances for different quiz areas
- **Custom Models**: Experiment with different AI models for varied response styles
- **Batch Processing**: Use short polling for rapid question sequences
- **Integration**: Pipe output to external tools using command-line interface

### Customization Options
- Modify AI prompts in the source code for specialized response formats
- Adjust OCR preprocessing parameters for different text types
- Create batch scripts for common region/provider combinations
- Set up shortcuts for frequently used configurations

---

## 📞 Support & Feedback

This is a feature-complete, production-ready solution for Japanese language learning assistance. The single-file design makes it easy to deploy, modify, and maintain.

**Happy learning! 🇯🇵**
