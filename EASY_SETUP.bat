@echo off
title Japanese Quiz Solver - One-Click Setup
color 0A
echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║          JAPANESE QUIZ SOLVER - EASY SETUP               ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.
echo  I'll do EVERYTHING except the API key (for security)!
echo.

echo [1/4] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Installing...
    echo Opening Python download page...
    start https://python.org/downloads
    echo Please install Python and run this script again.
    pause
    exit /b 1
) else (
    echo ✅ Python found!
)

echo.
echo [2/4] Installing all Python packages...
pip install mss pillow pytesseract google-generativeai openai cachetools --quiet
if errorlevel 1 (
    echo ❌ Package installation failed
    pause
    exit /b 1
) else (
    echo ✅ All packages installed!
)

echo.
echo [3/4] Checking Tesseract OCR...
tesseract --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Tesseract not found! Opening download page...
    start https://github.com/UB-Mannheim/tesseract/wiki
    echo.
    echo IMPORTANT: During Tesseract installation:
    echo - Check "Japanese" in language packs
    echo - Install to: C:\Program Files\Tesseract-OCR
    echo.
    echo Press any key after installing Tesseract...
    pause >nul
    
    tesseract --version >nul 2>&1
    if errorlevel 1 (
        echo ❌ Tesseract still not found. Please install and try again.
        pause
        exit /b 1
    )
)

tesseract --list-langs 2>nul | find "jpn" >nul
if errorlevel 1 (
    echo ⚠️  Japanese language not found in Tesseract
    echo Please reinstall Tesseract with Japanese language pack
    pause
    exit /b 1
) else (
    echo ✅ Tesseract with Japanese found!
)

echo.
echo [4/4] Setting up API key...
echo.
echo ┌─────────────────────────────────────────────────────────┐
echo │  ONLY ONE STEP LEFT - GET YOUR FREE API KEY:           │
echo │                                                         │
echo │  1. Go to: https://aistudio.google.com/app/apikey       │
echo │  2. Sign in with Google                                 │
echo │  3. Click "Create API Key"                              │
echo │  4. Copy the key (starts with AIza...)                 │
echo │                                                         │
echo │  Then run this command with YOUR key:                  │
echo │  setx GEMINI_API_KEY "your_key_here"                   │
echo └─────────────────────────────────────────────────────────┘
echo.

echo Opening API key page for you...
start https://aistudio.google.com/app/apikey

echo.
set /p "apikey=Paste your API key here (or press Enter to set it manually later): "

if not "%apikey%"=="" (
    echo Setting API key...
    setx GEMINI_API_KEY "%apikey%" >nul
    if errorlevel 1 (
        echo ❌ Failed to set API key
    ) else (
        echo ✅ API key set successfully!
        echo.
        echo 🎉 SETUP COMPLETE! Starting the application...
        echo.
        python main.py
        exit /b 0
    )
)

echo.
echo ⚠️  API key not set. Set it manually with:
echo setx GEMINI_API_KEY "your_actual_key_here"
echo.
echo Then run: python main.py
echo.
pause
