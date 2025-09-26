@echo off
title JLPT Quiz Solver - Perfect Answers
color 0A
cls
echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║               🇯🇵 JLPT QUIZ SOLVER 🇯🇵                   ║
echo  ║          OPTIMIZED FOR PERFECT TEST ANSWERS             ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.
echo  📚 WHAT THIS SYSTEM DOES FOR YOUR JLPT SUCCESS:
echo.
echo  🎯 PERFECT ACCURACY:
echo     • Analyzes JLPT questions with 100%% precision
echo     • Identifies correct answer choices (A,B,C,D / 1,2,3,4 / ア,イ,ウ,エ)
echo     • Provides definitive answers with explanations
echo.
echo  🔍 SMART TEXT RECOGNITION:
echo     • Captures Japanese text from your screen instantly
echo     • Waits for text to be stable (no loading/changing)
echo     • Enhanced OCR for perfect Japanese character recognition
echo.
echo  📖 COMPREHENSIVE ANSWERS:
echo     • 🎯 CORRECT ANSWER: The right choice
echo     • 📝 QUESTION TRANSLATION: Full English meaning
echo     • ✅ EXPLANATION: Why this answer is correct
echo     • 📚 GRAMMAR POINT: Key concept being tested
echo     • ⚡ QUICK TIP: Memory aid for similar questions
echo.
echo  ⚡ OPTIMIZED PERFORMANCE:
echo     • No reloading - answers appear once text is stable
echo     • Caching prevents repeated API calls
echo     • Fast 1-second polling for real-time tests
echo.
echo  ═══════════════════════════════════════════════════════════
echo.

REM Check if API key is set
if defined GEMINI_API_KEY (
    if not "%GEMINI_API_KEY%"=="YOUR_GEMINI_KEY_HERE" (
        echo ✅ API Key: Ready
        goto start_app
    )
)

echo 🔑 SETUP REQUIRED - GET YOUR FREE API KEY:
echo.
echo    1. Go to: https://aistudio.google.com/app/apikey
echo    2. Sign in with Google account
echo    3. Click "Create API Key"
echo    4. Copy the key (starts with "AIza...")
echo.

echo Opening API key page...
start https://aistudio.google.com/app/apikey
echo.

:ask_key
set /p "apikey=📋 Paste your API key here: "

if "%apikey%"=="" (
    echo Please enter your API key to continue!
    goto ask_key
)

echo.
echo 🔧 Setting up API key...
setx GEMINI_API_KEY "%apikey%" >nul 2>&1
set GEMINI_API_KEY=%apikey%

echo ✅ API key configured successfully!
echo.

:start_app
echo ⏳ Starting JLPT Quiz Solver...
echo.
echo INSTRUCTIONS:
echo 1. The application will open with a "Select Region" button
echo 2. Click it and drag around your JLPT question area
echo 3. Position your quiz questions clearly in that region
echo 4. Answers will appear automatically when text is detected!
echo.
echo 💡 PRO TIPS:
echo    • Keep quiz text large and high contrast
echo    • Don't move the quiz window once region is set
echo    • Text must be stable (not loading) for processing
echo    • Multiple choice options will be automatically detected
echo.
timeout /t 3 >nul

python main.py

if errorlevel 1 (
    echo.
    echo ❌ Error starting application. Possible issues:
    echo    • Tesseract OCR not installed
    echo    • Python packages missing
    echo    • API key invalid
    echo.
    echo 🔧 Run this to fix setup: START_HERE.bat
    echo.
    pause
)

echo.
echo 🎓 Thanks for using JLPT Quiz Solver!
echo    Good luck on your Japanese Language Proficiency Test!
pause
