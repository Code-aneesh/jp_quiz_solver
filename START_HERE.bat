@echo off
title Japanese Quiz Solver - Quick Start
color 0B
cls
echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║             🇯🇵 JAPANESE QUIZ SOLVER 🇯🇵                 ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.
echo  Everything is ready! Just need your API key...
echo.

REM Check if API key is already set
if defined GEMINI_API_KEY (
    if not "%GEMINI_API_KEY%"=="YOUR_GEMINI_KEY_HERE" (
        echo ✅ API key found! Starting application...
        python main.py
        exit /b 0
    )
)

echo ┌─────────────────────────────────────────────────────────┐
echo │  🔑 GET YOUR FREE API KEY (30 seconds):                │
echo │                                                         │
echo │  1. Go to: https://aistudio.google.com/app/apikey       │
echo │  2. Sign in with Google account                         │
echo │  3. Click "Create API Key"                              │
echo │  4. Copy the key                                        │
echo └─────────────────────────────────────────────────────────┘
echo.

echo Opening the API key page for you...
start https://aistudio.google.com/app/apikey
echo.
echo Waiting for you to get the key...
timeout /t 3 >nul
echo.

:ask_key
set /p "apikey=📋 Paste your API key here: "

if "%apikey%"=="" (
    echo Please enter your API key!
    goto ask_key
)

echo.
echo 🔧 Setting up your API key...
setx GEMINI_API_KEY "%apikey%" >nul 2>&1

if errorlevel 1 (
    echo ❌ Failed to set API key permanently
    echo Setting temporarily...
    set GEMINI_API_KEY=%apikey%
) else (
    echo ✅ API key saved permanently!
    set GEMINI_API_KEY=%apikey%
)

echo.
echo 🎉 READY TO GO! Starting the Japanese Quiz Solver...
echo.
timeout /t 2 >nul

python main.py

if errorlevel 1 (
    echo.
    echo ❌ Something went wrong. Try running: EASY_SETUP.bat
    pause
)

echo.
echo Thanks for using Japanese Quiz Solver! 
pause
