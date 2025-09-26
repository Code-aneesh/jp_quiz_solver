@echo off
title Japanese Quiz Solver
echo Starting Japanese Quiz Solver...
echo.

REM Quick dependency check
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "main.py" (
    echo Error: main.py not found. Please run from jp_quiz_solver directory.
    pause
    exit /b 1
)

REM Check if dependencies are installed
python -c "import mss, pytesseract, google.generativeai" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Error: Failed to install dependencies
        pause
        exit /b 1
    )
)

REM Run the application
echo Launching application...
python main.py

REM If the app exits, show a pause so user can see any error messages
if errorlevel 1 (
    echo.
    echo Application exited with error. Check the messages above.
    pause
)
