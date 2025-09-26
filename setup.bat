@echo off
echo Installing Japanese Quiz Solver...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

echo Python found. Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Checking Tesseract installation...
tesseract --version >nul 2>&1
if errorlevel 1 (
    echo Warning: Tesseract not found in PATH
    echo Please install Tesseract OCR with Japanese language pack
    echo Download from: https://github.com/UB-Mannheim/tesseract/wiki
    echo Default path: C:\Program Files\Tesseract-OCR\tesseract.exe
) else (
    echo Tesseract found. Checking Japanese language support...
    tesseract --list-langs | find "jpn" >nul
    if errorlevel 1 (
        echo Warning: Japanese language pack not found
        echo Please install Japanese (jpn) tessdata
    ) else (
        echo Japanese language support confirmed
    )
)

echo.
echo Setup complete!
echo.
echo Next steps:
echo 1. Edit config.py to set your API key
echo 2. Run: python main.py
echo 3. Click "Select Region" to choose quiz area
echo.
pause
