@echo off
title Auto-Installing Tesseract OCR with Japanese
color 0E
cls
echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║        🔧 AUTO-INSTALLING TESSERACT OCR + JAPANESE      ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.
echo  I'll download and install Tesseract automatically!
echo.

REM Check if already installed
if exist "C:\Program Files\Tesseract-OCR\tesseract.exe" (
    echo ✅ Tesseract already found at: C:\Program Files\Tesseract-OCR\tesseract.exe
    tesseract --list-langs 2>nul | find "jpn" >nul
    if not errorlevel 1 (
        echo ✅ Japanese language pack already installed!
        echo.
        echo 🎉 Tesseract is ready! You can close this window.
        pause
        exit /b 0
    ) else (
        echo ⚠️ Japanese language pack missing. Reinstalling...
    )
)

echo 📥 Step 1: Downloading Tesseract installer...
echo.

REM Create temp directory
if not exist "%TEMP%\tesseract_setup" mkdir "%TEMP%\tesseract_setup"
cd /d "%TEMP%\tesseract_setup"

REM Download the latest Tesseract installer (64-bit)
echo Downloading Tesseract OCR installer...
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.3.3.20231005.exe' -OutFile 'tesseract-installer.exe' -UserAgent 'Mozilla/5.0'}"

if not exist "tesseract-installer.exe" (
    echo ❌ Download failed! Opening manual download page...
    start https://github.com/UB-Mannheim/tesseract/wiki
    echo Please download and run the installer manually.
    pause
    exit /b 1
)

echo ✅ Download complete!
echo.
echo 🔧 Step 2: Installing Tesseract with Japanese language pack...
echo.

REM Silent installation with all languages including Japanese
echo Installing Tesseract OCR (this may take a minute)...
tesseract-installer.exe /S /D="C:\Program Files\Tesseract-OCR"

echo Waiting for installation to complete...
timeout /t 15 >nul

REM Check if installation was successful
if not exist "C:\Program Files\Tesseract-OCR\tesseract.exe" (
    echo ❌ Installation may have failed. Trying interactive installation...
    echo.
    echo I'll open the installer for you. Please:
    echo 1. Click Next through the installer
    echo 2. IMPORTANT: Check "Japanese" in language selection
    echo 3. Install to default location: C:\Program Files\Tesseract-OCR
    echo.
    pause
    tesseract-installer.exe
    echo.
    echo Press any key after installation is complete...
    pause >nul
)

echo.
echo 🧪 Step 3: Testing installation...

REM Test if tesseract is working
"C:\Program Files\Tesseract-OCR\tesseract.exe" --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Tesseract not working properly
    echo Please try running the installer manually: tesseract-installer.exe
    pause
    exit /b 1
)

echo ✅ Tesseract is working!

REM Test Japanese language support
"C:\Program Files\Tesseract-OCR\tesseract.exe" --list-langs 2>nul | find "jpn" >nul
if errorlevel 1 (
    echo ⚠️ Japanese language not found. This might mean:
    echo   - Installation is still in progress
    echo   - Japanese wasn't selected during installation
    echo.
    echo Checking tessdata folder...
    if exist "C:\Program Files\Tesseract-OCR\tessdata\jpn.traineddata" (
        echo ✅ Japanese data file found!
    ) else (
        echo ❌ Japanese language pack missing
        echo.
        echo 📥 Downloading Japanese language pack manually...
        cd /d "C:\Program Files\Tesseract-OCR\tessdata"
        powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://github.com/tesseract-ocr/tessdata/raw/main/jpn.traineddata' -OutFile 'jpn.traineddata'}"
        
        if exist "jpn.traineddata" (
            echo ✅ Japanese language pack downloaded!
        ) else (
            echo ❌ Failed to download Japanese pack
        )
    )
) else (
    echo ✅ Japanese language support confirmed!
)

echo.
echo 🧹 Step 4: Cleaning up...
cd /d "%~dp0"
rmdir /s /q "%TEMP%\tesseract_setup" 2>nul

echo.
echo ═══════════════════════════════════════════════════════════
echo  🎉 TESSERACT INSTALLATION COMPLETE!
echo ═══════════════════════════════════════════════════════════
echo.
echo Installation Details:
echo • Location: C:\Program Files\Tesseract-OCR\tesseract.exe
echo • Japanese Support: Enabled
echo • Ready for Japanese Quiz Solver!
echo.
echo Next: Run START_HERE.bat to set up your API key and start!
echo.
pause
