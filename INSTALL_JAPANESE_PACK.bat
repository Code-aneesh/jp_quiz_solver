@echo off
title Installing Japanese Language Pack for Tesseract
color 0A
echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║           🇯🇵 INSTALLING JAPANESE LANGUAGE PACK         ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo ✅ Running with administrator privileges
) else (
    echo ⚠️ Administrator privileges required
    echo Requesting elevation...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

echo.
echo 📁 Checking Tesseract installation...

if not exist "C:\Program Files\Tesseract-OCR\tesseract.exe" (
    echo ❌ Tesseract not found! Please install Tesseract first.
    pause
    exit /b 1
)

if not exist "C:\Program Files\Tesseract-OCR\tessdata" (
    echo 📂 Creating tessdata directory...
    mkdir "C:\Program Files\Tesseract-OCR\tessdata"
)

echo ✅ Tesseract found!
echo.

echo 📥 Copying Japanese language pack...

REM Check if Japanese pack is already there
if exist "C:\Program Files\Tesseract-OCR\tessdata\jpn.traineddata" (
    echo ⚠️ Japanese language pack already exists
    choice /c YN /m "Replace existing Japanese language pack"
    if errorlevel 2 goto skip_copy
)

REM Copy the downloaded Japanese pack
if exist "%TEMP%\jpn.traineddata" (
    copy "%TEMP%\jpn.traineddata" "C:\Program Files\Tesseract-OCR\tessdata\jpn.traineddata" >nul
    if %errorLevel% == 0 (
        echo ✅ Japanese language pack installed successfully!
    ) else (
        echo ❌ Failed to copy Japanese language pack
        pause
        exit /b 1
    )
) else (
    echo ⚠️ Japanese language pack not found in temp folder
    echo Downloading directly...
    
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://github.com/tesseract-ocr/tessdata/raw/main/jpn.traineddata' -OutFile 'C:\Program Files\Tesseract-OCR\tessdata\jpn.traineddata' -UserAgent 'Mozilla/5.0'}"
    
    if exist "C:\Program Files\Tesseract-OCR\tessdata\jpn.traineddata" (
        echo ✅ Japanese language pack downloaded and installed!
    ) else (
        echo ❌ Failed to download Japanese language pack
        pause
        exit /b 1
    )
)

:skip_copy
echo.
echo 🧪 Testing Japanese language support...

"C:\Program Files\Tesseract-OCR\tesseract.exe" --list-langs 2>nul | find "jpn" >nul
if %errorLevel% == 0 (
    echo ✅ Japanese language support confirmed!
    echo.
    echo 🎉 SUCCESS! Tesseract is ready for Japanese text recognition.
    echo.
    echo You can now use the Japanese Quiz Solver!
    echo Run: START_HERE.bat
) else (
    echo ❌ Japanese language not detected
    echo This might be a temporary issue. Try restarting the application.
)

echo.
echo 🧹 Cleaning up temporary files...
del "%TEMP%\jpn.traineddata" 2>nul
del "%TEMP%\tesseract-installer.exe" 2>nul

echo.
echo ═══════════════════════════════════════════════════════════
echo  Installation complete!
echo ═══════════════════════════════════════════════════════════
pause
