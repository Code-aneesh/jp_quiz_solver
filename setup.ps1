Write-Host "Japanese Quiz Solver Setup" -ForegroundColor Green
Write-Host "=========================" -ForegroundColor Green
Write-Host ""

# Check Python
try {
    $pythonVersion = python --version 2>$null
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found. Please install Python 3.10+ from https://python.org" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Install dependencies
Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
try {
    pip install -r requirements.txt
    Write-Host "✓ Dependencies installed successfully" -ForegroundColor Green
} catch {
    Write-Host "✗ Failed to install dependencies" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check Tesseract
Write-Host "Checking Tesseract OCR..." -ForegroundColor Yellow
try {
    $tesseractVersion = tesseract --version 2>$null
    Write-Host "✓ Tesseract found" -ForegroundColor Green
    
    # Check Japanese support
    $languages = tesseract --list-langs 2>$null
    if ($languages -match "jpn") {
        Write-Host "✓ Japanese language support confirmed" -ForegroundColor Green
    } else {
        Write-Host "⚠ Japanese language pack not found" -ForegroundColor Yellow
        Write-Host "  Please install Japanese tessdata file" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠ Tesseract not found in PATH" -ForegroundColor Yellow
    Write-Host "  Please install from: https://github.com/UB-Mannheim/tesseract/wiki" -ForegroundColor Yellow
    Write-Host "  Default path: C:\Program Files\Tesseract-OCR\tesseract.exe" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "1. Edit config.py to set your API key:" -ForegroundColor White
Write-Host "   - For Gemini: Set GEMINI_API_KEY" -ForegroundColor Gray
Write-Host "   - For OpenAI: Set OPENAI_API_KEY environment variable" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Run the application:" -ForegroundColor White
Write-Host "   python main.py" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Click 'Select Region' to choose the quiz area" -ForegroundColor White
Write-Host ""

# Offer to set API key
$setKey = Read-Host "Would you like to set your Gemini API key now? (y/n)"
if ($setKey -eq "y" -or $setKey -eq "Y") {
    $apiKey = Read-Host "AIzaSyCVjmHF0MOFf3goIuKli5ifc-xndDpEWT0" -MaskInput
    if ($apiKey) {
        $env:GEMINI_API_KEY = $apiKey
        [Environment]::SetEnvironmentVariable("AIzaSyCVjmHF0MOFf3goIuKli5ifc-xndDpEWT0", $apiKey, "User")
        Write-Host "✓ Gemini API key set" -ForegroundColor Green
    }
}

Write-Host ""
Read-Host "Press Enter to continue"
