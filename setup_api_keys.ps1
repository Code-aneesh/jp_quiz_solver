# PowerShell Script to Setup API Keys
# Run this script as Administrator for system-wide setup

Write-Host "🔑 API Key Setup for Ultimate Japanese Quiz Solver" -ForegroundColor Green
Write-Host "=" * 60

Write-Host "`n📋 Available API Providers:" -ForegroundColor Yellow
Write-Host "1. Gemini (Google) - Free tier: 50 requests/day, then $0.125/1M tokens"
Write-Host "2. OpenAI (GPT-4) - $0.03/1K input tokens, $0.06/1K output tokens"
Write-Host "3. Both (recommended for reliability)"

$choice = Read-Host "`nWhich API provider would you like to setup? (1/2/3)"

if ($choice -eq "1" -or $choice -eq "3") {
    Write-Host "`n🔑 Setting up Gemini API..." -ForegroundColor Cyan
    Write-Host "Get your API key from: https://aistudio.google.com/app/apikey"
    
    $geminiKey = Read-Host "Enter your Gemini API key"
    
    if ($geminiKey -and $geminiKey -ne "") {
        # Set for current user
        [Environment]::SetEnvironmentVariable("GEMINI_API_KEY", $geminiKey, "User")
        Write-Host "✅ Gemini API key set for current user" -ForegroundColor Green
        
        # Test the key
        $env:GEMINI_API_KEY = $geminiKey
        Write-Host "✅ Key available in current session" -ForegroundColor Green
    } else {
        Write-Host "❌ No Gemini API key provided" -ForegroundColor Red
    }
}

if ($choice -eq "2" -or $choice -eq "3") {
    Write-Host "`n🔑 Setting up OpenAI API..." -ForegroundColor Cyan
    Write-Host "Get your API key from: https://platform.openai.com/api-keys"
    
    $openaiKey = Read-Host "Enter your OpenAI API key"
    
    if ($openaiKey -and $openaiKey -ne "") {
        # Set for current user
        [Environment]::SetEnvironmentVariable("OPENAI_API_KEY", $openaiKey, "User")
        Write-Host "✅ OpenAI API key set for current user" -ForegroundColor Green
        
        # Test the key
        $env:OPENAI_API_KEY = $openaiKey
        Write-Host "✅ Key available in current session" -ForegroundColor Green
    } else {
        Write-Host "❌ No OpenAI API key provided" -ForegroundColor Red
    }
}

Write-Host "`n🔄 Current Environment Variables:" -ForegroundColor Yellow
if ($env:GEMINI_API_KEY) {
    $maskedGemini = $env:GEMINI_API_KEY.Substring(0, 8) + "..." + $env:GEMINI_API_KEY.Substring($env:GEMINI_API_KEY.Length - 4)
    Write-Host "GEMINI_API_KEY: $maskedGemini" -ForegroundColor Green
} else {
    Write-Host "GEMINI_API_KEY: Not set" -ForegroundColor Red
}

if ($env:OPENAI_API_KEY) {
    $maskedOpenAI = $env:OPENAI_API_KEY.Substring(0, 8) + "..." + $env:OPENAI_API_KEY.Substring($env:OPENAI_API_KEY.Length - 4)
    Write-Host "OPENAI_API_KEY: $maskedOpenAI" -ForegroundColor Green
} else {
    Write-Host "OPENAI_API_KEY: Not set" -ForegroundColor Red
}

Write-Host "`n📝 Next Steps:" -ForegroundColor Yellow
Write-Host "1. Restart PowerShell/Command Prompt to load new environment variables"
Write-Host "2. Or restart your entire application"
Write-Host "3. Run: python main_phase2a.py --mode gui"

Write-Host "`n✅ Setup complete!" -ForegroundColor Green
Read-Host "Press Enter to exit"
