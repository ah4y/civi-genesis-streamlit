# CIVI-GENESIS Setup Script for Windows PowerShell
# Run this script to set up the project automatically

Write-Host "🏙️  CIVI-GENESIS Setup Script" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Step 1: Checking Python version..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version
    Write-Host "✅ Found: $pythonVersion" -ForegroundColor Green
    
    # Parse version
    $versionMatch = $pythonVersion -match 'Python (\d+)\.(\d+)'
    if ($versionMatch) {
        $major = [int]$Matches[1]
        $minor = [int]$Matches[2]
        
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
            Write-Host "⚠️  Warning: Python 3.10+ is recommended" -ForegroundColor Red
        }
    }
} catch {
    Write-Host "❌ Python not found. Please install Python 3.10+ first." -ForegroundColor Red
    exit 1
}

Write-Host ""

# Install dependencies
Write-Host "Step 2: Installing Python dependencies..." -ForegroundColor Yellow
try {
    pip install -r requirements.txt
    Write-Host "✅ Dependencies installed successfully" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
    Write-Host "   Try running: pip install -r requirements.txt" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Check for .env file
Write-Host "Step 3: Checking environment configuration..." -ForegroundColor Yellow
if (Test-Path ".env") {
    Write-Host "✅ .env file found" -ForegroundColor Green
} else {
    Write-Host "⚠️  No .env file found" -ForegroundColor Yellow
    Write-Host "   Creating .env from .env.example..." -ForegroundColor Yellow
    
    if (Test-Path ".env.example") {
        Copy-Item .env.example .env
        Write-Host "✅ Created .env file" -ForegroundColor Green
        Write-Host ""
        Write-Host "⚠️  IMPORTANT: You need to edit .env and add your GEMINI_API_KEY!" -ForegroundColor Red
        Write-Host "   Get your API key from: https://makersuite.google.com/app/apikey" -ForegroundColor Cyan
    } else {
        Write-Host "❌ .env.example not found" -ForegroundColor Red
    }
}

Write-Host ""

# Check directories
Write-Host "Step 4: Checking project structure..." -ForegroundColor Yellow
$directories = @("data", "models")
foreach ($dir in $directories) {
    if (Test-Path $dir) {
        Write-Host "✅ $dir/ directory exists" -ForegroundColor Green
    } else {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "✅ Created $dir/ directory" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "🎉 Setup Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Edit .env and add your GEMINI_API_KEY" -ForegroundColor White
Write-Host "2. Run: streamlit run app.py" -ForegroundColor White
Write-Host ""
Write-Host "For detailed instructions, see QUICKSTART.md" -ForegroundColor Cyan
Write-Host ""

# Ask if user wants to run the app now
$runNow = Read-Host "Do you want to run the app now? (y/n)"
if ($runNow -eq "y" -or $runNow -eq "Y") {
    Write-Host ""
    Write-Host "Starting CIVI-GENESIS..." -ForegroundColor Cyan
    streamlit run app.py
}
