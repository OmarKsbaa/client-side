# Exoplanet API - One-Click Setup PowerShell Script
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   Exoplanet API - One-Click Setup" -ForegroundColor Cyan  
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Function to check if command exists
function Test-Command($command) {
    try {
        Get-Command $command -ErrorAction Stop | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

# Check if Python is installed
Write-Host "[1/4] Checking Python installation..." -ForegroundColor Yellow
if (-not (Test-Command "python")) {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ from https://python.org" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

$pythonVersion = python --version 2>&1
Write-Host "✓ Found: $pythonVersion" -ForegroundColor Green

# Create virtual environment
Write-Host ""
Write-Host "[2/4] Setting up virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Gray
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "✓ Virtual environment already exists" -ForegroundColor Green
}

# Install dependencies
Write-Host ""
Write-Host "[3/4] Installing dependencies..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "✓ Dependencies installed successfully" -ForegroundColor Green

# Start the API
Write-Host ""
Write-Host "[4/4] Starting Exoplanet API..." -ForegroundColor Yellow
Write-Host "✓ Setup complete! Starting server..." -ForegroundColor Green
Write-Host ""
Write-Host "🚀 API will be available at:" -ForegroundColor Cyan
Write-Host "   - API Documentation: " -NoNewline -ForegroundColor White
Write-Host "http://localhost:8001/docs" -ForegroundColor Blue
Write-Host "   - Web Interface: " -NoNewline -ForegroundColor White
Write-Host "http://localhost:8001/predict/form" -ForegroundColor Blue
Write-Host "   - Health Check: " -NoNewline -ForegroundColor White
Write-Host "http://localhost:8001/health" -ForegroundColor Blue
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Start the API
python kepler_api.py