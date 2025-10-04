@echo off
echo ========================================
echo    Exoplanet API - One-Click Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo [1/4] Checking Python installation...
python --version

echo.
echo [2/4] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✓ Virtual environment created
) else (
    echo ✓ Virtual environment already exists
)

echo.
echo [3/4] Installing dependencies...
call venv\Scripts\activate.bat
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo [4/4] Starting Exoplanet API...
echo ✓ Setup complete! Starting server...
echo.
echo 🚀 API will be available at:
echo    - API Documentation: http://localhost:8001/docs
echo    - Web Interface: http://localhost:8001/predict/form
echo    - Health Check: http://localhost:8001/health
echo.
echo Press Ctrl+C to stop the server
echo.

python kepler_api.py