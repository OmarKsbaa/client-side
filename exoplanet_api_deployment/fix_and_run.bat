@echo off
echo ========================================
echo    Quick Fix for NumPy Compatibility
echo ========================================
echo.

echo [1/3] Checking virtual environment...
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

echo.
echo [2/3] Activating environment and fixing NumPy...
call venv\Scripts\activate.bat

echo Uninstalling old NumPy...
pip uninstall numpy -y

echo Installing compatible NumPy version...
pip install "numpy>=1.24.0,<2.0.0" --force-reinstall

echo Reinstalling all dependencies...
pip install -r requirements.txt --force-reinstall

echo.
echo [3/3] Starting API...
echo ✓ NumPy fix applied! Starting server...
echo.
echo 🚀 API will be available at:
echo    - API Documentation: http://localhost:8001/docs
echo    - Web Interface: http://localhost:8001/predict/form
echo    - Health Check: http://localhost:8001/health
echo.
echo Press Ctrl+C to stop the server
echo.

python kepler_api.py