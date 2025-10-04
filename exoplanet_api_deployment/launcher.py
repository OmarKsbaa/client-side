#!/usr/bin/env python3
"""
Exoplanet API Launcher - One-Click Setup and Run
Automatically handles virtual environment, dependencies, and API startup
"""
import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def print_banner():
    """Print a nice banner"""
    print("=" * 50)
    print("    🚀 Exoplanet API - One-Click Launcher")
    print("=" * 50)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ ERROR: Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        print("   Please upgrade Python from https://python.org")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} detected")
    return True

def is_package_installed(package_name):
    """Check if a package is installed"""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def setup_virtual_environment():
    """Create and activate virtual environment"""
    print("\n📦 Setting up virtual environment...")
    
    venv_path = Path("venv")
    
    if not venv_path.exists():
        print("Creating virtual environment...")
        result = subprocess.run([
            sys.executable, "-m", "venv", "venv"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Failed to create virtual environment: {result.stderr}")
            return False
        print("✅ Virtual environment created")
    else:
        print("✅ Virtual environment already exists")
    
    return True

def install_requirements():
    """Install required packages in virtual environment"""
    print("\n📦 Installing dependencies in virtual environment...")
    
    # Use virtual environment's python and pip
    if os.name == 'nt':  # Windows
        venv_python = Path("venv/Scripts/python.exe")
        venv_pip = Path("venv/Scripts/pip.exe")
    else:  # Unix-like
        venv_python = Path("venv/bin/python")
        venv_pip = Path("venv/bin/pip")
    
    if not venv_python.exists():
        print("❌ Virtual environment not found. Please run setup first.")
        return False
    
    try:
        # Install from requirements.txt using virtual environment
        if Path("requirements.txt").exists():
            print("Installing from requirements.txt...")
            result = subprocess.run([
                str(venv_pip), "install", "-r", "requirements.txt", "--force-reinstall"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Dependencies installed successfully")
                return True
            else:
                print(f"⚠️  Warning: pip install failed: {result.stderr}")
                print("🔄 Trying basic packages...")
        
        # Fallback: install essential packages
        essential_packages = [
            "fastapi==0.104.1", 
            "uvicorn[standard]==0.24.0",
            "python-multipart==0.0.6",
            "pandas==2.1.3",
            "numpy>=1.24.0,<2.0.0",
            "scikit-learn==1.3.2",
            "joblib==1.3.2"
        ]
        
        for package in essential_packages:
            print(f"   Installing {package}...")
            result = subprocess.run([
                str(venv_pip), "install", package
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Failed to install {package}: {result.stderr}")
                return False
        
        print("✅ Essential dependencies installed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def check_model_files():
    """Check if model files exist"""
    print("\n🔍 Checking model files...")
    models_dir = Path("models")
    
    if not models_dir.exists():
        print("❌ ERROR: 'models' directory not found")
        print("   Please ensure the models directory with .pkl files exists")
        return False
    
    required_models = [
        "kepler_model_complete.pkl",
        "toi_model_enhanced_working.pkl"  # Will fallback to toi_model_complete.pkl if needed
    ]
    
    found_models = []
    for model_file in required_models:
        model_path = models_dir / model_file
        if model_path.exists():
            found_models.append(model_file)
            print(f"✅ Found: {model_file}")
    
    # Check for fallback TOI model
    if "toi_model_enhanced_working.pkl" not in found_models:
        fallback_toi = models_dir / "toi_model_complete.pkl"
        if fallback_toi.exists():
            found_models.append("toi_model_complete.pkl")
            print(f"✅ Found: toi_model_complete.pkl (fallback)")
    
    if len(found_models) < 2:  # Need at least Kepler + one TOI model
        print("❌ ERROR: Missing required model files")
        print("   Required: kepler_model_complete.pkl and at least one TOI model")
        return False
    
    print(f"✅ Found {len(found_models)} model files")
    return True

def start_api():
    """Start the FastAPI application using virtual environment"""
    print("\n🚀 Starting Exoplanet API...")
    print("\n" + "=" * 50)
    print("🌟 API will be available at:")
    print("   • API Documentation: http://localhost:8001/docs")
    print("   • Web Interface: http://localhost:8001/predict/form")
    print("   • Health Check: http://localhost:8001/health")
    print("=" * 50)
    print("\n⚡ Press Ctrl+C to stop the server\n")
    
    try:
        # Use virtual environment's python
        if os.name == 'nt':  # Windows
            venv_python = Path("venv/Scripts/python.exe")
        else:  # Unix-like
            venv_python = Path("venv/bin/python")
        
        if not venv_python.exists():
            print("❌ Virtual environment not found")
            return False
        
        # Start the API using virtual environment's Python
        print("🔄 Starting API with virtual environment...")
        subprocess.run([str(venv_python), "kepler_api.py"])
            
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting API: {e}")
        return False
    
    return True

def main():
    """Main launcher function"""
    print_banner()
    
    # Step 1: Check Python version
    print("🔍 Step 1/5: Checking Python compatibility...")
    if not check_python_version():
        input("\nPress Enter to exit...")
        return False
    
    # Step 2: Setup virtual environment
    print("\n🏗️ Step 2/5: Setting up virtual environment...")
    if not setup_virtual_environment():
        input("\nPress Enter to exit...")
        return False
    
    # Step 3: Install dependencies
    print("\n📦 Step 3/5: Installing dependencies...")
    if not install_requirements():
        input("\nPress Enter to exit...")
        return False
    
    # Step 4: Check model files
    print("\n🎯 Step 4/5: Validating model files...")
    if not check_model_files():
        input("\nPress Enter to exit...")
        return False
    
    # Step 5: Start API
    print("\n🚀 Step 5/5: Launching API server...")
    start_api()
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n👋 Launcher interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        input("\nPress Enter to exit...")
        sys.exit(1)