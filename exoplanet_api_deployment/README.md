# 🚀 Exoplanet Classifier API - Complete Deployment Package

A production-ready FastAPI service for classifying exoplanet candidates using machine learning models trained on NASA Kepler and TESS mission data.

## ⚡ **INSTANT START** - Just One Command!

**Don't want to read documentation?** Just run this:

```bash
python launcher.py
```

That's it! 🎉 Your API will be running at **http://localhost:8001/docs**

---

## 📊 **Performance Summary**

| Model | Accuracy | Dataset | Use Case |
|-------|----------|---------|----------|
| **Kepler** | 99.06% | NASA Kepler KOI | High-precision Kepler Objects of Interest |
| **TOI** | 88.13% | TESS TOI | TESS Objects of Interest (Enhanced) |

## 📁 **What's Included**

```
exoplanet_api_deployment/
├── 🚀 ONE-COMMAND LAUNCHERS
│   ├── launcher.py              # Smart Python launcher (RECOMMENDED)
│   ├── setup_and_run.bat        # Windows batch file
│   ├── fix_and_run.bat          # Windows batch with NumPy fix
│   └── setup_and_run.ps1        # PowerShell script
├── 🤖 API FILES
│   ├── kepler_api.py            # Main FastAPI application
│   ├── requirements.txt         # Python dependencies
│   └── test_api.py             # API tests
├── 🧠 MACHINE LEARNING MODELS
│   ├── kepler_model_complete.pkl      # Kepler classifier (99.06%)
│   ├── toi_model_complete.pkl         # Original TOI classifier
│   └── toi_model_enhanced_working.pkl # Enhanced TOI classifier (88.13%)
├── 📚 DOCUMENTATION
│   ├── README.md                # This file
│   └── docs/
│       ├── API_DOCUMENTATION.md # Comprehensive API docs
│       └── MODEL_TRAINING.md    # Model training documentation
└── 🔬 TRAINING SCRIPTS (Optional)
    ├── step2_binary_data_prep.py
    ├── step4_train_model.py
    └── [other training files...]
```

## 🚀 **Quick Start Options**

### 🎯 **Prerequisites** (Automatically checked)
- Python 3.8+ (the launcher will tell you if this is missing)
- 4GB+ RAM (for model loading)
- Internet connection (for downloading packages)

### ⚡ **ONE-COMMAND SETUP** (Pick Any One)

```bash
# 🏆 RECOMMENDED: Smart Python Launcher
python launcher.py

# 🪟 Windows Users: Double-click any of these files OR run:
.\setup_and_run.bat          # Basic setup
.\fix_and_run.bat           # Includes NumPy compatibility fix
powershell -ExecutionPolicy Bypass -File setup_and_run.ps1  # PowerShell version
```

**🎉 That's literally it!** The script automatically:
- ✅ Checks your Python version
- ✅ Creates virtual environment 
- ✅ Installs all dependencies
- ✅ Validates model files
- ✅ Starts the API server
- ✅ Shows you the links

**🔗 Your API will be ready at:**
- **📚 API Documentation**: http://localhost:8001/docs (START HERE!)
- **🌐 Web Interface**: http://localhost:8001/predict/form  
- **❤️ Health Check**: http://localhost:8001/health

### 📱 **Manual Setup** (Traditional way)

If you prefer manual setup:

```bash
# Clone or extract the deployment package
cd exoplanet_api_deployment

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the API Server
python kepler_api.py
```

### 🔄 **Alternative Startup Methods**

```bash
# Option 1: Direct Python execution
python kepler_api.py

# Option 2: Using Uvicorn (recommended for production)
uvicorn kepler_api:app --host 0.0.0.0 --port 8001

# Option 3: With auto-reload for development
uvicorn kepler_api:app --host 0.0.0.0 --port 8001 --reload
```

---

## 👨‍💻 **For API Engineers & Developers**

### 🔧 **Integration Endpoints**

```python
import requests

# Base URL
BASE_URL = "http://localhost:8001"

# 1. Health Check
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# 2. Get Available Models
models = requests.get(f"{BASE_URL}/models").json()
print(f"Available models: {list(models.keys())}")

# 3. Classify CSV Data
files = {'file': open('exoplanet_data.csv', 'rb')}
data = {'model_type': 'kepler'}  # or 'toi'
result = requests.post(f"{BASE_URL}/predict/csv", files=files, data=data)
predictions = result.json()

# 4. Test with Sample Data
sample_result = requests.post(f"{BASE_URL}/predict/sample/kepler")
print(sample_result.json())
```

### 📊 **Required CSV Format**

**Kepler Model** (10 features):
```csv
koi_score,koi_fpflag_nt,koi_fpflag_ec,koi_fpflag_co,koi_model_snr,koi_fpflag_ss,koi_prad,koi_period,koi_duration,koi_impact
0.95,0,0,0,25.5,0,1.1,365.25,6.2,0.3
```

**TOI Model** (10 features):
```csv
pl_eqt,pl_insol,pl_orbpererr2,pl_radeerr2,pl_tranmid,st_disterr2,st_tmag,pl_tranmiderr2,st_disterr1,pl_trandeperr1
1500,1.2,0.01,0.1,2458000,5,12.5,0.0001,5,0.5
```

### 🛠️ **Key Implementation Notes**

1. **Model Selection**: Use `model_type` parameter (`"kepler"` or `"toi"`)
2. **File Upload**: Multipart form data with `file` field
3. **Response Format**: JSON with predictions array and summary stats
4. **Error Handling**: 400 for client errors, 500 for server errors
5. **Rate Limiting**: Consider implementing for production use

### 🔒 **Production Deployment Checklist**

- [ ] Add HTTPS/SSL certificates
- [ ] Implement authentication (API keys/JWT)
- [ ] Add rate limiting (`slowapi` recommended)
- [ ] Configure proper logging
- [ ] Set up monitoring (health checks)
- [ ] Use production WSGI server (Gunicorn + Uvicorn)
- [ ] Configure environment variables
- [ ] Set up database for persistent storage (optional)

### 3. Verify Installation

**Visit these URLs in your browser:**
- **📚 API Documentation**: http://localhost:8001/docs (Interactive Swagger UI)
- **🌐 Web Interface**: http://localhost:8001/predict/form (Upload CSV files)  
- **❤️ Health Check**: http://localhost:8001/health (System status)

## 🎯 **Usage Examples**

## 🎯 **Usage Examples**

### 🌐 **Web Interface** (Easiest for beginners)

1. **Go to**: http://localhost:8001/predict/form
2. **Choose Model**: Kepler (99.06%) or TOI (88.13%)  
3. **Upload CSV**: Your exoplanet data file
4. **Get Results**: Instant classifications with confidence scores

### 🐍 **Python Integration** (For developers)

```python
import requests
import pandas as pd

BASE_URL = "http://localhost:8001"

# Quick test with sample data
response = requests.post(f'{BASE_URL}/predict/sample/kepler')
results = response.json()
print(f"Sample prediction: {results['predictions'][0]['prediction']}")

# Upload your CSV file
with open('your_exoplanet_data.csv', 'rb') as f:
    files = {'file': f}
    data = {'model_type': 'kepler'}  # or 'toi'
    
    response = requests.post(f'{BASE_URL}/predict/csv', files=files, data=data)
    results = response.json()
    
    print(f"✅ Classified {results['total_rows']} objects")
    print(f"🌟 Found {results['summary']['total_candidates']} exoplanet candidates")
    print(f"📊 Average confidence: {results['summary']['average_confidence']:.2%}")
```

### 🔧 **Command Line** (cURL examples)

```bash
# Test if API is running
curl http://localhost:8001/health

# See available models and their accuracy
curl http://localhost:8001/models

# Quick test with sample data
curl -X POST http://localhost:8001/predict/sample/kepler

# Upload your CSV file
curl -X POST "http://localhost:8001/predict/csv" \
  -F "file=@your_data.csv" \
  -F "model_type=toi"
```

---

## 🎯 **Troubleshooting**

### ❓ **Common Issues & Solutions**

| Problem | Solution |
|---------|----------|
| 🐍 "Python not found" | Install Python 3.8+ from [python.org](https://python.org) |
| 🔌 "Port 8001 already in use" | Stop other servers or change port in code |
| 💾 "Model loading failed" | Run `.\fix_and_run.bat` for NumPy compatibility |
| 📁 "File not found" | Make sure you're in the `exoplanet_api_deployment` folder |
| 🌐 "Can't access localhost:8001" | Check firewall settings, try `127.0.0.1:8001` |

### � **Need Help?**
1. **First**: Check the API logs in your terminal
2. **Second**: Visit http://localhost:8001/health to see system status  
3. **Third**: Try the `fix_and_run.bat` script for common fixes

---

## �📊 **CSV Data Format Requirements**

### 🔭 **Kepler Model** (NASA Kepler mission data)
**Requires exactly these 10 columns:**
```csv
koi_score,koi_fpflag_nt,koi_fpflag_ec,koi_fpflag_co,koi_model_snr,koi_fpflag_ss,koi_prad,koi_period,koi_duration,koi_impact
0.95,0,0,0,25.5,0,1.1,365.25,6.2,0.3
0.12,1,0,1,8.2,0,2.8,88.0,12.5,0.8
```

### 🛰️ **TOI Model** (TESS Objects of Interest)
**Requires exactly these 10 columns:**
```csv
pl_eqt,pl_insol,pl_orbpererr2,pl_radeerr2,pl_tranmid,st_disterr2,st_tmag,pl_tranmiderr2,st_disterr1,pl_trandeperr1
1500,1.2,0.01,0.1,2458000,5,12.5,0.0001,5,0.5
800,0.3,0.05,0.3,2459000,10,9.8,0.0005,8,1.2
```

**💡 Pro Tips:**
- Extra columns are ignored (only the required ones are used)
- Missing values are automatically handled by the model
- Column order doesn't matter, but column names must match exactly

## 🔧 **Complete API Reference**

| Endpoint | Method | Description | Example |
|----------|--------|-------------|---------|
| 📚 `/docs` | GET | **Interactive API documentation** | [Try it!](http://localhost:8001/docs) |
| 🏠 `/` | GET | API info and navigation links | Basic JSON response |
| ❤️ `/health` | GET | System health and model status | Check if models loaded |
| 🤖 `/models` | GET | Available models and performance | See accuracy rates |
| 📋 `/features/{model}` | GET | Required CSV columns for model | Get column names |
| 🌐 `/predict/form` | GET | User-friendly web upload form | Easy CSV upload |
| 📤 `/predict/csv` | POST | **Main prediction endpoint** | Upload CSV, get results |
| 🧪 `/predict/sample/{model}` | POST | Test with built-in sample data | Quick API test |

### 🚀 **Main Prediction Endpoint**
```http
POST /predict/csv
Content-Type: multipart/form-data

file: [your CSV file]
model_type: "kepler" or "toi"
```

**Response:**
```json
{
  "total_rows": 100,
  "predictions": [...],
  "summary": {
    "total_candidates": 23,
    "total_false_positives": 77,
    "average_confidence": 0.87,
    "candidate_percentage": 23.0
  },
  "classifier_used": "kepler",
  "classifier_accuracy": "99.06%"
}
```

## 🛠 **Model Training**

The training scripts demonstrate the complete pipeline used to create the models:

### Kepler Model Training
```bash
python training_scripts/step2_binary_data_prep.py
python training_scripts/step4_train_model.py
```

### TOI Model Training
```bash
python training_scripts/toi_step1_analysis.py
python training_scripts/toi_step2_cleaning.py
python training_scripts/toi_step3_feature_importance.py
python training_scripts/toi_step4_train_model.py
python training_scripts/toi_final_enhancement.py
```

### Model Enhancement
The TOI model was enhanced from 87.22% to 88.13% accuracy using:
```bash
python training_scripts/create_enhanced_model_fixed.py
```

## 🚦 **Production Deployment**

### Docker Deployment (Recommended)

Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8001

CMD ["uvicorn", "kepler_api:app", "--host", "0.0.0.0", "--port", "8001"]
```

Build and run:
```bash
docker build -t exoplanet-api .
docker run -p 8001:8001 exoplanet-api
```

### Cloud Deployment

**AWS EC2 / Google Cloud:**
```bash
# Install dependencies
sudo apt update
sudo apt install python3 python3-pip

# Setup application
git clone <your-repo>
cd exoplanet_api_deployment
pip3 install -r requirements.txt

# Run with production server
gunicorn -w 4 -k uvicorn.workers.UvicornWorker kepler_api:app --bind 0.0.0.0:8001
```

**Heroku:**
Create `Procfile`:
```
web: uvicorn kepler_api:app --host 0.0.0.0 --port $PORT
```

## 🔍 **Troubleshooting**

### Common Issues

**1. Port Already in Use**
```bash
# Use different port
uvicorn kepler_api:app --host 0.0.0.0 --port 8002
```

**2. Model Loading Errors**
- Check that all `.pkl` files are in the `models/` directory
- Verify Python version compatibility (3.8+)
- Ensure sufficient RAM (4GB+)

**3. Missing Dependencies**
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

**4. CSV Upload Issues**
- Verify CSV has required column names
- Check for missing values in critical features
- Ensure CSV file is properly formatted

### Performance Tuning

**Memory Optimization:**
- Both models require ~200MB RAM
- Consider loading models on-demand for memory-constrained environments

**Speed Optimization:**
- Use SSD storage for faster model loading
- Configure appropriate worker processes for high traffic
- Implement response caching for repeated queries

## 📈 **Monitoring & Logging**

### Health Monitoring
```bash
# Check system health
curl http://localhost:8001/health

# Monitor logs
tail -f api.log
```

### Performance Metrics
- **Model Loading**: ~3-5 seconds
- **Prediction Speed**: 100-500 objects/second
- **Memory Usage**: ~200MB for both models
- **Response Time**: <200ms for typical requests

## 🔒 **Security Considerations**

### Production Security
- Use HTTPS in production
- Implement rate limiting
- Add authentication for sensitive deployments
- Validate file uploads (size, type)
- Sanitize user inputs

### Example with Rate Limiting
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict/csv")
@limiter.limit("10/minute")
async def predict_from_csv(request: Request, ...):
    # Your existing code
```

## 📞 **Support & Maintenance**

### API Testing
```bash
# Run comprehensive tests
python -m pytest tests/ -v

# Load testing
ab -n 1000 -c 10 http://localhost:8001/health
```

### Model Updates
To update models:
1. Replace `.pkl` files in `models/` directory
2. Update feature requirements in API code
3. Test with sample data
4. Update documentation

### Backup Strategy
- Regular backups of `models/` directory
- Version control for training scripts
- Database backups if adding persistence
- Configuration backups

## 📝 **Development**

### Adding New Models
1. Train your model using the training scripts as templates
2. Save as `.pkl` file in `models/` directory
3. Update `ModelType` enum in `kepler_api.py`
4. Add model loading logic in `load_models()`
5. Update feature requirements and documentation

### Extending API
- Add new endpoints following FastAPI patterns
- Implement proper error handling
- Update API documentation
- Add comprehensive tests

## 📚 **Additional Resources**

- **Complete API Documentation**: `docs/API_DOCUMENTATION.md`
- **Interactive API Docs**: http://localhost:8001/docs
- **Model Performance Reports**: Generated during training
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Uvicorn Server**: https://www.uvicorn.org/

## 🏆 **Achievements**

- ✅ **Dual Model Support**: Kepler (99.06%) and TOI (88.13%) models
- ✅ **Enhanced Performance**: TOI model improved by +0.91 percentage points
- ✅ **Production Ready**: Complete API with documentation and testing
- ✅ **User Friendly**: Interactive web interface for easy access
- ✅ **Comprehensive**: Full training pipeline and deployment package

---

## 📋 **Quick Reference Card**

### 🏃‍♂️ **Ultra-Quick Start**
```bash
python launcher.py
# → API runs at http://localhost:8001/docs
```

### 🔗 **Essential URLs**
- **📚 Start Here**: http://localhost:8001/docs
- **🌐 Upload Files**: http://localhost:8001/predict/form  
- **❤️ Health Check**: http://localhost:8001/health

### 🎯 **Key Info**
- **Kepler Model**: 99.06% accuracy (10 features)
- **TOI Model**: 88.13% accuracy (10 features)  
- **Requirements**: Python 3.8+, 4GB RAM
- **Formats**: CSV upload, JSON API responses

### 👨‍💻 **For API Engineers**
```python
import requests
files = {'file': open('data.csv', 'rb')}
data = {'model_type': 'kepler'}
result = requests.post('http://localhost:8001/predict/csv', files=files, data=data)
predictions = result.json()
```

---

**🚀 You're all set! Run `python launcher.py` and start classifying exoplanets!** 🌟