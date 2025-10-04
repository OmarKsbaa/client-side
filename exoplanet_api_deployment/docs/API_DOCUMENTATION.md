# 🚀 Exoplanet Classifier API Documentation

## Overview

The Exoplanet Classifier API is a FastAPI-based web service that provides machine learning classification for exoplanet candidates using two high-performance models:

- **Kepler Model**: 99.06% accuracy for NASA Kepler mission data
- **TOI Model**: 88.13% accuracy for TESS Objects of Interest data

## Table of Contents

1. [Quick Start](#quick-start)
2. [API Endpoints](#api-endpoints)
3. [Model Information](#model-information)
4. [Data Formats](#data-formats)
5. [Error Handling](#error-handling)
6. [Examples](#examples)
7. [Performance Metrics](#performance-metrics)

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Running the API

```bash
uvicorn kepler_api:app --host 0.0.0.0 --port 8001
```

### Access Points

- **Web Interface**: http://localhost:8001/predict/form
- **API Documentation**: http://localhost:8001/docs
- **Health Check**: http://localhost:8001/health

## API Endpoints

### 1. Root Endpoint

**GET** `/`

Returns basic API information and navigation links.

**Response:**
```json
{
  "message": "Exoplanet Classifier API - Dual Model Support",
  "version": "2.0.0",
  "models": "Kepler & TOI",
  "form": "/predict/form",
  "docs": "/docs",
  "health": "/health"
}
```

### 2. Health Check

**GET** `/health`

Checks the status of both models and returns system health information.

**Response:**
```json
{
  "status": "healthy",
  "kepler_model_loaded": true,
  "toi_model_loaded": true,
  "kepler_features_count": 10,
  "toi_features_count": 10,
  "api_version": "2.0.0"
}
```

**Status Values:**
- `healthy`: Both models loaded successfully
- `partial`: One model loaded, one failed
- `unhealthy`: Both models failed to load

### 3. Available Models

**GET** `/models`

Returns information about all available classification models.

**Response:**
```json
{
  "kepler": {
    "classifier_type": "Kepler",
    "accuracy": "99.06%",
    "features_count": 10,
    "required_features": ["koi_score", "koi_fpflag_nt", ...],
    "description": "NASA Kepler exoplanet candidate classifier"
  },
  "toi": {
    "classifier_type": "TOI",
    "accuracy": "88.13%",
    "features_count": 10,
    "required_features": ["pl_eqt", "pl_insol", ...],
    "description": "TESS Objects of Interest exoplanet classifier"
  }
}
```

### 4. Model Features

**GET** `/features/{model_type}`

Returns the required features for a specific model.

**Parameters:**
- `model_type`: Either `kepler` or `toi`

**Response:**
```json
{
  "classifier_type": "kepler",
  "required_features": [
    "koi_score",
    "koi_fpflag_nt",
    "koi_fpflag_ec",
    "koi_fpflag_co",
    "koi_model_snr",
    "koi_fpflag_ss",
    "koi_prad",
    "koi_period",
    "koi_duration",
    "koi_impact"
  ],
  "feature_count": 10,
  "description": "Required features for KEPLER exoplanet classification"
}
```

### 5. Web Form Interface

**GET** `/predict/form`

Returns an HTML form for easy model selection and file upload.

Features:
- Model selection dropdown
- CSV file upload
- Performance information display
- One-click processing

### 6. CSV Prediction

**POST** `/predict/csv`

Upload a CSV file and get exoplanet predictions using the selected model.

**Parameters:**
- `file`: CSV file (multipart/form-data)
- `model_type`: Either `kepler` or `toi` (form data)

**Request Example:**
```bash
curl -X POST "http://localhost:8001/predict/csv" \
  -F "file=@exoplanet_data.csv" \
  -F "model_type=kepler"
```

**Response:**
```json
{
  "total_rows": 100,
  "predictions": [
    {
      "row_id": 1,
      "prediction": "CANDIDATE",
      "confidence": 0.95,
      "candidate_probability": 0.95,
      "false_positive_probability": 0.05
    }
  ],
  "summary": {
    "total_candidates": 75,
    "total_false_positives": 25,
    "average_confidence": 0.87,
    "candidate_percentage": 75.0
  },
  "processing_time_seconds": 0.15,
  "timestamp": "2025-10-04T12:00:00",
  "classifier_used": "kepler",
  "classifier_accuracy": "99.06%"
}
```

### 7. Sample Prediction

**POST** `/predict/sample/{model_type}`

Test the API with predefined sample data for a specific model.

**Parameters:**
- `model_type`: Either `kepler` or `toi`

**Response:**
```json
{
  "message": "Sample prediction successful for kepler model",
  "classifier_type": "kepler",
  "sample_data_shape": [3, 10],
  "required_features": ["koi_score", "koi_fpflag_nt", ...],
  "predictions": [
    {
      "row_id": 1,
      "prediction": "CANDIDATE",
      "confidence": 0.98,
      "candidate_probability": 0.98,
      "false_positive_probability": 0.02
    }
  ]
}
```

## Model Information

### Kepler Model
- **Accuracy**: 99.06%
- **Dataset**: NASA Kepler mission data
- **Features**: 10 carefully selected features
- **Use Case**: High-precision classification of Kepler Objects of Interest (KOI)

**Required Features:**
1. `koi_score` - Disposition score
2. `koi_fpflag_nt` - Not transit-like flag
3. `koi_fpflag_ec` - Ephemeris match flag
4. `koi_fpflag_co` - Centroid offset flag
5. `koi_model_snr` - Transit signal-to-noise ratio
6. `koi_fpflag_ss` - Stellar eclipse flag
7. `koi_prad` - Planetary radius
8. `koi_period` - Orbital period
9. `koi_duration` - Transit duration
10. `koi_impact` - Impact parameter

### TOI Model
- **Accuracy**: 88.13% (Enhanced version)
- **Dataset**: TESS Objects of Interest
- **Features**: 10 physical parameters
- **Use Case**: Classification of TESS mission discoveries

**Required Features:**
1. `pl_eqt` - Equilibrium temperature
2. `pl_insol` - Insolation flux
3. `pl_orbpererr2` - Orbital period upper error
4. `pl_radeerr2` - Planetary radius upper error
5. `pl_tranmid` - Transit midpoint
6. `st_disterr2` - Stellar distance upper error
7. `st_tmag` - TESS magnitude
8. `pl_tranmiderr2` - Transit midpoint upper error
9. `st_disterr1` - Stellar distance lower error
10. `pl_trandeperr1` - Transit depth lower error

## Data Formats

### Input CSV Format

Your CSV file should contain the required features as columns. Extra columns will be ignored.

**Kepler CSV Example:**
```csv
koi_score,koi_fpflag_nt,koi_fpflag_ec,koi_fpflag_co,koi_model_snr,koi_fpflag_ss,koi_prad,koi_period,koi_duration,koi_impact
0.95,0,0,0,25.5,0,1.1,365.25,6.2,0.3
0.12,1,0,1,8.2,0,2.8,88.0,12.5,0.8
```

**TOI CSV Example:**
```csv
pl_eqt,pl_insol,pl_orbpererr2,pl_radeerr2,pl_tranmid,st_disterr2,st_tmag,pl_tranmiderr2,st_disterr1,pl_trandeperr1
1500,1.2,0.01,0.1,2458000,5,12.5,0.0001,5,0.5
800,0.3,0.05,0.3,2459000,10,9.8,0.0005,8,1.2
```

### Prediction Output

Each prediction includes:
- `row_id`: Sequential row identifier
- `prediction`: Either "CANDIDATE" or "FALSE_POSITIVE"
- `confidence`: Confidence score (0-1)
- `candidate_probability`: Probability of being a candidate
- `false_positive_probability`: Probability of being a false positive

## Error Handling

### Common Error Codes

**400 Bad Request**
- Missing required features
- Invalid file format
- Empty CSV file
- Invalid model type

**503 Service Unavailable**
- Model not loaded
- System initialization failed

**500 Internal Server Error**
- Unexpected processing error
- Model prediction failure

### Error Response Format

```json
{
  "error": "HTTP 400",
  "message": "Missing required features for kepler model: {'koi_score', 'koi_period'}",
  "timestamp": "2025-10-04T12:00:00"
}
```

## Examples

### Python Client Example

```python
import requests
import pandas as pd

# Load your data
df = pd.read_csv('exoplanet_data.csv')

# Prepare the request
files = {'file': open('exoplanet_data.csv', 'rb')}
data = {'model_type': 'kepler'}  # or 'toi'

# Make prediction
response = requests.post(
    'http://localhost:8001/predict/csv',
    files=files,
    data=data
)

# Process results
if response.status_code == 200:
    results = response.json()
    print(f"Processed {results['total_rows']} objects")
    print(f"Found {results['summary']['total_candidates']} candidates")
    print(f"Model accuracy: {results['classifier_accuracy']}")
else:
    print(f"Error: {response.json()['message']}")
```

### JavaScript/Fetch Example

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('model_type', 'toi');

fetch('http://localhost:8001/predict/csv', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log(`Classified ${data.total_rows} objects`);
    console.log(`Candidates: ${data.summary.total_candidates}`);
    console.log(`Average confidence: ${data.summary.average_confidence}`);
});
```

### cURL Examples

```bash
# Health check
curl http://localhost:8001/health

# Get available models
curl http://localhost:8001/models

# Test with sample data
curl -X POST http://localhost:8001/predict/sample/kepler

# Upload CSV file
curl -X POST "http://localhost:8001/predict/csv" \
  -F "file=@data.csv" \
  -F "model_type=toi"
```

## Performance Metrics

### Model Comparison

| Metric | Kepler Model | TOI Model |
|--------|--------------|-----------|
| **Accuracy** | 99.06% | 88.13% |
| **Precision (Candidates)** | 99.2% | 88.0% |
| **Recall (Candidates)** | 99.1% | 86.0% |
| **F1-Score** | 99.1% | 87.0% |
| **ROC-AUC** | 99.5% | 86.0% |
| **Features** | 10 | 10 |
| **Training Samples** | 7,651 | 5,790 |
| **Test Samples** | 1,913 | 1,449 |

### API Performance

- **Startup Time**: ~3-5 seconds (model loading)
- **Prediction Speed**: ~100-500 objects/second
- **Memory Usage**: ~200MB (both models loaded)
- **CPU Usage**: Low (inference only)

### Response Times

| Endpoint | Typical Response Time |
|----------|----------------------|
| `/health` | < 10ms |
| `/models` | < 20ms |
| `/predict/sample` | < 100ms |
| `/predict/csv` (100 rows) | < 200ms |
| `/predict/csv` (1000 rows) | < 1000ms |

## Best Practices

### Model Selection
- Use **Kepler model** for KOI data from the Kepler mission
- Use **TOI model** for TESS Objects of Interest data
- Check required features before processing
- Validate data quality and completeness

### Performance Optimization
- Batch process large datasets
- Pre-validate CSV files
- Use appropriate model for your data source
- Monitor memory usage for very large files

### Error Handling
- Always check response status codes
- Validate required features before upload
- Handle network timeouts appropriately
- Log errors for debugging

## Support

For technical support or questions:
- Check the interactive documentation at `/docs`
- Test with sample data using `/predict/sample/{model}`
- Verify model status with `/health`
- Review this documentation for common issues

## Version History

- **v2.0.0** - Dual model support, enhanced TOI model, web interface
- **v1.0.0** - Initial Kepler-only API release