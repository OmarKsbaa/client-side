from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import io
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Exoplanet Classifier API - Dual Model Support",
    description="API for classifying exoplanet candidates using Kepler and TOI machine learning models",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Model type enumeration
class ModelType(str, Enum):
    KEPLER = "kepler"
    TOI = "toi"

# Global variables for models
kepler_model_data = None
toi_model_data = None

# Pydantic models for request/response
class PredictionResult(BaseModel):
    row_id: int
    prediction: str
    confidence: float
    candidate_probability: float
    false_positive_probability: float

class BatchPredictionResponse(BaseModel):
    total_rows: int
    predictions: List[PredictionResult]
    summary: Dict[str, Any]
    processing_time_seconds: float
    timestamp: str
    classifier_used: str
    classifier_accuracy: str

    class Config:
        protected_namespaces = ()

class HealthResponse(BaseModel):
    status: str
    kepler_model_loaded: bool
    toi_model_loaded: bool
    kepler_features_count: int
    toi_features_count: int
    api_version: str

class ModelInfo(BaseModel):
    classifier_type: str
    accuracy: str
    features_count: int
    required_features: List[str]
    description: str

    class Config:
        protected_namespaces = ()

class ErrorResponse(BaseModel):
    error: str
    message: str
    timestamp: str

# Load models on startup
@app.on_event("startup")
async def load_models():
    """Load both Kepler and TOI models and preprocessing components"""
    global kepler_model_data, toi_model_data
    
    try:
        # Load Kepler model
        kepler_path = "models/kepler_model_complete.pkl"
        kepler_model_data = joblib.load(kepler_path)
        logger.info(f"Kepler model loaded successfully with {len(kepler_model_data['required_features'])} features")
        
        # Load TOI model (enhanced version if available, otherwise original)
        try:
            toi_path = "models/toi_model_enhanced_working.pkl"
            toi_model_data = joblib.load(toi_path)
            logger.info(f"Enhanced TOI model loaded successfully with {len(toi_model_data['required_features'])} features")
        except FileNotFoundError:
            # Fallback to original TOI model
            toi_path = "models/toi_model_complete.pkl"
            toi_model_data = joblib.load(toi_path)
            logger.info(f"Original TOI model loaded successfully with {len(toi_model_data['required_features'])} features")
        
        logger.info("Both models loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        raise RuntimeError(f"Could not load models: {str(e)}")

def get_model_data(model_type: ModelType):
    """Get the appropriate model data based on model type"""
    if model_type == ModelType.KEPLER:
        if kepler_model_data is None:
            raise ValueError("Kepler model not loaded")
        return kepler_model_data
    elif model_type == ModelType.TOI:
        if toi_model_data is None:
            raise ValueError("TOI model not loaded")
        return toi_model_data
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def preprocess_data(df: pd.DataFrame, model_type: ModelType) -> np.ndarray:
    """
    Preprocess the input DataFrame for model prediction
    
    Args:
        df: Input DataFrame with exoplanet data
        model_type: Type of model to use (Kepler or TOI)
        
    Returns:
        Preprocessed numpy array ready for model prediction
    """
    try:
        # Get the appropriate model data
        model_data = get_model_data(model_type)
        required_features = model_data['required_features']
        
        # Check if all required features are present
        missing_features = set(required_features) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features for {model_type.value} model: {missing_features}")
        
        # Extract only required features
        X_features = df[required_features].copy()
        
        # Handle missing values using model's imputer if available, otherwise median
        if 'imputer' in model_data:
            X_features_filled = pd.DataFrame(
                model_data['imputer'].transform(X_features), 
                columns=X_features.columns
            )
        else:
            X_features_filled = X_features.fillna(X_features.median())
        
        # Scale features using the model's scaler
        scaler = model_data['scaler']
        X_scaled = scaler.transform(X_features_filled)
        
        return X_scaled
        
    except Exception as e:
        logger.error(f"Preprocessing error for {model_type.value}: {str(e)}")
        raise ValueError(f"Data preprocessing failed for {model_type.value} model: {str(e)}")

def make_predictions(X_processed: np.ndarray, model_type: ModelType) -> List[PredictionResult]:
    """
    Make predictions using the specified model
    
    Args:
        X_processed: Preprocessed feature array
        model_type: Type of model to use (Kepler or TOI)
        
    Returns:
        List of prediction results
    """
    try:
        # Get the appropriate model data and model
        model_data = get_model_data(model_type)
        ml_model = model_data['model']
        
        # Make predictions
        predictions = ml_model.predict(X_processed)
        probabilities = ml_model.predict_proba(X_processed)
        
        # Format results
        results = []
        for i in range(len(predictions)):
            # Both models: 0 = FALSE_POSITIVE, 1 = CANDIDATE
            pred_label = 'CANDIDATE' if predictions[i] == 1 else 'FALSE_POSITIVE'
            confidence = float(max(probabilities[i]))
            false_pos_prob = float(probabilities[i][0])
            candidate_prob = float(probabilities[i][1])
            
            results.append(PredictionResult(
                row_id=i + 1,
                prediction=pred_label,
                confidence=confidence,
                candidate_probability=candidate_prob,
                false_positive_probability=false_pos_prob
            ))
        
        return results
        
    except Exception as e:
        logger.error(f"Prediction error for {model_type.value}: {str(e)}")
        raise ValueError(f"Model prediction failed for {model_type.value} model: {str(e)}")

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic API information"""
    return {
        "message": "Exoplanet Classifier API - Dual Model Support",
        "version": "2.0.0",
        "models": "Kepler & TOI",
        "docs": "/docs - Interactive API Documentation",
        "form": "/predict/form",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    kepler_loaded = kepler_model_data is not None
    toi_loaded = toi_model_data is not None
    
    return HealthResponse(
        status="healthy" if (kepler_loaded and toi_loaded) else "partial" if (kepler_loaded or toi_loaded) else "unhealthy",
        kepler_model_loaded=kepler_loaded,
        toi_model_loaded=toi_loaded,
        kepler_features_count=len(kepler_model_data['required_features']) if kepler_loaded else 0,
        toi_features_count=len(toi_model_data['required_features']) if toi_loaded else 0,
        api_version="2.0.0"
    )

@app.get("/models", response_model=Dict[str, ModelInfo])
async def get_available_models():
    """Get information about available models"""
    models = {}
    
    if kepler_model_data is not None:
        kepler_perf = kepler_model_data.get('performance', {})
        models['kepler'] = ModelInfo(
            classifier_type="Kepler",
            accuracy=f"{kepler_perf.get('test_accuracy', 0.9906)*100:.2f}%",
            features_count=len(kepler_model_data['required_features']),
            required_features=kepler_model_data['required_features'],
            description="NASA Kepler exoplanet candidate classifier"
        )
    
    if toi_model_data is not None:
        toi_perf = toi_model_data.get('performance', {})
        models['toi'] = ModelInfo(
            classifier_type="TOI",
            accuracy=f"{toi_perf.get('test_accuracy', 0.8722)*100:.2f}%",
            features_count=len(toi_model_data['required_features']),
            required_features=toi_model_data['required_features'],
            description="TESS Objects of Interest exoplanet classifier"
        )
    
    if not models:
        raise HTTPException(status_code=503, detail="No models loaded")
    
    return models

@app.get("/features/{model_type}", response_model=Dict[str, Any])
async def get_required_features(model_type: ModelType):
    """Get the list of required features for the specified model"""
    try:
        model_data = get_model_data(model_type)
        required_features = model_data['required_features']
        
        return {
            "classifier_type": model_type.value,
            "required_features": required_features,
            "feature_count": len(required_features),
            "description": f"Required features for {model_type.value.upper()} exoplanet classification"
        }
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/predict/form", response_class=HTMLResponse)
async def prediction_form():
    """Serve HTML form for model selection and file upload"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Exoplanet Classifier</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            .form-group { margin: 20px 0; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            select, input[type="file"] { padding: 10px; width: 100%; max-width: 400px; }
            button { background: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background: #0056b3; }
            .model-info { background: #f8f9fa; padding: 15px; border-radius: 4px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <h1>🚀 Exoplanet Classifier API</h1>
        <p>Upload a CSV file and select a model to classify exoplanet candidates.</p>
        
        <div class="model-info">
            <h3>Available Models:</h3>
            <p><strong>Kepler Model:</strong> 99.06% accuracy - For NASA Kepler mission data</p>
            <p><strong>TOI Model:</strong> 88.13% accuracy - For TESS Objects of Interest data</p>
        </div>
        
        <form action="/predict/csv" method="post" enctype="multipart/form-data">
            <div class="form-group">
                <label for="model_type">Select Model:</label>
                <select name="model_type" id="model_type" required>
                    <option value="">Choose a model...</option>
                    <option value="kepler">Kepler Model (99.06% accuracy)</option>
                    <option value="toi">TOI Model (88.13% accuracy)</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="file">Upload CSV File:</label>
                <input type="file" name="file" id="file" accept=".csv" required>
            </div>
            
            <button type="submit">Classify Exoplanets</button>
        </form>
        
        <p><strong>📚 <a href="/docs" style="color: #007bff;">View API Documentation (/docs)</a></strong></p>
        <p><small>Health Check: <a href="/health">/health</a> | Root: <a href="/">/</a></small></p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict/csv", response_model=BatchPredictionResponse)
async def predict_from_csv(
    file: UploadFile = File(...),
    model_type: ModelType = Form(...)
):
    """
    Upload a CSV file and get exoplanet predictions for all rows using the selected model
    
    The CSV should contain exoplanet data with the required features for the selected model.
    Extra columns will be ignored.
    """
    start_time = datetime.now()
    
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=400, 
                detail="File must be a CSV file"
            )
        
        # Read CSV file
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        logger.info(f"Received CSV with shape: {df.shape} for {model_type.value} model")
        
        # Validate data
        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="CSV file is empty"
            )
        
        # Preprocess data for the selected model
        X_processed = preprocess_data(df, model_type)
        
        # Make predictions using the selected model
        predictions = make_predictions(X_processed, model_type)
        
        # Calculate summary statistics
        candidate_count = sum(1 for p in predictions if p.prediction == 'CANDIDATE')
        false_positive_count = len(predictions) - candidate_count
        avg_confidence = np.mean([p.confidence for p in predictions])
        
        # Get model accuracy
        model_data = get_model_data(model_type)
        model_accuracy = model_data.get('performance', {}).get('test_accuracy', 0)
        accuracy_str = f"{model_accuracy*100:.2f}%"
        
        summary = {
            "total_candidates": candidate_count,
            "total_false_positives": false_positive_count,
            "average_confidence": float(avg_confidence),
            "candidate_percentage": float(candidate_count / len(predictions) * 100) if predictions else 0,
        }
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return BatchPredictionResponse(
            total_rows=len(predictions),
            predictions=predictions,
            summary=summary,
            processing_time_seconds=processing_time,
            timestamp=datetime.now().isoformat(),
            classifier_used=model_type.value,
            classifier_accuracy=accuracy_str
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/predict/sample/{model_type}")
async def predict_sample(model_type: ModelType):
    """
    Test the API with sample data for the specified model
    
    This endpoint uses predefined sample data to test the prediction pipeline
    """
    try:
        # Get model data to determine required features
        model_data = get_model_data(model_type)
        required_features = model_data['required_features']
        
        if model_type == ModelType.KEPLER:
            # Create sample Kepler data
            sample_data = pd.DataFrame({
                'koi_score': [0.95, 0.12, 0.85],
                'koi_fpflag_nt': [0, 1, 0],
                'koi_fpflag_ec': [0, 0, 1],
                'koi_fpflag_co': [0, 1, 0],
                'koi_model_snr': [25.5, 8.2, 45.1],
                'koi_fpflag_ss': [0, 0, 1],
                'koi_prad': [1.1, 2.8, 0.95],
                'koi_period': [365.25, 88.0, 687.0],
                'koi_duration': [6.2, 12.5, 4.8],
                'koi_impact': [0.3, 0.8, 0.1]
            })
        else:  # TOI model
            # Create sample TOI data using the required features
            sample_data = pd.DataFrame({
                'pl_eqt': [1500, 800, 2000],
                'pl_insol': [1.2, 0.3, 5.8],
                'pl_orbpererr2': [0.01, 0.05, 0.02],
                'pl_radeerr2': [0.1, 0.3, 0.15],
                'pl_tranmid': [2458000, 2459000, 2458500],
                'st_disterr2': [5, 10, 8],
                'st_tmag': [12.5, 9.8, 14.2],
                'pl_tranmiderr2': [0.0001, 0.0005, 0.0002],
                'st_disterr1': [5, 8, 6],
                'pl_trandeperr1': [0.5, 1.2, 0.8]
            })
        
        # Process the sample data
        X_processed = preprocess_data(sample_data, model_type)
        predictions = make_predictions(X_processed, model_type)
        
        return {
            "message": f"Sample prediction successful for {model_type.value} model",
            "classifier_type": model_type.value,
            "sample_data_shape": sample_data.shape,
            "required_features": required_features,
            "predictions": predictions
        }
        
    except Exception as e:
        logger.error(f"Sample prediction error for {model_type.value}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Sample prediction failed for {model_type.value}: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP {exc.status_code}",
            message=exc.detail,
            timestamp=datetime.now().isoformat()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            message="An unexpected error occurred",
            timestamp=datetime.now().isoformat()
        ).dict()
    )

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Exoplanet Classifier API...")
    print("📚 API Documentation: http://localhost:8001/docs")
    print("🌐 Web Interface: http://localhost:8001/predict/form")
    print("❤️  Health Check: http://localhost:8001/health")
    print("\n⚡ Press Ctrl+C to stop the server\n")
    uvicorn.run(app, host="0.0.0.0", port=8001)