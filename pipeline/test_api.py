import requests
import json

# -----------------------------
# CONFIG
# -----------------------------
API_URL = "http://127.0.0.1:8003/exoplanet"  # Adjust port if you changed it
CSV_FILE_PATH = r"C:\Users\muhap\Downloads\combined.csv"        # Path to your CSV for testing
MODEL_NAME = "xgboost_model.pkl"  # Example model name for prediction

# -----------------------------
# TRAINING REQUEST
# -----------------------------
def test_train():
    options = {
        "mode": "predict",
        "mission_type": "TESS",
        "training_mode": "new",
        # Optional hyperparameters; can skip to use defaults
        "model_parameters": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        }
    }

    with open(CSV_FILE_PATH, "rb") as f:
        files = {"csv_file": f}
        data = {"options": json.dumps(options)}
        response = requests.post(API_URL, files=files, data=data)

    print("TRAIN RESPONSE:")
    print(response.status_code)
    print(response.json())

# -----------------------------
# PREDICTION REQUEST
# -----------------------------
def test_predict():
    options = {
        "mode": "predict",
        "model_name": MODEL_NAME
    }

    with open(CSV_FILE_PATH, "rb") as f:
        files = {"csv_file": f}
        data = {"options": json.dumps(options)}
        response = requests.post(API_URL, files=files, data=data)

    print("PREDICT RESPONSE:")
    print(response.status_code)
    print(response.json())

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    print("Testing TRAIN endpoint...")
    test_train()

    print("\nTesting PREDICT endpoint...")
    test_predict()
