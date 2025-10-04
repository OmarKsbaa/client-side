from flask import Flask, request, jsonify
import os
import json
from werkzeug.utils import secure_filename
from model_pipeline import ExoplanetPipeline, get_model_parameters, load_and_predict
import datetime

# Create folders if they don't exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('models', exist_ok=True)

app = Flask(__name__)


@app.route('/exoplanet', methods=['POST'])
def exoplanet_api():
    try:
        # Check if file part exists
        if 'csv_file' not in request.files:
            return jsonify({"status": "error", "message": "CSV file missing"}), 400

        csv_file = request.files['csv_file']
        if csv_file.filename == '':
            return jsonify({"status": "error", "message": "No file selected"}), 400

        # Save uploaded CSV
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = secure_filename(csv_file.filename)
        csv_path = os.path.join('uploads', f"{timestamp}_{filename}")
        csv_file.save(csv_path)

        # Get JSON options
        options = request.form.get('options')
        if not options:
            return jsonify({"status": "error", "message": "Options JSON missing"}), 400
        options = json.loads(options)

        mode = options.get('mode', '').lower()
        mission_type = options.get('mission_type', 'TESS')
        training_mode = options.get('training_mode', 'new')
        model_params = options.get('model_parameters', None)
        model_name = options.get("model_name", None)

        # Initialize pipeline
        pipeline = ExoplanetPipeline()

        if mode == 'train':
            # Step 1: Process CSV
            new_data = pipeline.process_data(csv_path, mission_type)

            # Step 2: Merge with existing dataset if append requested
            if training_mode == 'append':
                combined_data = pipeline.merge_with_existing_data(new_data)
            else:
                combined_data = new_data

            # Save combined dataset
            combined_data.to_csv('combined_exoplanet_data.csv', index=False)

            # Step 3: Prepare features
            X, y = pipeline.prepare_features(combined_data)

            # Step 4: Train model
            if model_params is None:
                # Use default parameters without user interaction
                model_params = {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'eval_metric': 'logloss'
                }
            results = pipeline.train_model(X, y, model_params=model_params)

            # Step 5: Save model
            # Generate model name with timestamp
            model_name = f"xgboost_model_{timestamp}"
            saved_model_paths = pipeline.save_model(model_name=model_name)

            # Return metrics and model name
            metrics = {
                "accuracy": float(results['accuracy']),
                "precision": float(results['precision']),
                "recall": float(results['recall']),
                "f1_score": float(results['f1_score']),
                "roc_auc": float(results['roc_auc'])
            }

            return jsonify({
                "status": "success",
                "metrics": metrics,
                "model_name": model_name
            })

        elif mode == 'predict':
            # Get model name from options or use default
            model_name = options.get('model_name', 'xgboost_model')

            # If model_name doesn't have extension, check for most recent model
            if not os.path.exists(os.path.join('models', f"{model_name}.pkl")):
                # Try to find the most recent model
                model_files = [f for f in os.listdir('models') if
                               f.endswith('.pkl') and not f.endswith('_scaler.pkl') and not f.endswith('_encoder.pkl')]
                if model_files:
                    # Sort by modification time and get the most recent
                    model_files.sort(key=lambda x: os.path.getmtime(os.path.join('models', x)), reverse=True)
                    model_name = model_files[0].replace('.pkl', '')
                else:
                    return jsonify({"status": "error", "message": "No trained model found"}), 400

            # Paths to model, scaler, encoder
            model_path = os.path.join('models', f"{model_name}.pkl")
            scaler_path = os.path.join('models', f"{model_name}_scaler.pkl")
            encoder_path = os.path.join('models', f"{model_name}_encoder.pkl")

            if not os.path.exists(model_path):
                return jsonify({"status": "error", "message": f"Model file {model_path} not found"}), 400

            # Make predictions
            try:
                predictions_df = load_and_predict(model_path, scaler_path, encoder_path, csv_path)

                # Return only object_id and predicted_disposition if they exist
                if 'object_id' in predictions_df.columns:
                    prediction_list = predictions_df[['object_id', 'predicted_disposition']].to_dict(orient='records')
                else:
                    # If object_id doesn't exist, return predictions with index
                    predictions_df['record_id'] = predictions_df.index
                    prediction_list = predictions_df[['record_id', 'predicted_disposition']].to_dict(orient='records')

                return jsonify({
                    "status": "success",
                    "predictions": prediction_list,
                    "model_used": model_name
                })
            except Exception as e:
                return jsonify({"status": "error", "message": f"Prediction failed: {str(e)}"}), 500

        else:
            return jsonify({"status": "error", "message": "Invalid mode, must be 'train' or 'predict'"}), 400

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    finally:
        # Clean up uploaded file if needed
        if 'csv_path' in locals() and os.path.exists(csv_path):
            try:
                os.remove(csv_path)
            except:
                pass


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8003)