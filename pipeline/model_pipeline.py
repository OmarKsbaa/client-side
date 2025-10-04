import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)
import xgboost as xgb
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ExoplanetPipeline:
    """
    Complete pipeline for exoplanet classification from data upload to model training.
    """

    def __init__(self):
        self.tess_column_mapping = {
            'tid': 'target_id',
            'toi': 'object_id',
            'ra': 'ra_deg',
            'dec': 'dec_deg',
            'st_pmra': 'pmra',
            'st_pmdec': 'pmdec',
            'pl_tranmid': 'transit_epoch_bjd',
            'pl_orbper': 'orbital_period_days',
            'pl_trandurh': 'transit_duration_hours',
            'pl_trandep': 'transit_depth_ppm',
            'pl_rade': 'planet_radius_earth',
            'pl_insol': 'insolation_flux_earth',
            'pl_eqt': 'equilibrium_temp_k',
            'st_tmag': 'stellar_magnitude',
            'st_dist': 'stellar_distance_pc',
            'st_teff': 'stellar_temp_k',
            'st_logg': 'stellar_logg',
            'st_rad': 'stellar_radius_sun',
            'tfopwg_disp': 'disposition_original',
            'toi_created': 'date_created',
            'rowupdate': 'date_updated'
        }

        self.kepler_column_mapping = {
            'kepid': 'target_id',
            'kepoi_name': 'object_id',
            'ra': 'ra_deg',
            'dec': 'dec_deg',
            'koi_time0': 'transit_epoch_bjd',
            'koi_period': 'orbital_period_days',
            'koi_duration': 'transit_duration_hours',
            'koi_depth': 'transit_depth_ppm',
            'koi_prad': 'planet_radius_earth',
            'koi_insol': 'insolation_flux_earth',
            'koi_teq': 'equilibrium_temp_k',
            'koi_kepmag': 'stellar_magnitude',
            'koi_steff': 'stellar_temp_k',
            'koi_slogg': 'stellar_logg',
            'koi_srad': 'stellar_radius_sun',
            'koi_pdisposition': 'disposition_original',
            'koi_vet_date': 'date_updated'
        }

        self.feature_columns = [
            'orbital_period_days', 'transit_duration_hours', 'transit_depth_ppm',
            'planet_radius_earth', 'insolation_flux_earth', 'equilibrium_temp_k',
            'stellar_magnitude', 'stellar_distance_pc', 'stellar_temp_k',
            'stellar_logg', 'stellar_radius_sun', 'pmra', 'pmdec'
        ]

        self.scaler = None
        self.label_encoder = None
        self.model = None

    def map_tess_disposition(self, disp):
        """Map TESS dispositions to binary classification."""
        if pd.isna(disp):
            return np.nan

        disp = str(disp).strip().upper()

        if disp in ['PC', 'CP', 'KP']:
            return 'Planet Candidate'
        elif disp == 'APC':
            return np.nan
        elif disp in ['FP', 'FA']:
            return 'False Positive'
        else:
            return np.nan

    def map_kepler_disposition(self, disp):
        """Map Kepler dispositions to binary classification."""
        if pd.isna(disp):
            return np.nan

        disp = str(disp).strip().upper()

        if disp in ['CANDIDATE', 'CONFIRMED']:
            return 'Planet Candidate'
        elif disp == 'FALSE POSITIVE':
            return 'False Positive'
        else:
            return np.nan

    def process_data(self, csv_path, mission_type):
        """
        Process uploaded CSV file based on mission type.

        Parameters:
        -----------
        csv_path : str
            Path to the CSV file
        mission_type : str
            Either 'TESS' or 'Kepler'

        Returns:
        --------
        pd.DataFrame : Processed dataframe
        """
        print(f"\n{'='*60}")
        print(f"Processing {mission_type} data...")
        print(f"{'='*60}")

        # Load data
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from {csv_path}")

        # Select appropriate mapping
        if mission_type.upper() == 'TESS':
            column_mapping = self.tess_column_mapping
            disposition_mapper = self.map_tess_disposition
        elif mission_type.upper() == 'KEPLER':
            column_mapping = self.kepler_column_mapping
            disposition_mapper = self.map_kepler_disposition
        else:
            raise ValueError("mission_type must be either 'TESS' or 'Kepler'")

        # Keep only columns that exist in the uploaded file
        cols_to_keep = [col for col in column_mapping.keys() if col in df.columns]
        df_processed = df[cols_to_keep].copy()

        # Rename columns
        df_processed.rename(columns=column_mapping, inplace=True)

        # Add mission identifier
        df_processed['mission'] = mission_type

        # Map disposition
        df_processed['disposition'] = df_processed['disposition_original'].apply(disposition_mapper)

        # Add missing columns with NaN
        all_expected_columns = [
            'target_id', 'object_id', 'ra_deg', 'dec_deg', 'pmra', 'pmdec',
            'transit_epoch_bjd', 'orbital_period_days', 'transit_duration_hours',
            'transit_depth_ppm', 'planet_radius_earth', 'insolation_flux_earth',
            'equilibrium_temp_k', 'stellar_magnitude', 'stellar_distance_pc',
            'stellar_temp_k', 'stellar_logg', 'stellar_radius_sun',
            'disposition_original', 'date_created', 'date_updated', 'mission', 'disposition'
        ]

        for col in all_expected_columns:
            if col not in df_processed.columns:
                df_processed[col] = np.nan

        # Remove rows with NaN disposition
        df_clean = df_processed.dropna(subset=['disposition'])

        print(f"After filtering: {len(df_clean)} rows with valid dispositions")
        print(f"\nDisposition distribution:")
        print(df_clean['disposition'].value_counts())

        return df_clean

    def merge_with_existing_data(self, new_data, existing_data_path='combined_exoplanet_data.csv'):
        """
        Merge new data with existing dataset.

        Parameters:
        -----------
        new_data : pd.DataFrame
            New processed data
        existing_data_path : str
            Path to existing combined dataset

        Returns:
        --------
        pd.DataFrame : Combined dataset
        """
        if os.path.exists(existing_data_path):
            print(f"\n{'='*60}")
            print("Merging with existing data...")
            print(f"{'='*60}")

            existing_data = pd.read_csv(existing_data_path)
            print(f"Existing data: {len(existing_data)} rows")
            print(f"New data: {len(new_data)} rows")

            # Ensure both dataframes have the same columns
            all_columns = sorted(set(existing_data.columns) | set(new_data.columns))
            for col in all_columns:
                if col not in existing_data.columns:
                    existing_data[col] = np.nan
                if col not in new_data.columns:
                    new_data[col] = np.nan

            # Reorder columns
            existing_data = existing_data[all_columns]
            new_data = new_data[all_columns]

            # Combine
            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
            print(f"Combined data: {len(combined_data)} rows")

            return combined_data
        else:
            print(f"\nNo existing data found at {existing_data_path}")
            print("Using only the uploaded data.")
            return new_data

    def prepare_features(self, df):
        """
        Prepare features for model training.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe

        Returns:
        --------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Encoded target labels
        """
        print(f"\n{'='*60}")
        print("Preparing features for training...")
        print(f"{'='*60}")

        # Extract features and target
        X = df[self.feature_columns].copy()
        y = df['disposition'].copy()

        print(f"Initial shape: {X.shape}")
        print(f"Target distribution:\n{y.value_counts()}")

        # Handle missing values - fill with median
        print("\nHandling missing values...")
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                median_val = X[col].median()
                X[col].fillna(median_val, inplace=True)
                print(f"  {col}: filled {X[col].isnull().sum()} missing values with median ({median_val:.4f})")

        # Encode target variable
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        print(f"\nTarget classes: {self.label_encoder.classes_}")

        return X.values, y_encoded

    def train_model(self, X, y, model_params=None):
        """
        Train XGBoost model with custom or recommended parameters.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        model_params : dict
            Custom model parameters (optional)

        Returns:
        --------
        dict : Training results and metrics
        """
        print(f"\n{'='*60}")
        print("Training XGBoost Model...")
        print(f"{'='*60}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"\nTraining set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Use provided parameters or recommended defaults
        if model_params is None:
            model_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'base_score': 0.5,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'eval_metric': 'logloss'
            }
        else:
            # Ensure required parameters are present
            if 'random_state' not in model_params:
                model_params['random_state'] = 42
            if 'eval_metric' not in model_params:
                model_params['eval_metric'] = 'logloss'

        print("\nModel Parameters:")
        for param, value in model_params.items():
            print(f"  {param}: {value}")

        print("\nTraining model...")
        self.model = xgb.XGBClassifier(**model_params)
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        # Calculate metrics
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'X_test_scaled': X_test_scaled
        }

        print(f"\n{'='*60}")
        print("Model Performance:")
        print(f"{'='*60}")
        print(f"Accuracy:  {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        print(f"F1-Score:  {results['f1_score']:.4f}")
        print(f"ROC-AUC:   {results['roc_auc']:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

        return results

    def save_model(self, output_dir='models', model_name=None):
        """
        Save trained model and preprocessing objects.

        Parameters:
        -----------
        output_dir : str
            Directory to save models
        model_name : str
            Custom name for the model (optional)
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Generate timestamp for unique naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if model_name is None:
            model_name = f"exoplanet_model_{timestamp}"

        # Save model components
        model_path = os.path.join(output_dir, f"{model_name}.pkl")
        scaler_path = os.path.join(output_dir, f"{model_name}_scaler.pkl")
        encoder_path = os.path.join(output_dir, f"{model_name}_encoder.pkl")

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoder, encoder_path)

        print(f"\n{'='*60}")
        print("Model Saved Successfully!")
        print(f"{'='*60}")
        print(f"Model:         {model_path}")
        print(f"Scaler:        {scaler_path}")
        print(f"Label Encoder: {encoder_path}")

        return model_path, scaler_path, encoder_path

    def generate_visualizations(self, results, output_dir='visualizations'):
        """
        Generate and save visualization plots.

        Parameters:
        -----------
        results : dict
            Training results from train_model
        output_dir : str
            Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print("Generating Visualizations...")
        print(f"{'='*60}")

        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(results['y_test'], results['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {cm_path}")

        # 2. ROC Curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(results['y_test'], results['y_pred_proba'])
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {results["roc_auc"]:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        roc_path = os.path.join(output_dir, 'roc_curve.png')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {roc_path}")

        # 3. Feature Importance
        plt.figure(figsize=(10, 8))
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
        plt.title('Feature Importance', fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        fi_path = os.path.join(output_dir, 'feature_importance.png')
        plt.savefig(fi_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {fi_path}")

        print("\nAll visualizations generated successfully!")


def get_model_parameters():
    """
    Interactive function to get model hyperparameters from user.

    Returns:
    --------
    dict : Model parameters
    """
    print("\n" + "="*70)
    print(" "*15 + "MODEL HYPERPARAMETERS CONFIGURATION")
    print("="*70)

    # Recommended parameters based on previous tuning
    recommended_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }

    print("\nRECOMMENDED PARAMETERS (Based on extensive tuning):")
    print("-" * 70)
    for param, value in recommended_params.items():
        print(f"  {param:20s}: {value}")
    print("-" * 70)

    print("\nParameter Descriptions:")
    print("  n_estimators       : Number of boosting rounds (trees)")
    print("                       Higher = more complex, may overfit")
    print("                       Typical range: 50-500")
    print()
    print("  max_depth          : Maximum tree depth")
    print("                       Higher = more complex trees")
    print("                       Typical range: 3-10")
    print()
    print("  learning_rate      : Step size shrinkage (eta)")
    print("                       Lower = more conservative, needs more trees")
    print("                       Typical range: 0.01-0.3")
    print()
    print("  subsample          : Fraction of samples for each tree")
    print("                       Lower = more regularization")
    print("                       Typical range: 0.5-1.0")
    print()
    print("  colsample_bytree   : Fraction of features for each tree")
    print("                       Lower = more regularization")
    print("                       Typical range: 0.5-1.0")

    print("\n" + "="*70)
    print("\nWould you like to:")
    print("  1. Use RECOMMENDED parameters (fastest, proven performance)")
    print("  2. CUSTOMIZE parameters (advanced users)")
    param_choice = input("\nEnter choice (1 or 2): ").strip()

    if param_choice == '2':
        print("\n" + "="*70)
        print("CUSTOM PARAMETER INPUT")
        print("="*70)
        print("(Press Enter to use recommended value)")

        custom_params = {}

        # n_estimators
        n_est_input = input(f"\nn_estimators [{recommended_params['n_estimators']}]: ").strip()
        if n_est_input:
            try:
                custom_params['n_estimators'] = int(n_est_input)
            except ValueError:
                print("  Invalid input, using recommended value")
                custom_params['n_estimators'] = recommended_params['n_estimators']
        else:
            custom_params['n_estimators'] = recommended_params['n_estimators']

        # max_depth
        max_depth_input = input(f"max_depth [{recommended_params['max_depth']}]: ").strip()
        if max_depth_input:
            try:
                custom_params['max_depth'] = int(max_depth_input)
            except ValueError:
                print("  Invalid input, using recommended value")
                custom_params['max_depth'] = recommended_params['max_depth']
        else:
            custom_params['max_depth'] = recommended_params['max_depth']

        # learning_rate
        lr_input = input(f"learning_rate [{recommended_params['learning_rate']}]: ").strip()
        if lr_input:
            try:
                custom_params['learning_rate'] = float(lr_input)
            except ValueError:
                print("  Invalid input, using recommended value")
                custom_params['learning_rate'] = recommended_params['learning_rate']
        else:
            custom_params['learning_rate'] = recommended_params['learning_rate']

        # subsample
        subsample_input = input(f"subsample [{recommended_params['subsample']}]: ").strip()
        if subsample_input:
            try:
                custom_params['subsample'] = float(subsample_input)
            except ValueError:
                print("  Invalid input, using recommended value")
                custom_params['subsample'] = recommended_params['subsample']
        else:
            custom_params['subsample'] = recommended_params['subsample']

        # colsample_bytree
        colsample_input = input(f"colsample_bytree [{recommended_params['colsample_bytree']}]: ").strip()
        if colsample_input:
            try:
                custom_params['colsample_bytree'] = float(colsample_input)
            except ValueError:
                print("  Invalid input, using recommended value")
                custom_params['colsample_bytree'] = recommended_params['colsample_bytree']
        else:
            custom_params['colsample_bytree'] = recommended_params['colsample_bytree']

        print("\n" + "="*70)
        print("FINAL PARAMETERS:")
        for param, value in custom_params.items():
            print(f"  {param:20s}: {value}")
        print("="*70)

        return custom_params
    else:
        return recommended_params


def run_pipeline():
    """
    Main function to run the interactive pipeline.
    """
    print("\n" + "="*70)
    print(" "*15 + "EXOPLANET CLASSIFICATION PIPELINE")
    print("="*70)

    # Initialize pipeline
    pipeline = ExoplanetPipeline()

    # Get user inputs
    print("\n[STEP 1] Upload CSV File")
    csv_path = input("Enter the path to your CSV file: ").strip()

    if not os.path.exists(csv_path):
        print(f"ERROR: File not found at {csv_path}")
        return

    print("\n[STEP 2] Select Mission Type")
    print("  1. TESS")
    print("  2. Kepler")
    mission_choice = input("Enter choice (1 or 2): ").strip()

    mission_type = 'TESS' if mission_choice == '1' else 'Kepler'

    print("\n[STEP 3] Training Mode")
    print("  1. Build NEW model (use only uploaded data)")
    print("  2. APPEND to existing data (combine with existing dataset)")
    mode_choice = input("Enter choice (1 or 2): ").strip()

    use_existing = (mode_choice == '2')

    # Process data
    new_data = pipeline.process_data(csv_path, mission_type)

    # Merge with existing data if requested
    if use_existing:
        combined_data = pipeline.merge_with_existing_data(new_data)
    else:
        combined_data = new_data

    # Save combined dataset
    combined_data.to_csv('combined_exoplanet_data.csv', index=False)
    print(f"\nCombined dataset saved as 'combined_exoplanet_data.csv'")

    # Prepare features
    X, y = pipeline.prepare_features(combined_data)

    # Get model parameters
    print("\n[STEP 4] Configure Model Parameters")
    model_params = get_model_parameters()

    # Train model
    results = pipeline.train_model(X, y, model_params=model_params)

    # Generate visualizations
    pipeline.generate_visualizations(results)

    # Save model
    print("\n[STEP 5] Save Model")
    custom_name = input("Enter custom model name (or press Enter for default): ").strip()

    if custom_name == '':
        custom_name = None

    pipeline.save_model(model_name=custom_name)

    print("\n" + "="*70)
    print(" "*20 + "PIPELINE COMPLETE!")
    print("="*70)
    print("\nYou can now use the saved model for predictions.")
    print(f"Load it using: joblib.load('models/{custom_name}.pkl')")


# Example usage for making predictions with saved model
def load_and_predict(model_path, scaler_path, encoder_path, new_data_csv):
    """
    Load saved model and make predictions on new data.

    Parameters:
    -----------
    model_path : str
        Path to saved model
    scaler_path : str
        Path to saved scaler
    encoder_path : str
        Path to saved label encoder
    new_data_csv : str
        Path to CSV with new data to predict

    Returns:
    --------
    pd.DataFrame : Predictions
    """
    # Load model components
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(encoder_path)

    # Load new data
    df = pd.read_csv(new_data_csv)

    # Feature columns (must match training)
    feature_columns = [
        'orbital_period_days', 'transit_duration_hours', 'transit_depth_ppm',
        'planet_radius_earth', 'insolation_flux_earth', 'equilibrium_temp_k',
        'stellar_magnitude', 'stellar_distance_pc', 'stellar_temp_k',
        'stellar_logg', 'stellar_radius_sun', 'pmra', 'pmdec'
    ]

    X = df[feature_columns].copy()

    # Handle missing values
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            X[col].fillna(X[col].median(), inplace=True)

    # Scale features
    X_scaled = scaler.transform(X)

    # Make predictions
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)

    # Decode predictions
    predicted_labels = label_encoder.inverse_transform(predictions)

    # Create results dataframe
    results_df = df.copy()
    results_df['predicted_disposition'] = predicted_labels
    results_df['confidence_false_positive'] = probabilities[:, 0]
    results_df['confidence_planet_candidate'] = probabilities[:, 1]

    return results_df


if __name__ == "__main__":
    run_pipeline()