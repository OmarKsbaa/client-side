# Step 4: Train Optimized Binary Classification Model
# Train XGBoost on top 10 features for API deployment

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import json
import pickle
import time

def load_processed_data():
    """Load the preprocessed binary classification data"""
    print("🚀 Step 4: Training Optimized Binary Classification Model")
    print("=" * 60)
    
    print("📂 Loading preprocessed data...")
    
    # Load training and test data
    train_data = pd.read_csv('binary_train_data.csv')
    test_data = pd.read_csv('binary_test_data.csv')
    
    # Separate features and targets
    X_train = train_data.drop('target', axis=1)
    y_train = train_data['target']
    X_test = test_data.drop('target', axis=1)
    y_test = test_data['target']
    
    # Load preprocessing config
    with open('binary_preprocessing_config.json', 'r') as f:
        config = json.load(f)
    
    print(f"✅ Data loaded successfully!")
    print(f"📊 Training set: {X_train.shape}")
    print(f"📊 Test set: {X_test.shape}")
    print(f"🎯 Target classes: {config['target_classes']}")
    print(f"🏆 Features used: {len(config['selected_features'])}")
    
    # Display features
    print(f"\n📋 Model Features (10 most important):")
    for i, feature in enumerate(config['selected_features'], 1):
        print(f"  {i:2d}. {feature}")
    
    return X_train, X_test, y_train, y_test, config

def train_baseline_xgboost(X_train, y_train, X_test, y_test):
    """Train baseline XGBoost model"""
    print(f"\n🤖 Training Baseline XGBoost Model...")
    print("=" * 40)
    
    # Initialize XGBoost classifier
    xgb_baseline = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    # Train the model
    print("🔄 Training baseline model...")
    start_time = time.time()
    xgb_baseline.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Make predictions
    y_pred_train = xgb_baseline.predict(X_train)
    y_pred_test = xgb_baseline.predict(X_test)
    y_pred_proba_test = xgb_baseline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    roc_auc = roc_auc_score(y_test, y_pred_proba_test)
    
    print(f"⏱️ Training time: {training_time:.2f} seconds")
    print(f"📈 Training accuracy: {train_acc:.4f}")
    print(f"📈 Test accuracy: {test_acc:.4f}")
    print(f"📈 ROC-AUC: {roc_auc:.4f}")
    
    # Cross-validation
    print(f"\n🔄 Performing 5-fold cross-validation...")
    cv_scores = cross_val_score(xgb_baseline, X_train, y_train, cv=5, scoring='accuracy')
    print(f"📊 CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return xgb_baseline, {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'roc_auc': roc_auc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'training_time': training_time
    }

def optimize_xgboost(X_train, y_train):
    """Optimize XGBoost hyperparameters"""
    print(f"\n🔧 Optimizing XGBoost Hyperparameters...")
    print("=" * 40)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    # Initialize XGBoost
    xgb_model = xgb.XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    # Grid search with cross-validation
    print("🔍 Performing grid search (this may take a few minutes)...")
    grid_search = GridSearchCV(
        xgb_model, 
        param_grid, 
        cv=3,  # 3-fold CV for speed
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    optimization_time = time.time() - start_time
    
    print(f"⏱️ Optimization time: {optimization_time:.1f} seconds")
    print(f"🏆 Best parameters: {grid_search.best_params_}")
    print(f"📈 Best CV accuracy: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

def evaluate_final_model(model, X_train, y_train, X_test, y_test, config):
    """Comprehensive evaluation of the final model"""
    print(f"\n📊 Final Model Evaluation")
    print("=" * 30)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"🎯 Final Model Performance:")
    print(f"  • Training Accuracy: {train_acc:.4f}")
    print(f"  • Test Accuracy: {test_acc:.4f}")
    print(f"  • ROC-AUC Score: {roc_auc:.4f}")
    
    # Overfitting check
    overfitting = train_acc - test_acc
    print(f"  • Overfitting: {overfitting:.4f} {'✅ Good' if overfitting < 0.02 else '⚠️ Potential overfitting'}")
    
    # Classification report
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(y_test, y_pred_test, target_names=config['target_classes']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    print(f"\n🔢 Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 CAND  FP")
    print(f"Actual CANDIDATE  {cm[0,0]:4d} {cm[0,1]:4d}")
    print(f"       FALSE POS  {cm[1,0]:4d} {cm[1,1]:4d}")
    
    # Feature importance
    print(f"\n🌟 Feature Importance:")
    feature_importance = model.feature_importances_
    feature_names = config['selected_features']
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    for i, (_, row) in enumerate(importance_df.iterrows(), 1):
        print(f"  {i:2d}. {row['Feature']:<15}: {row['Importance']:.4f}")
    
    return {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'roc_auc': roc_auc,
        'overfitting': overfitting,
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(y_test, y_pred_test, target_names=config['target_classes'], output_dict=True),
        'feature_importance': importance_df.to_dict('records')
    }

def save_final_model(model, config, results, best_params=None):
    """Save the final model and all artifacts for API deployment"""
    print(f"\n💾 Saving Final Model for API Deployment...")
    print("=" * 45)
    
    # Load preprocessing objects
    with open('binary_preprocessing_objects.pkl', 'rb') as f:
        preprocessing_objects = pickle.load(f)
    
    # Create complete model package for API
    model_package = {
        'model': model,
        'preprocessing_objects': preprocessing_objects,
        'config': config,
        'results': results,
        'best_params': best_params,
        'model_info': {
            'model_type': 'XGBoost Binary Classifier',
            'target_variable': 'koi_pdisposition',
            'classes': config['target_classes'],
            'features': config['selected_features'],
            'feature_count': len(config['selected_features']),
            'test_accuracy': results['test_accuracy'],
            'roc_auc': results['roc_auc']
        }
    }
    
    # Save complete package
    with open('kepler_binary_model_complete.pkl', 'wb') as f:
        pickle.dump(model_package, f)
    
    # Save just the model (for smaller file size if needed)
    with open('kepler_binary_model_only.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save model summary
    model_summary = {
        'model_type': 'XGBoost Binary Classifier',
        'dataset': 'NASA Kepler Exoplanet Archive',
        'target': 'koi_pdisposition (CANDIDATE vs FALSE POSITIVE)',
        'features_used': config['selected_features'],
        'feature_count': len(config['selected_features']),
        'training_samples': 7651,
        'test_samples': 1913,
        'performance': {
            'test_accuracy': results['test_accuracy'],
            'roc_auc': results['roc_auc'],
            'overfitting_check': results['overfitting']
        },
        'hyperparameters': best_params if best_params else 'baseline',
        'preprocessing_steps': config['preprocessing_steps'],
        'api_usage': {
            'input_format': 'CSV file with Kepler telescope data',
            'required_columns': config['selected_features'],
            'output_format': 'Binary classification (0=CANDIDATE, 1=FALSE POSITIVE)',
            'confidence_scores': 'Available via predict_proba'
        }
    }
    
    with open('kepler_model_summary.json', 'w') as f:
        json.dump(model_summary, f, indent=2)
    
    print("✅ Model saved successfully:")
    print("  • kepler_binary_model_complete.pkl (Full package for API)")
    print("  • kepler_binary_model_only.pkl (Model only)")
    print("  • kepler_model_summary.json (Model documentation)")
    
    return model_package

def main():
    """Main training function"""
    try:
        # Load processed data
        X_train, X_test, y_train, y_test, config = load_processed_data()
        
        # Train baseline model
        baseline_model, baseline_results = train_baseline_xgboost(X_train, y_train, X_test, y_test)
        
        # Optimize hyperparameters
        optimized_model, best_params, best_cv_score = optimize_xgboost(X_train, y_train)
        
        # Evaluate final model
        final_results = evaluate_final_model(optimized_model, X_train, y_train, X_test, y_test, config)
        
        # Save the final model
        model_package = save_final_model(optimized_model, config, final_results, best_params)
        
        # Final summary
        print(f"\n🎉 MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"🎯 Final Model Performance:")
        print(f"  • Test Accuracy: {final_results['test_accuracy']:.4f} ({final_results['test_accuracy']*100:.2f}%)")
        print(f"  • ROC-AUC Score: {final_results['roc_auc']:.4f}")
        print(f"  • Feature Count: {len(config['selected_features'])} (most important)")
        print(f"  • Model Type: Optimized XGBoost Binary Classifier")
        
        print(f"\n📋 Ready for API Deployment:")
        print(f"  • Input: CSV with {len(config['selected_features'])} required columns")
        print(f"  • Output: Binary classification + confidence scores")
        print(f"  • Features automatically extracted from full Kepler CSV")
        
        print(f"\n✅ Next Step: API Integration & Testing")
        
        return model_package
        
    except Exception as e:
        print(f"❌ Error in model training: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    model_package = main()