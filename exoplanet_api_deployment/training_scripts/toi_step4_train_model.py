"""
TOI Model Training - Step 4: Train Final TOI Classification Model
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime

print("🚀 TOI Final Model Training")
print("=" * 50)

# Load clean TOI data and top features
print("Loading data and configuration...")
toi_df = pd.read_csv('toi_clean_data.csv')

with open('toi_top_features.json', 'r') as f:
    feature_config = json.load(f)
    top_features = feature_config['top_features']

print(f"✅ Loaded dataset: {toi_df.shape}")
print(f"✅ Top features: {len(top_features)}")
print(f"🎯 Selected features: {top_features}")

# Prepare data
print(f"\n📊 Data Preparation")
X = toi_df[top_features].copy()
y = toi_df['binary_target'].copy()

# Check target distribution
target_counts = y.value_counts()
print(f"Target distribution:")
for target, count in target_counts.items():
    print(f"   {target}: {count} ({count/len(y)*100:.1f}%)")

# Handle missing values
print(f"\n🔧 Preprocessing Pipeline")
print("1. Handling missing values...")
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Encode target (CANDIDATE = 0, FALSE POSITIVE = 1 to match Kepler format)
print("2. Encoding target variable...")
y_encoded = (y == 'CANDIDATE').astype(int)
print(f"   CANDIDATE (1): {y_encoded.sum()}")
print(f"   FALSE POSITIVE (0): {len(y_encoded) - y_encoded.sum()}")

# Scale features
print("3. Scaling features...")
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)

# Train-test split with stratification (due to class imbalance)
print(f"\n🔄 Train-Test Split")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

# Model hyperparameter optimization
print(f"\n🔍 Hyperparameter Optimization")
print("Testing different XGBoost configurations...")

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 4, 6],
    'learning_rate': [0.05, 0.1, 0.15],
    'subsample': [0.8, 1.0]
}

# Use a smaller grid for faster computation
xgb_base = xgb.XGBClassifier(random_state=42, eval_metric='logloss')

# Grid search with cross-validation
grid_search = GridSearchCV(
    xgb_base, 
    param_grid, 
    cv=3,  # 3-fold CV for speed
    scoring='roc_auc',  # Use ROC-AUC due to class imbalance
    n_jobs=-1,
    verbose=1
)

print("Running grid search...")
grid_search.fit(X_train, y_train)

print(f"✅ Best parameters: {grid_search.best_params_}")
print(f"✅ Best CV score: {grid_search.best_score_:.4f}")

# Train final model with best parameters
print(f"\n🎯 Training Final Model")
best_model = grid_search.best_estimator_
print("Training with best parameters...")

# Evaluate model
train_pred = best_model.predict(X_train)
test_pred = best_model.predict(X_test)
test_proba = best_model.predict_proba(X_test)[:, 1]

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
roc_auc = roc_auc_score(y_test, test_proba)

print(f"\n📊 Final Model Performance")
print(f"   Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"   Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"   ROC-AUC Score: {roc_auc:.4f}")
print(f"   Overfitting: {train_acc - test_acc:.4f}")

if train_acc - test_acc > 0.05:
    print("   ⚠️ Some overfitting detected")
else:
    print("   ✅ Good generalization")

# Detailed classification report
print(f"\n📋 Classification Report:")
print(classification_report(y_test, test_pred, 
                          target_names=['FALSE POSITIVE', 'CANDIDATE']))

# Cross-validation scores for robustness
print(f"\n🔄 Cross-Validation Assessment")
cv_scores = cross_val_score(best_model, X_scaled, y_encoded, cv=5, scoring='accuracy')
print(f"   CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"   CV Scores: {[f'{score:.3f}' for score in cv_scores]}")

# Feature importance from final model
print(f"\n🏆 Final Model Feature Importance")
feature_importance = best_model.feature_importances_
importance_df = pd.DataFrame({
    'feature': top_features,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("Top features in final model:")
for i, (_, row) in enumerate(importance_df.iterrows()):
    print(f"{i+1:2d}. {row['feature']:<20} {row['importance']:.4f}")

# Create comprehensive model package for API
print(f"\n💾 Creating Model Package for API")

# Model metadata
model_metadata = {
    'model_type': 'XGBoost_TOI_Classifier',
    'dataset': 'TESS Objects of Interest (TOI)',
    'target_mapping': {
        'CP_KP_PC': 'CANDIDATE',
        'FP_FA': 'FALSE POSITIVE',
        'APC': 'SKIPPED'
    },
    'training_samples': len(y_train),
    'test_samples': len(y_test),
    'n_features': len(top_features),
    'class_distribution': {
        'CANDIDATE': int(y_encoded.sum()),
        'FALSE_POSITIVE': int(len(y_encoded) - y_encoded.sum())
    },
    'hyperparameters': grid_search.best_params_,
    'created_date': datetime.now().isoformat(),
    'training_method': 'GridSearchCV with 3-fold CV'
}

# Performance metrics
performance_metrics = {
    'training_accuracy': float(train_acc),
    'test_accuracy': float(test_acc),
    'roc_auc_score': float(roc_auc),
    'overfitting': float(train_acc - test_acc),
    'cv_accuracy_mean': float(cv_scores.mean()),
    'cv_accuracy_std': float(cv_scores.std())
}

# Create complete model package
model_package = {
    'model': best_model,
    'scaler': scaler,
    'imputer': imputer,
    'required_features': top_features,
    'target_classes': ['FALSE POSITIVE', 'CANDIDATE'],  # 0, 1
    'performance': performance_metrics,
    'preprocessing_info': {
        'imputation_strategy': 'median',
        'scaling_method': 'StandardScaler',
        'feature_selection': 'XGBoost feature importance top 10'
    },
    'model_metadata': model_metadata
}

# Save model package
model_filename = 'toi_model_complete.pkl'
joblib.dump(model_package, model_filename)
print(f"✅ Saved complete model to '{model_filename}'")

# Also save in joblib format for compatibility
joblib_filename = 'toi_model_complete.joblib'
joblib.dump(model_package, joblib_filename)
print(f"✅ Saved joblib model to '{joblib_filename}'")

# Save preprocessing components separately (for API documentation)
preprocessing_components = {
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_scale': scaler.scale_.tolist(),
    'imputer_statistics': imputer.statistics_.tolist(),
    'required_features': top_features,
    'feature_count': len(top_features)
}

with open('toi_preprocessing_components.json', 'w') as f:
    json.dump(preprocessing_components, f, indent=2)
print(f"✅ Saved preprocessing components to 'toi_preprocessing_components.json'")

# Save model summary
model_summary = {
    'model_name': 'TOI Exoplanet Classifier',
    'model_type': 'XGBoost Binary Classifier',
    'target': 'TESS Object Disposition (CANDIDATE vs FALSE POSITIVE)',
    'performance': performance_metrics,
    'features': {
        'count': len(top_features),
        'list': top_features
    },
    'dataset_info': {
        'total_samples': len(toi_df),
        'training_samples': len(y_train),
        'test_samples': len(y_test),
        'class_balance': f"{y_encoded.sum()/len(y_encoded)*100:.1f}% candidates"
    },
    'model_files': {
        'main_model': model_filename,
        'joblib_model': joblib_filename,
        'preprocessing': 'toi_preprocessing_components.json'
    }
}

with open('toi_model_summary.json', 'w') as f:
    json.dump(model_summary, f, indent=2)
print(f"✅ Saved model summary to 'toi_model_summary.json'")

# Test model loading (verification)
print(f"\n🧪 Model Loading Verification")
try:
    loaded_model_package = joblib.load(model_filename)
    loaded_model = loaded_model_package['model']
    loaded_features = loaded_model_package['required_features']
    
    # Test prediction
    test_sample = X_test.iloc[:1]
    prediction = loaded_model.predict(test_sample)[0]
    probability = loaded_model.predict_proba(test_sample)[0]
    
    print(f"✅ Model loads successfully")
    print(f"✅ Required features: {len(loaded_features)}")
    print(f"✅ Test prediction: {'CANDIDATE' if prediction == 1 else 'FALSE POSITIVE'}")
    print(f"✅ Test probabilities: [FP: {probability[0]:.3f}, CANDIDATE: {probability[1]:.3f}]")
    
except Exception as e:
    print(f"❌ Model loading error: {str(e)}")

# Final summary
print(f"\n📋 TOI MODEL TRAINING SUMMARY")
print("=" * 50)
print(f"✅ Dataset: {len(toi_df)} TOI objects processed")
print(f"✅ Classes: {target_counts['CANDIDATE']} candidates, {target_counts['FALSE POSITIVE']} false positives")
print(f"✅ Features: {len(top_features)} most important features selected")
print(f"✅ Performance: {test_acc:.3f} accuracy, {roc_auc:.3f} ROC-AUC")
print(f"✅ Cross-validation: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"✅ Model saved and verified for API deployment")
print(f"\n🎯 TOP TOI FEATURES:")
for i, feature in enumerate(top_features):
    print(f"   {i+1:2d}. {feature}")
print(f"\n🚀 Ready for API integration!")

# Create visualization of model performance
plt.figure(figsize=(12, 5))

# Subplot 1: Feature importance
plt.subplot(1, 2, 1)
plt.barh(range(len(importance_df)), importance_df['importance'])
plt.yticks(range(len(importance_df)), importance_df['feature'])
plt.xlabel('Feature Importance')
plt.title('TOI Model Feature Importance')
plt.gca().invert_yaxis()

# Subplot 2: Confusion matrix
plt.subplot(1, 2, 2)
cm = confusion_matrix(y_test, test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['FALSE POSITIVE', 'CANDIDATE'],
            yticklabels=['FALSE POSITIVE', 'CANDIDATE'])
plt.title('TOI Model Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('toi_model_performance.png', dpi=300, bbox_inches='tight')
print(f"💾 Saved performance plots to 'toi_model_performance.png'")

plt.show()