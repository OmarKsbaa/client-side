"""
Create a corrected version of the enhanced TOI model
Fix the ensemble wrapper pickle issue
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
from datetime import datetime

print("🔧 Creating Fixed Enhanced TOI Model")
print("=" * 40)

# Load baseline for reference
baseline_model = joblib.load('toi_model_complete.pkl')
baseline_features = baseline_model['required_features']
baseline_params = baseline_model['model_metadata']['hyperparameters']

print(f"Using baseline features: {len(baseline_features)}")
print(f"Using baseline params: {baseline_params}")

# Load and prepare data (exact same as successful script)
toi_df = pd.read_csv('TOI_2025.09.26_02.41.12.csv', comment='#')

def map_target(disposition):
    if pd.isna(disposition):
        return None
    if disposition in ['CP', 'KP', 'PC']:
        return 'CANDIDATE'
    elif disposition in ['FP', 'FA']:
        return 'FALSE_POSITIVE'
    elif disposition in ['APC']:
        return 'SKIPPED'
    else:
        return None

toi_df['binary_target'] = toi_df['tfopwg_disp'].apply(map_target)
toi_df = toi_df[toi_df['binary_target'].isin(['CANDIDATE', 'FALSE_POSITIVE'])]

X = toi_df[baseline_features].copy()
y = (toi_df['binary_target'] == 'CANDIDATE').astype(int)

# Preprocessing
imputer = SimpleImputer(strategy='median')
X_filled = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_filled), columns=X_filled.columns)

# Train-test split (same as baseline)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2001, random_state=42, stratify=y
)

print(f"Data prepared: Train={len(X_train)}, Test={len(X_test)}")

# Create the best single model from our enhancement
# This was the best single model: max_depth=5, n_estimators=150
best_params = {
    **baseline_params,
    'max_depth': 5,
    'n_estimators': 150
}

print(f"Training enhanced model with params: {best_params}")

enhanced_model = xgb.XGBClassifier(
    **best_params,
    random_state=42,
    eval_metric='logloss'
)

enhanced_model.fit(X_train, y_train)

# Evaluate
y_pred = enhanced_model.predict(X_test)
y_proba = enhanced_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Enhanced Model Performance:")
print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  ROC-AUC:  {roc_auc:.4f}")

# Save the corrected enhanced model
performance_metrics = {
    'training_accuracy': float(accuracy_score(y_train, enhanced_model.predict(X_train))),
    'test_accuracy': float(accuracy),
    'roc_auc_score': float(roc_auc),
    'baseline_accuracy': float(baseline_model['performance']['test_accuracy']),
    'improvement': float((accuracy - baseline_model['performance']['test_accuracy']) * 100)
}

model_package = {
    'model': enhanced_model,
    'scaler': scaler,
    'imputer': imputer,
    'required_features': baseline_features,
    'target_classes': ['FALSE_POSITIVE', 'CANDIDATE'],
    'performance': performance_metrics,
    'preprocessing_info': {
        'imputation_strategy': 'median',
        'scaling_method': 'StandardScaler',
        'approach': 'Conservative parameter enhancement'
    },
    'model_metadata': {
        'model_type': 'TOI_Enhanced_Single_XGBoost',
        'dataset': 'TESS Objects of Interest (TOI)',
        'target_mapping': {'CP_KP_PC': 'CANDIDATE', 'FP_FA': 'FALSE POSITIVE', 'APC': 'SKIPPED'},
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': len(baseline_features),
        'base_model': 'toi_model_complete.pkl',
        'enhancement': 'Increased max_depth and n_estimators',
        'hyperparameters': best_params,
        'created_date': datetime.now().isoformat()
    }
}

# Save the working model
joblib.dump(model_package, 'toi_model_enhanced_working.pkl')
print(f"✅ Saved working enhanced model to 'toi_model_enhanced_working.pkl'")

# Test that it loads correctly
print(f"\n🧪 Testing model loading...")
try:
    test_load = joblib.load('toi_model_enhanced_working.pkl')
    test_model = test_load['model']
    
    # Test prediction
    test_pred = test_model.predict(X_test[:5])
    test_proba = test_model.predict_proba(X_test[:5])
    
    print(f"✅ Model loads and predicts correctly")
    print(f"Sample predictions: {test_pred}")
    print(f"Sample probabilities shape: {test_proba.shape}")
    
    # Show improvement
    baseline_acc = baseline_model['performance']['test_accuracy']
    improvement = (accuracy - baseline_acc) * 100
    print(f"\nFinal Results:")
    print(f"  Baseline:    {baseline_acc*100:.2f}%")
    print(f"  Enhanced:    {accuracy*100:.2f}%")
    print(f"  Improvement: {improvement:+.2f} percentage points")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")

print(f"\n🏁 Enhanced model creation complete!")