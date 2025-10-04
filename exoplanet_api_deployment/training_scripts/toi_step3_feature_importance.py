"""
TOI Feature Importance Analysis - Step 3: Identify Most Important Features
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

print("🔍 TOI Feature Importance Analysis")
print("=" * 50)

# Load clean TOI data
print("Loading clean TOI dataset...")
toi_df = pd.read_csv('toi_clean_data.csv')
print(f"✅ Loaded dataset: {toi_df.shape}")

# Prepare data for analysis
print(f"\n📊 Data Preparation")
print("Target distribution:")
target_counts = toi_df['binary_target'].value_counts()
for target, count in target_counts.items():
    print(f"   {target}: {count} ({count/len(toi_df)*100:.1f}%)")

# Separate features and target
feature_cols = [col for col in toi_df.columns if col not in ['toi', 'tid', 'tfopwg_disp', 'binary_target']]
X = toi_df[feature_cols].copy()
y = toi_df['binary_target'].copy()

print(f"\n🎯 Features for analysis: {len(feature_cols)}")
print(f"Feature columns: {feature_cols[:10]}..." if len(feature_cols) > 10 else f"Feature columns: {feature_cols}")

# Handle missing values
print(f"\n🔧 Preprocessing Steps")
print("1. Handling missing values...")
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

missing_before = X.isna().sum().sum()
missing_after = X_imputed.isna().sum().sum()
print(f"   Missing values: {missing_before} → {missing_after}")

# Encode target
print("2. Encoding target variable...")
y_encoded = (y == 'CANDIDATE').astype(int)  # 1 for CANDIDATE, 0 for FALSE POSITIVE
print(f"   CANDIDATE (1): {y_encoded.sum()}")
print(f"   FALSE POSITIVE (0): {len(y_encoded) - y_encoded.sum()}")

# Scale features
print("3. Scaling features...")
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)

# Train-test split with stratification (due to class imbalance)
print(f"\n🔄 Train-Test Split (stratified)")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")
print(f"   Train target distribution: {y_train.value_counts().to_dict()}")
print(f"   Test target distribution: {y_test.value_counts().to_dict()}")

# Train XGBoost for feature importance
print(f"\n🚀 Training XGBoost for Feature Importance")
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42,
    eval_metric='logloss'
)

print("Training model...")
xgb_model.fit(X_train, y_train)

# Get feature importance
feature_importance = xgb_model.feature_importances_
feature_names = X_train.columns

# Create feature importance dataframe
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

# Calculate percentages
importance_df['importance_pct'] = importance_df['importance'] / importance_df['importance'].sum() * 100

print(f"✅ Model trained successfully!")

# Evaluate model performance
print(f"\n📊 Model Performance Evaluation")
train_pred = xgb_model.predict(X_train)
test_pred = xgb_model.predict(X_test)
test_proba = xgb_model.predict_proba(X_test)[:, 1]

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
roc_auc = roc_auc_score(y_test, test_proba)

print(f"   Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"   Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"   ROC-AUC Score: {roc_auc:.4f}")
print(f"   Overfitting Check: {train_acc - test_acc:.4f}")

if train_acc - test_acc > 0.05:
    print("   ⚠️ Potential overfitting detected")
else:
    print("   ✅ Good generalization")

# Show top features
print(f"\n🏆 Top 20 Most Important Features")
print("-" * 60)
for i, (_, row) in enumerate(importance_df.head(20).iterrows()):
    print(f"{i+1:2d}. {row['feature']:<25} {row['importance']:.4f} ({row['importance_pct']:.1f}%)")

# Select top features (similar to Kepler approach)
top_n = 10  # Select top 10 features like Kepler
top_features = importance_df.head(top_n)['feature'].tolist()

print(f"\n🎯 Selected Top {top_n} Features for TOI Model:")
print("-" * 50)
for i, feature in enumerate(top_features):
    importance_pct = importance_df[importance_df['feature'] == feature]['importance_pct'].iloc[0]
    print(f"{i+1:2d}. {feature} ({importance_pct:.1f}%)")

# Save feature importance results
importance_df.to_csv('toi_feature_importance.csv', index=False)
print(f"\n💾 Saved feature importance to 'toi_feature_importance.csv'")

# Create visualization
plt.figure(figsize=(12, 8))
top_20_features = importance_df.head(20)
plt.barh(range(len(top_20_features)), top_20_features['importance'])
plt.yticks(range(len(top_20_features)), top_20_features['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 20 Most Important Features for TOI Classification')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('toi_feature_importance.png', dpi=300, bbox_inches='tight')
print(f"💾 Saved feature importance plot to 'toi_feature_importance.png'")

# Test model with top features only
print(f"\n🧪 Testing Model with Top {top_n} Features Only")
X_train_top = X_train[top_features]
X_test_top = X_test[top_features]

xgb_top = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)

xgb_top.fit(X_train_top, y_train)

train_pred_top = xgb_top.predict(X_train_top)
test_pred_top = xgb_top.predict(X_test_top)
test_proba_top = xgb_top.predict_proba(X_test_top)[:, 1]

train_acc_top = accuracy_score(y_train, train_pred_top)
test_acc_top = accuracy_score(y_test, test_pred_top)
roc_auc_top = roc_auc_score(y_test, test_proba_top)

print(f"Performance with top {top_n} features:")
print(f"   Training Accuracy: {train_acc_top:.4f} ({train_acc_top*100:.2f}%)")
print(f"   Test Accuracy: {test_acc_top:.4f} ({test_acc_top*100:.2f}%)")
print(f"   ROC-AUC Score: {roc_auc_top:.4f}")

# Compare full vs top features
print(f"\n⚖️ Full Features vs Top {top_n} Features Comparison:")
print(f"   Full Model Test Accuracy: {test_acc:.4f}")
print(f"   Top {top_n} Model Test Accuracy: {test_acc_top:.4f}")
print(f"   Difference: {test_acc_top - test_acc:.4f}")

if abs(test_acc_top - test_acc) < 0.01:
    print(f"   ✅ Top {top_n} features maintain similar performance!")
else:
    print(f"   ⚠️ Performance difference with top {top_n} features")

# Summary
print(f"\n📋 FEATURE ANALYSIS SUMMARY")
print("=" * 40)
print(f"✅ TOI dataset: {len(toi_df)} samples, {len(feature_cols)} original features")
print(f"✅ Class distribution: {target_counts['CANDIDATE']} candidates, {target_counts['FALSE POSITIVE']} false positives")
print(f"✅ Model performance: {test_acc:.3f} accuracy, {roc_auc:.3f} ROC-AUC")
print(f"✅ Top {top_n} features selected for TOI classification")
print(f"✅ Ready for final model training!")

# Save top features list for next step
top_features_dict = {
    'top_features': top_features,
    'performance': {
        'test_accuracy': test_acc_top,
        'roc_auc': roc_auc_top,
        'n_features': len(top_features)
    }
}

import json
with open('toi_top_features.json', 'w') as f:
    json.dump(top_features_dict, f, indent=2)

print(f"💾 Saved top features to 'toi_top_features.json'")
print(f"\n🎯 Selected TOI Features: {top_features}")