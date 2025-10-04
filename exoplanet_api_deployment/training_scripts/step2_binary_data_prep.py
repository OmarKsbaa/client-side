# Step 2: Load and Prepare Binary Classification Data
# Focus on koi_pdisposition (CANDIDATE vs FALSE POSITIVE) with top features

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import json

def load_kepler_binary_data():
    """Load Kepler dataset for binary classification"""
    print("🚀 Step 2: Loading Kepler Dataset for Binary Classification")
    print("=" * 60)
    
    # Load the dataset
    print("📂 Loading cumulative_2025.09.25_10.52.58.csv...")
    kepler_data = pd.read_csv('cumulative_2025.09.25_10.52.58.csv', comment='#', low_memory=False)
    
    print(f"✅ Dataset loaded: {kepler_data.shape}")
    print(f"📊 Total samples: {kepler_data.shape[0]:,}")
    print(f"📋 Total columns: {kepler_data.shape[1]}")
    
    return kepler_data

def analyze_binary_target(data):
    """Analyze the binary target variable koi_pdisposition"""
    print(f"\n🎯 Binary Target Analysis: 'koi_pdisposition'")
    print("=" * 45)
    
    # Check target distribution
    target_counts = data['koi_pdisposition'].value_counts()
    target_pct = data['koi_pdisposition'].value_counts(normalize=True) * 100
    
    print("📊 Class Distribution:")
    for class_name, count in target_counts.items():
        percentage = target_pct[class_name]
        print(f"  • {class_name}: {count:,} samples ({percentage:.1f}%)")
    
    print(f"\n📈 Total valid samples: {target_counts.sum():,}")
    
    # Check for missing values in target
    missing_target = data['koi_pdisposition'].isnull().sum()
    print(f"🔍 Missing target values: {missing_target}")
    
    # Class balance analysis
    balance_ratio = min(target_counts) / max(target_counts)
    print(f"⚖️ Class balance ratio: {balance_ratio:.3f} {'✅ Well balanced' if balance_ratio > 0.8 else '⚠️ Imbalanced'}")
    
    return target_counts

def load_top_features():
    """Load the top features identified in Step 1"""
    print(f"\n📋 Loading Top Features from Step 1...")
    
    # Read feature importance results
    importance_df = pd.read_csv('kepler_feature_importance.csv')
    
    # Get top 10 features (95.4% of importance)
    top_10_features = importance_df.head(10)['Feature'].tolist()
    
    print(f"🏆 Selected Top 10 Features (95.4% importance):")
    for i, feature in enumerate(top_10_features, 1):
        importance = importance_df[importance_df['Feature'] == feature]['Importance_Normalized'].iloc[0]
        print(f"  {i:2d}. {feature:<15} ({importance:.1f}%)")
    
    return top_10_features

def prepare_binary_dataset(data, features):
    """Prepare the dataset for binary classification"""
    print(f"\n⚙️ Preparing Binary Classification Dataset...")
    print("=" * 45)
    
    # Select features and target
    X = data[features].copy()
    y = data['koi_pdisposition'].copy()
    
    print(f"📊 Initial data shape: {X.shape}")
    print(f"🎯 Target shape: {y.shape}")
    
    # Remove rows with missing target
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"📊 After removing missing targets: {X.shape}")
    
    # Check feature availability
    print(f"\n🔍 Feature Availability Check:")
    missing_features = []
    for feature in features:
        if feature not in X.columns:
            missing_features.append(feature)
        else:
            missing_count = X[feature].isnull().sum()
            missing_pct = (missing_count / len(X)) * 100
            print(f"  • {feature:<15}: {missing_count:,} missing ({missing_pct:.1f}%)")
    
    if missing_features:
        print(f"❌ Missing features: {missing_features}")
        return None, None
    
    return X, y

def preprocess_features(X, y):
    """Preprocess features for modeling"""
    print(f"\n🔧 Preprocessing Features...")
    print("=" * 30)
    
    # Handle missing values with median imputation
    print("📝 Handling missing values with median imputation...")
    imputer = SimpleImputer(strategy='median')
    
    # Replace infinite values with NaN first
    X_clean = X.replace([np.inf, -np.inf], np.nan)
    
    # Apply imputation
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X_clean),
        columns=X.columns,
        index=X.index
    )
    
    # Encode target variable
    print("🏷️ Encoding target variable...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y.astype(str))
    
    target_mapping = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))
    print(f"  Target mapping: {target_mapping}")
    
    # Train-test split
    print("✂️ Creating train-test split (80-20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Feature scaling
    print("📏 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames for easier handling
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
    
    print(f"✅ Preprocessing completed!")
    print(f"📊 Training set: {X_train_scaled.shape}")
    print(f"📊 Test set: {X_test_scaled.shape}")
    
    # Check class distribution in splits
    train_dist = np.bincount(y_train)
    test_dist = np.bincount(y_test)
    
    print(f"\n🎯 Class Distribution in Splits:")
    for i, class_name in enumerate(label_encoder.classes_):
        train_pct = (train_dist[i] / len(y_train)) * 100
        test_pct = (test_dist[i] / len(y_test)) * 100
        print(f"  • {class_name}:")
        print(f"    - Training: {train_dist[i]:,} ({train_pct:.1f}%)")
        print(f"    - Test: {test_dist[i]:,} ({test_pct:.1f}%)")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder, imputer

def save_preprocessing_artifacts(features, scaler, label_encoder, imputer):
    """Save preprocessing artifacts for later use"""
    print(f"\n💾 Saving Preprocessing Artifacts...")
    
    # Create preprocessing config
    preprocessing_config = {
        'selected_features': features,
        'feature_count': len(features),
        'target_variable': 'koi_pdisposition',
        'preprocessing_steps': [
            'median_imputation',
            'standard_scaling',
            'label_encoding'
        ],
        'target_classes': label_encoder.classes_.tolist(),
        'target_mapping': dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_.tolist()))
    }
    
    # Save configuration
    with open('binary_preprocessing_config.json', 'w') as f:
        json.dump(preprocessing_config, f, indent=2)
    
    # Save preprocessing objects
    import pickle
    
    preprocessing_objects = {
        'scaler': scaler,
        'label_encoder': label_encoder,
        'imputer': imputer,
        'features': features
    }
    
    with open('binary_preprocessing_objects.pkl', 'wb') as f:
        pickle.dump(preprocessing_objects, f)
    
    print("✅ Saved preprocessing artifacts:")
    print("  • binary_preprocessing_config.json")
    print("  • binary_preprocessing_objects.pkl")

def main():
    """Main function for Step 2"""
    try:
        # Load data
        data = load_kepler_binary_data()
        
        # Analyze binary target
        target_counts = analyze_binary_target(data)
        
        # Load top features from Step 1
        top_features = load_top_features()
        
        # Prepare dataset
        X, y = prepare_binary_dataset(data, top_features)
        
        if X is None:
            print("❌ Failed to prepare dataset")
            return None
        
        # Preprocess features
        X_train, X_test, y_train, y_test, scaler, label_encoder, imputer = preprocess_features(X, y)
        
        # Save preprocessing artifacts
        save_preprocessing_artifacts(top_features, scaler, label_encoder, imputer)
        
        # Save processed datasets
        print(f"\n💾 Saving Processed Datasets...")
        
        # Combine features and targets for saving
        train_data = X_train.copy()
        train_data['target'] = y_train
        
        test_data = X_test.copy()  
        test_data['target'] = y_test
        
        train_data.to_csv('binary_train_data.csv', index=False)
        test_data.to_csv('binary_test_data.csv', index=False)
        
        print("✅ Saved processed datasets:")
        print("  • binary_train_data.csv")
        print("  • binary_test_data.csv")
        
        # Summary
        print(f"\n📋 Step 2 Summary:")
        print(f"  • Dataset: {len(X):,} samples with binary target")
        print(f"  • Features: {len(top_features)} most important features")
        print(f"  • Classes: {list(label_encoder.classes_)}")
        print(f"  • Train/Test: {len(X_train):,} / {len(X_test):,} samples")
        print(f"  • All preprocessing artifacts saved ✅")
        
        print(f"\n✅ Step 2 Completed Successfully!")
        print(f"🚀 Ready for Step 3: Select final features and train model")
        
        return {
            'X_train': X_train,
            'X_test': X_test, 
            'y_train': y_train,
            'y_test': y_test,
            'features': top_features,
            'target_counts': target_counts,
            'preprocessing_objects': {
                'scaler': scaler,
                'label_encoder': label_encoder,
                'imputer': imputer
            }
        }
        
    except Exception as e:
        print(f"❌ Error in Step 2: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()