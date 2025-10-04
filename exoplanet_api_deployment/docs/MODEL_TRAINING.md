# 🧠 Model Training Documentation

## Overview

This document describes the training process for both Kepler and TOI exoplanet classification models included in this API.

## Kepler Model Training Pipeline

### Dataset
- **Source**: NASA Kepler Cumulative Exoplanet Dataset
- **Size**: 9,564 objects with 50 features
- **Classes**: CANDIDATE (49.3%) vs FALSE POSITIVE (50.7%)
- **Final Accuracy**: 99.06%

### Training Scripts

1. **step2_binary_data_prep.py**
   - Loads Kepler cumulative dataset
   - Performs binary classification preparation
   - Creates train/test splits
   - Handles missing values and feature engineering

2. **step4_train_model.py**
   - Trains XGBoost classifier
   - Performs hyperparameter optimization
   - Generates performance metrics
   - Saves final model as `kepler_model_complete.pkl`

### Key Features Selected
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

## TOI Model Training Pipeline

### Dataset
- **Source**: TESS Objects of Interest (TOI) Dataset
- **Size**: 7,238 objects with 66 features
- **Classes**: CANDIDATE (82.1%) vs FALSE POSITIVE (17.9%)
- **Final Accuracy**: 88.13% (Enhanced)

### Training Scripts

1. **toi_step1_analysis.py**
   - Exploratory data analysis
   - Class distribution analysis
   - Feature correlation study

2. **toi_step2_cleaning.py**
   - Data cleaning and preprocessing
   - Missing value handling
   - Outlier detection and treatment

3. **toi_step3_feature_importance.py**
   - Feature selection using multiple methods
   - Importance ranking and analysis
   - Feature correlation analysis

4. **toi_step4_train_model.py**
   - Initial model training
   - Baseline performance establishment
   - Model validation and testing

5. **toi_final_enhancement.py**
   - Model enhancement and optimization
   - Hyperparameter tuning
   - Ensemble methods testing
   - Final model selection

6. **create_enhanced_model_fixed.py**
   - Creates production-ready enhanced model
   - Fixes compatibility issues
   - Final model packaging

### Key Features Selected
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

## Model Enhancement Journey

### TOI Model Improvement
The TOI model underwent significant enhancement:

**Baseline**: 87.22% accuracy
**Enhanced**: 88.13% accuracy (+0.91 percentage points)

### Enhancement Strategies Tested

1. **❌ Random Upsampling**: 79.48% (-7.74pp) - Decreased performance
2. **❌ Advanced Feature Engineering**: 81.17% (-6.05pp) - Decreased performance
3. **❌ Complex Optimization**: Multiple attempts all decreased performance
4. **✅ Conservative Enhancement**: 88.13% (+0.91pp) - **SUCCESS!**

### Winning Strategy
- **Targeted Parameter Tuning**: Small adjustments to proven parameters
- **Ensemble Approach**: Multiple models with different random seeds
- **Feature Preservation**: Kept the same 10 proven features
- **Conservative Approach**: Built carefully on proven baseline

## Training Configuration

### Kepler Model
```python
XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)
```

### TOI Model (Enhanced)
```python
XGBClassifier(
    n_estimators=150,  # Increased from 100
    max_depth=5,       # Increased from 4
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
```

## Performance Comparison

| Metric | Kepler Model | TOI Original | TOI Enhanced |
|--------|--------------|--------------|--------------|
| **Accuracy** | 99.06% | 87.22% | 88.13% |
| **Precision** | 99.2% | 85.0% | 88.0% |
| **Recall** | 99.1% | 86.0% | 87.0% |
| **F1-Score** | 99.1% | 85.5% | 87.5% |
| **ROC-AUC** | 99.5% | 85.2% | 86.0% |

## Preprocessing Pipeline

### Common Steps
1. **Data Loading**: CSV with comment handling
2. **Target Mapping**: String labels to binary classification
3. **Feature Selection**: Domain-specific feature engineering
4. **Missing Value Handling**: Median imputation
5. **Feature Scaling**: StandardScaler normalization
6. **Train-Test Split**: Stratified 80/20 split

### Model-Specific Steps

**Kepler**:
- Binary disposition mapping (CANDIDATE/FALSE POSITIVE)
- Flag feature handling (0/1 encoding)
- Astronomical parameter normalization

**TOI**:
- Multi-class disposition consolidation
- Physical parameter error handling
- TESS-specific feature engineering

## Validation Strategy

### Cross-Validation
- **Method**: 5-fold Stratified Cross-Validation
- **Kepler**: 98.9% ± 0.2% CV accuracy
- **TOI**: 84.1% ± 1.9% CV accuracy

### Test Set Validation
- **Kepler**: 1,913 test samples
- **TOI**: 1,449 test samples
- **Hold-out Strategy**: Never used for training or hyperparameter tuning

## Model Artifacts

### Saved Components
Each model package includes:
- **Model**: Trained XGBoost classifier
- **Scaler**: StandardScaler for feature normalization
- **Imputer**: SimpleImputer for missing value handling
- **Feature List**: Required feature names and order
- **Metadata**: Training information and performance metrics

### File Structure
```python
model_package = {
    'model': xgb_classifier,
    'scaler': StandardScaler(),
    'imputer': SimpleImputer(),
    'required_features': [list_of_features],
    'target_classes': ['FALSE_POSITIVE', 'CANDIDATE'],
    'performance': {performance_metrics},
    'preprocessing_info': {preprocessing_details},
    'model_metadata': {training_metadata}
}
```

## Lessons Learned

### What Worked
1. **Conservative Enhancement**: Small, targeted improvements
2. **Domain Knowledge**: Physics-based feature selection
3. **Proper Validation**: Rigorous train/test separation
4. **Ensemble Methods**: Multiple random seeds for stability

### What Didn't Work
1. **Complex Feature Engineering**: Often hurt performance
2. **Aggressive Upsampling**: Introduced noise and overfitting
3. **Advanced Algorithms**: Simpler XGBoost outperformed complex ensembles
4. **Over-optimization**: Marginal gains often disappeared in production

### Best Practices
1. **Start Simple**: Establish strong baseline first
2. **Validate Everything**: Every change needs validation
3. **Preserve What Works**: Don't fix what's not broken
4. **Document Everything**: Track all experiments and results

## Reproducing Results

To retrain the models from scratch:

### Kepler Model
```bash
cd training_scripts/
python step2_binary_data_prep.py
python step4_train_model.py
```

### TOI Model
```bash
cd training_scripts/
python toi_step1_analysis.py
python toi_step2_cleaning.py
python toi_step3_feature_importance.py
python toi_step4_train_model.py
python toi_final_enhancement.py
python create_enhanced_model_fixed.py
```

## Future Improvements

### Potential Enhancements
1. **Multi-class Classification**: Beyond binary CANDIDATE/FALSE_POSITIVE
2. **Deep Learning**: Neural networks for complex feature interactions
3. **Ensemble Methods**: Combining multiple algorithm types
4. **Active Learning**: Iterative model improvement with new data
5. **Uncertainty Quantification**: Confidence intervals for predictions

### Data Improvements
1. **More Recent Data**: Latest mission discoveries
2. **Cross-Mission Validation**: Train on one mission, test on another
3. **Temporal Validation**: Train on older data, test on newer
4. **External Validation**: Independent confirmation data

---

This documentation provides a complete overview of the model training process and serves as a guide for future model development and improvement efforts.