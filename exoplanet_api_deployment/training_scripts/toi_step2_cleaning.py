"""
TOI Dataset Processing - Step 2: Data Cleaning and Target Remapping
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("🧹 TOI Data Cleaning and Target Remapping")
print("=" * 50)

# Load TOI dataset
print("Loading TOI dataset...")
toi_df = pd.read_csv('TOI_2025.09.26_02.41.12.csv', comment='#')
print(f"✅ Original dataset: {toi_df.shape}")

# 1. TARGET REMAPPING as requested
print(f"\n🎯 Step 1: Target Class Remapping")
print("Original class distribution:")
original_counts = toi_df['tfopwg_disp'].value_counts()
for cls, count in original_counts.items():
    print(f"   {cls}: {count} ({count/len(toi_df)*100:.1f}%)")

# Define remapping
target_mapping = {
    # Candidates (planets)
    'CP': 'CANDIDATE',  # Confirmed Planet
    'KP': 'CANDIDATE',  # Known Planet  
    'PC': 'CANDIDATE',  # Planet Candidate
    
    # False Positives (non-planets)
    'FP': 'FALSE POSITIVE',  # False Positive
    'FA': 'FALSE POSITIVE',  # False Alarm
    
    # Skip APC (Ambiguous Planet Candidate) as requested
    'APC': 'SKIP'
}

print(f"\n📋 Target Mapping:")
for original, mapped in target_mapping.items():
    count = original_counts.get(original, 0)
    print(f"   {original} → {mapped}: {count} samples")

# Apply mapping
toi_df['binary_target'] = toi_df['tfopwg_disp'].map(target_mapping)

# Remove APC entries (marked as 'SKIP')
print(f"\n🗑️ Removing APC entries...")
before_skip = len(toi_df)
toi_df = toi_df[toi_df['binary_target'] != 'SKIP'].copy()
after_skip = len(toi_df)
print(f"   Removed {before_skip - after_skip} APC entries")
print(f"   Dataset size after removal: {after_skip}")

# Final target distribution
print(f"\n🎯 Final Binary Target Distribution:")
final_counts = toi_df['binary_target'].value_counts()
for cls, count in final_counts.items():
    print(f"   {cls}: {count} ({count/len(toi_df)*100:.1f}%)")

# Check class balance
candidate_ratio = final_counts['CANDIDATE'] / len(toi_df)
print(f"\n⚖️ Class Balance:")
print(f"   CANDIDATE: {candidate_ratio:.3f} ({candidate_ratio*100:.1f}%)")
print(f"   FALSE POSITIVE: {1-candidate_ratio:.3f} ({(1-candidate_ratio)*100:.1f}%)")

if abs(candidate_ratio - 0.5) > 0.3:
    print("   ⚠️ Classes are imbalanced - will need stratified sampling")
else:
    print("   ✅ Classes are reasonably balanced")

# 2. DATA QUALITY ANALYSIS
print(f"\n🔍 Step 2: Data Quality Analysis")

# Missing values analysis
print("Missing values by column:")
missing_stats = []
for col in toi_df.columns:
    missing_count = toi_df[col].isna().sum()
    missing_pct = missing_count / len(toi_df) * 100
    if missing_count > 0:
        missing_stats.append({
            'column': col,
            'missing_count': missing_count,
            'missing_pct': missing_pct
        })

missing_df = pd.DataFrame(missing_stats).sort_values('missing_pct', ascending=False)

if len(missing_df) > 0:
    print(f"   Columns with missing values: {len(missing_df)}")
    print("   Top 10 columns with most missing values:")
    for _, row in missing_df.head(10).iterrows():
        print(f"     {row['column']}: {row['missing_count']} ({row['missing_pct']:.1f}%)")
else:
    print("   ✅ No missing values found!")

# 3. FEATURE SELECTION AND CLEANING
print(f"\n🔧 Step 3: Feature Selection and Cleaning")

# Get numerical columns for potential features
numerical_cols = toi_df.select_dtypes(include=[np.number]).columns.tolist()

# Remove ID columns and target columns
id_cols = ['loc_rowid', 'toi', 'tid']
target_cols = ['tfopwg_disp', 'binary_target']
feature_candidates = [col for col in numerical_cols if col not in id_cols + target_cols]

print(f"   Potential feature columns: {len(feature_candidates)}")
print(f"   First 15 features: {feature_candidates[:15]}")

# Remove columns with too many missing values (>50%)
high_quality_features = []
for col in feature_candidates:
    missing_pct = toi_df[col].isna().sum() / len(toi_df) * 100
    if missing_pct <= 50:  # Keep columns with ≤50% missing
        high_quality_features.append(col)

print(f"   High-quality features (≤50% missing): {len(high_quality_features)}")

# Remove constant or near-constant columns
final_features = []
for col in high_quality_features:
    if toi_df[col].dtype in ['int64', 'float64']:
        unique_vals = toi_df[col].nunique()
        if unique_vals > 1:  # More than 1 unique value
            # Check if it's not 99% the same value
            most_common_freq = toi_df[col].value_counts().iloc[0]
            if most_common_freq / len(toi_df) < 0.99:
                final_features.append(col)

print(f"   Final feature candidates: {len(final_features)} features")

# 4. CREATE CLEAN DATASET
print(f"\n💾 Step 4: Creating Clean Dataset")

# Select final columns
final_columns = ['toi', 'tid'] + final_features + ['tfopwg_disp', 'binary_target']
clean_toi_df = toi_df[final_columns].copy()

# Basic statistics
print(f"   Clean dataset shape: {clean_toi_df.shape}")
print(f"   Features for modeling: {len(final_features)}")
print(f"   Samples: {len(clean_toi_df)}")

# Save clean dataset
clean_toi_df.to_csv('toi_clean_data.csv', index=False)
print(f"   ✅ Saved to 'toi_clean_data.csv'")

# 5. SUMMARY
print(f"\n📊 PROCESSING SUMMARY")
print("=" * 30)
print(f"✅ Original TOI dataset: {toi_df.shape[0]} rows, {toi_df.shape[1]} columns")
print(f"✅ Target remapping completed:")
print(f"   • CP, KP, PC → CANDIDATE ({final_counts['CANDIDATE']} samples)")
print(f"   • FP, FA → FALSE POSITIVE ({final_counts['FALSE POSITIVE']} samples)")
print(f"   • APC → SKIPPED ({before_skip - after_skip} samples)")
print(f"✅ Clean dataset: {clean_toi_df.shape[0]} rows, {len(final_features)} features")
print(f"✅ Class balance: {candidate_ratio:.1%} candidates, {1-candidate_ratio:.1%} false positives")
print(f"✅ Ready for feature importance analysis!")

# Show sample of clean data
print(f"\n📋 Sample of clean data:")
sample_cols = ['toi', 'binary_target'] + final_features[:3]
print(clean_toi_df[sample_cols].head(3))