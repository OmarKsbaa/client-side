"""
TOI Dataset Analysis - Step 1: Explore TESS Objects of Interest Data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

print("🔍 TOI Dataset Analysis")
print("=" * 50)

# Load TOI dataset
print("Loading TOI dataset...")
toi_df = pd.read_csv('TOI_2025.09.26_02.41.12.csv', comment='#')

print(f"✅ Dataset loaded successfully!")
print(f"📊 Shape: {toi_df.shape}")
print(f"📋 Columns: {len(toi_df.columns)}")

# Basic info
print(f"\n📋 First 10 columns:")
print(list(toi_df.columns)[:10])

print(f"\n📋 Last 10 columns:")
print(list(toi_df.columns)[-10:])

# Look for target columns
target_columns = [col for col in toi_df.columns if 'disp' in col.lower() or 'class' in col.lower()]
print(f"\n🎯 Potential target columns: {target_columns}")

# Check for common target column names
if 'tfopwg_disp' in toi_df.columns:
    print(f"\n🎯 Target column found: 'tfopwg_disp'")
    target_counts = toi_df['tfopwg_disp'].value_counts()
    print(f"Target distribution:")
    for value, count in target_counts.items():
        print(f"   {value}: {count} ({count/len(toi_df)*100:.1f}%)")
    
    # Check for the classes you mentioned
    unique_classes = set(toi_df['tfopwg_disp'].dropna().unique())
    print(f"\n📊 Unique classes: {unique_classes}")
    
    # Count our target classes
    candidates = ['CP', 'KP', 'PC']  # Considered candidates
    false_positives = ['FP', 'FA']   # False positives
    
    candidate_count = sum(target_counts.get(cls, 0) for cls in candidates)
    fp_count = sum(target_counts.get(cls, 0) for cls in false_positives)
    apc_count = target_counts.get('APC', 0)
    
    print(f"\n🎯 Class Mapping Summary:")
    print(f"   CANDIDATES (CP, KP, PC): {candidate_count}")
    print(f"   FALSE POSITIVES (FP, FA): {fp_count}")
    print(f"   APC (to skip if few): {apc_count}")
    print(f"   Other classes: {len(toi_df) - candidate_count - fp_count - apc_count}")

else:
    print(f"\n❌ 'tfopwg_disp' column not found")
    print("Available columns containing 'disp':")
    disp_cols = [col for col in toi_df.columns if 'disp' in col.lower()]
    print(disp_cols)

# Check data quality
print(f"\n📊 Data Quality Overview:")
print(f"   Total rows: {len(toi_df)}")
print(f"   Total columns: {len(toi_df.columns)}")

# Missing values in key columns
key_columns = ['toi', 'tid', 'tfopwg_disp'] if 'tfopwg_disp' in toi_df.columns else ['toi', 'tid']
for col in key_columns:
    if col in toi_df.columns:
        missing = toi_df[col].isna().sum()
        print(f"   {col}: {missing} missing values ({missing/len(toi_df)*100:.1f}%)")

# Show a few sample rows
print(f"\n📋 Sample data (first 3 rows):")
sample_cols = ['toi', 'tid', 'tfopwg_disp'] if 'tfopwg_disp' in toi_df.columns else toi_df.columns[:3]
print(toi_df[sample_cols].head(3))

# Check for numerical columns that might be features
numerical_cols = toi_df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\n🔢 Numerical columns: {len(numerical_cols)}")
print(f"   First 10: {numerical_cols[:10]}")

print(f"\n✅ TOI dataset analysis complete!")
print(f"📝 Summary:")
print(f"   - Dataset has {len(toi_df)} rows and {len(toi_df.columns)} columns")
print(f"   - Target column: {'tfopwg_disp' if 'tfopwg_disp' in toi_df.columns else 'Not found'}")
print(f"   - Numerical features: {len(numerical_cols)}")