# ============================================================
#       Step 1: Load & Explore – Breast Cancer Dataset
# ============================================================

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

# ── Load Dataset ─────────────────────────────────────────────
data = load_breast_cancer()
df   = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df['diagnosis'] = df['target'].map({1: 'Benign', 0: 'Malignant'})

print("=" * 60)
print("       BREAST CANCER DATASET – EXPLORATION")
print("=" * 60)

# ── head() ───────────────────────────────────────────────────
print("\n📋 First 5 Rows (head):")
print(df[['mean radius', 'mean texture', 'mean perimeter',
          'mean area', 'mean smoothness', 'diagnosis']].head().to_string())

# ── info() ───────────────────────────────────────────────────
print("\n📊 Dataset Info (info):")
print(f"  Shape        : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  Features     : {data.data.shape[1]}")
print(f"  Target       : Binary (0 = Malignant, 1 = Benign)")
print(f"  Missing vals : {df.isnull().sum().sum()}")
print(f"  Data types   :")
for col in ['mean radius', 'mean texture', 'mean area']:
    print(f"    {col:<25} → {df[col].dtype}")

# ── describe() ───────────────────────────────────────────────
print("\n📐 Statistical Summary (describe) – first 5 features:")
print(df[data.feature_names[:5]].describe().round(2).to_string())

# ── Target Distribution ──────────────────────────────────────
print("\n🎯 Target Class Distribution:")
counts = df['diagnosis'].value_counts()
total  = len(df)
for label, count in counts.items():
    bar = "█" * int((count / total) * 40)
    print(f"  {label:<12} : {count:>3} samples ({count/total*100:.1f}%)  {bar}")

print("\n✅ Dataset is clean — no missing values found.")
print("✅ Classes are slightly imbalanced: more Benign than Malignant.")
