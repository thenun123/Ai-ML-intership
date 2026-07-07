# ============================================================
#    🏥  BREAST CANCER PREDICTION SYSTEM
#        Complete ML Pipeline with Hyperparameter Tuning
#        Mini Project – Day 12
# ============================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ═══════════════════════════════════════════════════════════════
print("=" * 62)
print("      🏥  BREAST CANCER PREDICTION SYSTEM  🏥")
print("=" * 62)

# ── STEP 1: Load & Explore ───────────────────────────────────
data    = load_breast_cancer()
df      = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print(f"\n📊 STEP 1: Dataset Overview")
print(f"  Samples     : {df.shape[0]}")
print(f"  Features    : {data.data.shape[1]}")
print(f"  Classes     : Malignant (0) = {sum(data.target==0)} | Benign (1) = {sum(data.target==1)}")
print(f"  Missing vals: {df.isnull().sum().sum()}")

print(f"\n  Top 5 Features (mean values by class):")
top_feats = ['mean radius', 'mean texture', 'mean perimeter',
             'mean area', 'mean smoothness']
for feat in top_feats:
    mal = df[df['target']==0][feat].mean()
    ben = df[df['target']==1][feat].mean()
    print(f"    {feat:<25} Malignant={mal:.2f}  Benign={ben:.2f}")

# ── STEP 2: Preprocess ───────────────────────────────────────
print(f"\n🔧 STEP 2: Preprocessing")
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"  Train set : {X_train.shape[0]} samples")
print(f"  Test set  : {X_test.shape[0]} samples")
print(f"  Scaling   : StandardScaler applied ✅")

# ── STEP 3: Baseline Model ───────────────────────────────────
print(f"\n📌 STEP 3: Baseline Logistic Regression")
baseline = LogisticRegression(random_state=42, max_iter=1000)
baseline.fit(X_train, y_train)
base_pred = baseline.predict(X_test)

base_acc  = accuracy_score(y_test, base_pred)
base_prec = precision_score(y_test, base_pred)
base_rec  = recall_score(y_test, base_pred)
base_f1   = f1_score(y_test, base_pred)

print(f"  Accuracy  : {base_acc  * 100:.2f}%")
print(f"  Precision : {base_prec * 100:.2f}%")
print(f"  Recall    : {base_rec  * 100:.2f}%")
print(f"  F1-Score  : {base_f1   * 100:.2f}%")

cv_scores = cross_val_score(baseline, X_train, y_train, cv=5)
print(f"  CV Mean   : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# ── STEP 4: GridSearchCV Tuning ──────────────────────────────
print(f"\n🔍 STEP 4: Hyperparameter Tuning (GridSearchCV)")

param_grid = {
    'C'       : [0.001, 0.01, 0.1, 1, 10, 100],
    'solver'  : ['lbfgs', 'liblinear'],
    'penalty' : ['l2'],
    'max_iter': [1000]
}

grid = GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
grid.fit(X_train, y_train)

best_model  = grid.best_estimator_
tuned_pred  = best_model.predict(X_test)

tuned_acc   = accuracy_score(y_test, tuned_pred)
tuned_prec  = precision_score(y_test, tuned_pred)
tuned_rec   = recall_score(y_test, tuned_pred)
tuned_f1    = f1_score(y_test, tuned_pred)

print(f"  Best Parameters : {grid.best_params_}")
print(f"  Best CV Score   : {grid.best_score_ * 100:.2f}%")
print(f"  Accuracy  : {tuned_acc  * 100:.2f}%")
print(f"  Precision : {tuned_prec * 100:.2f}%")
print(f"  Recall    : {tuned_rec  * 100:.2f}%")
print(f"  F1-Score  : {tuned_f1   * 100:.2f}%")

# ── STEP 5: Before vs After Comparison ───────────────────────
print(f"\n{'=' * 62}")
print(f"  📊 STEP 5: Baseline vs Tuned Model Comparison")
print(f"{'=' * 62}")
print(f"  {'Metric':<15} {'Baseline':>12} {'Tuned':>12} {'Change':>10}")
print(f"  {'-' * 52}")
metrics = {
    'Accuracy' : (base_acc,  tuned_acc),
    'Precision': (base_prec, tuned_prec),
    'Recall'   : (base_rec,  tuned_rec),
    'F1-Score' : (base_f1,   tuned_f1),
}
for m, (b, t) in metrics.items():
    diff = (t - b) * 100
    arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
    print(f"  {m:<15} {b*100:>11.2f}% {t*100:>11.2f}% {arrow}{abs(diff):>7.2f}%")

# ── STEP 6: Confusion Matrix (Bonus) ─────────────────────────
print(f"\n🖼️  STEP 6: Generating Confusion Matrix Heatmap...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Breast Cancer Prediction – Confusion Matrix Comparison",
             fontsize=13, fontweight='bold')

labels = ['Malignant', 'Benign']
for ax, (title, pred) in zip(axes, [
    (f"Baseline  (Acc: {base_acc*100:.1f}%)",  base_pred),
    (f"Tuned     (Acc: {tuned_acc*100:.1f}%)", tuned_pred),
]):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                ax=ax, linewidths=0.5)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
print(f"  ✅ Saved: confusion_matrix.png")

# ── STEP 7: Sample Predictions ───────────────────────────────
print(f"\n🔬 STEP 7: Sample Predictions (first 10 test samples)")
print(f"  {'#':<4} {'Actual':<14} {'Baseline':<14} {'Tuned':<14} {'Match'}")
print(f"  {'-' * 58}")
for i, (actual, bp, tp) in enumerate(zip(y_test[:10], base_pred[:10], tuned_pred[:10]), 1):
    a_label  = labels[actual]
    b_label  = labels[bp]
    t_label  = labels[tp]
    match    = "✅" if tp == actual else "❌"
    print(f"  {i:<4} {a_label:<14} {b_label:<14} {t_label:<14} {match}")

# ── STEP 8: Predict a Custom Sample ──────────────────────────
print(f"\n🧪 STEP 8: Predict a New Sample")
sample    = X_test[0:1]
pred_cls  = best_model.predict(sample)[0]
pred_prob = best_model.predict_proba(sample)[0]

print(f"  Predicted Class  : {'🔴 MALIGNANT' if pred_cls == 0 else '🟢 BENIGN'}")
print(f"  Malignant Prob   : {pred_prob[0]*100:.2f}%")
print(f"  Benign Prob      : {pred_prob[1]*100:.2f}%")
print(f"  Actual Class     : {'Malignant' if y_test[0] == 0 else 'Benign'}")

print(f"\n{'=' * 62}")
print(f"  ✅  Breast Cancer Prediction System Complete!")
print(f"{'=' * 62}")
