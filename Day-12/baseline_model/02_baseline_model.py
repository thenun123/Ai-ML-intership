# ============================================================
#      Step 2: Baseline Logistic Regression Model
#              Train → Evaluate → Diagnose
# ============================================================

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ── Load & Split ─────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Scale Features ───────────────────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── Train Baseline Model ─────────────────────────────────────
baseline = LogisticRegression(random_state=42, max_iter=1000)
baseline.fit(X_train, y_train)

# ── Predictions ──────────────────────────────────────────────
y_pred     = baseline.predict(X_test)
y_pred_tr  = baseline.predict(X_train)

# ── Metrics ──────────────────────────────────────────────────
train_acc = accuracy_score(y_train, y_pred_tr)
test_acc  = accuracy_score(y_test,  y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test,    y_pred)
f1        = f1_score(y_test,        y_pred)
cm        = confusion_matrix(y_test, y_pred)

print("=" * 55)
print("       BASELINE LOGISTIC REGRESSION MODEL")
print("=" * 55)

print(f"\n📈 Performance Metrics:")
print(f"  Training Accuracy : {train_acc * 100:.2f}%")
print(f"  Testing  Accuracy : {test_acc  * 100:.2f}%")
print(f"  Precision         : {precision * 100:.2f}%")
print(f"  Recall            : {recall    * 100:.2f}%")
print(f"  F1-Score          : {f1        * 100:.2f}%")

# ── Overfitting / Underfitting Check ─────────────────────────
gap = train_acc - test_acc
print(f"\n🔍 Overfitting / Underfitting Diagnosis:")
print(f"  Train-Test Gap : {gap * 100:.2f}%")
if gap > 0.10:
    print("  ⚠️  OVERFITTING — model memorized training data.")
elif train_acc < 0.80:
    print("  ⚠️  UNDERFITTING — model is too simple.")
else:
    print("  ✅ Model is well-balanced (no significant overfitting).")

# ── Cross Validation ─────────────────────────────────────────
cv_scores = cross_val_score(baseline, X_train, y_train, cv=5, scoring='accuracy')
print(f"\n📊 5-Fold Cross Validation:")
print(f"  Scores : {[f'{s*100:.2f}%' for s in cv_scores]}")
print(f"  Mean   : {cv_scores.mean()*100:.2f}%")
print(f"  Std Dev: ±{cv_scores.std()*100:.2f}%")

# ── Confusion Matrix ─────────────────────────────────────────
print(f"\n🔢 Confusion Matrix:")
print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
print(f"  FN={cm[1][0]}  TP={cm[1][1]}")

print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))
