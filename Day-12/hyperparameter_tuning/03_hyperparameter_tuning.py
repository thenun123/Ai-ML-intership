# ============================================================
#    Step 3: Hyperparameter Tuning
#            GridSearchCV  vs  RandomizedSearchCV
# ============================================================

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import time

# ── Load & Prepare ───────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print("=" * 60)
print("        HYPERPARAMETER TUNING – GridSearchCV")
print("=" * 60)

# ── Baseline (no tuning) ─────────────────────────────────────
baseline = LogisticRegression(random_state=42, max_iter=1000)
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)
baseline_acc  = accuracy_score(y_test, baseline_pred)
baseline_f1   = f1_score(y_test, baseline_pred)

print(f"\n📌 Baseline Model (default params):")
print(f"   Accuracy : {baseline_acc * 100:.2f}%")
print(f"   F1-Score : {baseline_f1  * 100:.2f}%")
print(f"   Params   : C=1.0, solver=lbfgs, penalty=l2")

# ── GridSearchCV ─────────────────────────────────────────────
param_grid = {
    'C'      : [0.001, 0.01, 0.1, 1, 10, 100],
    'solver' : ['lbfgs', 'liblinear'],
    'penalty': ['l2'],
    'max_iter': [1000]
}

print(f"\n🔍 Running GridSearchCV...")
print(f"   Parameter combinations: {6 * 2} × 5-fold CV = {6*2*5} fits")

start = time.time()
grid_search = GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train, y_train)
grid_time = time.time() - start

best_model    = grid_search.best_estimator_
tuned_pred    = best_model.predict(X_test)
tuned_acc     = accuracy_score(y_test, tuned_pred)
tuned_f1      = f1_score(y_test, tuned_pred)

print(f"\n✅ GridSearchCV Complete ({grid_time:.2f}s)")
print(f"   Best Parameters : {grid_search.best_params_}")
print(f"   Best CV Score   : {grid_search.best_score_ * 100:.2f}%")

# ── RandomizedSearchCV ───────────────────────────────────────
from scipy.stats import loguniform

param_dist = {
    'C'      : loguniform(0.001, 100),
    'solver' : ['lbfgs', 'liblinear'],
    'penalty': ['l2'],
    'max_iter': [1000]
}

print(f"\n🎲 Running RandomizedSearchCV (20 iterations)...")

start = time.time()
rand_search = RandomizedSearchCV(
    LogisticRegression(random_state=42),
    param_dist,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)
rand_search.fit(X_train, y_train)
rand_time = time.time() - start

rand_pred = rand_search.best_estimator_.predict(X_test)
rand_acc  = accuracy_score(y_test, rand_pred)

print(f"✅ RandomizedSearchCV Complete ({rand_time:.2f}s)")
print(f"   Best Parameters : {rand_search.best_params_}")
print(f"   Best CV Score   : {rand_search.best_score_ * 100:.2f}%")

# ── Comparison Table ─────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"  📊 COMPARISON: Baseline vs GridSearch vs RandomSearch")
print(f"{'=' * 60}")
print(f"  {'Model':<22} {'Accuracy':>10} {'F1-Score':>10} {'Time':>8}")
print(f"  {'-' * 52}")
print(f"  {'Baseline':<22} {baseline_acc*100:>9.2f}% {baseline_f1*100:>9.2f}%  {'N/A':>6}")
print(f"  {'GridSearchCV':<22} {tuned_acc*100:>9.2f}% {tuned_f1*100:>9.2f}%  {grid_time:>5.2f}s")
print(f"  {'RandomizedSearchCV':<22} {rand_acc*100:>9.2f}%  {'N/A':>8}  {rand_time:>5.2f}s")

improvement = tuned_acc - baseline_acc
print(f"\n  Improvement after tuning: {improvement*100:+.2f}%")
print(f"\n📝 Observation:")
print(f"   GridSearchCV exhaustively tries all combinations → most thorough.")
print(f"   RandomizedSearchCV is faster with similar results on large param spaces.")
print(f"   Hyperparameter tuning helped fine-tune C (regularization strength).")
