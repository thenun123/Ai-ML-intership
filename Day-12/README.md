# 📘 Day 12 – Model Evaluation & Hyperparameter Tuning

## 🗂️ Folder Structure

```
Day-12/
├── dataset_exploration/
│   └── 01_explore_dataset.py           # head(), info(), describe(), class distribution
├── baseline_model/
│   └── 02_baseline_model.py            # Logistic Regression + Cross Validation + Metrics
├── hyperparameter_tuning/
│   └── 03_hyperparameter_tuning.py     # GridSearchCV + RandomizedSearchCV + Comparison
├── breast_cancer_prediction_system/
│   ├── breast_cancer_pipeline.py       # Complete 8-step ML Pipeline
│   └── confusion_matrix.png            # Baseline vs Tuned Confusion Matrices
└── README.md
```

---

## 🧠 What I Learned About Model Evaluation

Model evaluation goes beyond just accuracy. A good ML Engineer must understand:

- **Train vs Test Performance** — If training accuracy is much higher than test accuracy, the model is overfitting.
- **Cross Validation** — Splits data into K folds and trains/tests K times to give a reliable average score. Reduces the risk of a lucky or unlucky data split.
- **Overfitting** — Model memorizes training data, fails on new data (high train acc, low test acc).
- **Underfitting** — Model is too simple to capture patterns (low both).
- **Choosing the right metric** — Accuracy alone is misleading on imbalanced datasets. For medical tasks, Recall matters more (we don't want to miss a Malignant case).

---

## 🔧 What is Hyperparameter Tuning and Why It Matters

**Hyperparameters** are settings we choose before training — the model doesn't learn them from data. Examples:
- `C` — Regularization strength in Logistic Regression
- `max_depth` — How deep a Decision Tree grows
- `solver` — Optimization algorithm used

**Why it matters:**
- Default parameters are rarely optimal.
- The right hyperparameters can be the difference between a mediocre and a production-ready model.
- GridSearchCV exhaustively tries all combinations; RandomizedSearchCV samples randomly — faster for large spaces.

---

## 🏆 Best Parameters Found by GridSearchCV

```
{'C': 0.1, 'solver': 'lbfgs', 'penalty': 'l2', 'max_iter': 1000}
```

GridSearchCV tested 12 parameter combinations × 5-fold CV = 60 model fits to find this.

---

## 📊 Model Comparison: Baseline vs Tuned

| Metric    | Baseline | Tuned  | Change  |
|-----------|----------|--------|---------|
| Accuracy  | 98.25%   | 97.37% | ↓ 0.88% |
| Precision | 98.61%   | 97.26% | ↓ 1.35% |
| Recall    | 98.61%   | 98.61% | → 0.00% |
| F1-Score  | 98.61%   | 97.93% | ↓ 0.68% |

---

## 🔍 Key Observations

1. **The baseline model was already excellent** at 98.25% accuracy — Logistic Regression works very well on this linearly separable dataset with proper feature scaling (StandardScaler).

2. **GridSearchCV selected C=0.1** (more regularization than default C=1.0), which slightly reduced variance but didn't improve test accuracy here — showing that tuning isn't always guaranteed to help on small, clean datasets.

3. **Recall stayed the same (98.61%)** for both models — meaning neither model missed any additional Malignant cases, which is the most critical metric for medical diagnosis.

4. **Cross Validation gave 98.02% ± 1.28%** — a very stable and reliable model with low variance across folds.

5. **Feature scaling was critical** — without StandardScaler, Logistic Regression's performance would degrade significantly because features like `mean area` (hundreds) dominate over `mean smoothness` (0.05–0.15).

6. **RandomizedSearchCV was faster** and found equally good parameters — making it the better choice for larger hyperparameter spaces.

---

## 📅 Date
**July 1, 2026**

> *"Building a model is only the first step. Improving and evaluating it is what makes ML solutions reliable in real-world applications."*
