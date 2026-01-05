# -*- coding: utf-8 -*-
"""daily_water_intake_ml_prediction - Husandeep Atwal.ipynb

Converted to a .py script (GitHub-friendly).

This project applies multiple machine learning models to predict Hydration Level (Good vs Poor)
using the Daily Water Intake dataset.
"""



"""IMPORTING LIBRARIES

Load the necessary Python libraries:

pandas, numpy for data handling

sklearn for preprocessing, models, and evaluation
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    f1_score,
)

"""LOADING DATASET

Load Daily_Water_Intake.csv and display the number of rows/columns and the first few rows of data.
"""

df = pd.read_csv("Daily_Water_Intake.csv")
print("Shape:", df.shape)
print("\nFirst 5 rows:\n")
print(df.head())

"""BASIC DATA CHECKS

Check missing values and class distribution.
"""

print("\nMissing values per column:\n")
print(df.isna().sum())

print("\nTarget distribution (Hydration Level):\n")
print(df["Hydration Level"].value_counts())

"""DATA PRE-PROCESSING

1) Explicitly encode the target Hydration Level:
   - Poor -> 0
   - Good -> 1

2) One-hot encode categorical input features:
   - Gender
   - Physical Activity Level
   - Weather

3) Standardize numeric features
"""

# Explicit target mapping (avoids LabelEncoder ambiguity)
df["Hydration Level"] = df["Hydration Level"].map({"Poor": 0, "Good": 1})
if df["Hydration Level"].isna().any():
    raise ValueError("Target mapping failed. Expected Hydration Level values: 'Poor' or 'Good' only.")

y = df["Hydration Level"]
X = df.drop(columns=["Hydration Level"])

cat_cols = ["Gender", "Physical Activity Level", "Weather"]
num_cols = [c for c in X.columns if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
    ]
)

"""TRAIN-TEST SPLIT

Splits the dataset into:
- 80% training data
- 20% testing data
"""

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def fit_evaluate(model, model_name: str):
    """Fit model in a pipeline and evaluate on the test set."""
    pipe = Pipeline([("prep", preprocess), ("model", model)])

    start_time = time.time()
    pipe.fit(X_train, y_train)
    train_time = time.time() - start_time

    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    wf1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)

    # ROC-AUC (requires predicted probabilities)
    auc = None
    fpr = tpr = None
    try:
        y_proba = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
    except Exception:
        pass

    print(f"\n==================== {model_name} ====================")
    print(f"Training time (s): {train_time:.6f}")
    print(f"Test Accuracy:     {acc:.6f}")
    if auc is not None:
        print(f"ROC-AUC:           {auc:.6f}")
    print(f"Weighted F1:       {wf1:.6f}\n")

    print("Classification Report:\n")
    print(classification_report(y_test, y_pred, digits=6))
    print("Confusion Matrix:\n", cm)

    return {
        "Model": model_name,
        "Test Accuracy": acc,
        "ROC-AUC": auc,
        "Weighted F1": wf1,
        "Training Time (s)": train_time,
        "roc": (fpr, tpr),
    }

"""MODEL TRAINING + EVALUATION

Train and compare:
- Logistic Regression
- Random Forest
- ANN (MLPClassifier)
"""

results = []
results.append(fit_evaluate(LogisticRegression(max_iter=2000, random_state=42), "Logistic Regression"))
results.append(fit_evaluate(RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1), "Random Forest"))
results.append(fit_evaluate(MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=100, random_state=42), "ANN (MLPClassifier)"))

"""SUMMARY TABLE"""

summary_df = pd.DataFrame([{k: v for k, v in r.items() if k != "roc"} for r in results])
print("\n==================== SUMMARY ====================")
print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

"""ROC CURVES (Optional)"""

plt.figure()
plotted = False
for r in results:
    fpr, tpr = r["roc"]
    if fpr is not None and tpr is not None:
        plt.plot(fpr, tpr, label=r["Model"])
        plotted = True

if plotted:
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curves (Daily Water Intake)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()
else:
    print("\nROC curves not available (model did not output probabilities).")
