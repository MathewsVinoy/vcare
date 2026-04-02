"""
Advanced Blood Sample Classification with CatBoost
Trains a CatBoost binary classifier for blood cancer risk

Features (16): Gender, Age, Hb, RBC, WBC, PLATELETS, LYMP, MONO, HCT, MCV, MCH, MCHC, RDW, PDW, MPV, PCT
Target: Cancer Risk (0 = Non-cancer, 1 = Cancer)

Run: python train_script/blood_sample.py
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from catboost import CatBoostClassifier
from joblib import dump
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Load Dataset
# ─────────────────────────────────────────────
print("=" * 60)
print("  Advanced Blood Sample Analysis — Training Pipeline")
print("=" * 60)

df1 = pd.read_csv("data/blood/BDCBC7196_Hematology_Dataset.csv")

print(f"\n📊 Dataset loaded: {df1.shape[0]} samples, {df1.shape[1]} features")
print(f"Columns: {list(df1.columns)}")
print(f"\nTarget distribution:\n{df1['Diagnosis'].value_counts()}")

# ─────────────────────────────────────────────
# Feature Definition
# ─────────────────────────────────────────────
FEATURE_NAMES = [
    'Gender', 'Age', 'Hb', 'RBC', 'WBC', 'PLATELETS', 
    'LYMP', 'MONO', 'HCT', 'MCV', 'MCH', 'MCHC', 
    'RDW', 'PDW', 'MPV', 'PCT'
]

# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────
print("\n🔄 Preprocessing data...")

# Extract features
X = df1.drop("Diagnosis", axis=1)

# Build binary target from diagnosis labels
# 1 = Blood cancer classes, 0 = Non-cancer classes
CANCER_LABELS = {"Chronic Leukemias", "Polycythemia Vera"}
y = df1["Diagnosis"].apply(lambda d: 1 if str(d).strip() in CANCER_LABELS else 0)

# Verify feature names
if list(X.columns) != FEATURE_NAMES:
    print(f"⚠️  Warning: Feature names mismatch")
    print(f"Expected: {FEATURE_NAMES}")
    print(f"Got: {list(X.columns)}")
else:
    print(f"✓ All {len(FEATURE_NAMES)} features verified")

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print("\nBinary target distribution:")
print(f"  • Non-cancer (0): {(y == 0).sum()}")
print(f"  • Cancer (1): {(y == 1).sum()}")
print(f"  • Cancer labels: {sorted(CANCER_LABELS)}")

# ─────────────────────────────────────────────
# Train-Test Split
# ─────────────────────────────────────────────
print("\n🔀 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Training target distribution:\n{pd.Series(y_train).value_counts()}")

# ─────────────────────────────────────────────
# CatBoost Classifier (Binary)
# ─────────────────────────────────────────────
print("\n🤖 Training CatBoost binary model...")

cat_model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    random_seed=42,
    verbose=0,
    loss_function='Logloss',
    eval_metric='AUC',
    auto_class_weights='Balanced'
)

# Train Model
cat_model.fit(X_train, y_train, verbose=False)

# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
print("\n📈 Evaluating model...")

y_pred = cat_model.predict(X_test).astype(int).flatten()
y_pred_proba = cat_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_pred_proba)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print("\nConfusion Matrix [ [TN, FP], [FN, TP] ]:")
print(cm)

# Feature importance
print("\n🎯 Top Features:")
feature_importance = cat_model.feature_importances_
feature_names = X_train.columns
top_features = sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)[:5]
for name, importance in top_features:
    print(f"  • {name}: {importance:.4f}")

# ─────────────────────────────────────────────
# Save Model
# ─────────────────────────────────────────────
print("\n💾 Saving model...")

os.makedirs('model', exist_ok=True)

model_path = 'model/catboost_model.joblib'
dump(cat_model, model_path)

print(f"✅ Model saved to '{model_path}'")
print(f"   Model type: {type(cat_model).__name__}")
print(f"   Model size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")

# Save model metadata
meta_path = 'model/blood_model_meta.joblib'
dump({'task': 'binary_cancer_classification', 'cancer_labels': sorted(CANCER_LABELS)}, meta_path)
print(f"✅ Model metadata saved to '{meta_path}'")

print("\n" + "=" * 60)
print("  ✨ Training Complete!")
print("=" * 60)
