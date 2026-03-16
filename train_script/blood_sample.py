"""
Advanced Blood Sample Classification with CatBoost
Trains a CatBoost classifier for blood disease diagnosis

Features (16): Gender, Age, Hb, RBC, WBC, PLATELETS, LYMP, MONO, HCT, MCV, MCH, MCHC, RDW, PDW, MPV, PCT
Target: Diagnosis (disease type classification)

Run: python train_script/blood_sample.py
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
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

# Extract features and target
X = df1.drop("Diagnosis", axis=1)
y = df1["Diagnosis"]

# Verify feature names
if list(X.columns) != FEATURE_NAMES:
    print(f"⚠️  Warning: Feature names mismatch")
    print(f"Expected: {FEATURE_NAMES}")
    print(f"Got: {list(X.columns)}")
else:
    print(f"✓ All {len(FEATURE_NAMES)} features verified")

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\nTarget classes ({len(le.classes_)}):")
for cls, enc in zip(le.classes_, le.transform(le.classes_)):
    print(f"  • {cls}: {enc}")

# ─────────────────────────────────────────────
# Train-Test Split
# ─────────────────────────────────────────────
print("\n🔀 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Training target distribution:\n{pd.Series(y_train).value_counts()}")

# ─────────────────────────────────────────────
# CatBoost Classifier
# ─────────────────────────────────────────────
print("\n🤖 Training CatBoost model...")

cat_model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    random_seed=42,
    verbose=0,
    loss_function='Logloss',  # Binary classification
    eval_metric='AUC'
)

# Train Model
cat_model.fit(X_train, y_train, verbose=False)

# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
print("\n📈 Evaluating model...")

y_pred = cat_model.predict(X_test)
y_pred_proba = cat_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

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

# Save label encoder
encoder_path = 'model/label_encoder.joblib'
dump(le, encoder_path)
print(f"✅ Label encoder saved to '{encoder_path}'")

print("\n" + "=" * 60)
print("  ✨ Training Complete!")
print("=" * 60)
