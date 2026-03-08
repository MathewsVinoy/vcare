import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from joblib import dump

df1 = pd.read_csv("data/blood/biased_leukemia_dataset.csv")

print("Gender:", df1['Gender'].unique())
print("Smoking_Status:", df1['Smoking_Status'].unique())
print("Genetic_Mutation:", df1['Genetic_Mutation'].unique())
print("Family_History:", df1['Family_History'].unique())
print("Radiation_Exposure:", df1['Radiation_Exposure'].unique())
print("Infection_History:", df1['Infection_History'].unique())
print("Leukemia_Status:", df1['Leukemia_Status'].unique())

df1['Gender'] = df1['Gender'].map({'Male': 1, 'Female': 0}).fillna(df1['Gender'])
df1['Smoking_Status'] = df1['Smoking_Status'].map({'Yes': 1, 'No': 0}).fillna(0)
df1['Genetic_Mutation'] = df1['Genetic_Mutation'].map({'Yes': 1, 'No': 0}).fillna(0)
df1['Family_History'] = df1['Family_History'].map({'Yes': 1, 'No': 0}).fillna(0)
df1['Radiation_Exposure'] = df1['Radiation_Exposure'].map({'Yes': 1, 'No': 0}).fillna(0)
df1['Infection_History'] = df1['Infection_History'].map({'Yes': 1, 'No': 0}).fillna(0)
df1['Leukemia_Status'] = df1['Leukemia_Status'].map({'Positive': 1, 'Negative': 0}).fillna(0)

selected_columns = [
    'Age',
    'Gender',
    'WBC_Count',
    'RBC_Count',
    'Platelet_Count',
    'Hemoglobin_Level',
    'Bone_Marrow_Blasts',
    'Family_History',
    'Smoking_Status',
    'Radiation_Exposure',
    'BMI',
    'Infection_History'
]

df = df1[selected_columns]
out = df1['Leukemia_Status']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(df, out, test_size=0.2)

# CatBoost Classifier
cat_model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    random_seed=42,
    verbose=0
)

# Train Model
cat_model.fit(X_train, y_train)

# Predict
y_pred = cat_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Create model folder
os.makedirs('model', exist_ok=True)

# Save model
dump(cat_model, 'model/catboost_model.joblib')

print("Model saved to 'model/catboost_model.joblib'")