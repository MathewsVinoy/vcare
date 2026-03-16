"""
Standalone Flask server for the Blood Sample Analysis feature.
Run from the project root:  python test-script/bloodsample_test.py
or from inside test-script/: python bloodsample_test.py
"""

import os
import json

import numpy as np
from joblib import load
from flask import Flask, render_template, request, jsonify

# ─────────────────────────────────────────────
# Paths – resolve relative to this file so the
# script works from any working directory.
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODEL_PATH = os.path.join(BASE_DIR, "model", "catboost_model.joblib")

# Fallback to random forest model if catboost not found
RANDOM_FOREST_MODEL_PATH = os.path.join(BASE_DIR, "model", "random_forest_model.joblib")

# ─────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR,
)

# ─────────────────────────────────────────────
# Model – CatBoost trained for blood
# sample leukemia classification (0 = Negative, 1 = Positive)
# ─────────────────────────────────────────────
blood_model = None
model_type = None  # 'catboost' or 'random_forest'

# Feature names must match training data
FEATURE_NAMES = [
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


def load_blood_model() -> bool:
    """Lazy-load the blood sample analysis model. Returns True on success."""
    global blood_model, model_type

    if blood_model is not None:
        return True

    # Try catboost model first
    if os.path.exists(MODEL_PATH):
        try:
            blood_model = load(MODEL_PATH)
            model_type = 'catboost'
            print(f"[INFO] ✅ CatBoost model loaded from {MODEL_PATH}")
            return True
        except Exception as exc:
            print(f"[WARNING] Failed to load CatBoost model: {exc}")

    # Fallback to random forest model
    if os.path.exists(RANDOM_FOREST_MODEL_PATH):
        try:
            blood_model = load(RANDOM_FOREST_MODEL_PATH)
            model_type = 'random_forest'
            print(f"[INFO] ✅ Random Forest model loaded from {RANDOM_FOREST_MODEL_PATH}")
            return True
        except Exception as exc:
            print(f"[ERROR] Failed to load Random Forest model: {exc}")
    
    print(f"[ERROR] No model found at {MODEL_PATH} or {RANDOM_FOREST_MODEL_PATH}")
    return False


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
@app.route("/blood_sample")
def blood_sample():
    """Render the Blood Sample Analysis UI."""
    return render_template("blood_sample.html")


@app.route("/predict_blood_sample", methods=["POST"])
def predict_blood_sample():
    """
    Accepts JSON POST with blood test parameters.
    Returns JSON: { success, raw_prediction, probability, prediction, confidence }
    
    Expected JSON fields:
    - age: float
    - gender: int (0 or 1)
    - bmi: float
    - wbc_count: float
    - rbc_count: float
    - platelet_count: float
    - hemoglobin_level: float
    - bone_marrow_blasts: float
    - family_history: int (0 or 1)
    - smoking_status: int (0 or 1)
    - radiation_exposure: int (0 or 1)
    - infection_history: int (0 or 1)
    """
    # Ensure model is loaded
    if not load_blood_model():
        return jsonify({
            "success": False,
            "error": "Model could not be loaded. Check that model/catboost_model.joblib or model/random_forest_model.joblib exists.",
        }), 500

    # Parse JSON request
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided."}), 400

        # Extract and validate fields
        try:
            age = float(data.get('age', 0))
            gender = int(data.get('gender', 0))
            bmi = float(data.get('bmi', 0))
            wbc_count = float(data.get('wbc_count', 0))
            rbc_count = float(data.get('rbc_count', 0))
            platelet_count = float(data.get('platelet_count', 0))
            hemoglobin_level = float(data.get('hemoglobin_level', 0))
            bone_marrow_blasts = float(data.get('bone_marrow_blasts', 0))
            family_history = int(data.get('family_history', 0))
            smoking_status = int(data.get('smoking_status', 0))
            radiation_exposure = int(data.get('radiation_exposure', 0))
            infection_history = int(data.get('infection_history', 0))
        except (TypeError, ValueError) as exc:
            return jsonify({
                "success": False,
                "error": f"Invalid data types: {exc}"
            }), 400

        # Build feature vector in the same order as FEATURE_NAMES
        feature_vector = np.array([[
            age,
            gender,
            wbc_count,
            rbc_count,
            platelet_count,
            hemoglobin_level,
            bone_marrow_blasts,
            family_history,
            smoking_status,
            radiation_exposure,
            bmi,
            infection_history
        ]])

        # ── Debug logging ──────────────────────────────────────
        print("[PREDICT] ──────────────────────────────────────")
        print(f"[PREDICT] Model Type     : {model_type}")
        print(f"[PREDICT] Age            : {age}")
        print(f"[PREDICT] Gender         : {gender}")
        print(f"[PREDICT] BMI            : {bmi}")
        print(f"[PREDICT] WBC Count      : {wbc_count}")
        print(f"[PREDICT] RBC Count      : {rbc_count}")
        print(f"[PREDICT] Platelet Count : {platelet_count}")
        print(f"[PREDICT] Hemoglobin     : {hemoglobin_level}")
        print(f"[PREDICT] Bone Marrow    : {bone_marrow_blasts}%")
        print(f"[PREDICT] Family History : {family_history}")
        print(f"[PREDICT] Smoking Status : {smoking_status}")
        print(f"[PREDICT] Radiation Exp. : {radiation_exposure}")
        print(f"[PREDICT] Infection Hist : {infection_history}")

        # Make prediction
        try:
            # Get prediction (0 or 1)
            raw_prediction = int(blood_model.predict(feature_vector)[0])
            
            # Get probability if available
            probability = None
            try:
                proba = blood_model.predict_proba(feature_vector)
                # proba is [P(negative), P(positive)] for binary classification
                # Get probability of the predicted class
                probability = float(proba[0][raw_prediction])
            except (AttributeError, IndexError):
                # Model doesn't have predict_proba or structure is different
                probability = None
            
            print(f"[PREDICT] Raw Prediction : {raw_prediction}")
            print(f"[PREDICT] Probability    : {probability}")
            print("[PREDICT] ──────────────────────────────────────")

            # Build human-readable prediction
            if raw_prediction == 1:
                prediction = "⚠️ Leukemia Risk — POSITIVE"
                confidence = "High Risk Detected"
            else:
                prediction = "✓ Normal — NEGATIVE"
                confidence = "Low Risk Detected"

            return jsonify({
                "success": True,
                "raw_prediction": raw_prediction,
                "probability": probability,
                "prediction": prediction,
                "confidence": confidence,
            })

        except Exception as exc:
            print(f"[ERROR] Prediction failed: {exc}")
            return jsonify({
                "success": False,
                "error": f"Prediction error: {exc}"
            }), 500

    except Exception as exc:
        return jsonify({
            "success": False,
            "error": f"Request processing error: {exc}"
        }), 500


@app.route("/health")
def health():
    """Quick health-check endpoint."""
    return jsonify({
        "status": "ok",
        "model_loaded": blood_model is not None,
        "model_type": model_type,
    })


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  VCare AI — Blood Sample Analysis Server")
    print("  UI  →  http://localhost:5002/blood_sample")
    print("=" * 55)

    # Pre-load model at startup so the first request is fast
    load_blood_model()

    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5002)
