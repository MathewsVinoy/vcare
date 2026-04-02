"""Blood Sample Disease Prediction Model Module — Advanced Version

Features (16):
Gender, Age, Hb, RBC, WBC, PLATELETS, LYMP, MONO, HCT, MCV, MCH, MCHC, RDW, PDW, MPV, PCT
"""

import os
import numpy as np
from pathlib import Path


class BloodModel:
    """Advanced blood sample prediction model with CatBoost support
    
    Input Features (16):
    - Gender: 0 or 1
    - Age: years
    - Hb: Hemoglobin (g/dL)
    - RBC: Red Blood Cells (M cells/µL)
    - WBC: White Blood Cells (cells/µL)
    - PLATELETS: Platelet count (cells/µL)
    - LYMP: Lymphocytes (%)
    - MONO: Monocytes (%)
    - HCT: Hematocrit (%)
    - MCV: Mean Corpuscular Volume (fL)
    - MCH: Mean Corpuscular Hemoglobin (pg)
    - MCHC: Mean Corpuscular Hemoglobin Concentration (g/dL)
    - RDW: Red Distribution Width (%)
    - PDW: Platelet Distribution Width (%)
    - MPV: Mean Platelet Volume (fL)
    - PCT: Plateletcrit (%)
    """
    
    FEATURE_NAMES = [
        'Gender', 'Age', 'Hb', 'RBC', 'WBC', 'PLATELETS',
        'LYMP', 'MONO', 'HCT', 'MCV', 'MCH', 'MCHC',
        'RDW', 'PDW', 'MPV', 'PCT'
    ]
    
    def __init__(self, model_path=None):
        """
        Initialize Blood Model
        
        Args:
            model_path: Path to model file. If None, tries CatBoost first, then Random Forest
        """
        self.catboost_path = os.path.join(
            os.path.dirname(__file__), 
            "../model/catboost_model.joblib"
        )
        self.random_forest_path = os.path.join(
            os.path.dirname(__file__), 
            "../model/random_forest_model.joblib"
        )
        self.model_path = model_path
        self.model = None
        self.model_type = None
        self.error = None
    
    def load(self):
        """Lazy load the blood sample model — tries CatBoost first, then Random Forest"""
        if self.model is not None:
            return True
        
        try:
            import joblib
            
            # Try CatBoost model first
            if os.path.exists(self.catboost_path):
                try:
                    self.model = joblib.load(self.catboost_path)
                    self.model_type = 'catboost'
                    print("✅ CatBoost model loaded successfully!")
                    return True
                except Exception as e:
                    print(f"⚠️  CatBoost model failed: {e}")
                    self.error = str(e)
            
            # Fallback to Random Forest
            if os.path.exists(self.random_forest_path):
                try:
                    self.model = joblib.load(self.random_forest_path)
                    self.model_type = 'random_forest'
                    print("✅ Random Forest model loaded successfully!")
                    return True
                except Exception as e:
                    print(f"⚠️  Random Forest model failed: {e}")
                    self.error = str(e)
            
            # Neither model exists
            self.error = f"No models found. CatBoost: {self.catboost_path}, Random Forest: {self.random_forest_path}"
            print(f"❌ {self.error}")
            return False
        
        except ImportError as e:
            self.error = str(e)
            print(f"❌ Import error: {e}")
            print("Please install: pip install joblib scikit-learn catboost")
            return False
        
        except Exception as e:
            self.error = str(e)
            print(f"❌ Error loading blood model: {e}")
            return False
    
    def predict(self, features_array):
        """
        Make prediction on blood sample features with advanced analysis
        
        Args:
            features_array: numpy array of shape (1, 16) with features in order:
                [Gender, Age, Hb, RBC, WBC, PLATELETS, LYMP, MONO,
                 HCT, MCV, MCH, MCHC, RDW, PDW, MPV, PCT]
        
        Returns:
            dict with comprehensive prediction results
        """
        if not self.load():
            return {
                'error': f'Model not loaded. {self.error}',
                'success': False,
                'raw_prediction': None,
                'probability': None
            }
        
        try:
            # Ensure input is correct shape
            if features_array.shape != (1, 16):
                raise ValueError(f"Expected shape (1, 16), got {features_array.shape}")
            
            # Make binary prediction
            raw_model_pred = self.model.predict(features_array)[0]
            predicted_class = int(float(raw_model_pred))

            # Get probability if available (positive class probability)
            probability = None
            positive_probability = None
            try:
                probabilities_array = self.model.predict_proba(features_array)[0]
                if len(probabilities_array) >= 2:
                    positive_probability = float(probabilities_array[1])
                    probability = positive_probability
                else:
                    probability = float(probabilities_array[predicted_class])
            except (AttributeError, IndexError, TypeError):
                # Model doesn't support predict_proba or different structure
                probability = None

            # Generate human-readable prediction
            if predicted_class == 1:
                diagnosis = "⚠️ Blood Cancer Risk — POSITIVE"
                confidence_text = "High Risk Detected"
            else:
                diagnosis = "✓ Blood Cancer Risk — NEGATIVE"
                confidence_text = "Low Risk Detected"
            
            return {
                'prediction': diagnosis,
                'confidence': confidence_text,
                'probability': probability,
                'probability_positive': positive_probability,
                'raw_prediction': predicted_class,
                'predicted_class': predicted_class,
                'model_type': self.model_type,
                'success': True
            }
        
        except Exception as e:
            return {
                'error': f'Prediction error: {str(e)}',
                'success': False,
                'raw_prediction': None,
                'probability': None,
                'model_type': self.model_type
            }
    
    def predict_batch(self, features_batch):
        """
        Make predictions on multiple samples
        
        Args:
            features_batch: numpy array of shape (N, 12)
        
        Returns:
            dict with batch predictions
        """
        if not self.load():
            return {
                'error': f'Model not loaded. {self.error}',
                'success': False
            }
        
        try:
            predictions = self.model.predict(features_batch)
            
            probabilities = None
            try:
                probabilities = self.model.predict_proba(features_batch)
            except (AttributeError, IndexError):
                pass
            
            return {
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist() if probabilities is not None else None,
                'success': True,
                'model_type': self.model_type
            }
        
        except Exception as e:
            return {
                'error': str(e),
                'success': False
            }
    
    def get_feature_importance(self):
        """Get feature importance if available"""
        if not self.load():
            return None
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                return {
                    'feature_names': [
                        'Age', 'Gender', 'WBC Count', 'RBC Count', 'Platelet Count',
                        'Hemoglobin Level', 'Bone Marrow Blasts', 'Family History',
                        'Smoking Status', 'Radiation Exposure', 'BMI', 'Infection History'
                    ],
                    'importances': self.model.feature_importances_.tolist(),
                    'model_type': self.model_type
                }
        except Exception as e:
            print(f"Could not get feature importance: {e}")
        
        return None
    
    def is_loaded(self):
        """Check if model is loaded"""
        return self.model is not None
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if not self.is_loaded():
            return None
        
        return {
            'model_type': self.model_type,
            'model_class': self.model.__class__.__name__,
            'has_predict_proba': hasattr(self.model, 'predict_proba'),
            'has_feature_importances': hasattr(self.model, 'feature_importances_')
        }

