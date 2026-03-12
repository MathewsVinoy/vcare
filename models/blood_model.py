"""Blood Sample Disease Prediction Model Module"""

import os
import numpy as np


class BloodModel:
    """Load and manage the blood sample prediction model"""
    
    def __init__(self, model_path=None):
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), "../model/random_forest_model.joblib")
        self.model = None
        self.error = None
    
    def load(self):
        """Lazy load the blood sample model"""
        if self.model is not None:
            return True
        
        try:
            import joblib
            self.model = joblib.load(self.model_path)
            print("✅ Blood sample model loaded successfully!")
            return True
        
        except ImportError as e:
            self.error = str(e)
            print(f"❌ Import error: {e}")
            print("Please install: pip install joblib scikit-learn")
            return False
        
        except FileNotFoundError:
            self.error = f"Model file not found at '{self.model_path}'"
            print(f"❌ {self.error}")
            return False
        
        except Exception as e:
            self.error = str(e)
            print(f"❌ Error loading blood model: {e}")
            return False
    
    def predict(self, features_array):
        """
        Make prediction on blood sample features
        
        Args:
            features_array: numpy array of shape (1, 12) with features:
                [age, gender, wbc_count, rbc_count, platelet_count, hemoglobin_level,
                 bone_marrow_blasts, family_history, smoking_status, radiation_exposure, bmi, infection_history]
        
        Returns:
            dict with prediction, probability, and diagnosis
        """
        if not self.load():
            return {
                'error': f'Model not loaded. {self.error}',
                'success': False
            }
        
        try:
            # Make prediction
            prediction = int(self.model.predict(features_array)[0])
            
            # Get probability if available
            try:
                probabilities = self.model.predict_proba(features_array)[0]
                confidence = float(max(probabilities))
            except:
                confidence = None
            
            # Map prediction to diagnosis
            # 0 = Negative/Healthy, 1 = Positive/Leukemia
            diagnosis = 'Leukemia Detected' if prediction == 1 else 'Healthy - No Leukemia Detected'
            
            return {
                'prediction': diagnosis,
                'probability': confidence,
                'raw_prediction': prediction,
                'success': True
            }
        
        except Exception as e:
            return {
                'error': f'Prediction error: {str(e)}',
                'success': False
            }
    
    def is_loaded(self):
        """Check if model is loaded"""
        return self.model is not None
