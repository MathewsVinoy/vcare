"""Skin Cancer Detection Image Model Module"""

import torch
from torch import nn
import torchvision
from torchvision import transforms
from PIL import Image
import io
import os
import timm

class ImageModel:
    """Load and manage the Vision Transformer skin cancer detection model"""
    
    def __init__(self, model_path=None, device=None):
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), "../model/best_efficientformer_model.pth")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.error = None
        
        self.CLASS_NAMES = ["mel", "bcc", "akiec", "nv", "bkl", "df", "vasc"]
        self.HIGH_RISK_CLASSES = {0, 1, 2}
        self.MODERATE_RISK_CLASSES = {6}
        self.LOW_RISK_CLASSES = {3, 4, 5}
        self.NUM_CLASSES = 7
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    
    def _create_vit_model(self, num_classes=7):
        """Construct an efficientformer_l1 matching the training script."""
        model = timm.create_model(
            "efficientformer_l1",
            pretrained=False,
            num_classes=num_classes,
            drop_rate=0.4,
            drop_path_rate=0.2
        )
        return model
    
    def load(self):
        """Lazy load the skin cancer detection model"""
        if self.model is not None:
            return True
        
        try:
            self.model = self._create_vit_model(num_classes=self.NUM_CLASSES)
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"✅ Skin cancer detection model loaded successfully on {self.device}!")
            return True
        
        except FileNotFoundError:
            self.error = f"PyTorch model file not found at '{self.model_path}'"
            print(f"❌ {self.error}")
            return False
        
        except Exception as e:
            self.error = str(e)
            print(f"❌ Error loading image model: {e}")
            return False
    
    def predict(self, image_bytes):
        """
        Predict skin cancer from image bytes
        """
        if not self.load():
            return {
                'error': f'Image model not loaded. {self.error}',
                'success': False
            }
        
        try:
            # Read and preprocess the image
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                prediction = torch.argmax(probabilities).item()
                predicted_name = self.CLASS_NAMES[prediction]
                
                # Aggregate risk from all 7 class probabilities
                cancer_probability = float(sum(probabilities[c].item() for c in self.HIGH_RISK_CLASSES)) * 100
                moderate_risk_prob = float(sum(probabilities[c].item() for c in self.MODERATE_RISK_CLASSES)) * 100
            
            # Generate diagnosis based on prediction
            if cancer_probability > 50:
                diagnosis = "High Risk — Skin Cancer Suspected"
                description = "The analysis identifies features strongly associated with malignant skin lesions (e.g. melanoma, basal cell carcinoma, or actinic keratosis): irregular borders, asymmetry, and colour variation. <strong>Please consult a dermatologist immediately.</strong>"
            elif cancer_probability > 25 or moderate_risk_prob > 40:
                diagnosis = "Moderate Risk — Further Examination Needed"
                description = "The lesion shows some concerning characteristics. Schedule a professional dermatological evaluation to rule out malignancy."
            else:
                diagnosis = "Low Risk — Likely Benign"
                description = "The lesion appears consistent with a benign skin condition (e.g. common nevus, benign keratosis, or dermatofibroma). Continue routine self-monitoring and see a dermatologist if it changes."
            
            return {
                'probability': float(cancer_probability),
                'diagnosis': diagnosis,
                'description': description,
                'prediction': int(prediction),
                'predicted_condition': predicted_name,
                'success': True
            }
        
        except Exception as e:
            return {
                'error': f'Error processing image: {str(e)}',
                'success': False
            }
    
    def is_loaded(self):
        """Check if model is loaded"""
        return self.model is not None
