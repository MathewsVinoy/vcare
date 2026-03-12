"""Skin Cancer Detection Image Model Module"""

import torch
from torch import nn
import torchvision
from torchvision import transforms
from PIL import Image
import io
import os


class ImageModel:
    """Load and manage the Vision Transformer skin cancer detection model"""
    
    def __init__(self, model_path=None, device=None):
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), "../model/model.pth")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.error = None
    
    def _create_vit_model(self, num_classes=2):
        """Create ViT model architecture"""
        model = torchvision.models.vit_b_16(weights="IMAGENET1K_V1")
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        return model
    
    def load(self):
        """Lazy load the skin cancer detection model"""
        if self.model is not None:
            return True
        
        try:
            self.model = self._create_vit_model(num_classes=2)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
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
        
        Args:
            image_bytes: raw image bytes
        
        Returns:
            dict with probability, diagnosis, description, and prediction
        """
        if not self.load():
            return {
                'error': f'Image model not loaded. {self.error}',
                'success': False
            }
        
        try:
            # Read and preprocess the image
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Apply transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                prediction = torch.argmax(probabilities).item()
                cancer_probability = probabilities[1].item() * 100  # Class 1 is cancer
            
            # Generate diagnosis based on prediction
            if cancer_probability > 70:
                diagnosis = 'High Risk - Skin Cancer Suspected'
                description = 'The analysis indicates characteristics commonly associated with skin cancer (melanoma, basal cell carcinoma, or actinic keratosis), including irregular borders, asymmetry, and color variations. Immediate consultation with a dermatologist is strongly recommended.'
            elif cancer_probability > 40:
                diagnosis = 'Moderate Risk - Further Examination Needed'
                description = 'The lesion shows some concerning features that warrant professional evaluation. Schedule an appointment with a dermatologist for a thorough examination.'
            else:
                diagnosis = 'Low Risk - Likely Benign'
                description = 'The lesion appears to have characteristics of a benign skin condition (nevus, benign keratosis, dermatofibroma, or vascular lesion). However, regular monitoring and professional evaluation are still recommended as a precautionary measure.'
            
            return {
                'probability': float(cancer_probability),
                'diagnosis': diagnosis,
                'description': description,
                'prediction': int(prediction),
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
