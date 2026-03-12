from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import os
import numpy as np
import timm
from torchvision import transforms
from PIL import Image
import io


app = Flask(__name__)

# Model configuration
model_name = "microsoft/Phi-3-mini-4k-instruct"
lora_adapter_path = os.path.join(os.path.dirname(__file__), "phi3_lora_model")
cache_dir = os.path.join(os.path.dirname(__file__), "model_cache")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Global variables for model and pipeline
model = None
tokenizer = None
pipe = None
blood_model = None
image_model = None

def load_blood_model():
    """Lazy load the blood sample model when needed"""
    global blood_model
    
    if blood_model is not None:
        return True
    
    try:
        import joblib
        blood_model = joblib.load('model/random_forest_model.joblib')
        print("✅ Blood sample model loaded successfully!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install: pip install joblib scikit-learn")
        return False
    except FileNotFoundError:
        print("❌ Model file not found at 'model/random_forest_model.joblib'")
        return False
    except Exception as e:
        print(f"❌ Error loading blood model: {e}")
        return False

def get_skin_cancer_model(num_classes=2):
    """Create EfficientFormer model architecture (same as training script)"""
    model = timm.create_model(
        "efficientformer_l1",
        pretrained=False,
        num_classes=num_classes
    )
    return model

def load_image_model():
    """Lazy load the skin cancer detection model when needed"""
    global image_model
    
    if image_model is not None:
        return True
    
    try:
        image_model = get_skin_cancer_model(num_classes=2)
        image_model.load_state_dict(torch.load('model/efficientformer_model.pth', map_location=device))
        image_model = image_model.to(device)
        image_model.eval()
        print(f"✅ Skin cancer detection model loaded successfully on {device}!")
        return True
    except FileNotFoundError:
        print("❌ PyTorch model file not found at 'model/efficientformer_model.pth'")
        return False
    except Exception as e:
        print(f"❌ Error loading image model: {e}")
        return False

def load_model():
    """Load the Phi-3 model with LoRA adapter"""
    global model, tokenizer, pipe
    
    print("Loading Phi-3 base model with cancer diagnosis fine-tuning...")
    print(f"Base Model: {model_name}")
    print(f"LoRA Adapter: {lora_adapter_path}")
    print(f"Cache directory: {cache_dir}")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load tokenizer from the fine-tuned model directory
    tokenizer = AutoTokenizer.from_pretrained(
        lora_adapter_path,
        trust_remote_code=False
    )
    
    # Ensure offload directory exists
    offload_dir = os.path.join(os.path.dirname(__file__), "offload_dir")
    os.makedirs(offload_dir, exist_ok=True)
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=False,
        cache_dir=cache_dir,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        device_map="auto",
        offload_folder=offload_dir
    )
    
    # Load LoRA adapter weights on top of base model
    model = PeftModel.from_pretrained(
        base_model,
        lora_adapter_path,
        device_map="auto",
        offload_folder=offload_dir
    )
    
    print(f"Fine-tuned model loaded successfully on device: {model.device}")
    
    # Create a text generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

def chat(prompt):
    """Generate a response using Phi-3"""
    if pipe is None:
        return "Model not loaded. Please wait..."
    
    messages = [
        {"role": "user", "content": prompt},
    ]
    
    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.7,
        "do_sample": True,
    }
    
    output = pipe(messages, **generation_args)
    return output[0]['generated_text']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/blood_sample')
def blood_sample():
    return render_template('blood_sample.html')

@app.route('/image_detection')
def image_detection():
    return render_template('image_detection.html')

@app.route('/predict_blood_sample', methods=['POST'])
def predict_blood_sample():
    try:
        # Lazy load blood model only when needed
        if not load_blood_model():
            return jsonify({
                'error': 'Model not loaded. Please install dependencies: pip install joblib scikit-learn',
                'success': False
            }), 500
            
        data = request.json
        
        # Extract features in the correct order (as per the trained model)
        features = [
            float(data['age']),
            int(data['gender']),
            float(data['wbc_count']),
            float(data['rbc_count']),
            float(data['platelet_count']),
            float(data['hemoglobin_level']),
            float(data['bone_marrow_blasts']),
            int(data['family_history']),
            int(data['smoking_status']),
            int(data['radiation_exposure']),
            float(data['bmi']),
            int(data['infection_history'])
        ]
        
        # Convert to numpy array with shape (1, 12)
        features_array = np.array([features])
        
        # Make prediction
        prediction = int(blood_model.predict(features_array)[0])
        
        # Get probability if available
        try:
            probabilities = blood_model.predict_proba(features_array)[0]
            confidence = float(max(probabilities))
        except:
            confidence = None
        
        # Map prediction to diagnosis
        # 0 = Negative/Healthy, 1 = Positive/Leukemia
        diagnosis = 'Leukemia Detected' if prediction == 1 else 'Healthy - No Leukemia Detected'
        
        return jsonify({
            'prediction': diagnosis,
            'probability': confidence,
            'raw_prediction': prediction,
            'success': True
        })
            
    except KeyError as e:
        return jsonify({
            'error': f'Missing required field: {str(e)}',
            'success': False
        }), 400
    except ValueError as e:
        return jsonify({
            'error': f'Invalid value provided: {str(e)}',
            'success': False
        }), 400
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}',
            'success': False
        }), 500

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """Handle chat requests"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message.strip():
            return jsonify({'error': 'Empty message'}), 400
        
        # Generate response
        response = chat(user_message)
        
        return jsonify({
            'response': response,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/health')
def health():
    """Check if the model is loaded"""
    return jsonify({
        'status': 'ready' if model is not None else 'loading',
        'model_loaded': model is not None
    })

@app.route('/predict_skin_cancer', methods=['POST'])
def predict_skin_cancer():
    try:
        # Lazy load image model only when needed
        if not load_image_model():
            return jsonify({
                'error': 'Image model not loaded. Please check if model/efficientformer_model.pth exists.',
                'success': False
            }), 500
            
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read and preprocess the image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Apply the same transforms used during training
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = image_model(image_tensor)
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
        
        return jsonify({
            'probability': float(cancer_probability),
            'diagnosis': diagnosis,
            'description': description,
            'prediction': int(prediction),
            'success': True
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Error processing image: {str(e)}',
            'success': False
        }), 500

if __name__ == '__main__':
    # Load model before starting the server
    load_model()
    
    # Run the Flask app
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
