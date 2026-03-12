from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import os
import json
import numpy as np
from torch import nn
import torchvision
from torchvision import transforms
from PIL import Image
import io


app = Flask(__name__)

# Model configuration
model_name = "microsoft/Phi-3-mini-4k-instruct"
cache_dir = os.path.join(os.path.dirname(__file__), "model_cache")
offload_dir = os.path.join(os.path.dirname(__file__), "offload_dir")
generation_args = {
    "max_new_tokens": 300,
    "temperature": 0.7,
    "do_sample": True,
    "return_full_text": False
}

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Global variables for model and pipeline
model = None
tokenizer = None
pipe = None
blood_model = None
image_model = None
model_error = None

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

def get_vit_model(num_classes=2):
    """Create ViT model architecture"""
    model = torchvision.models.vit_b_16(weights="IMAGENET1K_V1")
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model

def load_image_model():
    """Lazy load the skin cancer detection model when needed"""
    global image_model
    
    if image_model is not None:
        return True
    
    try:
        image_model = get_vit_model(num_classes=2)
        image_model.load_state_dict(torch.load('model/model.pth', map_location=device))
        image_model = image_model.to(device)
        image_model.eval()
        print(f"✅ Skin cancer detection model loaded successfully on {device}!")
        return True
    except FileNotFoundError:
        print("❌ PyTorch model file not found at 'model/model.pth'")
        return False
    except Exception as e:
        print(f"❌ Error loading image model: {e}")
        return False

def load_model():
    """Load the Phi-3 chat model using the same setup as test_llm.py"""
    global model, tokenizer, pipe, model_error

    if model is not None and tokenizer is not None and pipe is not None:
        return True

    try:
        model_error = None
        print("Loading Phi-3 model...")
        print(f"Model: {model_name}")
        print(f"Cache directory: {cache_dir}")

        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(offload_dir, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=False
        )

        quantization_config = None
        max_memory = None

        if torch.cuda.is_available():
            max_memory = {0: "2GiB", "cpu": "12GiB"}
            print(f"Setting GPU budget to {max_memory[0]} and CPU budget to {max_memory['cpu']}")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                max_memory=max_memory,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                cache_dir=cache_dir,
                offload_folder=offload_dir,
                trust_remote_code=False
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="cpu",
                torch_dtype=torch.float32,
                cache_dir=cache_dir,
                offload_folder=offload_dir,
                trust_remote_code=False
            )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )

        print("Model loaded successfully!")
        return True
    except Exception as e:
        model = None
        tokenizer = None
        pipe = None
        model_error = str(e)
        print(f"❌ Error loading chat model: {model_error}")
        return False

def chat(prompt):
    """Generate a response using Phi-3"""
    if pipe is None and not load_model():
        return f"Chat model could not be loaded right now. {model_error or ''}".strip()
    
    messages = [
        {"role": "user", "content": prompt},
    ]

    output = pipe(messages, **generation_args)
    return output[0]['generated_text']

def stream_chat_response(response_text):
    """Yield chat output in SSE format for the frontend."""
    chunk_size = 24
    for i in range(0, len(response_text), chunk_size):
        chunk = response_text[i:i + chunk_size]
        yield f"data: {json.dumps({'token': chunk})}\n\n"
    yield "data: [DONE]\n\n"

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

@app.route('/chat/stream', methods=['POST'])
def chat_stream_endpoint():
    """Handle streaming chat requests for the frontend."""
    try:
        data = request.get_json()
        user_message = data.get('message', '')

        if not user_message.strip():
            return jsonify({'error': 'Empty message'}), 400

        response_text = chat(user_message)

        if response_text.startswith("Chat model could not be loaded right now."):
            def error_stream():
                yield f"data: {json.dumps({'error': response_text})}\n\n"
                yield "data: [DONE]\n\n"

            return Response(
                stream_with_context(error_stream()),
                mimetype='text/event-stream',
                headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
            )

        return Response(
            stream_with_context(stream_chat_response(response_text)),
            mimetype='text/event-stream',
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
        )

    except Exception as e:
        def error_stream():
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

        return Response(
            stream_with_context(error_stream()),
            mimetype='text/event-stream',
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
        )

@app.route('/health')
def health():
    """Check if the model is loaded"""
    return jsonify({
        'status': 'ready' if model is not None else 'loading',
        'model_loaded': model is not None,
        'error': model_error
    })

@app.route('/predict_skin_cancer', methods=['POST'])
def predict_skin_cancer():
    try:
        # Lazy load image model only when needed
        if not load_image_model():
            return jsonify({
                'error': 'Image model not loaded. Please check if model/model.pth exists.',
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
    # Run the Flask app
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
