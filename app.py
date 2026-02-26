from flask import Flask, render_template, request, jsonify
import numpy as np
import torch
from torch import nn
import torchvision
from torchvision import transforms
from PIL import Image
import io
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from config import CHAT_MODEL_CONFIG, FLASK_CONFIG

app = Flask(__name__)

# Local directory for model cache
MODEL_CACHE_DIR = os.path.join(os.path.dirname(__file__), 'model_cache')
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Offload directory for model layers that don't fit in memory
OFFLOAD_DIR = os.path.join(os.path.dirname(__file__), 'model_offload')
os.makedirs(OFFLOAD_DIR, exist_ok=True)

# Load the Random Forest model for blood sample
blood_model = None
blood_model_loaded = False

# Load the PyTorch model for skin cancer detection
image_model = None
image_model_loaded = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the fine-tuned Phi-3 model for chat (lazy loaded on first use)
chat_model = None
chat_tokenizer = None
chat_model_loaded = False
chat_model_loading = False

# Load Random Forest model for blood sample
try:
    import joblib
    blood_model = joblib.load('model/random_forest_model.joblib')
    blood_model_loaded = True
    print("✅ Blood sample model loaded successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install: pip install joblib scikit-learn")
except FileNotFoundError:
    print("❌ Model file not found at 'model/random_forest_model.joblib'")
except Exception as e:
    print(f"❌ Error loading blood model: {e}")

# Load PyTorch Vision Transformer model for skin cancer detection
try:
    def get_vit_model(num_classes=2):
        model = torchvision.models.vit_b_16(weights="IMAGENET1K_V1")
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        return model
    
    image_model = get_vit_model(num_classes=2)
    image_model.load_state_dict(torch.load('model/model.pth', map_location=device))
    image_model = image_model.to(device)
    image_model.eval()
    image_model_loaded = True
    print(f"✅ Skin cancer detection model loaded successfully on {device}!")
except FileNotFoundError:
    print("❌ PyTorch model file not found at 'model/model.pth'")
except Exception as e:
    print(f"❌ Error loading image model: {e}")

# Lazy load function for Phi-3 chat model
def load_chat_model():
    """Load the Phi-3 model on first use to avoid OOM at startup"""
    global chat_model, chat_tokenizer, chat_model_loaded, chat_model_loading
    
    if chat_model_loaded:
        return True
    
    if chat_model_loading:
        return False
    
    chat_model_loading = True
    
    try:
        print("Loading Phi-3 chat model (this may take a minute)...")
        base_model_name = CHAT_MODEL_CONFIG['base_model']
        adapter_path = CHAT_MODEL_CONFIG['adapter_path']
        use_8bit = CHAT_MODEL_CONFIG['use_8bit']
        use_4bit = CHAT_MODEL_CONFIG['use_4bit']
        
        print(f"Using local model cache: {MODEL_CACHE_DIR}")
        
        # Load tokenizer
        chat_tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, 
            trust_remote_code=True,
            cache_dir=MODEL_CACHE_DIR
        )
        
        # Determine optimal loading parameters
        use_cuda = torch.cuda.is_available()
        
        # Prepare quantization config if requested
        quantization_config = None
        if use_4bit and use_8bit:
            print("⚠️ Both 4-bit and 8-bit quantization requested. Using 4-bit (more aggressive).")
            use_8bit = False
        
        if use_4bit:
            try:
                print("Configuring 4-bit quantization (reduces memory by ~8x)...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16 if use_cuda else torch.float32,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
            except Exception as e:
                print(f"⚠️ 4-bit quantization failed: {e}. Install bitsandbytes: pip install bitsandbytes")
                quantization_config = None
        elif use_8bit:
            try:
                print("Configuring 8-bit quantization (reduces memory by ~4x)...")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            except Exception as e:
                print(f"⚠️ 8-bit quantization failed: {e}. Install bitsandbytes: pip install bitsandbytes")
                quantization_config = None
        
        # Load base model with optimizations
        if use_cuda:
            # GPU: Use float16 for efficiency
            chat_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                dtype=torch.float16,
                device_map="auto",
                quantization_config=quantization_config,
                trust_remote_code=True,
                cache_dir=MODEL_CACHE_DIR,
                offload_folder=OFFLOAD_DIR  # Enable disk offloading for layers that don't fit
            )
        else:
            # CPU: Use float32 but with low_cpu_mem_usage for efficiency
            print("Loading on CPU with memory optimization...")
            if quantization_config:
                print("⚠️ Quantization is not well supported on CPU. Loading without quantization.")
                quantization_config = None
            
            chat_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                cache_dir=MODEL_CACHE_DIR,
                device_map="auto",
                offload_folder=OFFLOAD_DIR  # Enable disk offloading for layers that don't fit
            )
        
        # Load LoRA adapter
        print("Loading LoRA adapter...")
        chat_model = PeftModel.from_pretrained(
            chat_model, 
            adapter_path,
            offload_folder=OFFLOAD_DIR  # Enable disk offloading for adapter layers
        )
        
        # Only merge if not quantized (merging with quantized models can cause issues)
        if quantization_config is None:
            chat_model = chat_model.merge_and_unload()  # Merge adapter weights with base model
        
        chat_model.eval()
        chat_model_loaded = True
        chat_model_loading = False
        
        quant_info = ""
        if use_4bit:
            quant_info = " (4-bit quantized)"
        elif use_8bit:
            quant_info = " (8-bit quantized)"
        
        print(f"✅ Phi-3 chat model loaded successfully on {device}{quant_info}!")
        return True
        
    except FileNotFoundError as e:
        print(f"❌ Chat model files not found: {e}")
        print("Please ensure 'phi3_lora_model' directory exists with the adapter files")
        chat_model_loading = False
        return False
    except Exception as e:
        print(f"❌ Error loading chat model: {e}")
        import traceback
        traceback.print_exc()
        chat_model_loading = False
        return False

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
        if not blood_model_loaded or blood_model is None:
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
def chat():
    try:
        # Load model on first use if not already loaded
        if not chat_model_loaded:
            if chat_model_loading:
                return jsonify({
                    'response': 'Chat model is currently loading. Please try again in a moment.',
                    'error': True
                }), 503
            
            if not load_chat_model():
                return jsonify({
                    'response': 'Failed to load chat model. Please check the server logs.',
                    'error': True
                }), 500
        
        if chat_model is None or chat_tokenizer is None:
            return jsonify({
                'response': 'Chat model is not available.',
                'error': True
            }), 500
        
        user_message = request.json.get('message', '')
        
        if not user_message.strip():
            return jsonify({
                'response': 'Please provide a message.',
                'error': True
            }), 400
        
        # Format the input using Phi-3 chat template
        messages = [
            {"role": "user", "content": user_message}
        ]
        
        # Apply chat template
        input_text = chat_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = chat_tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(chat_model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = chat_model.generate(
                **inputs,
                max_new_tokens=CHAT_MODEL_CONFIG['max_new_tokens'],
                temperature=CHAT_MODEL_CONFIG['temperature'],
                top_p=CHAT_MODEL_CONFIG['top_p'],
                do_sample=True,
                pad_token_id=chat_tokenizer.eos_token_id,
                eos_token_id=chat_tokenizer.eos_token_id
            )
        
        # Decode the response
        full_response = chat_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response (remove the input prompt)
        # The response format is typically: <prompt><|assistant|>response
        if "<|assistant|>" in full_response:
            response = full_response.split("<|assistant|>")[-1].strip()
        else:
            # Fallback: remove the input text
            response = full_response[len(input_text):].strip()
        
        return jsonify({
            'response': response,
            'error': False
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'response': f'Sorry, an error occurred: {str(e)}',
            'error': True
        }), 500

@app.route('/predict_skin_cancer', methods=['POST'])
def predict_skin_cancer():
    try:
        if not image_model_loaded or image_model is None:
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
    print("\n" + "="*60)
    print("🏥 Medical Diagnosis Application Starting")
    print("="*60)
    print(f"Blood Sample Model: {'✅ Loaded' if blood_model_loaded else '❌ Not loaded'}")
    print(f"Image Detection Model: {'✅ Loaded' if image_model_loaded else '❌ Not loaded'}")
    print(f"Chat Model: ⏳ Will load on first use (lazy loading)")
    print(f"Device: {device}")
    print(f"\nQuantization Config:")
    print(f"  - 4-bit: {CHAT_MODEL_CONFIG['use_4bit']}")
    print(f"  - 8-bit: {CHAT_MODEL_CONFIG['use_8bit']}")
    print(f"\nTo enable quantization (reduces memory by 4-8x):")
    print(f"  export USE_4BIT_QUANTIZATION=true  # Most aggressive")
    print(f"  export USE_8BIT_QUANTIZATION=true  # Balanced")
    print("="*60 + "\n")
    
    app.run(
        debug=FLASK_CONFIG['debug'],
        host=FLASK_CONFIG['host'],
        port=FLASK_CONFIG['port']
    )
