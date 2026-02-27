from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import os

app = Flask(__name__)

# Model configuration
model_name = "microsoft/Phi-3-mini-4k-instruct"
lora_adapter_path = os.path.join(os.path.dirname(__file__), "phi3_lora_model")
cache_dir = os.path.join(os.path.dirname(__file__), "model_cache")

# Global variables for model and pipeline
model = None
tokenizer = None
pipe = None

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
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=False,
        cache_dir=cache_dir,
        attn_implementation="eager",
        low_cpu_mem_usage=True
    )
    
    # Load LoRA adapter weights on top of base model
    model = PeftModel.from_pretrained(
        base_model,
        lora_adapter_path,
        device_map="auto"
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
    """Render the main page"""
    return render_template('index.html')

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

if __name__ == '__main__':
    # Load model before starting the server
    load_model()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
