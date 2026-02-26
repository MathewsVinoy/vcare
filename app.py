from flask import Flask, render_template, request, jsonify, session
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None

def load_model():
    """Load the fine-tuned Phi-3 model with LoRA adapters"""
    global model, tokenizer, device
    
    print("Loading model...")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Base model name from the fine-tuning notebook
    base_model_name = "microsoft/Phi-3-mini-4k-instruct"
    adapter_path = "./phi3_lora_model"
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        
        # Load LoRA adapters
        if os.path.exists(adapter_path):
            print(f"Loading LoRA adapters from {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()  # Merge LoRA weights with base model
        
        # Move to device if CPU
        if device == "cpu":
            model = model.to(device)
        
        model.eval()
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using fallback mode...")
        model = None
        tokenizer = None

def generate_response(user_message, conversation_history=None):
    """Generate a response from the model"""
    if model is None or tokenizer is None:
        return "Model is not loaded. Please check the server logs."
    
    try:
        # Prepare conversation history
        if conversation_history is None:
            conversation_history = []
        
        # Add system message for cancer-related context
        messages = [
            {"role": "system", "content": "You are a helpful medical AI assistant specializing in cancer information. Provide accurate, evidence-based information about cancer, its diagnosis, treatment, and prevention. Always remind users to consult healthcare professionals for medical advice."}
        ]
        
        # Add conversation history
        messages.extend(conversation_history)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        # Apply chat template
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        # The response typically includes the full conversation, so we extract the last part
        if "<|assistant|>" in full_response:
            response = full_response.split("<|assistant|>")[-1].strip()
        else:
            response = full_response.split(user_message)[-1].strip()
        
        return response
        
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"I encountered an error processing your request. Please try again."

@app.route('/')
def index():
    """Render the main chat interface"""
    session['conversation'] = []
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Get conversation history from session
        conversation = session.get('conversation', [])
        
        # Generate response
        response = generate_response(user_message, conversation)
        
        # Update conversation history
        conversation.append({"role": "user", "content": user_message})
        conversation.append({"role": "assistant", "content": response})
        session['conversation'] = conversation[-10:]  # Keep last 10 messages
        
        return jsonify({
            'response': response,
            'success': True
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset():
    """Reset the conversation history"""
    session['conversation'] = []
    return jsonify({'success': True})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device else 'unknown'
    })

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
