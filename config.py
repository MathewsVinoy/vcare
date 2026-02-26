# Configuration for the Medical Diagnosis Application

import os

# Chat Model Configuration
CHAT_MODEL_CONFIG = {
    # Set to True to enable 8-bit quantization (requires bitsandbytes)
    # This reduces memory usage by ~4x but may slightly reduce quality
    'use_8bit': os.environ.get('USE_8BIT_QUANTIZATION', 'false').lower() == 'true',
    
    # Set to True to enable 4-bit quantization (requires bitsandbytes)
    # This reduces memory usage by ~8x but may reduce quality more
    'use_4bit': os.environ.get('USE_4BIT_QUANTIZATION', 'false').lower() == 'true',
    
    # Base model name
    'base_model': 'microsoft/Phi-3-mini-4k-instruct',
    
    # Adapter path
    'adapter_path': 'phi3_lora_model',
    
    # Max tokens to generate
    'max_new_tokens': 512,
    
    # Temperature for sampling
    'temperature': 0.7,
    
    # Top-p for nucleus sampling
    'top_p': 0.9,
}

# Flask Configuration
FLASK_CONFIG = {
    'debug': True,
    'host': '0.0.0.0',
    'port': 5000
}
