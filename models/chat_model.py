"""Phi-3 Chat Model Module"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import os
import random


class ChatModel:
    """Load and manage the Phi-3 chat model with quantization and offloading"""
    
    def __init__(self, model_name="microsoft/Phi-3-mini-4k-instruct", cache_dir=None, offload_dir=None):
        self.model_name = model_name
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "../model_cache")
        self.offload_dir = offload_dir or os.path.join(os.path.dirname(__file__), "../offload_dir")
        
        self.model = None
        self.tokenizer = None
        self.pipe = None
        self.error = None
        
        self.generation_args = {
            "max_new_tokens": 300,
            "temperature": 0.7,
            "do_sample": True,
            "return_full_text": False
        }
        
        # Greeting keywords and custom responses
        self.greeting_keywords = ['hello', 'hi', 'hai', 'hey', 'greetings', 'howdy']
        
        self.greeting_responses = {
            'friendly': [
                "Hello! 👋 Welcome to VCare AI Medical Assistant. How can I help you with your health concerns today?",
                "Hi there! 😊 I'm here to assist you with medical information and health guidance. What would you like to know?",
                "Hey! Welcome! 🏥 I'm ready to help with any medical questions or health-related advice you might need."
            ],
            'professional': [
                "Good day. I'm VCare AI Medical Assistant. How may I assist you with your healthcare needs?",
                "Greetings. Welcome to VCare AI. Please tell me how I can help with your medical inquiries.",
                "Welcome. I'm your AI medical advisor. What health information can I provide for you today?"
            ],
            'warm': [
                "Welcome! 🤍 I hope you're doing well. I'm here to answer all your health and medical questions.",
                "Hi! So glad you're here. 💙 Let's talk about your health and wellness. What's on your mind?",
                "Hello friend! 🌟 VCare AI is at your service. How can I support your health journey today?"
            ]
        }
    
    def is_greeting(self, text):
        """Check if the input text is a greeting"""
        text_lower = text.strip().lower()
        
        # Check if it matches greeting keywords (exact or starts with)
        for keyword in self.greeting_keywords:
            if text_lower == keyword or text_lower.startswith(keyword):
                return True
        
        return False
    
    def get_greeting_choices(self):
        """Get 3 greeting response choices from different styles"""
        choices = []
        
        # Select one response from each style
        for style in ['friendly', 'professional', 'warm']:
            response = random.choice(self.greeting_responses[style])
            choices.append(response)
        
        return choices
    
    def load(self):
        """Load the chat model and pipeline"""
        if self.model is not None and self.tokenizer is not None and self.pipe is not None:
            return True
        
        try:
            self.error = None
            print("Loading Phi-3 model...")
            print(f"Model: {self.model_name}")
            print(f"Cache directory: {self.cache_dir}")
            
            os.makedirs(self.cache_dir, exist_ok=True)
            os.makedirs(self.offload_dir, exist_ok=True)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=False
            )
            
            # Prepare model loading parameters
            quantization_config = None
            max_memory = None
            
            if torch.cuda.is_available():
                max_memory = {0: "2GiB", "cpu": "12GiB"}
                print(f"Setting GPU budget to {max_memory[0]} and CPU budget to {max_memory['cpu']}")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    max_memory=max_memory,
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16,
                    cache_dir=self.cache_dir,
                    offload_folder=self.offload_dir,
                    trust_remote_code=False
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    cache_dir=self.cache_dir,
                    offload_folder=self.offload_dir,
                    trust_remote_code=False
                )
            
            # Create pipeline
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer
            )
            
            print("Model loaded successfully!")
            return True
        
        except Exception as e:
            self.model = None
            self.tokenizer = None
            self.pipe = None
            self.error = str(e)
            print(f"❌ Error loading chat model: {self.error}")
            return False
    
    def chat(self, prompt):
        """Generate a chat response or greeting choices"""
        # Check if input is a greeting
        if self.is_greeting(prompt):
            return {
                'is_greeting': True,
                'choices': self.get_greeting_choices()
            }
        
        # For non-greeting messages, use the full model
        if self.pipe is None:
            if not self.load():
                return {
                    'is_greeting': False,
                    'response': f"Chat model could not be loaded right now. {self.error or ''}".strip()
                }
        
        messages = [{"role": "user", "content": prompt}]
        output = self.pipe(messages, **self.generation_args)
        
        return {
            'is_greeting': False,
            'response': output[0]['generated_text']
        }
    
    def is_loaded(self):
        """Check if model is loaded"""
        return self.model is not None and self.tokenizer is not None and self.pipe is not None
