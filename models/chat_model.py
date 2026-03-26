"""Phi-3 Chat Model Module with External LLM Support"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    pipeline,
)
import os
import random
import re
from collections import OrderedDict
from contextlib import nullcontext
from threading import Lock
from threading import Thread

import psutil
from .external_llm_client import ExternalLLMClient


class ChatModel:
    """Load and manage the chat stack with lightweight routing and Phi-3 generation."""

    def __init__(
        self,
        model_name="microsoft/Phi-3-mini-4k-instruct",
        router_model_name="google/flan-t5-small",
        cache_dir=None,
        offload_dir=None,
        external_llm_url=None,
    ):
        self.model_name = model_name
        self.router_model_name = router_model_name
        self.cache_dir = os.path.abspath(cache_dir or os.path.join(os.path.dirname(__file__), "../model_cache"))
        self.offload_dir = os.path.abspath(offload_dir or os.path.join(os.path.dirname(__file__), "../offload_dir"))
        self.router_cache_dir = self.cache_dir

        # Initialize external LLM client (REQUIRED for chat)
        self.external_llm = ExternalLLMClient(external_llm_url)
        self.use_external_llm = external_llm_url is not None
        self.external_llm_url = external_llm_url
        
        # Chat requires external LLM - check if configured
        if not external_llm_url:
            print("\n" + "="*60)
            print("⚠️  WARNING: Chat requires Colab LLM configuration")
            print("="*60)
            print("\nTo enable chat, set EXTERNAL_LLM_URL:")
            print("  export EXTERNAL_LLM_URL='https://xxxxx.ngrok.io'")
            print("\nThen restart this app.")
            print("="*60 + "\n")

        self.model = None
        self.tokenizer = None
        self.pipe = None
        self.error = None

        self.router_model = None
        self.router_tokenizer = None
        self.router_pipe = None
        self.router_error = None

        self.generation_args = {
            "max_new_tokens": 220,
            "do_sample": False,
            "return_full_text": False,
            "repetition_penalty": 1.08,
            "use_cache": True,
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

        self.cancer_keywords = {
            'cancer', 'tumor', 'tumour', 'oncology', 'oncologist', 'chemotherapy',
            'radiation', 'metastasis', 'biopsy', 'malignant', 'benign', 'leukemia',
            'leukaemia', 'lymphoma', 'melanoma', 'carcinoma', 'sarcoma', 'neoplasm',
            'breast cancer', 'lung cancer', 'skin cancer', 'blood cancer', 'colon cancer',
            'prostate cancer', 'cervical cancer', 'brain tumor', 'cancer stage',
            'cancer symptoms', 'cancer treatment', 'cancer diagnosis'
        }

        self.medical_keywords = {
            'doctor', 'hospital', 'medicine', 'medical', 'health', 'healthcare', 'symptom',
            'symptoms', 'diagnosis', 'treatment', 'disease', 'illness', 'pain', 'fever',
            'infection', 'blood', 'scan', 'mri', 'ct', 'xray', 'x-ray', 'ultrasound',
            'test', 'lab', 'report', 'prescription', 'surgery', 'clinic', 'patient',
            'therapy', 'dose', 'drug', 'tablet', 'doctor appointment', 'rash', 'headache',
            'cough', 'diabetes', 'heart', 'liver', 'kidney', 'brain', 'skin', 'biopsy',
            'wbc', 'rbc', 'platelet', 'hemoglobin', 'medical advice'
        }

        self.math_keywords = {
            'math', 'mathematics', 'solve', 'equation', 'algebra', 'geometry',
            'trigonometry', 'calculus', 'integral', 'derivative', 'multiply',
            'division', 'divide', 'addition', 'subtract', 'subtraction', 'sum',
            'product', 'percentage', 'formula', 'simplify', 'factorial'
        }

        self.out_of_scope_message = (
            "Sorry, I can only assist with cancer-related or medical questions. "
            "Please ask about symptoms, diagnosis, treatment, reports, or other health concerns."
        )

        self.max_cached_responses = 128
        self.response_cache = OrderedDict()
        self.generation_lock = Lock()

        self.cancer_patterns = self._compile_keyword_patterns(self.cancer_keywords)
        self.medical_patterns = self._compile_keyword_patterns(self.medical_keywords)
        self.math_patterns = self._compile_keyword_patterns(self.math_keywords)
        self.math_expression_patterns = [
            re.compile(r'\b\d+\s*[-+*/x=]\s*\d+\b'),
            re.compile(r'\bwhat\s+is\s+\d+'),
            re.compile(r'\bcalculate\b'),
            re.compile(r'\bsolve\s+\d+'),
            re.compile(r'\b\d+\s*%\s+of\s+\d+\b')
        ]
        self.random_patterns = [
            re.compile(r'^[a-z]{1,4}$'),
            re.compile(r'^(.)\1{4,}$'),
            re.compile(r'^[bcdfghjklmnpqrstvwxyz]{5,}$')
        ]

    def tune_runtime(self):
        """Apply runtime-only performance tuning (no model change)."""
        try:
            cpu_count = os.cpu_count() or 2
            torch.set_num_threads(max(1, min(8, cpu_count // 2)))
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        if torch.cuda.is_available():
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
            except Exception:
                pass
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

    def _chunk_text(self, text, chunk_size=28):
        """Yield small chunks for non-model streaming responses."""
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]

    def _compile_keyword_patterns(self, keywords):
        """Compile keyword regex patterns once for faster matching."""
        return [re.compile(r'(?<!\w)' + re.escape(keyword) + r'(?!\w)') for keyword in keywords]

    def _get_cached_response(self, prompt):
        """Return cached response for repeated prompts."""
        cached = self.response_cache.get(prompt)
        if cached is not None:
            self.response_cache.move_to_end(prompt)
        return cached

    def _cache_response(self, prompt, response):
        """Store response in a small LRU-style cache."""
        self.response_cache[prompt] = response
        self.response_cache.move_to_end(prompt)
        if len(self.response_cache) > self.max_cached_responses:
            self.response_cache.popitem(last=False)

    def _count_pattern_matches(self, text, patterns):
        """Count the number of pattern hits for a prompt."""
        text_lower = self._normalize_text(text)
        return sum(1 for pattern in patterns if pattern.search(text_lower))
    
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

    def _normalize_text(self, text):
        """Normalize text for lightweight intent matching."""
        normalized = text.strip().lower()
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized

    def _contains_keyword(self, text, keywords):
        """Match keywords using word boundaries where possible."""
        text_lower = self._normalize_text(text)

        patterns = keywords
        if isinstance(keywords, set):
            patterns = self._compile_keyword_patterns(keywords)

        for pattern in patterns:
            if pattern.search(text_lower):
                return True

        return False

    def is_cancer_related(self, text):
        """Check if the input is cancer-related."""
        return self._contains_keyword(text, self.cancer_patterns)

    def is_medical_related(self, text):
        """Check if the input is medical-related."""
        return self._contains_keyword(text, self.medical_patterns) or self.is_cancer_related(text)

    def is_math_related(self, text):
        """Reject explicit mathematics or generic calculation prompts."""
        text_lower = self._normalize_text(text)

        if self._contains_keyword(text_lower, self.math_patterns):
            return True

        return any(pattern.search(text_lower) for pattern in self.math_expression_patterns)

    def is_random_text(self, text):
        """Detect likely gibberish or random non-medical text."""
        text_lower = self._normalize_text(text)

        if not text_lower:
            return True

        tokens = re.findall(r'[a-zA-Z]+', text_lower)
        if not tokens:
            return True

        if len(tokens) == 1:
            token = tokens[0]
            vowel_count = sum(1 for char in token if char in 'aeiou')
            if len(token) >= 5 and vowel_count == 0:
                return True
            if len(token) >= 6 and len(set(token)) <= 2:
                return True

        return any(pattern.fullmatch(text_lower) for pattern in self.random_patterns)

    def lightweight_classify(self, text):
        """Fast rule-based classifier before using the router model."""
        if self.is_random_text(text):
            return "non-medical"

        if self.is_math_related(text) and not self.is_medical_related(text):
            return "non-medical"

        cancer_hits = self._count_pattern_matches(text, self.cancer_patterns)
        medical_hits = self._count_pattern_matches(text, self.medical_patterns)

        if cancer_hits > 0:
            return "cancer"

        if medical_hits > 0:
            return "medical"

        return "unknown"

    def load_router(self):
        """Load a small instruct model for routing ambiguous prompts."""
        if self.router_model is not None and self.router_tokenizer is not None and self.router_pipe is not None:
            return True

        try:
            self.router_error = None
            os.makedirs(self.router_cache_dir, exist_ok=True)

            self.router_tokenizer = AutoTokenizer.from_pretrained(
                self.router_model_name,
                cache_dir=self.router_cache_dir,
                trust_remote_code=False
            )
            self.router_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.router_model_name,
                cache_dir=self.router_cache_dir,
                trust_remote_code=False
            )
            self.router_model.eval()
            self.router_pipe = pipeline(
                "text2text-generation",
                model=self.router_model,
                tokenizer=self.router_tokenizer,
                device=-1,
            )
            return True
        except Exception as e:
            self.router_model = None
            self.router_tokenizer = None
            self.router_pipe = None
            self.router_error = str(e)
            return False

    def route_with_small_model(self, prompt):
        """Use the small instruct model to classify ambiguous prompts."""
        if not self.load_router():
            return "non-medical"

        router_prompt = (
            "Classify the user message into exactly one label: cancer, medical, or non-medical. "
            "Return only the label.\n"
            f"Message: {prompt}\n"
            "Label:"
        )

        try:
            output = self.router_pipe(
                router_prompt,
                max_new_tokens=4,
                do_sample=False,
                truncation=True,
            )[0]["generated_text"].strip().lower()
        except Exception:
            return "non-medical"

        if "non-medical" in output:
            return "non-medical"
        if "cancer" in output:
            return "cancer"
        if "medical" in output:
            return "medical"
        return "non-medical"

    def classify_prompt(self, prompt):
        """Classify the prompt domain using rules, then the small router if needed."""
        normalized_prompt = self._normalize_text(prompt)

        # Strict fast-path gate for obviously out-of-domain long prompts.
        if len(normalized_prompt) > 220 and not self.is_medical_related(normalized_prompt):
            return "non-medical"

        label = self.lightweight_classify(prompt)
        if label != "unknown":
            return label
        return self.route_with_small_model(prompt)

    def can_preload_main_model(self):
        """Check whether the main model can be loaded eagerly based on available memory."""
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        if torch.cuda.is_available():
            return available_gb >= 10
        return available_gb >= 20

    def initialize(self, preload_main=False):
        """Initialize router first, and optionally preload the main model."""
        self.tune_runtime()
        self.load_router()
        if preload_main:
            if self.load():
                self.warmup()

    def warmup(self):
        """Run a tiny warmup pass to reduce first-token latency."""
        if self.pipe is None:
            return

        try:
            warmup_messages = self.build_domain_prompt("What is cancer?", "cancer")
            self.pipe(
                warmup_messages,
                max_new_tokens=8,
                do_sample=False,
                return_full_text=False,
                use_cache=True,
            )
        except Exception:
            # Warmup failure should never block server startup.
            pass

    def build_domain_prompt(self, prompt, label):
        """Add a domain instruction based on the detected topic."""
        if label == "cancer":
            domain_instruction = (
                "You are VCare AI, a cancer-focused medical assistant. "
                "Answer only in the context of cancer, oncology, diagnosis support, symptoms, screening, reports, "
                "risk factors, and treatment guidance. Keep the response clear, supportive, and medically relevant. "
                "Always remind the user to consult a qualified doctor for diagnosis or treatment decisions."
            )
        else:
            domain_instruction = (
                "You are VCare AI, a medical assistant. "
                "Answer only medical or health-related questions in a clear and careful way. "
                "Do not provide unrelated information. Encourage consulting a qualified doctor for urgent or serious concerns."
            )

        return [
            {"role": "system", "content": domain_instruction},
            {"role": "user", "content": prompt}
        ]

    def _prepare_prompt(self, prompt):
        """Resolve greeting, cached, rejected, or model-backed prompt handling."""
        normalized_prompt = self._normalize_text(prompt)

        if self.is_greeting(normalized_prompt):
            result = {
                'is_greeting': True,
                'choices': self.get_greeting_choices()
            }
            self._cache_response(normalized_prompt, result)
            return {"mode": "final", "cache_key": normalized_prompt, "result": result}

        cached_response = self._get_cached_response(normalized_prompt)
        if cached_response is not None:
            return {"mode": "final", "cache_key": normalized_prompt, "result": cached_response}

        label = self.classify_prompt(prompt)
        if label == "non-medical":
            result = {
                'is_greeting': False,
                'response': self.out_of_scope_message
            }
            self._cache_response(normalized_prompt, result)
            return {"mode": "final", "cache_key": normalized_prompt, "result": result}

        if self.pipe is None and not self.load():
            result = {
                'is_greeting': False,
                'response': f"Chat model could not be loaded right now. {self.error or ''}".strip()
            }
            self._cache_response(normalized_prompt, result)
            return {"mode": "final", "cache_key": normalized_prompt, "result": result}

        return {
            "mode": "generate",
            "cache_key": normalized_prompt,
            "label": label,
            "messages": self.build_domain_prompt(prompt, label),
        }
    
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
            self.model.eval()
            
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
        """Generate a chat response using Colab LLM (required)\n        Chat REQUIRES external LLM from Colab - no fallback to local model.
        """
        # Chat REQUIRES external LLM from Colab - no fallback to local
        if not self.external_llm_url:
            return {
                'is_greeting': False,
                'response': 'Chat is unavailable. Please configure EXTERNAL_LLM_URL to connect to Colab LLM server.\n\nSet: export EXTERNAL_LLM_URL="https://xxxxx.ngrok.io"',
                'status': 'error'
            }
        
        if not self.use_external_llm:
            return {
                'is_greeting': False,
                'response': 'Chat is unavailable. Cannot connect to Colab LLM server at ' + self.external_llm_url + '. Ensure the Colab notebook is running.',
                'status': 'error'
            }
        
        # Use external LLM only
        return self._chat_external(prompt)

    def _chat_external(self, prompt):
        """Chat using external LLM server"""
        # Check for greeting
        normalized_prompt = self._normalize_text(prompt)
        
        if self.is_greeting(normalized_prompt):
            return {
                'is_greeting': True,
                'choices': self.get_greeting_choices()
            }
        
        # Check cache
        cached = self._get_cached_response(normalized_prompt)
        if cached is not None:
            return cached
        
        # Check if out of scope
        label = self.lightweight_classify(prompt)
        if label == "non-medical":
            result = {
                'is_greeting': False,
                'response': self.out_of_scope_message
            }
            self._cache_response(normalized_prompt, result)
            return result
        
        # Call external LLM
        response = self.external_llm.generate_response(prompt, max_tokens=1000)
        
        if response['success']:
            result = {
                'is_greeting': False,
                'response': response['response']
            }
            self._cache_response(normalized_prompt, result)
            return result
        else:
            return {
                'is_greeting': False,
                'response': f"Error from LLM server: {response['error']}"
            }

    def stream_chat(self, prompt):
        """Stream chat responses using Colab LLM (required)
        Chat REQUIRES external LLM from Colab - no fallback to local model.
        """
        # Chat REQUIRES external LLM from Colab - no fallback to local
        if not self.external_llm_url:
            yield {'error': 'Chat is unavailable. Please configure EXTERNAL_LLM_URL to connect to Colab LLM server.'}
            return
        
        if not self.use_external_llm:
            yield {'error': 'Chat is unavailable. Cannot connect to Colab LLM server. Ensure the Colab notebook is running.'}
            return
        
        # Use external LLM streaming only
        yield from self._stream_chat_external(prompt)
    
    def _stream_chat_external(self, prompt):
        """Stream chat responses from external LLM server"""
        # Do all context-sensitive operations BEFORE yielding
        normalized_prompt = self._normalize_text(prompt)
        
        # Check for greeting
        is_greeting = self.is_greeting(normalized_prompt)
        if is_greeting:
            choices = self.get_greeting_choices()
            intro = "Here are a few ways I can help you get started:\n\n"
            yield {"token": intro}
            for index, choice in enumerate(choices, start=1):
                prefix = f"{index}. "
                suffix = "\n\n" if index < len(choices) else ""
                yield {"token": f"{prefix}{choice}{suffix}"}
            return
        
        # Check cache (before any yields)
        cached = self._get_cached_response(normalized_prompt)
        if cached is not None:
            response_text = cached.get('response', '')
            for chunk in self._chunk_text(response_text):
                yield {"token": chunk}
            return
        
        # Check if out of scope (before any streaming yields)
        label = self.lightweight_classify(prompt)
        is_out_of_scope = (label == "non-medical")
        
        if is_out_of_scope:
            for chunk in self._chunk_text(self.out_of_scope_message):
                yield {"token": chunk}
            # Cache after all yields
            self._cache_response(normalized_prompt, {
                'is_greeting': False,
                'response': self.out_of_scope_message
            })
            return
        
        # Stream from external LLM
        full_response = ""
        for event in self.external_llm.stream_response(prompt, max_tokens=1000):
            if event.get('error'):
                error_msg = f"Error: {event['error']}"
                yield {"error": error_msg}
                return
            elif event.get('token') and event['token'] != '[DONE]':
                token = event['token']
                full_response += token
                yield {"token": token}
        
        # Cache the full response after streaming is complete
        if full_response:
            self._cache_response(normalized_prompt, {
                'is_greeting': False,
                'response': full_response
            })
    
    def is_loaded(self):
        """Check if model is loaded"""
        if self.use_external_llm:
            return True
        return self.model is not None and self.tokenizer is not None and self.pipe is not None
