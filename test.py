import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import os

# Model configuration
model_name = "microsoft/Phi-3-mini-4k-instruct"
lora_adapter_path = os.path.join(os.path.dirname(__file__), "phi3_lora_model")

# Set local cache directory
cache_dir = os.path.join(os.path.dirname(__file__), "model_cache")
os.makedirs(cache_dir, exist_ok=True)

print("Loading Phi-3 base model with cancer diagnosis fine-tuning...")
print(f"Base Model: {model_name}")
print(f"LoRA Adapter: {lora_adapter_path}")
print(f"Cache directory: {cache_dir}")

# Load tokenizer from the fine-tuned model directory (includes special tokens)
tokenizer = AutoTokenizer.from_pretrained(
    lora_adapter_path,
    trust_remote_code=False
)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use float16 for efficiency
    trust_remote_code=False,  # Use transformers' built-in implementation
    cache_dir=cache_dir,
    attn_implementation="eager",  # Use eager attention for better compatibility
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

# Generation settings
generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.7,
    "do_sample": True,
}

# Example usage
def chat(prompt):
    """Generate a response using Phi-3"""
    messages = [
        {"role": "user", "content": prompt},
    ]
    
    output = pipe(messages, **generation_args)
    return output[0]['generated_text']


# Test the model
if __name__ == "__main__":
    print("\n" + "="*50)
    print("Phi-3 Cancer Diagnosis Model Ready!")
    print("="*50 + "\n")
    
    # Example prompt for cancer diagnosis
    test_prompt = "What are the common symptoms of breast cancer?"
    
    print(f"User: {test_prompt}\n")
    response = chat(test_prompt)
    print(f"Phi-3: {response}\n")
    
    # Interactive mode
    print("\nEntering interactive mode. Type 'quit' or 'exit' to stop.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        if user_input.strip():
            response = chat(user_input)
            print(f"\nPhi-3: {response}\n")
