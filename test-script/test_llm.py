import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import os

# Model configuration
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

# Set local cache directory
cache_dir = os.path.join(os.path.dirname(__file__), "model_cache")
os.makedirs(cache_dir, exist_ok=True)

print("Loading Mistral model...")
print(f"Model: {model_name}")
print(f"Cache directory: {cache_dir}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir
)

# Configure quantization with CPU offload support (8-bit is often more stable for offloading)
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

# Offload directory
offload_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "offload"))
os.makedirs(offload_dir, exist_ok=True)

# Calculate max memory to prevent over-allocation on GPU
if torch.cuda.is_available():
    # Use a conservative GPU budget to avoid meta-tensor issues
    max_memory = {0: "2GiB", "cpu": "12GiB"}
    print(f"Setting GPU budget to {max_memory[0]} and CPU budget to {max_memory['cpu']}")
else:
    max_memory = None

# Load model (8-bit with CPU offloading)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    max_memory=max_memory,
    quantization_config=quantization_config,
    torch_dtype=torch.float16, # Better for memory than default float32
    cache_dir=cache_dir,
    offload_folder=offload_dir
)

print("Model loaded successfully!")

# Create pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

generation_args = {
    "max_new_tokens": 300,
    "temperature": 0.7,
    "do_sample": True,
    "return_full_text": False
}

def chat(prompt):
    messages = [
        {"role": "user", "content": prompt}
    ]

    output = pipe(messages, **generation_args)
    return output[0]["generated_text"]

if __name__ == "__main__":

    print("\nMistral Model Ready!\n")

    test_prompt = "Explain machine learning in simple terms."

    print(f"User: {test_prompt}\n")
    response = chat(test_prompt)
    print(f"Mistral: {response}\n")

    print("\nInteractive mode (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if user_input.strip():
            response = chat(user_input)
            print(f"\nMistral: {response}\n")