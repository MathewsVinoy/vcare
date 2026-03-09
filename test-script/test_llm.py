"""
VCare AI — Flask chat server powered by Phi-3-mini-4k-instruct
Run:  python test-script/test_llm.py
"""

import os
import threading
import traceback

import torch
from flask import Flask, jsonify, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ---------------------------------------------------------------------------
# Paths — point Flask at the project-level templates / static folders
# ---------------------------------------------------------------------------
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR   = os.path.join(BASE_DIR, "static")
CACHE_DIR    = os.path.join(BASE_DIR, "model_cache")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

# ---------------------------------------------------------------------------
# CUDA / GPU detection
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    _gpu_name = torch.cuda.get_device_name(0)
    _vram_gb  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"✅  CUDA available — GPU: {_gpu_name}  ({_vram_gb:.1f} GB VRAM)")
    device = torch.device("cuda")
else:
    print("⚠️  CUDA not available — running on CPU (slow). "
          "On Windows make sure you installed the CUDA-enabled PyTorch wheel "
          "(pip install torch --index-url https://download.pytorch.org/whl/cu121) "
          "and that your NVIDIA drivers are up-to-date.")
    device = torch.device("cpu")

tokenizer      = None
pipe           = None
model_loaded   = False
model_loading  = False
load_error     = None

SYSTEM_PROMPT = (
    "You are VCare AI, an intelligent medical assistant specialising in cancer "
    "diagnosis, blood analysis, and skin cancer detection. Provide accurate, "
    "helpful medical information while always advising the user to consult a "
    "qualified healthcare professional for any actual diagnosis or treatment."
)

# ---------------------------------------------------------------------------
# Model loading (runs in a background thread so the server starts immediately)
# ---------------------------------------------------------------------------
def load_model():
    global tokenizer, pipe, model_loaded, model_loading, load_error

    model_loading = True
    print(f"\n🔄  Loading {MODEL_NAME} …  (device: {device})\n")
    OFFLOAD_DIR  = os.path.join(BASE_DIR, "offload_dir")
    os.makedirs(OFFLOAD_DIR, exist_ok=True)

    try:
        # Free any leftover GPU memory before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Use bfloat16 on Ampere+ GPUs (RTX 30xx/40xx), float16 on older,
        # float32 on CPU — all selected automatically.
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            torch_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
        else:
            torch_dtype = torch.float32

        print(f"   dtype : {torch_dtype}")

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
        )

        llm = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            load_in_4bit=True,             # 4-bit quantization to save RAM
            cache_dir=CACHE_DIR,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager",   # avoids flash-attn / sdpa dependency
            offload_folder=OFFLOAD_DIR,
        )

        # ⚠️  Do NOT pass device_map to pipeline when the model is already
        #     device-mapped — doing so causes silent CPU fall-back or errors
        #     on Windows / WSL2.
        pipe = pipeline(
            "text-generation",
            model=llm,
            tokenizer=tokenizer,
            # dtype=torch_dtype,  # pipeline also accepts dtype but often infers it from model
            device_map="auto",
        )

        if torch.cuda.is_available():
            used  = torch.cuda.memory_reserved(0) / (1024 ** 3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"   VRAM : {used:.1f} GB used / {total:.1f} GB total")

        model_loaded  = True
        model_loading = False
        print("✅  Phi-3-mini-4k-instruct loaded successfully!\n")

    except Exception as exc:
        load_error    = str(exc)
        model_loading = False
        traceback.print_exc()          # prints full stack trace to stderr
        print(f"❌  Failed to load model: {exc}\n")


# ---------------------------------------------------------------------------
# Chat helper
# ---------------------------------------------------------------------------
def generate_response(user_message: str, history: list[dict] | None = None) -> str:
    """Build the message list and call the pipeline."""
    if pipe is None:
        if model_loading:
            return "⏳ The model is still loading — please try again in a moment."
        if load_error:
            return f"❌ Model failed to load: {load_error}"
        return "Model is not loaded yet. Please wait…"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        for entry in history:
            if isinstance(entry, dict) and "role" in entry and "content" in entry:
                messages.append(entry)

    messages.append({"role": "user", "content": user_message})

    try:
        result = pipe(
            messages,
            max_new_tokens=512,
            return_full_text=False,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
        )
        answer = result[0]["generated_text"].strip()
        # Release any fragmented GPU memory after inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return answer
    except Exception as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return f"Error generating response: {exc}"


# ---------------------------------------------------------------------------
# Routes — page rendering
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/blood_sample")
def blood_sample():
    return render_template("blood_sample.html")


@app.route("/image_detection")
def image_detection():
    return render_template("image_detection.html")


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------
@app.route("/health")
def health():
    """Polled by script.js to show the model-status badge in the header."""
    info = {
        "status":        "running",
        "model_loaded":  model_loaded,
        "model_loading": model_loading,
        "model_name":    MODEL_NAME,
        "device":        str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_name"]    = torch.cuda.get_device_name(0)
        info["vram_used_gb"] = round(torch.cuda.memory_reserved(0) / (1024**3), 2)
        info["vram_total_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
        )
    return jsonify(info)


@app.route("/chat", methods=["POST"])
def chat():
    """
    Expected JSON body:
      { "message": "<user text>", "history": [ {role, content}, … ] }

    Returns:
      { "response": "<assistant text>", "success": true }
    """
    try:
        data = request.get_json(silent=True) or {}
        user_message = (data.get("message") or "").strip()

        if not user_message:
            return jsonify({"error": "No message provided", "success": False}), 400

        history  = data.get("history", [])
        response = generate_response(user_message, history)
        return jsonify({"response": response, "success": True})

    except Exception as exc:
        return jsonify({"error": str(exc), "success": False}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Start model loading in background so HTTP responses don't block
    t = threading.Thread(target=load_model, daemon=False)
    t.start()

    print("🚀  VCare AI (Phi-3-mini-4k-instruct) starting at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
