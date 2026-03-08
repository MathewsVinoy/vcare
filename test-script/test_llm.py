"""
VCare AI — Flask chat server powered by Mistral-7B-Instruct
Run:  python test-script/test_llm.py
"""

import os
import threading

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
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    os.makedirs(CACHE_DIR, exist_ok=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR,
            trust_remote_code=False,
        )

        llm = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            cache_dir=CACHE_DIR,
            low_cpu_mem_usage=True,
        )

        pipe = pipeline(
            "text-generation",
            model=llm,
            tokenizer=tokenizer,
        )

        model_loaded  = True
        model_loading = False
        print("✅  Mistral-7B loaded successfully!\n")

    except Exception as exc:
        load_error    = str(exc)
        model_loading = False
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
        return result[0]["generated_text"].strip()
    except Exception as exc:
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
    return jsonify({
        "status":       "running",
        "model_loaded": model_loaded,
        "model_loading": model_loading,
        "model_name":   MODEL_NAME,
        "device":       str(device),
    })


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
    t = threading.Thread(target=load_model, daemon=True)
    t.start()

    print("🚀  VCare AI (Mistral-7B) starting at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
