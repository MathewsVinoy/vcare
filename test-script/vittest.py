"""
Standalone Flask server for the Skin Cancer (Image Detection) feature.
Run from the project root:  python test-script/vittest.py
or from inside test-script/: python vittest.py
"""

import os
import io

import torch
from torch import nn
from torchvision import transforms
import torchvision
from PIL import Image
from flask import Flask, render_template, request, jsonify

# ─────────────────────────────────────────────
# Paths – resolve relative to this file so the
# script works from any working directory.
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODEL_PATH = os.path.join(BASE_DIR, "model", "efficientformer_model.pth")

# ─────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR,
)

# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ─────────────────────────────────────────────
# Model – ViT-B/16 fine-tuned for binary skin
# cancer classification (0 = benign, 1 = cancer)
# ─────────────────────────────────────────────
image_model = None


def build_vit_model(num_classes: int = 2) -> nn.Module:
    """Construct a ViT-B/16 with a custom classification head."""
    vit = torchvision.models.vit_b_16(weights=None)          # no pre-trained weights; we load our own
    in_features = vit.heads.head.in_features
    vit.heads.head = nn.Linear(in_features, num_classes)
    return vit


def load_image_model() -> bool:
    """Lazy-load the skin-cancer detection model. Returns True on success."""
    global image_model

    if image_model is not None:
        return True

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        return False

    try:
        image_model = build_vit_model(num_classes=2)
        state_dict = torch.load(MODEL_PATH, map_location=device)
        image_model.load_state_dict(state_dict)
        image_model.to(device)
        image_model.eval()
        print(f"[INFO] ✅ Skin-cancer model loaded from {MODEL_PATH} on {device}")
        return True
    except Exception as exc:
        print(f"[ERROR] Failed to load model: {exc}")
        return False


# ─────────────────────────────────────────────
# Image pre-processing (must match training)
# ─────────────────────────────────────────────
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
@app.route("/image_detection")
def image_detection():
    """Render the Skin Cancer Detection UI."""
    return render_template("image_detection.html")


@app.route("/predict_skin_cancer", methods=["POST"])
def predict_skin_cancer():
    """
    Accepts a multipart/form-data POST with an 'image' field.
    Returns JSON: { success, probability, diagnosis, description, prediction }
    """
    # Ensure model is loaded
    if not load_image_model():
        return jsonify({
            "success": False,
            "error": "Model could not be loaded. Check that model/model.pth exists.",
        }), 500

    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image field in request."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename."}), 400

    try:
        # Read & pre-process
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = IMG_TRANSFORM(image).unsqueeze(0).to(device)   # (1, 3, 224, 224)

        # Inference
        with torch.no_grad():
            logits = image_model(tensor)                         # (1, 2)
            probs = torch.softmax(logits, dim=1)[0]              # (2,)
            prediction = int(torch.argmax(probs).item())
            cancer_prob = float(probs[1].item()) * 100           # class-1 = cancer

        # ── Debug logging ──────────────────────────────────────
        print("[PREDICT] ──────────────────────────────────────")
        print(f"[PREDICT] File         : {file.filename}")
        print(f"[PREDICT] Logits       : {logits[0].tolist()}")
        print(f"[PREDICT] Probabilities: benign={probs[0].item()*100:.2f}%  cancer={probs[1].item()*100:.2f}%")
        print(f"[PREDICT] Predicted cls: {prediction}  ({'cancer' if prediction == 1 else 'benign'})")
        print(f"[PREDICT] Cancer prob  : {cancer_prob:.2f}%")
        print("[PREDICT] ──────────────────────────────────────")

        # Build human-readable result
        if cancer_prob > 70:
            diagnosis = "High Risk — Skin Cancer Suspected"
            description = (
                "The analysis identifies features strongly associated with malignant skin "
                "lesions (e.g. melanoma, basal cell carcinoma, or actinic keratosis): "
                "irregular borders, asymmetry, and colour variation. "
                "<strong>Please consult a dermatologist immediately.</strong>"
            )
        elif cancer_prob > 40:
            diagnosis = "Moderate Risk — Further Examination Needed"
            description = (
                "The lesion shows some concerning characteristics. "
                "Schedule a professional dermatological evaluation to rule out malignancy."
            )
        else:
            diagnosis = "Low Risk — Likely Benign"
            description = (
                "The lesion appears consistent with a benign skin condition "
                "(e.g. common nevus, benign keratosis, or dermatofibroma). "
                "Continue routine self-monitoring and see a dermatologist if it changes."
            )

        return jsonify({
            "success": True,
            "probability": round(cancer_prob, 2),
            "prediction": prediction,
            "diagnosis": diagnosis,
            "description": description,
        })

    except Exception as exc:
        return jsonify({"success": False, "error": f"Processing error: {exc}"}), 500


@app.route("/health")
def health():
    """Quick health-check endpoint."""
    return jsonify({
        "status": "ok",
        "model_loaded": image_model is not None,
        "device": str(device),
    })


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  VCare AI — Skin Cancer Detection Server")
    print("  UI  →  http://localhost:5001/image_detection")
    print("=" * 55)

    # Pre-load model at startup so the first request is fast
    load_image_model()

    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5001)
