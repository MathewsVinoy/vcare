import json
import os

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    stream_with_context,
)

from models.blood_model import BloodModel
from models.chat_model import ChatModel
from models.image_model import ImageModel

app = Flask(__name__)

# ========== EXTERNAL LLM CONFIGURATION ==========
# ⚠️  REQUIRED: Chat feature requires Colab LLM connection
# Set the public ngrok URL from your running Colab notebook:
EXTERNAL_LLM_URL = os.environ.get(
    "EXTERNAL_LLM_URL", "https://patients-assigned-founder-extras.trycloudflare.com"
)
#
# Setup Steps:
# 1. Run the colab_llm_server.ipynb in Google Colab
# 2. Copy the public URL from cell output
# 3. Set environment variable: export EXTERNAL_LLM_URL="https://xxxxx.ngrok.io"
# 4. Restart this Flask app
#
# Note: Blood sample and image detection work without Colab.
#       Chat feature requires Colab LLM connection.
# =============================================

# Initialize model instances
chat_model = ChatModel(external_llm_url=EXTERNAL_LLM_URL)
blood_model = BloodModel()
image_model = ImageModel()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/blood_sample")
def blood_sample():
    return render_template("blood_sample.html")


@app.route("/image_detection")
def image_detection():
    return render_template("image_detection.html")


@app.route("/predict_blood_sample", methods=["POST"])
def predict_blood_sample():
    """
    Blood sample prediction endpoint with 16 features

    Expected JSON fields (in order):
    - gender: int (0=Male, 1=Female)
    - age: float (years)
    - hb: float (Hemoglobin, g/dL)
    - rbc: float (Red Blood Cells, M cells/µL)
    - wbc: float (White Blood Cells, cells/µL)
    - platelets: float (Platelet count, cells/µL)
    - lymp: float (Lymphocytes, %)
    - mono: float (Monocytes, %)
    - hct: float (Hematocrit, %)
    - mcv: float (Mean Corpuscular Volume, fL)
    - mch: float (Mean Corpuscular Hemoglobin, pg)
    - mchc: float (Mean Corpuscular Hemoglobin Concentration, g/dL)
    - rdw: float (Red Distribution Width, %)
    - pdw: float (Platelet Distribution Width, %)
    - mpv: float (Mean Platelet Volume, fL)
    - pct: float (Plateletcrit, %)
    """
    try:
        data = request.json

        if not data:
            return jsonify({"error": "No JSON data provided", "success": False}), 400

        # Extract and validate 16 features in correct order
        try:
            features = [
                int(data.get("gender", 0)),  # 0: Gender
                float(data.get("age", 0)),  # 1: Age
                float(data.get("hb", 0)),  # 2: Hemoglobin
                float(data.get("rbc", 0)),  # 3: RBC
                float(data.get("wbc", 0)),  # 4: WBC
                float(data.get("platelets", 0)),  # 5: Platelets
                float(data.get("lymp", 0)),  # 6: Lymphocytes
                float(data.get("mono", 0)),  # 7: Monocytes
                float(data.get("hct", 0)),  # 8: Hematocrit
                float(data.get("mcv", 0)),  # 9: MCV
                float(data.get("mch", 0)),  # 10: MCH
                float(data.get("mchc", 0)),  # 11: MCHC
                float(data.get("rdw", 0)),  # 12: RDW
                float(data.get("pdw", 0)),  # 13: PDW
                float(data.get("mpv", 0)),  # 14: MPV
                float(data.get("pct", 0)),  # 15: PCT
            ]
        except (TypeError, ValueError) as e:
            return jsonify(
                {"error": f"Invalid data types in request: {str(e)}", "success": False}
            ), 400

        # No strict numeric range restrictions: allow any valid numeric values.

        # Convert to numpy array with shape (1, 16)
        import numpy as np

        features_array = np.array([features])

        # Debug logging
        print("[BLOOD_PREDICT] ─────────────────────────────────")
        print(f"[BLOOD_PREDICT] Gender: {features[0]}, Age: {features[1]}")
        print(
            f"[BLOOD_PREDICT] Hb: {features[2]}, RBC: {features[3]}, WBC: {features[4]}, Platelets: {features[5]}"
        )
        print(
            f"[BLOOD_PREDICT] Lymp: {features[6]}, Mono: {features[7]}, HCT: {features[8]}"
        )
        print(
            f"[BLOOD_PREDICT] MCV: {features[9]}, MCH: {features[10]}, MCHC: {features[11]}"
        )
        print(
            f"[BLOOD_PREDICT] RDW: {features[12]}, PDW: {features[13]}, MPV: {features[14]}, PCT: {features[15]}"
        )

        # Make prediction using model module
        result = blood_model.predict(features_array)

        if not result.get("success", False):
            return jsonify(
                {"error": result.get("error", "Prediction failed"), "success": False}
            ), 500

        print(
            f"[BLOOD_PREDICT] Prediction: {result.get('raw_prediction')}, Probability: {result.get('probability')}"
        )
        print("[BLOOD_PREDICT] ─────────────────────────────────")

        return jsonify(result)

    except KeyError as e:
        return jsonify(
            {"error": f"Missing required field: {str(e)}", "success": False}
        ), 400
    except ValueError as e:
        return jsonify(
            {"error": f"Invalid value provided: {str(e)}", "success": False}
        ), 400
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return jsonify({"error": f"Prediction error: {str(e)}", "success": False}), 500


@app.route("/chat", methods=["POST"])
def chat_endpoint():
    """Handle chat requests"""
    try:
        data = request.get_json()
        user_message = data.get("message", "")

        if not user_message.strip():
            return jsonify({"error": "Empty message"}), 400

        # Generate response using chat model module
        result = chat_model.chat(user_message)

        # Handle greeting responses with choices
        if result.get("is_greeting"):
            return jsonify(
                {
                    "is_greeting": True,
                    "choices": result.get("choices", []),
                    "status": "success",
                }
            )

        # Handle regular responses
        return jsonify(
            {
                "response": result.get("response", "No response generated"),
                "is_greeting": False,
                "status": "success",
            }
        )

    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/chat/stream", methods=["POST"])
def chat_stream_endpoint():
    """Handle streaming chat requests for the frontend."""
    try:
        data = request.get_json()
        user_message = data.get("message", "")

        if not user_message.strip():
            return jsonify({"error": "Empty message"}), 400

        def token_stream():
            try:
                for event in chat_model.stream_chat(user_message):
                    if event.get("error"):
                        yield f"data: {json.dumps({'error': event['error']})}\n\n"
                    elif event.get("token"):
                        yield f"data: {json.dumps({'token': event['token']})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                yield "data: [DONE]\n\n"

        return Response(
            stream_with_context(token_stream()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    except Exception:

        def error_stream():
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

        return Response(
            stream_with_context(error_stream()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )


@app.route("/health")
def health():
    """Check if the model is loaded"""
    return jsonify(
        {
            "status": "ready" if chat_model.is_loaded() else "loading",
            "model_loaded": chat_model.is_loaded(),
            "router_loaded": chat_model.router_pipe is not None,
            "error": chat_model.error or chat_model.router_error,
        }
    )


@app.route("/predict_skin_cancer", methods=["POST"])
def predict_skin_cancer():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Read image bytes
        image_bytes = file.read()

        # Make prediction using image model module
        result = image_model.predict(image_bytes)

        if not result.get("success", False):
            return jsonify(
                {"error": result.get("error", "Model not loaded"), "success": False}
            ), 500

        return jsonify(result)

    except Exception as e:
        return jsonify(
            {"error": f"Error processing image: {str(e)}", "success": False}
        ), 500


if __name__ == "__main__":
    preload_setting = os.getenv("PRELOAD_CHAT_MODEL", "auto").strip().lower()
    if preload_setting in {"1", "true", "yes", "on"}:
        preload_main = True
    elif preload_setting in {"0", "false", "no", "off"}:
        preload_main = False
    else:
        preload_main = chat_model.can_preload_main_model()

    chat_model.initialize(preload_main=preload_main)

    debug_mode = os.getenv("FLASK_DEBUG", "0").strip() == "1"

    # Run the Flask app
    app.run(
        debug=debug_mode,
        use_reloader=False,
        threaded=True,
        host="0.0.0.0",
        port=5000,
    )
