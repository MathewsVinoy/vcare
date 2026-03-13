from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import os
import json

from models.chat_model import ChatModel
from models.blood_model import BloodModel
from models.image_model import ImageModel

app = Flask(__name__)

# Initialize model instances
chat_model = ChatModel()
blood_model = BloodModel()
image_model = ImageModel()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/blood_sample')
def blood_sample():
    return render_template('blood_sample.html')

@app.route('/image_detection')
def image_detection():
    return render_template('image_detection.html')

@app.route('/predict_blood_sample', methods=['POST'])
def predict_blood_sample():
    try:
        data = request.json
        
        # Extract features in the correct order (as per the trained model)
        features = [
            float(data['age']),
            int(data['gender']),
            float(data['wbc_count']),
            float(data['rbc_count']),
            float(data['platelet_count']),
            float(data['hemoglobin_level']),
            float(data['bone_marrow_blasts']),
            int(data['family_history']),
            int(data['smoking_status']),
            int(data['radiation_exposure']),
            float(data['bmi']),
            int(data['infection_history'])
        ]
        
        # Convert to numpy array with shape (1, 12)
        import numpy as np
        features_array = np.array([features])
        
        # Make prediction using model module
        result = blood_model.predict(features_array)
        
        if not result.get('success', False):
            return jsonify({
                'error': result.get('error', 'Model not loaded'),
                'success': False
            }), 500
        
        return jsonify(result)
            
    except KeyError as e:
        return jsonify({
            'error': f'Missing required field: {str(e)}',
            'success': False
        }), 400
    except ValueError as e:
        return jsonify({
            'error': f'Invalid value provided: {str(e)}',
            'success': False
        }), 400
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}',
            'success': False
        }), 500

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """Handle chat requests"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message.strip():
            return jsonify({'error': 'Empty message'}), 400
        
        # Generate response using chat model module
        result = chat_model.chat(user_message)
        
        # Handle greeting responses with choices
        if result.get('is_greeting'):
            return jsonify({
                'is_greeting': True,
                'choices': result.get('choices', []),
                'status': 'success'
            })
        
        # Handle regular responses
        return jsonify({
            'response': result.get('response', 'No response generated'),
            'is_greeting': False,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/chat/stream', methods=['POST'])
def chat_stream_endpoint():
    """Handle streaming chat requests for the frontend."""
    try:
        data = request.get_json()
        user_message = data.get('message', '')

        if not user_message.strip():
            return jsonify({'error': 'Empty message'}), 400

        def token_stream():
            for event in chat_model.stream_chat(user_message):
                if event.get('error'):
                    yield f"data: {json.dumps({'error': event['error']})}\n\n"
                elif event.get('token'):
                    yield f"data: {json.dumps({'token': event['token']})}\n\n"
            yield "data: [DONE]\n\n"

        return Response(
            stream_with_context(token_stream()),
            mimetype='text/event-stream',
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
        )

    except Exception as e:
        def error_stream():
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

        return Response(
            stream_with_context(error_stream()),
            mimetype='text/event-stream',
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
        )

@app.route('/health')
def health():
    """Check if the model is loaded"""
    return jsonify({
        'status': 'ready' if chat_model.is_loaded() else 'loading',
        'model_loaded': chat_model.is_loaded(),
        'router_loaded': chat_model.router_pipe is not None,
        'error': chat_model.error or chat_model.router_error
    })

@app.route('/predict_skin_cancer', methods=['POST'])
def predict_skin_cancer():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read image bytes
        image_bytes = file.read()
        
        # Make prediction using image model module
        result = image_model.predict(image_bytes)
        
        if not result.get('success', False):
            return jsonify({
                'error': result.get('error', 'Model not loaded'),
                'success': False
            }), 500
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': f'Error processing image: {str(e)}',
            'success': False
        }), 500

if __name__ == '__main__':
    chat_model.initialize(preload_main=chat_model.can_preload_main_model())

    # Run the Flask app
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
