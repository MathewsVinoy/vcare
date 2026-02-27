# Phi-3 Cancer Diagnosis Web Application

A Flask-based web application that provides an interactive chat interface for the Phi-3 medical AI assistant fine-tuned for cancer diagnosis information.

## Features

- 🎨 Modern, responsive chat interface
- 💬 Real-time conversation with Phi-3 AI model
- 🏥 Specialized in cancer-related medical information
- 📱 Mobile-friendly design
- ⚡ Fast and efficient responses

## Setup

1. **Install dependencies** (if not already installed):

```bash
pip install -r requirements.txt
```

2. **Run the Flask application**:

```bash
python app.py
```

3. **Access the web interface**:
   Open your browser and navigate to:

```
http://localhost:5000
```

## Project Structure

```
├── app.py                  # Flask application with API routes
├── test.py                 # Original CLI test script
├── templates/
│   └── index.html         # Chat interface HTML
├── static/
│   └── style.css          # Styling for the interface
├── phi3_lora_model/       # Fine-tuned LoRA model
└── model_cache/           # Cached base model
```

## API Endpoints

- `GET /` - Main chat interface
- `POST /chat` - Send message and get AI response
  - Request: `{"message": "your question here"}`
  - Response: `{"response": "AI response", "status": "success"}`
- `GET /health` - Check model loading status

## Usage

1. Open the web interface in your browser
2. Type your cancer-related medical question in the input box
3. Press Enter or click Send
4. Wait for the AI to generate a response
5. Continue the conversation as needed

## Example Questions

- "What are the common symptoms of breast cancer?"
- "How is leukemia diagnosed?"
- "What are the risk factors for lung cancer?"
- "Explain the difference between benign and malignant tumors"

## Important Notice

⚠️ **Medical Disclaimer**: This AI assistant is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns.

## Technical Details

- **Framework**: Flask 3.0.0
- **Model**: Microsoft Phi-3-mini-4k-instruct with LoRA fine-tuning
- **Frontend**: Vanilla JavaScript with modern CSS
- **Backend**: Python with PyTorch and Transformers

## Troubleshooting

- If the model takes time to load, wait for the "Model loaded successfully" message in the terminal
- For GPU issues, the model will automatically fall back to CPU
- Check the terminal for detailed error messages if something goes wrong
