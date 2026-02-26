#!/bin/bash

echo "Starting Cancer Care AI Assistant..."
echo "=================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Check if model exists
if [ ! -d "phi3_lora_model" ]; then
    echo "WARNING: phi3_lora_model directory not found!"
    echo "Please ensure your fine-tuned model is in the phi3_lora_model directory"
    exit 1
fi

# Run the Flask app
echo "Starting Flask application..."
echo "The app will be available at http://localhost:5000"
python app.py
