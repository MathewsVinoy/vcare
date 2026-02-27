# Phi-3 Cancer Diagnosis Web Application - Docker Setup

A Flask-based web application that provides an interactive chat interface for the Phi-3 medical AI assistant fine-tuned for cancer diagnosis information.

## 🐳 Quick Start with Docker (Recommended for Windows)

### Prerequisites

1. Install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
2. Ensure Docker Desktop is running

### Running with Docker Compose (Easiest)

1. **Open PowerShell or Command Prompt** in the project directory

2. **Build and start the container**:
```bash
docker-compose up --build
```

3. **Access the web interface**:
   - Open your browser and go to: http://localhost:5000

4. **Stop the application**:
   - Press `Ctrl+C` in the terminal, then run:
```bash
docker-compose down
```

### Running with Docker (Manual)

1. **Build the Docker image**:
```bash
docker build -t phi3-cancer-diagnosis .
```

2. **Run the container**:
```bash
docker run -p 5000:5000 -v "%cd%\model_cache:/app/model_cache" -v "%cd%\phi3_lora_model:/app/phi3_lora_model" phi3-cancer-diagnosis
```

For PowerShell, use:
```powershell
docker run -p 5000:5000 -v "${PWD}\model_cache:/app/model_cache" -v "${PWD}\phi3_lora_model:/app/phi3_lora_model" phi3-cancer-diagnosis
```

3. **Access the web interface**:
   - Open your browser and go to: http://localhost:5000

4. **Stop the container**:
   - Press `Ctrl+C` or run:
```bash
docker stop phi3-cancer-diagnosis
```

## 📁 Project Structure

```
├── app.py                  # Flask application with API routes
├── test.py                 # Original CLI test script
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose configuration
├── .dockerignore          # Docker ignore file
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html         # Chat interface HTML
├── static/
│   └── style.css          # Styling for the interface
├── phi3_lora_model/       # Fine-tuned LoRA model
└── model_cache/           # Cached base model
```

## 🌟 Features

- 🎨 Modern, responsive chat interface
- 💬 Real-time conversation with Phi-3 AI model
- 🏥 Specialized in cancer-related medical information
- 📱 Mobile-friendly design
- ⚡ Fast and efficient responses
- 🐳 Docker support for easy deployment

## 🔧 Troubleshooting

### Windows-Specific Issues

1. **Docker Desktop not starting**:
   - Ensure WSL2 is installed and enabled
   - Restart Docker Desktop

2. **Port 5000 already in use**:
   - Change the port in docker-compose.yml:
   ```yaml
   ports:
     - "8080:5000"  # Use port 8080 instead
   ```

3. **Model loading takes too long**:
   - First run will download the base model (~7GB)
   - Subsequent runs will be faster as models are cached
   - Check Docker Desktop resources (Settings > Resources)

4. **Out of memory errors**:
   - Increase Docker memory in Docker Desktop settings
   - Recommended: At least 8GB RAM allocated to Docker

### Volume Mounting Issues

If you see "model not found" errors:
```bash
# Ensure the model directories exist and are accessible
docker-compose down -v
docker-compose up --build
```

## 💻 Alternative: Run Without Docker

If Docker doesn't work, you can run directly on Windows:

1. **Install Python 3.10+** from [python.org](https://www.python.org/)

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the app**:
```bash
python app.py
```

4. **Access**: http://localhost:5000

## 📝 API Endpoints

- `GET /` - Main chat interface
- `POST /chat` - Send message and get AI response
  - Request: `{"message": "your question here"}`
  - Response: `{"response": "AI response", "status": "success"}`
- `GET /health` - Check model loading status

## 💡 Example Questions

- "What are the common symptoms of breast cancer?"
- "How is leukemia diagnosed?"
- "What are the risk factors for lung cancer?"
- "Explain the difference between benign and malignant tumors"

## ⚠️ Important Notice

**Medical Disclaimer**: This AI assistant is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns.

## 🔧 Technical Details

- **Framework**: Flask 3.0.0
- **Model**: Microsoft Phi-3-mini-4k-instruct with LoRA fine-tuning
- **Frontend**: Vanilla JavaScript with modern CSS
- **Backend**: Python with PyTorch and Transformers
- **Container**: Docker with Python 3.10

## 🚀 Advanced Configuration

### Enable GPU Support (NVIDIA only)

Uncomment the GPU section in docker-compose.yml and install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Development Mode

Uncomment volume mounts in docker-compose.yml to enable hot-reloading:
```yaml
volumes:
  - ./app.py:/app/app.py
  - ./templates:/app/templates
  - ./static:/app/static
```

## 📞 Support

For issues specific to:
- **Docker**: Check Docker Desktop logs
- **Model loading**: Ensure phi3_lora_model directory exists
- **Network**: Try http://127.0.0.1:5000 instead of localhost
