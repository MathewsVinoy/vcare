#!/bin/bash

# Cancer Care AI Flask App - Quick Diagnostic Script
# Run this if you encounter any issues

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     Cancer Care AI - Quick Diagnostic & Troubleshooting     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check status
check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $1"
        return 0
    else
        echo -e "${RED}✗${NC} $1"
        return 1
    fi
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. Checking Python Installation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 --version > /dev/null 2>&1
check_status "Python 3 is installed"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. Checking Required Files"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

[ -f "app.py" ]
check_status "app.py exists"

[ -f "templates/index.html" ]
check_status "templates/index.html exists"

[ -f "requirements.txt" ]
check_status "requirements.txt exists"

[ -d "phi3_lora_model" ]
check_status "phi3_lora_model/ directory exists"

[ -f "phi3_lora_model/adapter_model.safetensors" ]
check_status "Model weights file exists"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. Checking Dependencies"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 -c "import flask" > /dev/null 2>&1
check_status "Flask is installed" || echo "   → Install: pip install flask"

python3 -c "import torch" > /dev/null 2>&1
check_status "PyTorch is installed" || echo "   → Install: pip install torch"

python3 -c "import transformers" > /dev/null 2>&1
check_status "Transformers is installed" || echo "   → Install: pip install transformers"

python3 -c "import peft" > /dev/null 2>&1
check_status "PEFT is installed" || echo "   → Install: pip install peft"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4. Checking Port Availability"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠${NC} Port 5000 is already in use"
    echo "   → Kill process: lsof -ti:5000 | xargs kill -9"
    echo "   → Or change port in app.py"
else
    echo -e "${GREEN}✓${NC} Port 5000 is available"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5. Checking GPU"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} GPU detected and available"
    else
        echo -e "${YELLOW}⚠${NC} GPU drivers installed but not accessible"
    fi
else
    echo -e "${YELLOW}⚠${NC} No GPU detected - will use CPU (slower)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "6. Common Issues & Solutions"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "❓ Problem: 'Model not loaded'"
echo "   Solution: Ensure phi3_lora_model/ directory exists with all files"
echo ""
echo "❓ Problem: 'Out of memory'"
echo "   Solution: Close other applications or reduce max_new_tokens in app.py"
echo ""
echo "❓ Problem: 'Port already in use'"
echo "   Solution: lsof -ti:5000 | xargs kill -9"
echo ""
echo "❓ Problem: 'Slow responses'"
echo "   Solution: This is normal on CPU (30-90 seconds). Use GPU for faster responses."
echo ""
echo "❓ Problem: 'Module not found'"
echo "   Solution: pip install -r requirements.txt"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "7. Quick Fix Commands"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Install all dependencies:"
echo "  pip install -r requirements.txt"
echo ""
echo "Kill process on port 5000:"
echo "  lsof -ti:5000 | xargs kill -9"
echo ""
echo "Start the application:"
echo "  python app.py"
echo ""
echo "Check if server is running:"
echo "  curl http://localhost:5000/health"
echo ""
echo "Run full system check:"
echo "  python test_setup.py"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "8. Recommended Next Steps"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Read documentation:"
echo "   - START_HERE.md (complete overview)"
echo "   - QUICKSTART.md (quick reference)"
echo ""
echo "2. Install missing dependencies:"
echo "   pip install -r requirements.txt"
echo ""
echo "3. Run full diagnostic:"
echo "   python test_setup.py"
echo ""
echo "4. Start the application:"
echo "   python app.py"
echo ""
echo "5. Open in browser:"
echo "   http://localhost:5000"
echo ""

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                  Diagnostic Complete!                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
