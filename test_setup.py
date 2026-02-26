#!/usr/bin/env python3
"""
Test script to verify the Cancer Care AI Flask app setup
"""

import os
import sys

def print_status(message, status):
    """Print colored status message"""
    symbols = {"✓": "✓", "✗": "✗", "→": "→"}
    colors = {"green": "", "red": "", "yellow": "", "reset": ""}
    
    if status == "success":
        print(f"{symbols['✓']} {message}")
    elif status == "error":
        print(f"{symbols['✗']} {message}")
    else:
        print(f"{symbols['→']} {message}")

def check_python_version():
    """Check if Python version is adequate"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print_status(f"Python version {version.major}.{version.minor}.{version.micro} OK", "success")
        return True
    else:
        print_status(f"Python version {version.major}.{version.minor}.{version.micro} - Need 3.8+", "error")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = {
        'flask': 'Flask',
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'peft': 'PEFT (LoRA)',
    }
    
    all_installed = True
    for module, name in required_packages.items():
        try:
            __import__(module)
            print_status(f"{name} installed", "success")
        except ImportError:
            print_status(f"{name} NOT installed", "error")
            all_installed = False
    
    return all_installed

def check_model_files():
    """Check if model files exist"""
    model_dir = "phi3_lora_model"
    required_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json"
    ]
    
    if not os.path.exists(model_dir):
        print_status(f"Model directory '{model_dir}' NOT found", "error")
        return False
    
    print_status(f"Model directory '{model_dir}' found", "success")
    
    all_files_exist = True
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            size_mb = size / (1024 * 1024)
            print_status(f"  {file} ({size_mb:.2f} MB)", "success")
        else:
            print_status(f"  {file} NOT found", "error")
            all_files_exist = False
    
    return all_files_exist

def check_templates():
    """Check if templates exist"""
    template_file = "templates/index.html"
    if os.path.exists(template_file):
        print_status(f"Template file '{template_file}' found", "success")
        return True
    else:
        print_status(f"Template file '{template_file}' NOT found", "error")
        return False

def check_app_file():
    """Check if app.py exists"""
    if os.path.exists("app.py"):
        print_status("app.py found", "success")
        return True
    else:
        print_status("app.py NOT found", "error")
        return False

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print_status(f"CUDA available - {gpu_name}", "success")
            return True
        else:
            print_status("CUDA not available - will use CPU (slower)", "info")
            return False
    except:
        return False

def main():
    print("=" * 60)
    print("Cancer Care AI Flask App - System Check")
    print("=" * 60)
    print()
    
    results = {
        "Python Version": check_python_version(),
        "Dependencies": check_dependencies(),
        "Model Files": check_model_files(),
        "Template Files": check_templates(),
        "App File": check_app_file(),
    }
    
    print()
    print("=" * 60)
    print("GPU Check:")
    print("=" * 60)
    check_cuda()
    
    print()
    print("=" * 60)
    print("Summary:")
    print("=" * 60)
    
    if all(results.values()):
        print_status("All checks passed! Ready to run the app.", "success")
        print()
        print("To start the application, run:")
        print("  python app.py")
        print()
        print("Then open: http://localhost:5000")
        return 0
    else:
        print_status("Some checks failed. Please fix the issues above.", "error")
        print()
        print("Common fixes:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Ensure model files are in phi3_lora_model/ directory")
        print("  - Verify all project files are present")
        return 1

if __name__ == "__main__":
    exit(main())
