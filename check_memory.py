#!/usr/bin/env python3
"""
Memory Check Script - Test what your system can handle
"""

import sys
import psutil

def get_memory_info():
    """Get system memory information"""
    mem = psutil.virtual_memory()
    return {
        'total': mem.total / (1024**3),  # GB
        'available': mem.available / (1024**3),  # GB
        'percent': mem.percent
    }

def recommend_configuration():
    """Recommend best configuration based on available memory"""
    mem = get_memory_info()
    
    print("="*60)
    print("🔍 MEMORY CHECK FOR MEDICAL DIAGNOSIS APP")
    print("="*60)
    print(f"\n💾 System Memory:")
    print(f"  Total:     {mem['total']:.1f} GB")
    print(f"  Available: {mem['available']:.1f} GB")
    print(f"  Used:      {mem['percent']:.1f}%")
    print()
    
    # Memory requirements
    blood_model = 0.1
    image_model = 0.3
    base_app = 0.2
    chat_full = 5.0
    chat_8bit = 1.2
    chat_4bit = 0.6
    
    base_required = blood_model + image_model + base_app
    
    print("📊 Model Memory Requirements:")
    print(f"  Base App + Models:    {base_required:.1f} GB")
    print(f"  Chat (Standard):      {chat_full:.1f} GB")
    print(f"  Chat (8-bit):         {chat_8bit:.1f} GB")
    print(f"  Chat (4-bit):         {chat_4bit:.1f} GB")
    print()
    
    print("="*60)
    print("🎯 RECOMMENDED CONFIGURATION")
    print("="*60)
    
    # Recommendations
    if mem['available'] >= (base_required + chat_full + 1):
        print("\n✅ Your system has plenty of memory!")
        print(f"\nRecommended: Standard Mode (No Quantization)")
        print(f"  Available: {mem['available']:.1f} GB")
        print(f"  Required:  {base_required + chat_full:.1f} GB")
        print(f"  Buffer:    {mem['available'] - (base_required + chat_full):.1f} GB")
        print(f"\nCommand:")
        print(f"  python app.py")
        
    elif mem['available'] >= (base_required + chat_8bit + 0.5):
        print("\n⚠️  Moderate memory available")
        print(f"\nRecommended: 8-bit Quantization")
        print(f"  Available: {mem['available']:.1f} GB")
        print(f"  Required:  {base_required + chat_8bit:.1f} GB")
        print(f"  Buffer:    {mem['available'] - (base_required + chat_8bit):.1f} GB")
        print(f"\nCommand:")
        print(f"  pip install bitsandbytes")
        print(f"  USE_8BIT_QUANTIZATION=true python app.py")
        
    elif mem['available'] >= (base_required + chat_4bit + 0.3):
        print("\n⚠️  Low memory available")
        print(f"\nRecommended: 4-bit Quantization")
        print(f"  Available: {mem['available']:.1f} GB")
        print(f"  Required:  {base_required + chat_4bit:.1f} GB")
        print(f"  Buffer:    {mem['available'] - (base_required + chat_4bit):.1f} GB")
        print(f"\nCommand:")
        print(f"  pip install bitsandbytes")
        print(f"  USE_4BIT_QUANTIZATION=true python app.py")
        
    else:
        print("\n❌ Insufficient memory!")
        print(f"\nYour system may not have enough memory to run this app.")
        print(f"  Available: {mem['available']:.1f} GB")
        print(f"  Minimum:   {base_required + chat_4bit:.1f} GB (with 4-bit)")
        print(f"\nOptions:")
        print(f"  1. Close other applications to free up memory")
        print(f"  2. Use a system with more RAM")
        print(f"  3. Run without the chat feature (comment out model loading)")
    
    print("\n" + "="*60)
    print("\n💡 Tips:")
    print("  • The app uses lazy loading - chat model loads on first use")
    print("  • Close browser tabs and other apps before starting")
    print("  • Monitor memory during runtime with: watch -n 1 free -h")
    print("  • Quantization slightly reduces quality but saves memory")
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    try:
        recommend_configuration()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure psutil is installed: pip install psutil")
        sys.exit(1)
