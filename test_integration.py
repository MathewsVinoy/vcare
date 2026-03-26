#!/usr/bin/env python3
"""
Test script to validate Colab LLM integration
Tests both local app and Colab connection
"""

import requests
import json
import sys
import time
from typing import Dict, Any

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_test(name: str):
    print(f"\n{Colors.BLUE}▶ {name}{Colors.END}")

def print_pass(msg: str):
    print(f"  {Colors.GREEN}✓ {msg}{Colors.END}")

def print_fail(msg: str):
    print(f"  {Colors.RED}✗ {msg}{Colors.END}")

def print_info(msg: str):
    print(f"  {Colors.YELLOW}ℹ {msg}{Colors.END}")

def test_local_health() -> bool:
    """Test local Flask app health"""
    print_test("Testing Local Flask Health")
    try:
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print_pass(f"Local app responding")
            print_info(f"Status: {data.get('status', 'unknown')}")
            return True
        else:
            print_fail(f"Unexpected status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_fail("Cannot connect to local Flask app at http://localhost:5000")
        print_info("Make sure app.py is running: python app.py")
        return False
    except requests.exceptions.Timeout:
        print_fail("Local app health check timeout")
        return False
    except Exception as e:
        print_fail(f"Error: {str(e)}")
        return False

def test_external_llm_health(url: str) -> bool:
    """Test Colab LLM server health"""
    print_test(f"Testing External LLM Health")
    print_info(f"URL: {url}")
    
    try:
        response = requests.get(f"{url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print_pass(f"Colab LLM server responding")
            print_info(f"Status: {data.get('status', 'unknown')}")
            print_info(f"Model: {data.get('model_name', 'unknown')}")
            print_info(f"GPU available: {data.get('gpu_available', False)}")
            return True
        else:
            print_fail(f"Server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_fail(f"Cannot connect to Colab LLM server at {url}")
        print_info("Make sure:")
        print_info("1. colab_llm_server.ipynb is running in Colab")
        print_info("2. URL is correct")
        print_info("3. ngrok tunnel is active")
        return False
    except requests.exceptions.Timeout:
        print_fail(f"Colab LLM server health check timeout")
        return False
    except Exception as e:
        print_fail(f"Error: {str(e)}")
        return False

def test_local_chat() -> bool:
    """Test local chat endpoint"""
    print_test("Testing Local Chat Endpoint")
    
    test_messages = [
        "Hello",
        "What is diabetes?",
        "Hi there"
    ]
    
    for message in test_messages:
        try:
            response = requests.post(
                "http://localhost:5000/chat",
                json={"message": message},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                status = data.get('status', 'unknown')
                is_greeting = data.get('is_greeting', False)
                
                if is_greeting:
                    print_pass(f"'{message}' → Greeting (3 choices)")
                else:
                    response_text = data.get('response', '')[:50]
                    print_pass(f"'{message}' → {response_text}...")
            else:
                print_fail(f"'{message}' returned status {response.status_code}")
                return False
                
        except requests.exceptions.Timeout:
            print_fail(f"'{message}' request timeout")
            return False
        except Exception as e:
            print_fail(f"'{message}' error: {str(e)}")
            return False
    
    return True

def test_blood_sample() -> bool:
    """Test blood sample prediction endpoint"""
    print_test("Testing Blood Sample Prediction")
    
    sample_data = {
        "gender": 0,
        "age": 35,
        "hb": 13.5,
        "rbc": 4.5,
        "wbc": 7000,
        "platelets": 250000,
        "lymp": 30,
        "mono": 5,
        "hct": 40,
        "mcv": 90,
        "mch": 30,
        "mchc": 33,
        "rdw": 12,
        "pdw": 15,
        "mpv": 8,
        "pct": 0.2
    }
    
    try:
        response = requests.post(
            "http://localhost:5000/predict_blood_sample",
            json=sample_data,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                prediction = data.get('raw_prediction')
                probability = data.get('probability')
                print_pass(f"Blood prediction: {prediction} (prob: {probability:.2%})")
                return True
            else:
                print_fail(f"Prediction failed: {data.get('error')}")
                return False
        else:
            print_fail(f"Server returned status {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print_fail("Blood prediction timeout")
        return False
    except Exception as e:
        print_fail(f"Error: {str(e)}")
        return False

def test_external_chat(url: str) -> bool:
    """Test chat on external Colab LLM"""
    print_test(f"Testing External LLM Chat")
    
    test_message = "What are symptoms of diabetes?"
    
    try:
        response = requests.post(
            f"{url}/chat",
            json={"prompt": test_message, "max_tokens": 100},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            response_text = data.get('response', '')
            if response_text:
                print_pass(f"LLM responded with {len(response_text)} characters")
                print_info(f"Response preview: {response_text[:100]}...")
                return True
            else:
                print_fail(f"Empty response from LLM")
                return False
        else:
            print_fail(f"Server returned status {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print_fail("LLM chat timeout (model may be loading)")
        return False
    except Exception as e:
        print_fail(f"Error: {str(e)}")
        return False

def main():
    import os
    
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}VCare AI - Integration Test Suite{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")
    
    # Get external LLM URL
    external_url = os.getenv("EXTERNAL_LLM_URL")
    
    if external_url:
        print_info(f"Using external LLM: {external_url}")
    else:
        print_info("No external LLM configured (using local model)")
    
    # Test counter
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Local health
    tests_total += 1
    if test_local_health():
        tests_passed += 1
    
    # Test 2: Local chat
    tests_total += 1
    if test_local_chat():
        tests_passed += 1
    
    # Test 3: Blood sample
    tests_total += 1
    if test_blood_sample():
        tests_passed += 1
    
    # Test 4: External LLM (if configured)
    if external_url:
        tests_total += 1
        if test_external_llm_health(external_url):
            tests_passed += 1
        
        tests_total += 1
        if test_external_chat(external_url):
            tests_passed += 1
    
    # Summary
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}Test Summary{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")
    
    if tests_passed == tests_total:
        print_pass(f"All tests passed! ({tests_passed}/{tests_total})")
        print(f"\n{Colors.GREEN}✓ System ready for production!{Colors.END}\n")
        return 0
    else:
        print_fail(f"Some tests failed ({tests_passed}/{tests_total})")
        print(f"\n{Colors.RED}✗ Please fix issues above{Colors.END}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
