"""
Cancer Care AI Flask App - API Examples and Testing
====================================================

This file contains example API requests and responses for testing
and integrating with the Cancer Care AI assistant.
"""

# =============================================================================
# PYTHON EXAMPLES
# =============================================================================

# Example 1: Simple chat request
import requests
import json

def send_message(message):
    """Send a message to the AI and get a response"""
    url = "http://localhost:5000/chat"
    headers = {"Content-Type": "application/json"}
    data = {"message": message}
    
    response = requests.post(url, json=data, headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        return result["response"]
    else:
        return f"Error: {response.status_code}"

# Usage:
response = send_message("What is leukemia?")
print(response)


# Example 2: Multiple conversation turns
def chat_conversation():
    """Demonstrate a multi-turn conversation"""
    session = requests.Session()
    base_url = "http://localhost:5000"
    
    messages = [
        "What are the symptoms of breast cancer?",
        "How is it diagnosed?",
        "What are the treatment options?"
    ]
    
    for message in messages:
        print(f"\nUser: {message}")
        
        response = session.post(
            f"{base_url}/chat",
            json={"message": message}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"AI: {result['response'][:100]}...")
    
    # Reset conversation
    session.post(f"{base_url}/reset")

# Usage:
# chat_conversation()


# Example 3: Health check
def check_health():
    """Check if the server and model are ready"""
    response = requests.get("http://localhost:5000/health")
    if response.status_code == 200:
        health = response.json()
        print(f"Status: {health['status']}")
        print(f"Model Loaded: {health['model_loaded']}")
        print(f"Device: {health['device']}")
    return response.json()

# Usage:
# check_health()


# =============================================================================
# CURL EXAMPLES
# =============================================================================

"""
# Send a chat message
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is lung cancer?"}'

# Response:
{
  "response": "Lung cancer is a type of cancer that begins in the lungs...",
  "success": true
}


# Reset conversation
curl -X POST http://localhost:5000/reset

# Response:
{
  "success": true
}


# Health check
curl http://localhost:5000/health

# Response:
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
"""


# =============================================================================
# JAVASCRIPT/FETCH EXAMPLES
# =============================================================================

"""
// Example 1: Send a message
async function sendMessage(message) {
    const response = await fetch('http://localhost:5000/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: message })
    });
    
    const data = await response.json();
    return data.response;
}

// Usage:
sendMessage("What is melanoma?").then(response => {
    console.log(response);
});


// Example 2: Check health
async function checkHealth() {
    const response = await fetch('http://localhost:5000/health');
    const data = await response.json();
    console.log('Server status:', data);
    return data;
}


// Example 3: Reset conversation
async function resetChat() {
    const response = await fetch('http://localhost:5000/reset', {
        method: 'POST'
    });
    return response.json();
}
"""


# =============================================================================
# TESTING SCENARIOS
# =============================================================================

def test_scenarios():
    """Test various cancer-related queries"""
    
    test_cases = [
        # Basic questions
        {
            "message": "What is cancer?",
            "expected_keywords": ["cells", "abnormal", "growth"]
        },
        
        # Specific cancer types
        {
            "message": "Tell me about leukemia",
            "expected_keywords": ["blood", "bone marrow", "white blood cells"]
        },
        
        # Symptoms
        {
            "message": "What are symptoms of lung cancer?",
            "expected_keywords": ["cough", "chest", "breathing"]
        },
        
        # Treatment
        {
            "message": "How is breast cancer treated?",
            "expected_keywords": ["surgery", "chemotherapy", "radiation"]
        },
        
        # Prevention
        {
            "message": "How can I prevent skin cancer?",
            "expected_keywords": ["sunscreen", "UV", "sun"]
        },
        
        # Diagnosis
        {
            "message": "How is colorectal cancer diagnosed?",
            "expected_keywords": ["colonoscopy", "biopsy", "screening"]
        },
        
        # Risk factors
        {
            "message": "What are risk factors for prostate cancer?",
            "expected_keywords": ["age", "family history", "men"]
        },
        
        # Follow-up questions (with context)
        {
            "message": "What are the stages?",
            "expected_keywords": ["stage", "I", "II", "III", "IV"]
        },
    ]
    
    session = requests.Session()
    base_url = "http://localhost:5000"
    
    print("=" * 60)
    print("CANCER CARE AI - TEST SCENARIOS")
    print("=" * 60)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test['message']}")
        print("-" * 60)
        
        response = session.post(
            f"{base_url}/chat",
            json={"message": test["message"]}
        )
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result["response"]
            
            # Check for expected keywords
            found_keywords = []
            for keyword in test["expected_keywords"]:
                if keyword.lower() in ai_response.lower():
                    found_keywords.append(keyword)
            
            print(f"✓ Response received ({len(ai_response)} chars)")
            print(f"✓ Keywords found: {found_keywords}")
            print(f"Preview: {ai_response[:150]}...")
        else:
            print(f"✗ Error: {response.status_code}")
    
    # Reset after tests
    session.post(f"{base_url}/reset")
    print("\n" + "=" * 60)
    print("Testing complete!")

# Usage:
# test_scenarios()


# =============================================================================
# PERFORMANCE TESTING
# =============================================================================

def test_performance():
    """Test response times and throughput"""
    import time
    
    session = requests.Session()
    base_url = "http://localhost:5000"
    
    test_messages = [
        "What is cancer?",
        "What are symptoms of leukemia?",
        "How is breast cancer treated?",
        "What are risk factors for lung cancer?",
        "How can I prevent colorectal cancer?"
    ]
    
    print("=" * 60)
    print("PERFORMANCE TEST")
    print("=" * 60)
    
    times = []
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n{i}. Sending: {message}")
        
        start_time = time.time()
        response = session.post(
            f"{base_url}/chat",
            json={"message": message}
        )
        end_time = time.time()
        
        elapsed = end_time - start_time
        times.append(elapsed)
        
        if response.status_code == 200:
            result = response.json()
            response_length = len(result["response"])
            print(f"   ✓ Time: {elapsed:.2f}s | Length: {response_length} chars")
        else:
            print(f"   ✗ Error: {response.status_code}")
    
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    print(f"Average response time: {sum(times)/len(times):.2f}s")
    print(f"Fastest response: {min(times):.2f}s")
    print(f"Slowest response: {max(times):.2f}s")
    
    # Reset after tests
    session.post(f"{base_url}/reset")

# Usage:
# test_performance()


# =============================================================================
# STRESS TESTING
# =============================================================================

def stress_test(num_requests=10):
    """Send multiple rapid requests to test stability"""
    import time
    import concurrent.futures
    
    def send_request(i):
        try:
            response = requests.post(
                "http://localhost:5000/chat",
                json={"message": f"What is cancer type #{i}?"},
                timeout=60
            )
            return (i, response.status_code, response.elapsed.total_seconds())
        except Exception as e:
            return (i, "Error", str(e))
    
    print("=" * 60)
    print(f"STRESS TEST - {num_requests} concurrent requests")
    print("=" * 60)
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(send_request, range(num_requests)))
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Analyze results
    successful = sum(1 for r in results if r[1] == 200)
    failed = num_requests - successful
    
    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Successful: {successful}/{num_requests}")
    print(f"Failed: {failed}/{num_requests}")
    print(f"Requests per second: {num_requests/total_time:.2f}")

# Usage (be careful with this - can overload the server):
# stress_test(10)


# =============================================================================
# INTEGRATION EXAMPLE - CHAT WIDGET
# =============================================================================

"""
<!-- Embeddable Chat Widget Example -->

<!DOCTYPE html>
<html>
<head>
    <style>
        #cancer-ai-widget {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 350px;
            height: 500px;
            border: 2px solid #667eea;
            border-radius: 10px;
            background: white;
            box-shadow: 0 5px 20px rgba(0,0,0,0.3);
            display: flex;
            flex-direction: column;
        }
        
        #widget-messages {
            flex: 1;
            overflow-y: auto;
            padding: 10px;
        }
        
        #widget-input {
            display: flex;
            padding: 10px;
            border-top: 1px solid #ddd;
        }
        
        #widget-input input {
            flex: 1;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        
        #widget-input button {
            margin-left: 5px;
            padding: 8px 15px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div id="cancer-ai-widget">
        <div id="widget-messages"></div>
        <div id="widget-input">
            <input type="text" id="widget-text" placeholder="Ask about cancer...">
            <button onclick="sendWidgetMessage()">Send</button>
        </div>
    </div>
    
    <script>
        async function sendWidgetMessage() {
            const input = document.getElementById('widget-text');
            const message = input.value.trim();
            
            if (!message) return;
            
            // Add user message
            addMessage('user', message);
            input.value = '';
            
            // Send to API
            const response = await fetch('http://localhost:5000/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message: message})
            });
            
            const data = await response.json();
            
            // Add AI response
            addMessage('ai', data.response);
        }
        
        function addMessage(type, text) {
            const messages = document.getElementById('widget-messages');
            const div = document.createElement('div');
            div.style.marginBottom = '10px';
            div.style.padding = '8px';
            div.style.borderRadius = '5px';
            
            if (type === 'user') {
                div.style.background = '#667eea';
                div.style.color = 'white';
                div.style.textAlign = 'right';
            } else {
                div.style.background = '#f0f0f0';
            }
            
            div.textContent = text;
            messages.appendChild(div);
            messages.scrollTop = messages.scrollHeight;
        }
    </script>
</body>
</html>
"""


# =============================================================================
# MAIN TESTING FUNCTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CANCER CARE AI - API TESTING SUITE")
    print("=" * 60)
    print("\nAvailable tests:")
    print("1. test_scenarios() - Test various cancer queries")
    print("2. test_performance() - Measure response times")
    print("3. stress_test() - Load testing")
    print("4. check_health() - Server health check")
    print("\nMake sure the Flask app is running on http://localhost:5000")
    print("=" * 60)
    
    # Uncomment to run tests:
    # check_health()
    # test_scenarios()
    # test_performance()
