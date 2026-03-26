# System Architecture Diagrams

## High-Level Overview

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                          WEB BROWSER                            ┃
┃                   http://localhost:5000                         ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                                 │
                    ┌────────────┴────────────┐
                    │                         │
        ┌───────────▼────────────┐   ┌───────▼──────────┐
        │   Flask Web Server     │   │  Static Assets   │
        │   (app.py)             │   │  HTML/CSS/JS     │
        │   Port 5000            │   │                  │
        └───────────┬────────────┘   └──────────────────┘
                    │
        ┌───────────┴──────────┬─────────────┬──────────────┐
        │                      │             │              │
    ┌───▼────┐          ┌──────▼──┐  ┌──────▼────┐  ┌──────▼────┐
    │ Chat   │          │ Blood   │  │ Image     │  │ External  │
    │ Routes │          │ Routes  │  │ Routes    │  │ LLM Check │
    └───┬────┘          └─────────┘  └───────────┘  └──────┬────┘
        │                                                   │
        ├──────────────────────────────────────────────────┤
        │                                                   │
    ┌───▼────────────────────────────┐         ┌──────────▼──┐
    │   ChatModel                    │         │ External    │
    │   (models/chat_model.py)       │         │ LLM Client  │
    │                                │         │             │
    │ ├─ Greeting detection (local)  │         │ Checks:     │
    │ ├─ Cache lookup (local)        │         │ ├─ Health   │
    │ ├─ Validation (local)          │         │ ├─ Timeout  │
    │ ├─ Router fallback (local)     │         │ └─ Errors   │
    │ └─ External LLM routing        ─────────►│             │
    └────────────────────────────────┘         └──────┬──────┘
        │                                             │
        │          ┌──────────────────────────────────┤
        │          │                                  │
    ┌───▼────┐    │                        ┌─────────▼────────┐
    │ Local  │    │   HTTPS via ngrok      │ Google Colab     │
    │ Phi-3  │    │   (External LLM URL)   │                  │
    │ Model  │    │                        │ ├─ Mistral-7B    │
    │ (GPU)  │    └───────────────────────►│ ├─ Flask Server  │
    └────────┘                             │ ├─ GPU Accel.    │
                                           │ └─ Public Tunnel │
    Blood Model                            └──────────────────┘
    Image Model
    (Both local, always available)
```

---

## Request Flow Diagram

```
┌─────────────────────────┐
│  User types message     │
│  in web browser         │
└────────────┬────────────┘
             │
             ▼
    ┌────────────────────┐
    │  POST /chat        │
    │  (Local app)       │
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────────────┐
    │  ChatModel.chat()          │
    │  (models/chat_model.py)    │
    └────────┬───────────────────┘
             │
    ┌────────▼────────────────────────────────────────┐
    │  Check if it's a greeting                       │
    └────┬───────────────────────────────────────────┘
         │
         ├─────► Yes ─────► Generate greeting locally ──┐
         │                  (3 choices)                 │
         │                                              │
         └─────► No ────────► Check cache ─────────────┐
                             │                         │
                    ┌────────▼────────────────┐         │
                    │  Found in cache? │       │         │
                    └────┬──────────┬──────────┘         │
                         │          │                   │
                    Yes ─┘          └─ No               │
                    │                   │               │
         ┌──────────▼────────┐ ┌────────▼───────────────┐
         │ Return from cache │ │ Check if medical      │
         │ (instant!)        │ │                       │
         └──────────┬────────┘ └────────┬───────────────┘
                    │                   │
                    │              ┌────▼──────────────┐
                    │              │ Is medical?       │
                    │              └────┬────────┬─────┘
                    │                   │        │
                    │          Yes ─────┘   No ──┘
                    │          │              │
         ┌──────────┴──────────▼────┐    ┌───▼────────────┐
         │  Check if external LLM   │    │ Return error   │
         │  is configured           │    │ "Out of scope" │
         └────┬──────────┬──────────┘    └────┬───────────┘
              │          │                    │
         Yes ─┘     No ──┘                    │
         │          │                         │
    ┌────▼──────┐   │                         │
    │   Use     │   │                         │
    │ Colab LLM │   │                         │
    │           │   │                         │
    │ External  │   │                         │
    │ LLMClient │   │                         │
    │  request  │   │                         │
    │  ↓ HTTPS  │   │                         │
    │  ↓ ngrok  │   │                         │
    │  to Colab │   │                         │
    │  LLM      │   │                         │
    └────┬──────┘   │                         │
         │      ┌───▼──────┐                  │
         │      │   Use    │                  │
         │      │  Local   │                  │
         │      │ Phi-3    │                  │
         │      │ Model    │                  │
         │      └───┬──────┘                  │
         │          │                         │
    ┌────▼──────┬───▼──────┬─────────────────▼──┐
    │            Response Generated              │
    │            (from LLM or local or error)    │
    └────┬──────────────────────────────────────┘
         │
         ▼
    ┌──────────────────┐
    │  Cache response  │
    │  (for next time) │
    └────┬─────────────┘
         │
         ▼
    ┌──────────────────┐
    │  Return JSON     │
    │  to browser      │
    └────┬─────────────┘
         │
         ▼
    ┌──────────────────┐
    │  Browser shows   │
    │  response to user│
    └──────────────────┘
```

---

## Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Flask App (app.py)                        │
│                                                                   │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐   │
│  │ Routes:        │  │ Blood Model    │  │ Image Model      │   │
│  │ ├─ /           │  │ ├─ Random      │  │ ├─ EfficientNet  │   │
│  │ ├─ /health     │  │ │ Forest       │  │ │ vision model   │   │
│  │ ├─ /chat       │  │ ├─ CatBoost    │  │ ├─ HAM10000      │   │
│  │ ├─ /chat/stream│  │ └─ Prediction  │  │ │ dataset        │   │
│  │ └─ Blood/Image │  └────────────────┘  └──────────────────┘   │
│  │   endpoints    │                                              │
│  └────────┬───────┘                                              │
│           │                                                      │
│           ▼                                                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  ChatModel (models/chat_model.py)                         │ │
│  │                                                            │ │
│  │  ┌──────────────────┐  ┌──────────────────────────────┐  │ │
│  │  │ Local Processing │  │ External LLM (if available) │  │ │
│  │  │                  │  │                              │  │ │
│  │  │ ├─ Greeting      │  │ ├─ Check health             │  │ │
│  │  │ ├─ Cache check   │  │ ├─ Send request             │  │ │
│  │  │ ├─ Validation    │  │ ├─ Parse response           │  │ │
│  │  │ ├─ Router model  │  │ └─ Stream tokens            │  │ │
│  │  │ └─ Phi-3 (local) │  │                              │  │ │
│  │  │                  │  │   ExternalLLMClient         │  │ │
│  │  └────────┬─────────┘  └──────────┬───────────────────┘  │ │
│  │           │                       │                       │ │
│  │           └───────────┬───────────┘                       │ │
│  │                       │                                   │ │
│  └───────────────────────┼───────────────────────────────────┘ │
│                          │                                      │
└──────────────────────────┼──────────────────────────────────────┘
                           │
                   ┌───────▼────────┐
                   │  HTTPS/ngrok   │
                   │   (External    │
                   │   LLM URL)     │
                   └───────┬────────┘
                           │
                   ┌───────▼──────────────┐
                   │  Google Colab        │
                   │                      │
                   │ ┌──────────────────┐ │
                   │ │ Mistral-7B LLM   │ │
                   │ │ (7B parameters)  │ │
                   │ │ 4-bit quantized  │ │
                   │ └──────────────────┘ │
                   │ ┌──────────────────┐ │
                   │ │ Flask Server     │ │
                   │ │ (on GPU T4/A100) │ │
                   │ └──────────────────┘ │
                   │ ┌──────────────────┐ │
                   │ │ ngrok Tunnel     │ │
                   │ │ (public URL)     │ │
                   │ └──────────────────┘ │
                   └──────────────────────┘
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────┐
│         User Browser                │
│         localhost:5000              │
│                                     │
│  Chat Form                          │
│  ├─ Message input field             │
│  ├─ Send button                     │
│  └─ Response display area           │
└────────────────┬────────────────────┘
                 │ HTTP POST
                 │ {"message": "..."}
                 │
┌────────────────▼────────────────────┐
│      Flask App (app.py)             │
│      Port 5000                      │
│                                     │
│  /chat endpoint                     │
│  ├─ Parse JSON request              │
│  ├─ Validate message                │
│  └─ Call chat_model.chat()          │
└────────────────┬────────────────────┘
                 │
┌────────────────▼────────────────────┐
│  ChatModel (chat_model.py)          │
│                                     │
│  Processing Decision Tree:          │
│  1. Is greeting?                    │
│  2. Is cached?                      │
│  3. Is medical?                     │
│  4. Use external or local?          │
│  5. Generate/retrieve response      │
└────────────────┬────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
   ┌────▼─────┐      ┌───▼──────────┐
   │   Cache  │      │ Determination│
   │   (LRU)  │      │              │
   │          │      │ ├─ Greeting  │
   │ 128 max  │      │ ├─ Local     │
   │ responses│      │ └─ External  │
   └────┬─────┘      └───┬──────────┘
        │                │
   ┌────▼────────────────▼──────┐
   │  Response Generation       │
   │                            │
   │  Option 1: From Cache      │ (1ms)
   │  Option 2: Local Model     │ (2-5s)
   │  Option 3: External LLM    │ (3-5s)
   │  Option 4: Error Message   │ (instant)
   └────┬─────────────────────────┘
        │
        │ HTTPS for external
        │
   ┌────▼──────────────────────────┐
   │ Cache Updated                │
   │ (for repeated questions)      │
   └────┬──────────────────────────┘
        │
   ┌────▼──────────────────────────┐
   │ JSON Response                │
   │ {                            │
   │   "response": "...",         │
   │   "is_greeting": false,      │
   │   "status": "success"        │
   │ }                            │
   └────┬──────────────────────────┘
        │ HTTP 200
        │
┌───────▼──────────────────┐
│  Browser Display         │
│  ├─ Show response text   │
│  ├─ Update chat history  │
│  └─ Clear input field    │
└──────────────────────────┘
```

---

## Failure Mode Diagram

```
┌─────────────────────────────────────┐
│  External LLM Request Initiated      │
└────────────────┬────────────────────┘
                 │
                 ▼
         ┌───────────────┐
         │ Check health  │
         │ (at init)     │
         └───┬───────┬───┘
             │       │
        ✓ OK │       │ ✗ Down
           │       │
       ┌───▼─┐   ┌──▼────────────┐
       │ Use │   │ Set flag:     │
       │Colab│   │ use_external  │
       │ LLM │   │ = False       │
       └─────┘   └──┬─────────────┘
                    │
                    └──────────┐
                               │
         ┌─────────────────────▼──────┐
         │  Send request to Colab     │
         │  (with 30s timeout)        │
         └────┬──────────┬─────────────┘
              │          │
          ✓ OK│          │ ✗ Timeout/Error
            │          │
        ┌───▼┐      ┌──▼──────────┐
        │Use │      │ Fallback:   │
        │Resp│      │ Use local   │
        │onse│      │ Phi-3 model │
        └────┘      └──┬──────────┘
                       │
                       └─────────────┐
                                     │
                            ┌────────▼──────┐
                            │ Generate      │
                            │ response      │
                            │ locally       │
                            └───────────────┘
```

---

## Configuration Diagram

```
┌────────────────────────────────────────────────────────────┐
│  Environment Setup                                         │
│                                                            │
│  Option A: Environment Variable (Recommended)             │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ export EXTERNAL_LLM_URL="https://xxxxx.ngrok.io"   │ │
│  │ python app.py                                       │ │
│  └──────────────────────────────────────────────────────┘ │
│                           │                               │
│  Option B: Hardcode in app.py                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ EXTERNAL_LLM_URL = "https://xxxxx.ngrok.io"        │ │
│  │ python app.py                                       │ │
│  └──────────────────────────────────────────────────────┘ │
│                           │                               │
│  Option C: No Config (Local Model Only)                  │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ # Don't set EXTERNAL_LLM_URL                         │ │
│  │ python app.py                                       │ │
│  └──────────────────────────────────────────────────────┘ │
│                           │                               │
│                           ▼                               │
│  ┌───────────────────────────────────────────────────┐   │
│  │ ChatModel Initialization                          │   │
│  │                                                   │   │
│  │ if external_llm_url:                              │   │
│  │   ├─ Create ExternalLLMClient(url)                │   │
│  │   ├─ Check health                                 │   │
│  │   └─ Set use_external_llm = True/False            │   │
│  │ else:                                              │   │
│  │   └─ Set use_external_llm = False                 │   │
│  │       (will use local model)                       │   │
│  │                                                   │   │
│  └──────────────────┬────────────────────────────────┘   │
│                     │                                    │
│                     ▼                                    │
│  ┌───────────────────────────────────────────────────┐   │
│  │ At Runtime                                        │   │
│  │                                                   │   │
│  │ if use_external_llm:                              │   │
│  │   └─ Route to ExternalLLMClient                   │   │
│  │ else:                                              │   │
│  │   └─ Route to local Phi-3                         │   │
│  └───────────────────────────────────────────────────┘   │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## Deployment Diagram

```
┌──────────────────────────┐
│  Your Local Machine      │
│  ┌────────────────────┐  │
│  │ Flask App          │  │
│  │ (app.py)           │  │
│  │ Port 5000          │  │
│  │                    │  │
│  │ Models:            │  │
│  │ ├─ ChatModel       │  │
│  │ ├─ BloodModel      │  │
│  │ └─ ImageModel      │  │
│  └────────┬───────────┘  │
│           │              │
└───────────┼──────────────┘
            │ HTTP/HTTPS
            │ (via ngrok)
            │
      ┌─────▼──────────────┐
      │  Internet          │
      │  (ngrok tunnel)    │
      └─────┬──────────────┘
            │
┌───────────▼──────────────────┐
│  Google Colab Notebook       │
│  ┌────────────────────────┐  │
│  │ Mistral-7B Model       │  │
│  │ (7B parameters)        │  │
│  │                        │  │
│  │ 4-bit Quantization     │  │
│  │ Memory: ~4GB           │  │
│  │                        │  │
│  │ T4/V100/A100 GPU       │  │
│  │ (Free!)                │  │
│  │                        │  │
│  │ Flask Server           │  │
│  │ (colab_llm_server.     │  │
│  │  ipynb)                │  │
│  └────────────────────────┘  │
└───────────────────────────────┘
```

---

These diagrams help visualize:

1. **High-level architecture** - Component overview
2. **Request flow** - Step-by-step message processing
3. **Component interaction** - How parts communicate
4. **Data flow** - Information movement through system
5. **Failure modes** - What happens when things go wrong
6. **Configuration** - Setup options
7. **Deployment** - Physical system layout

Use these when explaining the system to others or planning modifications!
