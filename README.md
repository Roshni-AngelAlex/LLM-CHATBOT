---
title: LLM Chatbot
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# Transformer-based LLM Chatbot

Chatbot using **Qwen2.5-0.5B-Instruct** (Hugging Face Transformers) with a **Gradio** chat interface.

## Features
- Pre-trained Transformer LLM with tokenizer and chat template
- Conversation history support
- Adjustable temperature and maximum output length

## Run locally
```
pip install -r requirements.txt
python app.py
```

## Limitations
- Can hallucinate facts
- Weak at maths and complex reasoning (small model)
- Slower on CPU; latency grows with output length
- Knowledge limited to training data
