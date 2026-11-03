import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM_PROMPT = "You are a helpful, concise AI assistant."

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
model.eval()


def respond(message, history, temperature, max_tokens):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history and isinstance(history[0], dict):
        messages += [{"role": m["role"], "content": m["content"]} for m in history]
    else:
        for u, b in (history or []):
            messages += [{"role": "user", "content": u}, {"role": "assistant", "content": b}]
    messages.append({"role": "user", "content": message})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Slider(0.0, 1.5, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(20, 400, value=200, step=10, label="Max new tokens"),
    ],
    title="LLM Chatbot",
    description="Transformer-based chatbot (Qwen2.5-0.5B-Instruct) built with Hugging Face Transformers and Gradio.",
    examples=[["What is overfitting?"], ["Explain gradient descent simply."]],
)

if __name__ == "__main__":
    demo.launch()
