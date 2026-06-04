import os

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor

BASE_MODEL = os.environ.get("BASE_MODEL", "google/gemma-4-12B-it")
ADAPTER_MODEL = os.environ.get("ADAPTER_MODEL", "TUSTResearcher/gemma4tunnel")

model = None
processor = None


def load_model():
    global model, processor
    if model is not None:
        return model, processor
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    base = AutoModelForMultimodalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, ADAPTER_MODEL)
    model.eval()
    return model, processor


def respond(message, history, max_new_tokens=512, temperature=0.3):
    m, proc = load_model()
    messages = [
        {
            "role": "system",
            "content": "You are Gemma4Tunnel, a careful tunnel boring machine and tunnelling engineering assistant. State assumptions and do not invent citations.",
        }
    ]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    inputs = proc.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(m.device)

    with torch.no_grad():
        output = m.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=0.9,
        )
    text = proc.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return text.strip()


demo = gr.ChatInterface(
    fn=respond,
    title="Gemma4Tunnel",
    description="Research assistant for tunnel boring machine and tunnelling engineering. Use source-grounded RAG for paper-specific citations.",
    additional_inputs=[
        gr.Slider(64, 2048, value=512, step=64, label="max_new_tokens"),
        gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="temperature"),
    ],
)

if __name__ == "__main__":
    demo.launch()
