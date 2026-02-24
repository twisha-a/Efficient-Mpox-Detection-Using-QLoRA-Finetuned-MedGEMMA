import torch
import numpy as np
from PIL import Image
import gradio as gr
from transformers import AutoProcessor, AutoModelForImageTextToText
import os

HF_TOKEN = os.environ.get("HF_TOKEN", "")
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

CLASSES = ["A: mpox", "B: not mpox"]

PROMPT = (
    "What is the most likely diagnosis for the skin lesion in the image?\n"
    "Choose exactly one of the following:\n"
    + "\n".join(CLASSES)
    + "\nAnswer with the full label string exactly as above."
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_dir = "/gpfs/data/coffem01lab/Aneesh/medgemma-4b-it-qlora-mpox-2gpu/checkpoint-1253"

processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, token=HF_TOKEN)
model = AutoModelForImageTextToText.from_pretrained(
    model_dir,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    trust_remote_code=True,
    token=HF_TOKEN,
).to(device).eval()


@torch.inference_mode()
def predict(image):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": PROMPT},
        ],
    }]

    prompt_templ = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    scores = []
    for opt in CLASSES:
        enc = processor(images=image, text=prompt_templ + opt, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        labels = enc["input_ids"].clone()
        labels[:, :-len(opt.split())] = -100

        out = model(**enc, labels=labels)
        scores.append(float(out.loss))

    nlls = np.array(scores)
    probs = np.exp(-nlls)
    probs = probs / probs.sum()

    result = {
        "Prediction": CLASSES[np.argmin(nlls)],
    }

    return result


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="json",
    title="Mpox Detection Demo",
    description="Upload a skin lesion image to classify as mpox or not mpox."
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
