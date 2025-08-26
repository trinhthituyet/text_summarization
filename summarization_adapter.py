from fasthtml.common import *
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from utils import load_model, predict, clean_text


# Config
base_model_id = "HuggingFaceTB/SmolLM-360M"
adapter_path = "./smol-lm-360m-finetuned_r8_alpha32"

# Load base tokenizer
tokenizer_base = AutoTokenizer.from_pretrained(base_model_id)
tokenizer_ft = AutoTokenizer.from_pretrained(adapter_path)

# Load base model (no adapter)
base_model = load_model(base_model_id)
base_model.eval()

# Load fine-tuned model (base + adapter)
ft_model = load_model(base_model_id)
ft_model = PeftModel.from_pretrained(ft_model, adapter_path)
ft_model.eval()

app, rt = fast_app()

@rt("/")
def home():
    return Titled("Summarizer Service",
        Form(
            H3("Document:"),
            Textarea(name="text", rows=8, cols=80, placeholder="Paste your document here..."),
            Select(
                Option("Base Model", value="base"),
                Option("Fine-tuned Model", value="ft", selected=True),
                name="model_choice",
                style="flex"
            ),
            Button("Summarize"),
            method="post",
            hx_post="/summarize",
            hx_target="#summary-box",
            hx_swap="innerHTML"
        ),
        Div(id="summary-box", cls="mt-4")
    )

@rt("/summarize", methods=["POST"])
def summarize(text: str, model_choice: str = "ft"):
    # Pick model
    if model_choice == "base":
        model = base_model
        tokenizer = tokenizer_base
    else:
        model = ft_model
        tokenizer = tokenizer_ft

    # Build prompt
    text = clean_text(text)
    prompt = f"Summarize this document:\n\n{text}\n\nSummary:"

    summary = predict(model, tokenizer, prompt)

    return Div(
        H3("Summary:", style="margin-top:20px;"),
        Textarea(
            summary,
            style="overflow:hidden; min-height:200px;",
        )
    )

if __name__ == "__main__":
    serve()
