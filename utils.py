import polars as pl
import pandas as pd
import torch, gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import re
import unicodedata
from bs4 import BeautifulSoup

def load_data_from_hugging_face(split, n_samples=1000):
    splits = {'train': '1.0.0/train-*.parquet', 'validation': '1.0.0/validation-00000-of-00001.parquet', 'test': '1.0.0/test-00000-of-00001.parquet'}
    df = pl.read_parquet('hf://datasets/abisee/cnn_dailymail/' + splits[split])
    df_sample = df.sample(n_samples)
    df_sample.write_parquet(f"data/cnn_dailymail_{split}.parquet")
    print(df_sample.head())

def load_data(path):
    df = pd.read_parquet(path)
    print(df.head())
    return df

def load_model(model_name):
    # Configure 4-bit quantization
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    # Load base model with quantization
    print(f"Loading base model: {model_name} with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
    )
    return model

def get_tokenizer(model_name):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def predict(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response[len(prompt):]
    del outputs  # delete unused tensors
    gc.collect()
    torch.cuda.empty_cache()
    return answer

def clean_text(text: str) -> str:
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    #  Normalize unicode (handles \u00A0 and other odd spaces)
    text = unicodedata.normalize("NFKC", text)
    # Replace smart quotes and dashes with plain versions
    replacements = {
        "“": '"', "”": '"', "‘": "'", "’": "'",
        "–": "-", "—": "-", "…": "...",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    # Remove emojis / non-word symbols (except common punctuation)
    text = re.sub(r"[^\w\s.,!?;:'\"-]", "", text)
    # Replace multiple spaces or newlines with a single space
    text = re.sub(r"\s+", " ", text)
    # Collapse multiple newlines → one newline
    text = re.sub(r"\n+", "\n", text)
    # Strip leading/trailing spaces
    return text.strip()