import torch, gc
import pandas as pd
import evaluate
from utils import load_data, load_model, get_tokenizer, predict
from peft import PeftModel


def compute_rouge_similarity(predicted, target):
    rouge = evaluate.load("rouge")
    rouge_scores = rouge.compute(predictions=predicted, references=target)
    print("ROUGE:", rouge_scores)
    return rouge_scores

def evaluate_model(test_path, base_model="HuggingFaceTB/SmolLM-360M", adapter_path="None"):
    model = load_model(base_model)
    if adapter_path != "None":
        model = PeftModel.from_pretrained(model, adapter_path)
        tokenizer = get_tokenizer(adapter_path)
    else:
        tokenizer = get_tokenizer(base_model)
    model.eval()

    
    test_data = load_data(test_path)

    predicted_data = test_data["article"].apply(lambda x: predict(
        model=model, 
        tokenizer=tokenizer, 
        prompt=f"Summarize this document:\n\n{x}\n\nSummary:"
        )
    )

    rouge = evaluate.load("rouge")
    results = rouge.compute(
        predictions=predicted_data.tolist(),
        references=test_data["highlights"].tolist(),
        use_stemmer=True
    )
    print(results)


if __name__ == "__main__":
    evaluate_model(
        base_model="HuggingFaceTB/SmolLM-360M", 
        #adapter_path="./smol-lm-360m-finetuned_r8_alpha64", 
        test_path="data/cnn_dailymail_test_200.parquet"
    )
