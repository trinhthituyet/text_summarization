import pandas as pd
import torch
import gc
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq
)
from huggingface_hub import login
from utils import load_data_from_hugging_face, load_model
import evaluate

def preprocess_text(tokenizer, examples):
    prompt = f"Summarize this document:\n\n{examples['article']}\n\nSummary:"
    target = examples["highlights"]

    tokenized_prompt = tokenizer(prompt, truncation=True, max_length=768, padding=False)
    tokenized_target = tokenizer(target, truncation=True, max_length=64, add_special_tokens=False)

    # Combine the tokenized prompt and target
    input_ids = tokenized_prompt["input_ids"] + tokenized_target["input_ids"]
    attention_mask = [1] * len(input_ids)

    # Create labels where the prompt part is masked with -100
    labels = [-100] * len(tokenized_prompt["input_ids"]) + tokenized_target["input_ids"]
    
    # Append EOS token to the end of input_ids and labels
    input_ids.append(tokenizer.eos_token_id)
    attention_mask.append(1)
    labels.append(tokenizer.eos_token_id)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

def train_summarizer(train_path, validation_path, model_name="HuggingFaceTB/SmolLM-360M"):
    # Load model tokenizer
    print(f"Loading tokenizer for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and preprocess datasets
    print("Preparing datasets...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(validation_path)
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # Apply the preprocessing function to the datasets
    train_dataset = train_dataset.map(
        lambda x: preprocess_text(tokenizer, x),
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        lambda x: preprocess_text(tokenizer, x),
        remove_columns=val_dataset.column_names
    )

    model = load_model(model_name)

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable() # Conserves memory
    print("Trainable parameters:")
    model.print_trainable_parameters()

    # Data collator for language modeling
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./smol-lm-360m",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32, 
        optim="paged_adamw_8bit", # Use 8-bit optimizer to save memory
        num_train_epochs=1, 
        learning_rate=1e-4,
        fp16=True, 
        save_strategy="steps",
        save_steps=500,
        eval_strategy="steps",
        eval_steps=200,
        logging_steps=100,
        report_to="none",
        save_total_limit=2,
    )
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Start training
    print("Starting model training...")
    trainer.train()

    # Save the fine-tuned LoRA adapter
    final_model_path = "./smol-lm-360m-finetuned"
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Fine-tuned model adapter saved to {final_model_path}")

    # Clean up memory
    del model, trainer, train_dataset, val_dataset
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    train_summarizer(
        model_name="HuggingFaceTB/SmolLM-360M", 
        train_path="data/cnn_dailymail_train_10000.parquet", 
        validation_path="data/cnn_dailymail_validation.parquet"
    )
    #load_data_from_hugging_face("test", 500)

