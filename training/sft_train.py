"""Supervised fine-tuning for compliance action generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def format_example(example: dict) -> dict:
    text = f"{example['prompt']}\n{example['response']}"
    return {"text": text}


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT for compliance environment.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--dataset-path", default="training/data/sft_dataset.jsonl")
    parser.add_argument("--output-dir", default="training/checkpoints/sft")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch: dict) -> dict:
        encoded = tokenizer(
            batch["text"],
            truncation=True,
            max_length=1024,
            padding="max_length",
        )
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype="auto", device_map="auto")
    lora = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"SFT checkpoint saved to {args.output_dir}")


if __name__ == "__main__":
    main()
