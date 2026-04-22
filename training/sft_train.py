"""Supervised fine-tuning for compliance action generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def format_example(example: dict) -> dict:
    text = f"{example['prompt']}\n{example['response']}"
    return {"text": text}


def resolve_precision(precision: str) -> tuple[bool, bool]:
    if precision == "bf16":
        return True, False
    if precision == "fp16":
        return False, True
    if precision == "fp32":
        return False, False
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return True, False
    if torch.cuda.is_available():
        return False, True
    return False, False


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT for compliance environment.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--dataset-path", default="training/data/sft_dataset.jsonl")
    parser.add_argument("--output-dir", default="training/checkpoints/sft")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--precision", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
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
            max_length=args.max_length,
            padding="max_length",
        )
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    use_bf16, use_fp16 = resolve_precision(args.precision)
    if use_bf16:
        model_dtype = torch.bfloat16
    elif use_fp16:
        model_dtype = torch.float16
    else:
        model_dtype = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=model_dtype)
    model.config.use_cache = False
    lora = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)

    # With gradient checkpointing + LoRA on frozen backbones, ensure checkpointed
    # activations keep a grad path to trainable adapter weights.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    if trainable_params == 0:
        raise RuntimeError("No trainable LoRA parameters found. Check target_modules/model compatibility.")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        report_to="none",
    )

    precision_name = "bf16" if use_bf16 else "fp16" if use_fp16 else "fp32"
    print(f"SFT precision={precision_name} max_length={args.max_length} trainable_params={trainable_params}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"SFT checkpoint saved to {args.output_dir}")


if __name__ == "__main__":
    main()
