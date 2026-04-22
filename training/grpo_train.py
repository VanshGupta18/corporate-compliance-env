"""GRPO post-training for compliance decision making."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import requests
import torch
from datasets import load_dataset
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel


class JsonlMetricsCallback(TrainerCallback):
    """Write trainer log events to JSONL for the monitor dashboard."""

    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        payload = {"step": int(state.global_step), **logs}
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def action_json_reward(completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
    rewards: List[float] = []
    for completion in completions:
        content = completion[0]["content"]
        try:
            payload = json.loads(content)
            valid = payload.get("action_type") in {"SearchPolicy", "RequestInformation", "ResolveTicket"}
            rewards.append(1.0 if valid else 0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


def environment_reward(completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
    api_url = kwargs.get("api_url", "http://127.0.0.1:7860")
    task_ids = kwargs.get("task_id", [])
    rewards: List[float] = []
    for completion, task_id in zip(completions, task_ids):
        content = completion[0]["content"]
        try:
            payload = json.loads(content)
            requests.post(f"{api_url}/reset", params={"task_id": task_id}, timeout=30).raise_for_status()
            result = requests.post(f"{api_url}/step", json={"action": payload}, timeout=30)
            result.raise_for_status()
            rewards.append(float(result.json().get("reward", 0.0)))
        except Exception:
            rewards.append(-0.5)
    return rewards


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
    parser = argparse.ArgumentParser(description="Train GRPO policy on compliance environment.")
    parser.add_argument("--model-id", default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
    parser.add_argument("--dataset-path", default="training/data/sft_dataset.jsonl")
    parser.add_argument("--output-dir", default="training/checkpoints/grpo")
    parser.add_argument("--api-url", default="http://127.0.0.1:7860")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--log-file", default="training/logs/grpo_metrics.jsonl")
    parser.add_argument("--precision", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    args = parser.parse_args()

    use_bf16, use_fp16 = resolve_precision(args.precision)
    precision_name = "bf16" if use_bf16 else "fp16" if use_fp16 else "fp32"
    print(f"GRPO precision={precision_name} max_seq_length={args.max_seq_length} num_generations={args.num_generations}")

    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    def to_prompt(example: Dict[str, Any]) -> Dict[str, Any]:
        return {"prompt": example["prompt"], "task_id": example["task_id"]}

    dataset = dataset.map(to_prompt, remove_columns=dataset.column_names)

    model, tokenizer = FastLanguageModel.from_pretrained(
        args.model_id,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=32,
        lora_dropout=0.0,
        use_gradient_checkpointing="unsloth",
    )

    config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=128,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[action_json_reward, environment_reward],
        args=config,
        train_dataset=dataset,
        reward_kwargs={"api_url": args.api_url},
        callbacks=[JsonlMetricsCallback(args.log_file)],
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Saved GRPO adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
