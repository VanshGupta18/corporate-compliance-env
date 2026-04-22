"""Publish a trained PEFT adapter and tokenizer to Hugging Face Hub."""

from __future__ import annotations

import argparse

from huggingface_hub import login
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish PEFT adapter to HF Hub.")
    parser.add_argument("--checkpoint", default="training/checkpoints/grpo")
    parser.add_argument("--repo-id", required=True, help="HF repo id, e.g. user/compliance-grpo-adapter")
    parser.add_argument("--token", default=None, help="HF token (optional if already logged in)")
    args = parser.parse_args()

    if args.token:
        login(token=args.token)

    model = AutoPeftModelForCausalLM.from_pretrained(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)

    model.push_to_hub(args.repo_id)
    tokenizer.push_to_hub(args.repo_id)
    print(f"Published adapter to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
