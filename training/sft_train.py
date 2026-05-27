"""Supervised fine-tuning with Unsloth QLoRA (Colab-friendly)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import unsloth  # noqa: F401  # must import before trl for Unsloth patches
except ImportError:
    pass

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from training.training_utils import load_unsloth_model, resolve_precision


def format_example(example: dict) -> dict:
    return {"text": f"{example['prompt']}\n{example['response']}"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Unsloth SFT for compliance environment.")
    parser.add_argument(
        "--model-id",
        default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        help="Base model (4-bit recommended for Colab T4).",
    )
    parser.add_argument("--dataset-path", default="training/data/sft_dataset_balanced.jsonl")
    parser.add_argument("--output-dir", default="training/checkpoints/sft")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--precision", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate dataset wiring without loading the model.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"SFT dataset not found: {dataset_path}")

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    required = {"prompt", "response"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"SFT dataset missing columns {missing}; run training/prepare_data.py first.")

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    print(f"SFT examples={len(dataset)} max_length={args.max_length}")

    if args.dry_run:
        print("Dry run OK — dataset ready for Unsloth SFT.")
        return

    model, tokenizer = load_unsloth_model(
        args.model_id,
        args.max_length,
        load_in_4bit=True,
        for_training=True,
    )

    use_bf16, use_fp16 = resolve_precision(args.precision)
    precision_name = "bf16" if use_bf16 else "fp16" if use_fp16 else "fp32"
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_params == 0:
        raise RuntimeError("No trainable LoRA parameters found.")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        report_to="none",
        max_length=args.max_length,
        dataset_text_field="text",
        packing=False,
    )

    print(
        f"SFT precision={precision_name} trainable_params={trainable_params} "
        f"output={args.output_dir}"
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"SFT checkpoint saved to {args.output_dir} (load with grpo_train.py --sft-checkpoint)")


if __name__ == "__main__":
    main()
