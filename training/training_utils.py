"""Shared helpers for Colab/Unsloth training scripts."""

from __future__ import annotations

import inspect
import json
import re
from typing import Any, Dict, List, Optional, Sequence

# Curriculum stage definitions (OpenEnv course module-5 style progression)
CURRICULUM_STAGES: Dict[str, Dict[str, Any]] = {
    "stage_0_baseline": {"tasks": ["easy", "medium", "hard"], "weights": None},
    "stage_1_easy": {"tasks": ["easy"], "weights": {"easy": 1.0}},
    "stage_2_medium": {
        "tasks": ["easy", "medium"],
        "weights": {"easy": 0.3, "medium": 0.7},
    },
    "stage_3_hard": {
        "tasks": ["easy", "medium", "hard"],
        "weights": {"easy": 0.15, "medium": 0.35, "hard": 0.50},
    },
}

DEFAULT_LORA_TARGET = ["q_proj", "k_proj", "v_proj", "o_proj"]


def resolve_precision(precision: str) -> tuple[bool, bool]:
    import torch

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


def parse_json_payload(text: str) -> Optional[Dict[str, Any]]:
    candidate = text.strip()
    if not candidate:
        return None
    try:
        payload = json.loads(candidate)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        return None


def extract_task_id(text: str, default: str = "easy") -> str:
    match = re.search(r"Task:\s*(easy|medium|hard)", text)
    return match.group(1) if match else default


def build_step_prompt(task_id: str, observation: Dict[str, Any]) -> str:
    clean = {
        k: observation[k]
        for k in (
            "ticket_id",
            "employee_name",
            "employee_role",
            "employee_level",
            "amount",
            "currency",
            "description",
            "has_receipt",
            "missing_document",
            "rule_keyword",
            "risk_score",
            "env_message",
            "step_count",
            "max_steps",
        )
        if k in observation
    }
    return (
        "You are an AI compliance officer. Return only valid action JSON.\n"
        f"Task: {task_id}\n"
        f"Ticket: {json.dumps(clean, ensure_ascii=True)}"
    )


def filter_dataset_by_curriculum(dataset, stage: str):
    """Filter HF dataset rows by curriculum stage task list."""
    spec = CURRICULUM_STAGES.get(stage)
    if not spec:
        raise ValueError(
            f"Unknown stage '{stage}'. Choose from: {', '.join(CURRICULUM_STAGES)}"
        )
    allowed = set(spec["tasks"])

    def _keep(example: Dict[str, Any]) -> bool:
        return example.get("task_id", "easy") in allowed

    filtered = dataset.filter(_keep)
    weights = spec.get("weights")
    if not weights:
        return filtered

    # Oversample by repeating rows per task weights (simple curriculum weighting)
    rows = [dict(row) for row in filtered]
    buckets: Dict[str, List[Dict[str, Any]]] = {t: [] for t in allowed}
    for row in rows:
        buckets[row.get("task_id", "easy")].append(row)

    expanded: List[Dict[str, Any]] = []
    for task_id, task_rows in buckets.items():
        w = float(weights.get(task_id, 1.0))
        repeat = max(1, int(round(w * 10)))
        for row in task_rows:
            expanded.extend([row] * repeat)

    from datasets import Dataset

    return Dataset.from_list(expanded)


def get_trl_version() -> str:
    try:
        import trl  # type: ignore

        return getattr(trl, "__version__", "unknown")
    except Exception:
        return "missing"


def native_grpo_supports_rollout_func() -> bool:
    """Return True if upstream TRL GRPOTrainer exposes rollout_func natively."""
    try:
        from trl.trainer.grpo_trainer import GRPOTrainer
    except Exception:
        try:
            from trl import GRPOTrainer  # type: ignore
        except Exception:
            return False
    try:
        return "rollout_func" in inspect.signature(GRPOTrainer.__init__).parameters
    except Exception:
        return False


def _compat_grpo_available() -> bool:
    try:
        from training.grpo_trainer_compat import OpenEnvGRPOTrainer  # noqa: F401

        return True
    except Exception:
        return False


def grpo_supports_rollout_func() -> bool:
    return native_grpo_supports_rollout_func() or _compat_grpo_available()


def resolve_grpo_trainer():
    """Resolve native GRPOTrainer when available, otherwise use compat shim."""
    if native_grpo_supports_rollout_func():
        from trl import GRPOTrainer  # type: ignore

        return GRPOTrainer
    from training.grpo_trainer_compat import OpenEnvGRPOTrainer

    return OpenEnvGRPOTrainer


def require_rollout_dependencies(
    require_generate: bool = True,
    *,
    check_unsloth: bool = True,
) -> None:
    """Fail fast when Colab training prerequisites are missing."""
    if check_unsloth:
        try:
            from unsloth import FastLanguageModel  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "unsloth is required for Colab training. Install with training/requirements-training.txt"
            ) from exc

    if not grpo_supports_rollout_func():
        raise RuntimeError(
            "Installed TRL GRPOTrainer does not support rollout_func. "
            "Install training/requirements-training.txt for the supported Colab stack."
        )

    if require_generate:
        try:
            from training.rollout_generation import generate_rollout_completions  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "training.rollout_generation.generate_rollout_completions is required for "
                "multi-turn GRPO. Ensure training dependencies are installed."
            ) from exc


def load_unsloth_model(
    model_id: str,
    max_seq_length: int,
    *,
    load_in_4bit: bool = True,
    for_training: bool = True,
):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_id,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if for_training:
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=DEFAULT_LORA_TARGET,
            lora_alpha=32,
            lora_dropout=0.0,
            use_gradient_checkpointing="unsloth",
        )
    return model, tokenizer
