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

COMPLIANCE_SYSTEM_PROMPT = """\
You are an AI Compliance Officer. Audit employee expense claims against company policy.

Use the available action JSON types:
- SearchPolicy when policy details are hidden or unclear.
- RequestInformation when a required document is missing.
- ResolveTicket when you are ready to decide Approve, Reject, or Escalate.
- If missing_document is "required", infer the likely concrete document type
  (manager_approval, gst_invoice, vp_approval, or utility_bill) before requesting it.

Return only one valid JSON action for the next step.
"""

VALID_ACTION_TYPES = frozenset(
    {"SearchPolicy", "RequestInformation", "ResolveTicket"}
)
VALID_DECISIONS = frozenset({"Approve", "Reject", "Escalate"})
_OBSERVATION_FIELD_NAMES = frozenset(
    {
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
        "reward",
        "done",
        "ground_truth_decision",
    }
)
_INVALID_ACTION_FALLBACK: Dict[str, Any] = {
    "action_type": "ResolveTicket",
    "decision": "Reject",
    "reason": "Invalid action format",
}


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


def _infer_action_type(
    raw: Dict[str, Any], observation: Optional[Dict[str, Any]] = None
) -> str:
    action_type = raw.get("action_type")
    if isinstance(action_type, str) and action_type in VALID_ACTION_TYPES:
        return action_type

    if isinstance(action_type, str):
        compact = action_type.lower().replace("_", "").replace(" ", "")
        if "search" in compact or "policy" in compact:
            return "SearchPolicy"
        if "request" in compact or "information" in compact or "notify" in compact:
            return "RequestInformation"
        if "resolve" in compact or "ticket" in compact:
            return "ResolveTicket"

    action_label = str(raw.get("action") or "").lower()
    if "search" in action_label or "policy" in action_label:
        return "SearchPolicy"
    if (
        "request" in action_label
        or "inform" in action_label
        or "notify" in action_label
    ):
        return "RequestInformation"
    if (
        "resolve" in action_label
        or "approve" in action_label
        or "reject" in action_label
        or "escalate" in action_label
    ):
        return "ResolveTicket"
    if raw.get("query"):
        return "SearchPolicy"
    if raw.get("message"):
        return "RequestInformation"
    if raw.get("decision"):
        return "ResolveTicket"
    if observation:
        max_steps = int(observation.get("max_steps", 8) or 8)
        step_count = int(observation.get("step_count", 1) or 1)
        if step_count >= max_steps - 1:
            return "ResolveTicket"
        if observation.get("rule_keyword") == "hidden":
            return "SearchPolicy"
    if observation and int(observation.get("step_count", 1) or 1) <= 2:
        return "SearchPolicy"
    return "ResolveTicket"


def infer_required_document(observation: Optional[Dict[str, Any]]) -> str:
    if not observation:
        return "manager_approval"
    rule_keyword = str(observation.get("rule_keyword") or "").lower()
    description = str(observation.get("description") or "").lower()
    text = f"{rule_keyword} {description}"
    if "international" in text or "vp" in text:
        return "vp_approval"
    if "gst" in text or float(observation.get("amount", 0) or 0) > 5000:
        return "gst_invoice"
    if "wfh" in text or "internet" in text or "electricity" in text:
        return "utility_bill"
    return "manager_approval"


def normalize_compliance_action(
    raw: Any,
    observation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Map model output (often observation-shaped) into a valid ComplianceAction dict."""
    from app.models import ComplianceAction

    if not isinstance(raw, dict):
        return dict(_INVALID_ACTION_FALLBACK)

    cleaned = {
        k: v
        for k, v in raw.items()
        if k not in _OBSERVATION_FIELD_NAMES and k != "action"
    }
    action_type = _infer_action_type(raw, observation)

    normalized: Dict[str, Any] = {"action_type": action_type}
    if action_type == "SearchPolicy":
        query = cleaned.get("query") or raw.get("query")
        if not query and observation:
            query = observation.get("rule_keyword") or "policy"
        normalized["query"] = str(query or "policy")[:500]
    elif action_type == "RequestInformation":
        message = cleaned.get("message") or raw.get("message")
        if not message:
            action_label = str(raw.get("action") or "")
            label_low = action_label.lower()
            if action_label and not any(
                token in label_low
                for token in ("notify", "request", "inform", "employee")
            ):
                message = action_label
        if (not message or not str(message).strip()) and observation:
            missing = observation.get("missing_document")
            if missing:
                if str(missing).lower() == "required":
                    doc = infer_required_document(observation)
                else:
                    doc = str(missing).replace("_", " ")
                message = f"Please provide {doc}"
        normalized["message"] = str(message or "Please provide the required document.")[
            :500
        ]
    else:
        decision = cleaned.get("decision") or raw.get("decision")
        if isinstance(decision, str):
            title = decision.strip().title()
            normalized["decision"] = (
                title if title in VALID_DECISIONS else "Reject"
            )
        else:
            normalized["decision"] = "Reject"
        normalized["reason"] = str(
            cleaned.get("reason") or raw.get("reason") or "Model decision"
        )[:500]

    metadata = cleaned.get("metadata")
    if isinstance(metadata, dict):
        normalized["metadata"] = metadata

    try:
        return ComplianceAction.model_validate(normalized).model_dump()
    except Exception:
        return dict(_INVALID_ACTION_FALLBACK)


def parse_model_action(
    text: str, observation: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Parse generation text into a validated ComplianceAction dict."""
    payload = parse_json_payload(text)
    if payload:
        return normalize_compliance_action(payload, observation)

    if "SearchPolicy" in text:
        query_match = re.search(r'query["\']?\s*[:\-]?\s*["\']([^"\']+)["\']', text)
        return normalize_compliance_action(
            {
                "action_type": "SearchPolicy",
                "query": query_match.group(1) if query_match else "policy",
            },
            observation,
        )
    if "RequestInformation" in text:
        msg_match = re.search(r'message["\']?\s*[:\-]?\s*["\']([^"\']+)["\']', text)
        return normalize_compliance_action(
            {
                "action_type": "RequestInformation",
                "message": msg_match.group(1)
                if msg_match
                else "Please provide missing information",
            },
            observation,
        )
    if "ResolveTicket" in text:
        decision = "Reject"
        if "Approve" in text:
            decision = "Approve"
        elif "Escalate" in text:
            decision = "Escalate"
        reason_match = re.search(r'reason["\']?\s*[:\-]?\s*["\']([^"\']+)["\']', text)
        return normalize_compliance_action(
            {
                "action_type": "ResolveTicket",
                "decision": decision,
                "reason": reason_match.group(1) if reason_match else "Based on policy review",
            },
            observation,
        )

    return dict(_INVALID_ACTION_FALLBACK)


def render_compliance_prompt(
    tokenizer: Any,
    task_id: str,
    observation: Dict[str, Any],
) -> str:
    """Render chat-template prompt aligned with GRPO training."""
    user_prompt = build_step_prompt(task_id, observation)
    messages = [
        {"role": "system", "content": COMPLIANCE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
    return f"{COMPLIANCE_SYSTEM_PROMPT}\n\n{user_prompt}\n"


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
    extra = ""
    missing = clean.get("missing_document")
    step_count = int(clean.get("step_count", 1) or 1)
    max_steps = int(clean.get("max_steps", 8) or 8)
    if missing == "required":
        extra += (
            "\nThe ticket requires a document. Request the specific document type, "
            "not the word 'required'."
        )
    if step_count >= max_steps - 1:
        extra += "\nYou are near max steps; resolve now unless a required document is still pending."
    return (
        "You are an AI compliance officer. Return only valid action JSON.\n"
        f"Task: {task_id}\n"
        f"Ticket: {json.dumps(clean, ensure_ascii=True)}"
        f"{extra}"
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
