"""Shared helpers for Colab/Unsloth training scripts."""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Curriculum stage definitions (OpenEnv course module-5 style progression)
CURRICULUM_STAGES: Dict[str, Dict[str, Any]] = {
    "stage_0_baseline": {"tasks": ["easy", "medium", "hard"], "weights": None},
    # Content-driven complexity stages (no rule_keyword signal — model deduces from claim facts)
    "stage_1_direct": {
        "tasks": ["easy"],
        "weights": {"easy": 1.0},
    },
    "stage_2_search": {
        "tasks": ["easy", "medium"],
        "weights": {"easy": 0.40, "medium": 0.60},
    },
    "stage_3_full": {
        "tasks": ["easy", "medium", "hard"],
        "weights": {"easy": 0.20, "medium": 0.35, "hard": 0.45},
    },
    # Legacy stage names kept for backward compatibility
    "stage_1_easy": {"tasks": ["easy"], "weights": {"easy": 1.0}},
    "stage_2_medium": {"tasks": ["easy", "medium"], "weights": {"easy": 0.3, "medium": 0.7}},
    "stage_3_hard": {
        "tasks": ["easy", "medium", "hard"],
        "weights": {"easy": 0.15, "medium": 0.35, "hard": 0.50},
    },
}

DEFAULT_LORA_TARGET = ["q_proj", "k_proj", "v_proj", "o_proj"]

COMPLIANCE_SYSTEM_PROMPT = """\
You are an AI Compliance Officer. Review employee expense claims against company policy.

For each claim, decide which action to take next based solely on the claim facts:

Use SearchPolicy when the claim details suggest a policy threshold you need to verify
before deciding — for example: meal expenses near or above ₹500/₹2,000, amounts above
₹5,000 (GST rule), cab rides (day vs night), WFH allowances, or international travel.
Use a short topical query: meal, gst, cab, wfh, international.

Use RequestInformation when a required document is flagged as missing
(missing_document is set) and you have already retrieved the relevant policy.
Name the specific document type explicitly.

Use ResolveTicket when you have all information needed to decide.
Return Approve, Reject, or Escalate with a brief reason citing the applicable rule.

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
        "policy_retrieved",
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

_INVALID_SEARCH_QUERIES = frozenset({"hidden", "unknown", ""})


def sanitize_search_query(
    query: Any,
    observation: Optional[Dict[str, Any]] = None,
    claim: Optional[Dict[str, Any]] = None,
) -> str:
    """Map missing or placeholder queries to a real policy keyword."""
    from app.agent_helpers import search_query_for_hidden_policy

    text = str(query or "").strip()
    if text.lower() not in _INVALID_SEARCH_QUERIES:
        return text[:500]

    # No rule_keyword hint available anymore — derive from claim content
    return search_query_for_hidden_policy(observation or {}, claim)[:500]


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
        if (
            observation.get("rule_keyword") == "hidden"
            and not observation.get("policy_retrieved")
        ):
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
    claim: Optional[Dict[str, Any]] = None,
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
        normalized["query"] = sanitize_search_query(query, observation, claim)
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


def _extract_decision_from_text(
    text: str, payload: Optional[Dict[str, Any]] = None
) -> str:
    """Parse ResolveTicket decision without substring false positives."""
    if payload:
        raw = payload.get("decision")
        if isinstance(raw, str) and raw.strip():
            title = raw.strip().title()
            if title in VALID_DECISIONS:
                return title

    field_match = re.search(
        r'decision["\']?\s*[:\-]?\s*["\']?(Approve|Reject|Escalate)["\']?',
        text,
        re.IGNORECASE,
    )
    if field_match:
        return field_match.group(1).title()

    for label in ("Reject", "Escalate", "Approve"):
        if re.search(rf"\b{label}\b", text):
            return label
    return "Reject"


def parse_model_action(
    text: str,
    observation: Optional[Dict[str, Any]] = None,
    claim: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Parse generation text into a validated ComplianceAction dict."""
    payload = parse_json_payload(text)
    if payload:
        return normalize_compliance_action(payload, observation, claim)

    if "SearchPolicy" in text:
        query_match = re.search(r'query["\']?\s*[:\-]?\s*["\']([^"\']+)["\']', text)
        return normalize_compliance_action(
            {
                "action_type": "SearchPolicy",
                "query": query_match.group(1) if query_match else "policy",
            },
            observation,
            claim,
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
            claim,
        )
    if "ResolveTicket" in text:
        reason_match = re.search(r'reason["\']?\s*[:\-]?\s*["\']([^"\']+)["\']', text)
        return normalize_compliance_action(
            {
                "action_type": "ResolveTicket",
                "decision": _extract_decision_from_text(text),
                "reason": reason_match.group(1) if reason_match else "Based on policy review",
            },
            observation,
            claim,
        )

    return dict(_INVALID_ACTION_FALLBACK)


def render_compliance_prompt(
    tokenizer: Any,
    observation: Dict[str, Any],
    task_id: str | None = None,
) -> str:
    """Render chat-template prompt aligned with GRPO training."""
    del task_id  # curriculum label is internal; not shown to the model
    user_prompt = build_step_prompt(observation)
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


def build_step_prompt(observation: Dict[str, Any]) -> str:
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
            "policy_retrieved",
            "risk_score",
            "env_message",
            "step_count",
            "max_steps",
        )
        if k in observation
    }
    extra = ""
    if clean.get("policy_retrieved"):
        extra += "\nPolicy already retrieved. Do not SearchPolicy again."
        if not clean.get("missing_document"):
            extra += " ResolveTicket now; no further document requests."
    missing = clean.get("missing_document")
    step_count = int(clean.get("step_count", 1) or 1)
    max_steps = int(clean.get("max_steps", 8) or 8)
    if missing == "required":
        from app.document_utils import infer_required_document

        hint = infer_required_document(clean)
        extra += (
            f"\nDocument required. After search, request '{hint}' explicitly "
            "(not the word 'required'), then resolve."
        )
    elif missing:
        extra += f"\nRequest missing document '{missing}', then resolve."
    if step_count >= max_steps - 1:
        extra += "\nYou are near max steps; resolve now unless a required document is still pending."
    return (
        "You are an AI compliance officer. Return only valid action JSON.\n"
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


def clear_unsloth_compiled_cache() -> None:
    """Remove stale Unsloth compiled trainers (avoids GRPO NameError bugs)."""
    import shutil

    cache = Path.cwd() / "unsloth_compiled_cache"
    if cache.is_dir():
        shutil.rmtree(cache, ignore_errors=True)


def inject_unsloth_grpo_helpers() -> None:
    """
    Inject align_completion_tool_mask into Unsloth's compiled GRPO module.

    Unsloth 2026.4.x + TRL 0.24 can compile grpo_accumulated_loss that calls
    align_completion_tool_mask without defining it in module scope (NameError).
    """
    try:
        from unsloth_zoo.rl_replacements import (
            align_completion_tool_mask,
            align_logprobs_with_mask,
        )
    except ImportError:
        return

    import sys

    helpers = {
        "align_completion_tool_mask": align_completion_tool_mask,
        "align_logprobs_with_mask": align_logprobs_with_mask,
    }
    for mod in sys.modules.values():
        if mod is None:
            continue
        if callable(getattr(mod, "grpo_accumulated_loss", None)):
            for name, fn in helpers.items():
                setattr(mod, name, fn)


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


def _looks_like_local_checkpoint(model_id: str) -> bool:
    if model_id.startswith((".", "/")) or "checkpoints" in model_id:
        return True
    path = Path(model_id)
    return path.exists()


def validate_training_checkpoint(model_id: str) -> None:
    """Fail fast when a local SFT/GRPO checkpoint path is missing or incomplete."""
    if not _looks_like_local_checkpoint(model_id):
        return

    path = Path(model_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Training checkpoint not found: {model_id}\n"
            "Run SFT warm-start before GRPO (notebook section 4):\n"
            "  python -m training.sft_train "
            "--dataset-path training/data/sft_dataset_balanced.jsonl "
            "--output-dir training/checkpoints/sft"
        )

    has_adapter = (path / "adapter_config.json").is_file()
    has_base_config = (path / "config.json").is_file()
    if not has_adapter and not has_base_config:
        raise FileNotFoundError(
            f"Incomplete checkpoint at {model_id}: missing adapter_config.json.\n"
            "Re-run SFT and confirm training/checkpoints/sft contains adapter weights."
        )


def load_unsloth_model(
    model_id: str,
    max_seq_length: int,
    *,
    load_in_4bit: bool = True,
    for_training: bool = True,
):
    from unsloth import FastLanguageModel

    validate_training_checkpoint(model_id)
    path = Path(model_id)
    is_saved_adapter = path.is_dir() and (path / "adapter_config.json").is_file()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_id,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Saved LoRA adapters already include PEFT weights; do not stack a new adapter.
    if for_training and not is_saved_adapter:
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=DEFAULT_LORA_TARGET,
            lora_alpha=32,
            lora_dropout=0.0,
            use_gradient_checkpointing="unsloth",
        )
    return model, tokenizer
