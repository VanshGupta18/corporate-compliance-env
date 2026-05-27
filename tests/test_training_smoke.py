"""Unit tests for training utilities (no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from datasets import Dataset

from training.training_utils import (
    CURRICULUM_STAGES,
    build_step_prompt,
    extract_task_id,
    filter_dataset_by_curriculum,
    infer_required_document,
    normalize_compliance_action,
    parse_json_payload,
    parse_model_action,
    validate_training_checkpoint,
)


def test_validate_training_checkpoint_missing_local_path():
    with pytest.raises(FileNotFoundError, match="Training checkpoint not found"):
        validate_training_checkpoint("training/checkpoints/sft")


def test_normalize_compliance_action_maps_notify_and_strips_observation_fields():
    obs = {
        "employee_name": "Arjun Nair",
        "missing_document": "receipt",
        "rule_keyword": "travel",
        "step_count": 1,
    }
    raw = {
        "action": "Notify Employee",
        "employee_name": "Arjun Nair",
        "employee_role": "Sales Executive",
        "employee_level": "L4",
        "risk_score": 0.34,
    }
    action = normalize_compliance_action(raw, obs)
    assert action["action_type"] == "RequestInformation"
    assert "employee_name" not in action
    assert "receipt" in action["message"].lower()


def test_parse_model_action_invalid_json_fallback():
    action = parse_model_action("not json at all")
    assert action["action_type"] == "ResolveTicket"
    assert action["decision"] == "Reject"


def test_normalize_compliance_action_infers_required_doc_type():
    obs = {
        "missing_document": "required",
        "description": "Vendor software license renewal",
        "rule_keyword": "gst",
        "amount": 12000,
    }
    action = normalize_compliance_action({"action_type": "RequestInformation"}, obs)
    assert action["action_type"] == "RequestInformation"
    assert "gst_invoice" in action["message"]
    assert infer_required_document(obs) == "gst_invoice"


def test_parse_json_payload_extracts_object():
    text = 'noise {"action_type": "SearchPolicy", "query": "travel"} tail'
    payload = parse_json_payload(text)
    assert payload is not None
    assert payload["action_type"] == "SearchPolicy"


def test_extract_task_id():
    assert extract_task_id("Task: hard\nTicket: {}") == "hard"
    assert extract_task_id("no task") == "easy"


def test_build_step_prompt_no_leakage():
    obs = {
        "ticket_id": "t1",
        "employee_name": "A",
        "employee_role": "Eng",
        "employee_level": "L3",
        "amount": 100.0,
        "currency": "USD",
        "description": "trip",
        "has_receipt": True,
        "missing_document": None,
        "rule_keyword": "travel",
        "risk_score": 0.2,
        "env_message": "",
        "step_count": 1,
        "max_steps": 5,
        "reward": 1.0,
        "done": False,
    }
    prompt = build_step_prompt("medium", obs)
    assert "reward" not in prompt
    assert "done" not in prompt
    assert "Task: medium" in prompt


@pytest.mark.parametrize(
    "stage,expected",
    [
        ("stage_1_easy", {"easy"}),
        ("stage_2_medium", {"easy", "medium"}),
        ("stage_3_hard", {"easy", "medium", "hard"}),
    ],
)
def test_curriculum_filter(stage, expected):
    ds = Dataset.from_list(
        [
            {"prompt": "e", "task_id": "easy"},
            {"prompt": "m", "task_id": "medium"},
            {"prompt": "h", "task_id": "hard"},
        ]
    )
    filtered = filter_dataset_by_curriculum(ds, stage)
    got = {r["task_id"] for r in filtered}
    assert expected <= got


def test_sft_dataset_schema_if_present():
    path = Path("training/data/sft_dataset.jsonl")
    if not path.exists():
        pytest.skip("sft_dataset.jsonl not generated yet")
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert {"prompt", "response"} <= set(row.keys())


def test_curriculum_stages_defined():
    assert "stage_1_easy" in CURRICULUM_STAGES
    assert "stage_3_hard" in CURRICULUM_STAGES


def test_local_rollout_returns_batched_tutorial_contract(monkeypatch):
    from training import grpo_train

    def fake_generate_step(_trainer, _prompt):
        return {
            "prompt_ids": [1, 2],
            "completion_ids": [3, 4],
            "logprobs": [-0.1, -0.2],
            "text": json.dumps(
                {
                    "action_type": "ResolveTicket",
                    "decision": "Approve",
                    "reason": "Smoke-test action.",
                }
            ),
        }

    monkeypatch.setattr(grpo_train, "_generate_step", fake_generate_step)
    result = grpo_train.rollout_local(["Task: easy\nTicket: {}"], trainer=object(), task_ids=["easy"])

    assert set(result) == {
        "prompt_ids",
        "completion_ids",
        "logprobs",
        "grader_score",
        "env_reward",
        "format_reward",
        "loop_penalty",
        "unresolved_penalty",
    }
    assert all(isinstance(value, list) and len(value) == 1 for value in result.values())
    assert result["prompt_ids"][0] == [1, 2]
    assert result["completion_ids"][0] == [3, 4]
    assert result["format_reward"][0] == 0.05
