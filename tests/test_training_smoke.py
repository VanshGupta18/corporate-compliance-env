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
    parse_json_payload,
)


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
