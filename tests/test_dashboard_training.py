"""Dashboard training replay should match baseline/inference log parsing."""

from __future__ import annotations

from pathlib import Path

from app.dashboard import (
    _enrich_episodes,
    _load_claims_index,
    _load_dashboard_data,
    _load_training_replay,
    _load_metrics,
    _parse_episode_log,
)
from app.paths import TRAINING_LOG, TRAINING_RESULTS


def test_training_log_parses_claim_ids_and_steps():
    eps, acts = _parse_episode_log(TRAINING_LOG, "training")
    assert len(eps) == 120
    assert len(acts) > 0
    sample = eps[0]
    assert sample["method"] == "training"
    assert sample["claim_id"].startswith("EXP-")
    assert sample["task_id"] in ("easy", "medium", "hard")
    assert "grader_score" in sample
    search = [a for a in acts if a.get("action_type") == "SEARCH_POLICY"]
    assert search and search[0].get("query")
    assert search[0]["query"] != "hidden"


def test_training_replay_enriches_ticket_like_baseline():
    metrics_file = _load_metrics(TRAINING_RESULTS, "training")
    t_eps, t_acts, training, _rows = _load_training_replay(metrics_file)
    claims_idx = _load_claims_index()
    enriched = _enrich_episodes(t_eps[:1], claims_idx)[0]
    assert enriched.get("employee_name")
    assert enriched.get("amount", 0) > 0
    assert enriched.get("expected_policy_search")
    assert enriched.get("ground_truth")
    assert training.total == 120
    assert abs(training.overall - 0.713) < 0.01


def test_load_dashboard_data_training_episode_count():
    data = _load_dashboard_data()
    t_eps = [e for e in data["episodes"] if e["method"] == "training"]
    assert len(t_eps) == 120
    assert data["metrics"]["training"]["n"] == 120
    assert all(e.get("claim_id") for e in t_eps)
    assert all(e.get("employee_name") for e in t_eps)


def test_rl_story_marketing_metrics():
    data = _load_dashboard_data()
    story = data["rl_story"]
    assert story["has_training"] is True
    assert abs(story["medium_gain"] - 0.038) < 0.002
    assert story["hard_gain"] == 0.035
    assert abs(story["complex_task_gain"] - 0.036) < 0.002
    assert story["search_policy_before"] == 53
    assert story["search_policy_after"] == 78
    assert story["request_information_before"] == 29
    assert story["request_information_after"] == 40
    assert "headline" in story
    assert "why_it_matters" in story


def test_tool_action_counts_before_after_training():
    data = _load_dashboard_data()
    tools = data["tool_actions"]
    before = tools["before_training"]
    after = tools["after_training"]
    assert before["request_information"] == 29
    assert after["request_information"] == 40
    assert before["search_policy"] == 53
    assert after["search_policy"] == 78


def test_merge_jsonl_fills_missing_claim_id():
    from app.dashboard import _merge_training_jsonl_meta, _read_training_jsonl

    rows = _read_training_jsonl()
    assert rows and rows[0].get("claim_id")
    episodes = [
        {
            "method": "training",
            "episode_id": "training-0010",
            "claim_id": "",
            "task_id": "easy",
            "steps": 0,
            "grader_score": 0.0,
            "success": False,
        }
    ]
    _merge_training_jsonl_meta(episodes, rows)
    assert episodes[0]["claim_id"] == "EXP-20072"
    assert episodes[0]["task_id"] == "medium"
