"""Tests for training eval metrics and dashboard log parsing."""

from __future__ import annotations

from app.graders import (
    episode_success,
    grade_episode,
    normalize_actions_history,
    normalize_decision_value,
    normalize_history_action,
)
from app.dashboard import _CLAIM_RUN_RE
from training.training_utils import (
    normalize_compliance_action,
    parse_model_action,
    sanitize_search_query,
)


def test_normalize_history_action_enum_repr():
    row = {
        "action_type": "ActionType.RESOLVE_TICKET",
        "decision": "TicketDecision.REJECT",
        "reason": "Alcohol on bill.",
    }
    out = normalize_history_action(row)
    assert out["action_type"] == "ResolveTicket"
    assert out["decision"] == "Reject"


def test_episode_success_matches_grader_correct_decision():
    history = [
        {
            "action_type": "ResolveTicket",
            "decision": "TicketDecision.REJECT",
            "reason": "Alcohol prohibited on expense.",
        }
    ]
    grader = grade_episode("easy", history, "Reject")
    assert float(grader["components"]["correct_decision"]) > 0.0
    assert episode_success(grader, done=True) is True


def test_episode_success_false_when_wrong_decision():
    history = [
        {
            "action_type": "ResolveTicket",
            "decision": "Approve",
            "reason": "Looks fine to me honestly.",
        }
    ]
    grader = grade_episode("easy", history, "Reject")
    assert episode_success(grader, done=True) is False


def test_normalize_decision_value():
    assert normalize_decision_value("TicketDecision.ESCALATE") == "Escalate"
    assert normalize_decision_value("APPROVE") == "Approve"


def test_claim_run_regex_parses_training():
    line = "Running training for claim EXP-20004 (easy)..."
    match = _CLAIM_RUN_RE.match(line)
    assert match is not None
    assert match.group(1) == "EXP-20004"
    assert match.group(2).lower() == "easy"


def test_normalize_actions_history_chain():
    raw = [{"action_type": "SEARCH_POLICY", "query": "gst"}]
    norm = normalize_actions_history(raw)
    assert norm[0]["action_type"] == "SearchPolicy"


def test_sanitize_search_query_replaces_hidden():
    obs = {
        "rule_keyword": "hidden",
        "description": "Team dinner at restaurant, Rs 3200",
        "amount": 3200,
    }
    claim = {"rule_keyword": "large meal", "policy_category": "large meal"}
    assert sanitize_search_query("hidden", obs, claim) == "large meal"
    assert sanitize_search_query(None, obs, claim) == "large meal"
    assert sanitize_search_query("gst", obs, claim) == "gst"


def test_normalize_compliance_action_never_hidden_query():
    obs = {
        "rule_keyword": "hidden",
        "description": "GST invoice missing for software license",
        "amount": 12000,
    }
    claim = {"rule_keyword": "gst", "policy_category": "gst"}
    out = normalize_compliance_action(
        {"action_type": "SearchPolicy", "query": "hidden"},
        obs,
        claim,
    )
    assert out["query"] != "hidden"
    assert out["query"] == "gst"


def test_parse_model_action_decision_not_substring_approve():
    obs = {"rule_keyword": "meal", "policy_retrieved": True}
    text = (
        '{"action_type": "ResolveTicket", "decision": "Reject", '
        '"reason": "Do not Approve alcohol expenses."}'
    )
    out = parse_model_action(text, obs)
    assert out["decision"] == "Reject"


def test_concat_episode_rollout_tokens_merges_steps():
    from training.rollout_generation import concat_episode_rollout_tokens

    steps = [
        {"prompt_ids": [1, 2], "completion_ids": [10, 11], "logprobs": [-0.1, -0.2]},
        {"prompt_ids": [3, 4], "completion_ids": [20], "logprobs": [-0.3]},
    ]
    prompt_ids, completion_ids, logprobs = concat_episode_rollout_tokens(
        steps, max_prompt=8, max_completion=8
    )
    assert prompt_ids == [1, 2]
    assert completion_ids == [10, 11, 20]
    assert len(logprobs) == 3
