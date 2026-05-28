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
