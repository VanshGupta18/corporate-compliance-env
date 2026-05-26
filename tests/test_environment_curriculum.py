"""Unit tests for curriculum-hard environment mechanics."""

import pytest

from app.models import ComplianceAction, ActionType, TicketDecision
from app.server.environment import ComplianceEnv


@pytest.fixture
def env():
    return ComplianceEnv()


def test_easy_hides_nothing_and_penalizes_search(env):
    claim = next(c for c in env.claims if c.get("task_difficulty") == "easy")
    env.reset(task_id="easy", claim_id=claim["id"])
    obs = env.step(ComplianceAction(action_type=ActionType.SEARCH_POLICY, query="meal"))
    assert obs.rule_keyword != "hidden"
    assert obs.reward is not None and obs.reward < 0


def test_medium_hides_rule_keyword_until_search(env):
    claim = next(c for c in env.claims if c.get("task_difficulty") == "medium")
    env.reset(task_id="medium", claim_id=claim["id"])
    obs0 = env._get_observation()
    assert obs0.rule_keyword == "hidden"
    if claim.get("missing_document") or claim.get("required_document"):
        assert obs0.missing_document == "required"

    obs1 = env.step(
        ComplianceAction(
            action_type=ActionType.SEARCH_POLICY,
            query=claim.get("rule_keyword", "cab"),
        )
    )
    assert obs1.rule_keyword != "hidden"
    assert len(obs1.env_message) > 10


def test_terminal_observation_does_not_expose_ground_truth(env):
    claim = next(c for c in env.claims if c.get("task_difficulty") == "easy")
    env.reset(task_id="easy", claim_id=claim["id"])
    obs = env.step(
        ComplianceAction(
            action_type=ActionType.RESOLVE_TICKET,
            decision=TicketDecision(claim["ground_truth_decision"]),
            reason="Apply visible policy fields.",
        )
    )
    assert obs.done
    assert obs.ground_truth_decision is None


def test_random_reset_respects_split(env):
    obs = env.reset(task_id="easy", split="test", seed=1)
    claim = env._current_claim
    assert claim["split"] == "test"


def test_max_steps_terminates(env):
    claim = next(c for c in env.claims if c.get("task_difficulty") == "easy")
    env.reset(task_id="easy", claim_id=claim["id"])
    for _ in range(5):
        obs = env.step(
            ComplianceAction(action_type=ActionType.SEARCH_POLICY, query="x")
        )
        if obs.done:
            break
    assert env.state.is_done


def test_hard_document_request_clears_missing(env):
    claim = next(
        c
        for c in env.claims
        if c.get("task_difficulty") == "hard"
        and (c.get("required_document") or c.get("missing_document"))
        and c.get("policy_category") != "seniority"
    )
    env.reset(task_id="hard", claim_id=claim["id"])
    env.step(
        ComplianceAction(
            action_type=ActionType.SEARCH_POLICY,
            query=claim.get("rule_keyword", "meal"),
        )
    )
    req = claim.get("missing_document") or claim.get("required_document")
    assert req, "selected hard claim must declare a required document"
    obs = env.step(
        ComplianceAction(
            action_type=ActionType.REQUEST_INFORMATION,
            message=f"Please provide {req.replace('_', ' ')}",
        )
    )
    msg = obs.env_message.lower()
    assert (
        "received" in msg
        or "not yet" in msg
        or "not provided" in msg
        or "missing" in msg
    )
