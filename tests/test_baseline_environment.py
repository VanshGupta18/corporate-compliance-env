"""Baseline agent tests using in-process environment."""

import pytest

from app.baseline import BaselineAgent, _normalize_action
from app.models import ComplianceAction, ActionType
from app.server.environment import ComplianceEnv
from app.graders import grade_episode


def test_normalize_escalate_action():
    raw = {"action_type": "Escalate", "decision": "", "reason": "L7+"}
    fixed = _normalize_action(raw)
    assert fixed["action_type"] == "ResolveTicket"
    assert fixed["decision"] == "Escalate"


def test_baseline_inprocess_episode():
    """Run baseline logic against in-process env (no HTTP)."""
    env = ComplianceEnv()
    agent = BaselineAgent()
    claim = next(c for c in env.claims if c["task_difficulty"] == "easy")
    env.reset(task_id="easy", claim_id=claim["id"])
    obs = env._get_observation().model_dump()
    done = False
    steps = 0
    while not done and steps < 5:
        steps += 1
        action_dict = _normalize_action(agent.decide_action(obs))
        result = env.step(ComplianceAction(**action_dict))
        obs = result.model_dump()
        done = result.done

    grader = grade_episode(
        "easy",
        env.state.actions_history,
        claim["ground_truth_decision"],
        claim=claim,
    )
    assert 0.01 <= grader["score"] <= 0.99


def test_http_reset_returns_observation():
    """HTTP reset works; multi-step HTTP uses new env per request (use WebSocket)."""
    from fastapi.testclient import TestClient
    from app.server.app import app

    client = TestClient(app)
    response = client.post("/reset", json={"task_id": "easy"})
    assert response.status_code == 200
    assert "observation" in response.json()
