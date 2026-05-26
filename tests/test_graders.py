"""Grader tests — unit tests run offline; integration tests need live server."""

import os
import pytest

from app.graders import grade_easy, grade_medium, grade_hard, grade_episode

pytestmark = pytest.mark.integration


def test_easy_grader_correct_approval_unit():
    actions = [
        {
            "action_type": "ResolveTicket",
            "decision": "Approve",
            "reason": "Meal under Rs500 per policy rule 1.",
        }
    ]
    r = grade_easy(actions, "Approve")
    assert r["score"] >= 0.8


def test_medium_grader_useful_search_unit():
    claim = {"rule_keyword": "daytime cab"}
    actions = [
        {"action_type": "SearchPolicy", "query": "daytime cab manager"},
        {
            "action_type": "ResolveTicket",
            "decision": "Reject",
            "reason": "Missing manager note.",
        },
    ]
    r = grade_medium(actions, "Reject", claim)
    assert r["components"]["useful_search"] > 0


def test_hard_grader_document_request_unit():
    claim = {"rule_keyword": "large meal", "required_document": "manager_approval"}
    actions = [
        {"action_type": "SearchPolicy", "query": "large meal"},
        {
            "action_type": "RequestInformation",
            "message": "Please provide manager_approval",
        },
        {
            "action_type": "ResolveTicket",
            "decision": "Approve",
            "reason": "Documents complete.",
        },
    ]
    r = grade_hard(actions, "Approve", claim)
    assert r["components"]["correct_document_request"] > 0


@pytest.mark.skipif(
    not os.getenv("COMPLIANCE_API_URL"),
    reason="Set COMPLIANCE_API_URL to run live WebSocket grader integration tests",
)
def test_live_easy_episode():
    from app.client import ComplianceEnvClient
    from app.models import ComplianceAction

    api_url = os.getenv("COMPLIANCE_API_URL", "http://127.0.0.1:7860")
    with ComplianceEnvClient(base_url=api_url).sync() as client:
        result = client.reset(task_id="easy")
        step = client.step(
            ComplianceAction(
                action_type="ResolveTicket",
                decision="Approve",
                reason="Integration test.",
            )
        )
        assert step.done is True
