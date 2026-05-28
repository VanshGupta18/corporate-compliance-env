"""Tests for shared agent policy helpers and baseline GST behavior."""

from app.agent_helpers import resolve_after_missing_document, search_query_for_hidden_policy
from app.baseline import BaselineAgent


def test_search_query_for_gst_claim():
    claim = {"rule_keyword": "gst", "policy_category": "gst", "description": "Vendor software"}
    obs = {"rule_keyword": "hidden", "description": "Software procurement expense"}
    assert search_query_for_hidden_policy(obs, claim) == "gst"


def test_resolve_after_missing_gst_document():
    obs = {
        "policy_retrieved": True,
        "env_message": "GST invoice not provided. Resolve the ticket now.",
        "rule_keyword": "gst",
    }
    action = resolve_after_missing_document(obs, {"policy_category": "gst"})
    assert action["decision"] == "Reject"
    assert "GST" in action["reason"]


def test_baseline_rejects_gst_after_document_denied():
    agent = BaselineAgent(api_url="http://127.0.0.1:7860")
    agent._active_claim = {
        "rule_keyword": "gst",
        "policy_category": "gst",
        "ground_truth_decision": "Reject",
    }
    obs = {
        "amount": 12000,
        "has_receipt": True,
        "missing_document": None,
        "employee_level": "L4",
        "description": "Annual software subscription",
        "rule_keyword": "gst",
        "policy_retrieved": True,
        "env_message": "GST invoice not provided. Resolve the ticket now.",
    }
    action = agent.decide_action(obs)
    assert action["action_type"] == "ResolveTicket"
    assert action["decision"] == "Reject"
