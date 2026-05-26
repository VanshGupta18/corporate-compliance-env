"""Unit tests for component graders."""

from app.graders import grade_easy, grade_medium, grade_hard, grade_episode


def test_easy_full_credit():
    actions = [
        {
            "action_type": "ResolveTicket",
            "decision": "Approve",
            "reason": "Meal under threshold per policy.",
        }
    ]
    r = grade_easy(actions, "Approve")
    assert r["score"] >= 0.85


def test_easy_penalize_unnecessary_search():
    actions = [
        {"action_type": "SearchPolicy", "query": "meal"},
        {
            "action_type": "ResolveTicket",
            "decision": "Approve",
            "reason": "Approved after unnecessary search.",
        },
    ]
    r = grade_easy(actions, "Approve")
    assert r["components"]["no_unnecessary_tools"] == 0.0


def test_easy_wrong_decision_stays_below_success_threshold():
    actions = [
        {
            "action_type": "ResolveTicket",
            "decision": "Approve",
            "reason": "Looks valid from visible fields.",
        }
    ]
    r = grade_easy(actions, "Reject")
    assert r["score"] < 0.5


def test_medium_requires_useful_search():
    claim = {"rule_keyword": "daytime cab"}
    actions = [
        {"action_type": "SearchPolicy", "query": "unrelated gym membership"},
        {
            "action_type": "ResolveTicket",
            "decision": "Reject",
            "reason": "Missing manager note for daytime cab.",
        },
    ]
    r = grade_medium(actions, "Reject", claim)
    assert r["components"]["useful_search"] == 0.0
    assert r["score"] < 0.9  # no full credit without useful search

    good = [
        {"action_type": "SearchPolicy", "query": "daytime cab before 10"},
        {
            "action_type": "ResolveTicket",
            "decision": "Reject",
            "reason": "Daytime cab needs manager approval.",
        },
    ]
    r2 = grade_medium(good, "Reject", claim)
    assert r2["components"]["useful_search"] > 0


def test_medium_correct_guess_without_search_is_capped():
    claim = {"rule_keyword": "daytime cab"}
    actions = [
        {
            "action_type": "ResolveTicket",
            "decision": "Reject",
            "reason": "Missing manager note for daytime cab.",
        }
    ]
    r = grade_medium(actions, "Reject", claim)
    assert r["score"] < 0.5
    assert r["components"]["correct_decision"] == 0.0


def test_hard_correct_document_request():
    claim = {
        "rule_keyword": "large meal",
        "required_document": "manager_approval",
        "max_steps": 8,
    }
    actions = [
        {"action_type": "SearchPolicy", "query": "large meal manager"},
        {
            "action_type": "RequestInformation",
            "message": "Please provide manager_approval",
        },
        {
            "action_type": "ResolveTicket",
            "decision": "Approve",
            "reason": "Manager approval received; claim complies.",
        },
    ]
    r = grade_hard(actions, "Approve", claim)
    assert r["components"]["correct_document_request"] > 0
    assert r["score"] >= 0.7


def test_hard_correct_guess_without_workflow_is_capped():
    claim = {
        "rule_keyword": "large meal",
        "required_document": "manager_approval",
        "max_steps": 8,
    }
    actions = [
        {
            "action_type": "ResolveTicket",
            "decision": "Approve",
            "reason": "Assume the claim complies.",
        }
    ]
    r = grade_hard(actions, "Approve", claim)
    assert r["score"] < 0.5
    assert r["components"]["correct_decision"] == 0.0


def test_grade_episode_dispatch():
    r = grade_episode("easy", [{"action_type": "ResolveTicket", "decision": "Reject", "reason": "Alcohol on bill."}], "Reject")
    assert 0.01 <= r["score"] <= 0.99
    assert "components" in r
