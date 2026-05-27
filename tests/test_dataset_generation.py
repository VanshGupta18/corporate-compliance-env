"""Tests for dataset generation invariants."""

from data.generate_dataset import generate_hard_claim, get_ground_truth


def test_doc_missing_reject_scenario_has_reject_ground_truth():
    claim = generate_hard_claim(9001, scenario="doc_missing_reject")
    assert claim["ground_truth_decision"] == "Reject"
    assert claim["required_document"] == "manager_approval"
    assert claim["document_outcome"] == "not_provided"


def test_ground_truth_uses_vague_description_when_description_missing():
    claim = {
        "amount": 3200,
        "description": "",
        "vague_description": "Team dinner reimbursement",
        "employee_level": "L4",
        "has_receipt": True,
        "document_outcome": "not_provided",
        "missing_document": "manager_approval",
        "policy_category": "meal",
    }
    decision, reason = get_ground_truth(claim)
    assert decision == "Reject"
    assert "manager" in reason.lower()
