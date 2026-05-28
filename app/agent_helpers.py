"""Shared policy heuristics for baseline and inference fallback agents."""

from __future__ import annotations

from typing import Any, Dict, Optional

from app.document_utils import infer_required_document


def effective_rule_keyword(
    observation: Dict[str, Any], claim: Optional[Dict[str, Any]] = None
) -> str:
    rule_keyword = str(observation.get("rule_keyword") or "").lower()
    if rule_keyword and rule_keyword != "hidden":
        return rule_keyword
    claim = claim or {}
    return str(claim.get("rule_keyword") or claim.get("policy_category") or "").lower()


def document_unavailable(observation: Dict[str, Any]) -> bool:
    msg = (observation.get("env_message") or "").lower()
    return any(
        phrase in msg
        for phrase in (
            "not provided",
            "not yet received",
            "not submitted",
            "email missing",
            "invoice not",
        )
    )


def resolve_after_missing_document(
    observation: Dict[str, Any], claim: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """Return ResolveTicket when env confirms the required document was not provided."""
    if not observation.get("policy_retrieved") or not document_unavailable(observation):
        return None
    claim = claim or {}
    rule = effective_rule_keyword(observation, claim)
    amount = float(observation.get("amount") or claim.get("amount") or 0)
    description = (observation.get("description") or "").lower()

    if rule == "gst" or claim.get("policy_category") == "gst":
        return {
            "action_type": "ResolveTicket",
            "decision": "Reject",
            "reason": "GST invoice missing (Rule 12)",
        }
    if rule == "large meal" or (amount > 2000 and "meal" in description):
        return {
            "action_type": "ResolveTicket",
            "decision": "Reject",
            "reason": "Manager approval missing (Rule 3)",
        }
    return {
        "action_type": "ResolveTicket",
        "decision": "Reject",
        "reason": "Required document not provided",
    }


def search_query_for_hidden_policy(
    observation: Dict[str, Any], claim: Optional[Dict[str, Any]] = None
) -> str:
    claim = claim or {}
    description = (observation.get("description") or "").lower()
    rule = effective_rule_keyword(observation, claim)
    if rule in ("gst", "international", "wfh", "personal"):
        return rule.replace(" ", "")
    if "cab" in description or "ride" in description:
        return (
            "daytime cab"
            if "before" in description or "business hours" in description
            else "cab"
        )
    if "meal" in description or "dinner" in description or "lunch" in description:
        return "meal" if rule != "large meal" else "large meal"
    return rule or "policy"


def task_prompt_prefix(task_id: str, observation: Dict[str, Any]) -> str:
    if task_id == "easy":
        return (
            "\nEASY task: rule_keyword is visible. Do NOT use SearchPolicy. "
            "Resolve directly unless missing_document requires a specific request."
        )
    if task_id in ("medium", "hard") and observation.get("rule_keyword") == "hidden":
        return (
            "\nMEDIUM/HARD: rule_keyword is hidden. Use SearchPolicy once first "
            "(short query: meal, large meal, gst, cab)."
        )
    return ""
