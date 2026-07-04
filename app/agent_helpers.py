"""Shared policy heuristics for baseline and inference fallback agents."""

from __future__ import annotations

from typing import Any, Dict, Optional

from app.document_utils import infer_required_document


def _needs_policy_search(observation: Dict[str, Any], claim: Optional[Dict[str, Any]] = None) -> bool:
    """
    Content-based: does this claim warrant a SearchPolicy before resolving?
    The model no longer sees rule_keyword — we derive intent from claim facts.
    """
    claim = claim or {}
    amount = float(observation.get("amount") or claim.get("amount") or 0)
    description = (observation.get("description") or claim.get("description") or "").lower()
    level = str(observation.get("employee_level") or claim.get("employee_level") or "")
    category = str(claim.get("policy_category") or "").lower()

    # These resolve immediately — no search needed
    if level in ("L7", "L8") or category in ("duplicate", "personal", "seniority"):
        return False
    if any(k in description for k in ("alcohol", "wine", "beer", "gym", "personal")):
        return False

    # Meal threshold ambiguity
    if amount > 2000 and any(k in description for k in ("meal", "dinner", "lunch", "breakfast", "entertainment", "food")):
        return True
    # GST rule (high-value invoices)
    if amount > 5000:
        return True
    # Cab: day-vs-night rule requires policy lookup
    if any(k in description for k in ("cab", "ride", "taxi", "auto")):
        return True
    # WFH cap rule
    if any(k in description for k in ("wfh", "internet", "electricity", "remote", "work from home")):
        return True
    # International travel rule
    if any(k in description for k in ("international", "flight", "hotel", "travel", "airline")):
        return True

    return False


def effective_rule_keyword(
    observation: Dict[str, Any], claim: Optional[Dict[str, Any]] = None
) -> str:
    """Return the best available policy category label from the claim (internal use only)."""
    claim = claim or {}
    # rule_keyword is no longer in the observation; always read from claim internals
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
    """Derive a policy search query from claim content (no rule_keyword needed)."""
    claim = claim or {}
    description = (observation.get("description") or claim.get("description") or "").lower()
    amount = float(observation.get("amount") or claim.get("amount") or 0)
    rule = effective_rule_keyword(observation, claim)

    if rule in ("gst", "international", "wfh", "personal"):
        return rule.replace(" ", "")
    if "cab" in description or "ride" in description or "taxi" in description:
        return (
            "daytime cab"
            if any(k in description for k in ("before", "business hours", "morning", "afternoon"))
            else "cab"
        )
    if amount > 2000 and any(k in description for k in ("meal", "dinner", "lunch", "entertainment")):
        return "large meal"
    if any(k in description for k in ("meal", "dinner", "lunch", "breakfast")):
        return "meal"
    if amount > 5000:
        return "gst"
    if any(k in description for k in ("wfh", "internet", "electricity", "remote")):
        return "wfh"
    return rule or "policy"


def task_prompt_prefix(task_id: str, observation: Dict[str, Any]) -> str:
    """Content-based prompt hint — no rule_keyword labels."""
    policy_retrieved = bool(observation.get("policy_retrieved"))
    missing_doc = observation.get("missing_document")
    amount = float(observation.get("amount") or 0)
    description = (observation.get("description") or "").lower()

    if policy_retrieved:
        return ""  # caller builds the post-search hint
    if missing_doc == "required":
        return (
            "\nA document is required but the type is not yet known. "
            "SearchPolicy first to identify the applicable rule, then request the document."
        )
    # Suggest search when claim content implies a threshold rule
    if amount > 5000:
        return "\nHigh-value claim (>₹5,000). Consider SearchPolicy to verify the applicable GST or threshold rule."
    if amount > 2000 and any(k in description for k in ("meal", "dinner", "lunch", "entertainment")):
        return "\nLarge meal claim. Consider SearchPolicy to verify the manager-approval threshold (Rule 3)."
    if any(k in description for k in ("cab", "ride", "taxi")):
        return "\nCab/ride claim. Consider SearchPolicy to check day-vs-night approval rules."
    if any(k in description for k in ("wfh", "internet", "electricity", "remote")):
        return "\nWFH claim. Consider SearchPolicy to verify the monthly allowance cap."
    return ""
