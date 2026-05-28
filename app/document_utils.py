"""Helpers for inferring required documents from ticket observations."""

from __future__ import annotations

from typing import Any, Dict, Optional


def infer_required_document(
    observation: Dict[str, Any],
    claim: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Guess the concrete document type when the env shows missing_document as 'required'.
    Uses revealed rule_keyword, policy category, amount, and description text.
    """
    claim = claim or {}
    rule_keyword = (
        observation.get("rule_keyword")
        or claim.get("rule_keyword")
        or ""
    ).lower()
    if rule_keyword == "hidden":
        rule_keyword = (claim.get("rule_keyword") or "").lower()
    policy_category = (claim.get("policy_category") or rule_keyword).lower()
    description = (observation.get("description") or claim.get("vague_description") or "").lower()
    amount = float(observation.get("amount") or claim.get("amount") or 0)
    text = f"{rule_keyword} {policy_category} {description}"

    if policy_category == "gst" or "gst" in text or amount > 5000:
        return "gst_invoice"
    if policy_category == "international" or "international" in text:
        return "vp_approval"
    if policy_category == "wfh" or "wfh" in text or "internet" in text or "electricity" in text:
        return "utility_bill"
    if "international" in text or "vp approval" in text:
        return "vp_approval"
    return "manager_approval"
