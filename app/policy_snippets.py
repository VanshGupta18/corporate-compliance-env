"""Policy snippets keyed by rule category for SearchPolicy retrieval."""

from __future__ import annotations

from typing import Dict, List, Tuple

# (keywords in query, snippet text)
POLICY_SNIPPETS: Dict[str, List[Tuple[List[str], str]]] = {
    "meal": [
        (["meal", "food", "lunch", "dinner", "breakfast", "snack"], "Rule 1-3: Meals under Rs500 need no receipt; Rs500-2000 need receipt; above Rs2000 need receipt and manager note."),
        (["alcohol", "wine", "beer"], "Rule 4: Alcohol is never reimbursable; entire claim rejected if alcohol on bill."),
    ],
    "meal receipt": [
        (["receipt", "meal"], "Rule 2: Meals between Rs500 and Rs2000 require a valid receipt."),
    ],
    "large meal": [
        (["large", "meal", "manager", "2000"], "Rule 3: Meals above Rs2000 require receipt AND manager approval note."),
    ],
    "daytime cab": [
        (["cab", "day", "daytime", "before", "10"], "Rule 7: Cab rides before 10:00 PM require a manager approval note explaining business purpose."),
    ],
    "night cab": [
        (["cab", "night", "late", "10 pm", "after"], "Rule 6: Cab rides after 10:00 PM are pre-approved with receipt."),
    ],
    "auto metro": [
        (["auto", "metro", "rickshaw"], "Rule 5: Auto/metro under Rs500 approved without receipt."),
    ],
    "flight economy": [
        (["flight", "economy", "business", "l1", "l6"], "Rule 8: L1-L6 must fly economy; business class rejected."),
    ],
    "flight executive": [
        (["flight", "l7", "vp", "business"], "Rule 9: L7+ may book business class; escalate for review."),
    ],
    "international": [
        (["international", "50000", "vp", "travel"], "Rule 10: International travel over Rs50000 requires VP approval email."),
    ],
    "wfh": [
        (["wfh", "internet", "electricity", "remote"], "Rule 11: WFH utility claims capped at Rs1000/month."),
    ],
    "gst": [
        (["gst", "invoice", "5000"], "Rule 12: Claims above Rs5000 require GST-compliant invoice with GSTIN."),
    ],
    "duplicate": [
        (["duplicate", "same day", "same amount"], "Rule 13: Same employee, same amount, same date = auto reject second claim."),
    ],
    "seniority": [
        (["l7", "vp", "senior", "executive"], "Rule 14: L7+ claims must Escalate; never Approve/Reject directly."),
    ],
    "personal": [
        (["personal", "gym", "spa", "gift"], "Rule 15: Personal expenses are never approved."),
    ],
}

DISTRACTOR_SNIPPET = (
    "Rule 5: Auto-rickshaw and metro travel under Rs500 is approved without a receipt. "
    "(This snippet may not apply to your ticket.)"
)


def match_policy_snippet(rule_keyword: str, query: str) -> Tuple[str, bool]:
    """
    Return (snippet_text, is_relevant).
    is_relevant True when query tokens overlap the rule's keyword set.
    """
    if not query or not query.strip():
        return "No query provided. Try searching for meal, cab, flight, gst, or seniority rules.", False

    q = query.lower().strip()
    entries = POLICY_SNIPPETS.get(rule_keyword, [])

    best_match = None
    best_score = 0
    for keywords, snippet in entries:
        score = sum(1 for kw in keywords if kw in q)
        if score > best_score:
            best_score = score
            best_match = snippet

    if best_score > 0 and best_match:
        return best_match, True

    # Generic fallback: any category partial match on rule_keyword itself
    if rule_keyword.replace(" ", "") in q.replace(" ", "") or any(
        part in q for part in rule_keyword.split()
    ):
        if entries:
            return entries[0][1], True

    return DISTRACTOR_SNIPPET, False


def document_simulation(doc_type: str, claim: Dict) -> str:
    """Simulate returned document content after RequestInformation."""
    if doc_type == "manager_approval":
        if claim.get("document_outcome") == "provided":
            return (
                "Manager approval received: Approved for business purpose. "
                "Signed by reporting manager."
            )
        return "Manager approval not yet received from employee."
    if doc_type == "gst_invoice":
        if claim.get("document_outcome") == "provided":
            return "GST invoice received with valid GSTIN and tax breakdown."
        return "GST invoice not provided."
    if doc_type == "vp_approval":
        if claim.get("document_outcome") == "provided":
            return "VP approval email received for international travel."
        return "VP approval email missing."
    if doc_type == "utility_bill":
        if claim.get("document_outcome") == "provided":
            return "Utility bill received for WFH allowance."
        return "Utility bill not submitted."
    return f"Document type '{doc_type}' requested; awaiting employee response."
