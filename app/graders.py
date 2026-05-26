"""
Component-based grading for curriculum easy / medium / hard tasks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.policy_snippets import match_policy_snippet


def _final_resolve(actions_history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for action in reversed(actions_history):
        if action.get("action_type") == "ResolveTicket":
            return action
    return None


def _useful_search(actions_history: List[Dict[str, Any]], rule_keyword: str) -> bool:
    for action in actions_history:
        if action.get("action_type") != "SearchPolicy":
            continue
        query = action.get("query") or ""
        _, relevant = match_policy_snippet(rule_keyword, query)
        if relevant:
            return True
    return False


def _correct_document_request(
    actions_history: List[Dict[str, Any]], required_document: Optional[str]
) -> bool:
    if not required_document:
        return False
    required = str(required_document)
    req_space = required.replace("_", " ").lower()
    req_compact = req_space.replace(" ", "")
    for action in actions_history:
        if action.get("action_type") != "RequestInformation":
            continue
        msg = (action.get("message") or "").lower()
        msg_compact = msg.replace("_", "").replace(" ", "")
        if req_space in msg or required.lower() in msg or req_compact in msg_compact:
            return True
    return False


def _decision_matches(decision: Any, ground_truth_decision: str) -> bool:
    return decision == ground_truth_decision or str(decision) == str(ground_truth_decision)


def grade_easy(
    actions_history: List[Dict[str, Any]],
    ground_truth_decision: str,
    claim: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    components = {
        "valid_resolve": 0.0,
        "correct_decision": 0.0,
        "valid_reason": 0.0,
        "no_unnecessary_tools": 0.0,
    }
    final = _final_resolve(actions_history)
    if not final:
        return {"score": 0.01, "components": components}

    unnecessary = any(
        a.get("action_type") in ("SearchPolicy", "RequestInformation")
        for a in actions_history[:-1]
    )
    components["no_unnecessary_tools"] = 0.0 if unnecessary else 0.15
    components["valid_resolve"] = 0.15

    decision = final.get("decision")
    if _decision_matches(decision, ground_truth_decision):
        components["correct_decision"] = 0.55
    if final.get("reason") and len(str(final.get("reason", ""))) >= 8:
        components["valid_reason"] = 0.15

    score = sum(components.values())
    return {"score": max(0.01, min(0.99, score)), "components": components}


def grade_medium(
    actions_history: List[Dict[str, Any]],
    ground_truth_decision: str,
    claim: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    claim = claim or {}
    rule_keyword = claim.get("rule_keyword", "")
    components = {
        "useful_search": 0.0,
        "correct_decision": 0.0,
        "valid_reason": 0.0,
        "minimal_steps": 0.0,
    }

    useful_search = _useful_search(actions_history, rule_keyword)
    final = _final_resolve(actions_history)
    correct_decision = bool(final and _decision_matches(final.get("decision"), ground_truth_decision))

    if useful_search:
        components["useful_search"] = 0.35

    if final:
        if correct_decision and useful_search:
            components["correct_decision"] = 0.45
        if final.get("reason") and len(str(final.get("reason", ""))) >= 8:
            components["valid_reason"] = 0.10

    if final and len(actions_history) <= 3:
        components["minimal_steps"] = 0.10

    score = sum(components.values())
    if correct_decision and not useful_search:
        score = min(score, 0.45)
    elif not correct_decision:
        score = min(score, 0.35)
    return {"score": max(0.01, min(0.99, score)), "components": components}


def grade_hard(
    actions_history: List[Dict[str, Any]],
    ground_truth_decision: str,
    claim: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    claim = claim or {}
    rule_keyword = claim.get("rule_keyword", "")
    required = claim.get("missing_document") or claim.get("required_document")

    components = {
        "useful_search": 0.0,
        "correct_document_request": 0.0,
        "correct_decision": 0.0,
        "valid_reason": 0.0,
        "no_max_step_failure": 0.0,
    }

    useful_search = _useful_search(actions_history, rule_keyword)
    correct_document_request = _correct_document_request(actions_history, required)
    final = _final_resolve(actions_history)
    correct_decision = bool(final and _decision_matches(final.get("decision"), ground_truth_decision))

    if useful_search:
        components["useful_search"] = 0.20

    if correct_document_request:
        components["correct_document_request"] = 0.25

    if final:
        if correct_decision and useful_search and correct_document_request:
            components["correct_decision"] = 0.40
        if final.get("reason") and len(str(final.get("reason", ""))) >= 10:
            components["valid_reason"] = 0.10

    max_steps = claim.get("max_steps") or 8
    if final and len(actions_history) <= max_steps:
        components["no_max_step_failure"] = 0.05

    score = sum(components.values())
    if correct_decision and not (useful_search and correct_document_request):
        score = min(score, 0.45)
    elif not correct_decision:
        score = min(score, 0.35)
    return {"score": max(0.01, min(0.99, score)), "components": components}


def grade_episode(
    task_id: str,
    actions_history: List[Dict[str, Any]],
    ground_truth_decision: str,
    claim: Optional[Dict[str, Any]] = None,
    requested_document: bool = False,
) -> Dict[str, Any]:
    if task_id == "easy":
        result = grade_easy(actions_history, ground_truth_decision, claim)
    elif task_id == "medium":
        result = grade_medium(actions_history, ground_truth_decision, claim)
    elif task_id == "hard":
        result = grade_hard(actions_history, ground_truth_decision, claim)
    else:
        result = {"score": 0.01, "components": {"unknown_task": 0.0}}

    return {
        "score": result["score"],
        "task_id": task_id,
        "num_steps": len(actions_history),
        "components": result.get("components", {}),
        "details": f"Graded {task_id} with component scores",
    }
