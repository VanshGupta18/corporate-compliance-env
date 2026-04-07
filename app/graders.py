"""
Grading logic for different task difficulties.
Scores episodes based on task type (easy/medium/hard).
"""

from typing import List, Dict, Any


def grade_easy(actions_history: List[Dict[str, Any]], ground_truth_decision: str) -> float:
    """
    Easy task: Single-step classification.
    Score = 0.99 if correct decision, 0.01 otherwise (strictly between 0 and 1).
    """
    if not actions_history:
        return 0.01
    
    final_action = actions_history[-1]
    if final_action.get("action_type") == "ResolveTicket":
        decision = final_action.get("decision")
        if decision == ground_truth_decision:
            return 0.99
    
    return 0.01


def grade_medium(actions_history: List[Dict[str, Any]], ground_truth_decision: str) -> float:
    """
    Medium task: Policy retrieval + classification.
    
    Score breakdown:
    - 0.99: Correct decision AND searched policy
    - 0.5: Correct decision but NO search (lucky guess)
    - 0.01: Wrong decision (strictly between 0 and 1)
    """
    if not actions_history:
        return 0.01
    
    # Check if agent searched policy
    searched_policy = any(
        action.get("action_type") == "SearchPolicy" 
        for action in actions_history
    )
    
    # Check if final decision is correct
    final_action = actions_history[-1]
    correct_decision = (
        final_action.get("action_type") == "ResolveTicket" and
        final_action.get("decision") == ground_truth_decision
    )
    
    if correct_decision and searched_policy:
        return 0.99
    elif correct_decision and not searched_policy:
        return 0.5  # Lucky guess (already strictly between 0 and 1)
    else:
        return 0.01


def grade_hard(
    actions_history: List[Dict[str, Any]],
    ground_truth_decision: str,
    requested_document: bool = False
) -> float:
    """
    Hard task: Multi-turn contextual decision.
    
    Score breakdown (component-based, strictly between 0 and 1):
    - identified_missing_doc: 0.3 (RequestInformation was called)
    - correct_final_decision: 0.69 (ResolveTicket matches ground truth)
    
    Total: strictly between 0.01 and 0.99
    """
    if not actions_history:
        return 0.01
    
    score = 0.0
    
    # Check if RequestInformation was called
    requested_info = any(
        action.get("action_type") == "RequestInformation"
        for action in actions_history
    )
    
    if requested_info:
        score += 0.3  # Full credit for attempting to get information
    
    # Check if final decision is correct
    final_action = actions_history[-1]
    if (
        final_action.get("action_type") == "ResolveTicket" and
        final_action.get("decision") == ground_truth_decision
    ):
        score += 0.69  # Changed from 0.7 to 0.69 so max total is 0.99, not 1.0
    
    # Clamp to strictly between 0.01 and 0.99
    return max(0.01, min(0.99, score))


def grade_episode(
    task_id: str,
    actions_history: List[Dict[str, Any]],
    ground_truth_decision: str,
    requested_document: bool = False
) -> Dict[str, Any]:
    """
    Main grading function for a completed episode.
    
    Args:
        task_id: "easy", "medium", or "hard"
        actions_history: List of actions taken during episode
        ground_truth_decision: Expected correct decision
        requested_document: Whether a document was requested (for hard tasks)
    
    Returns:
        Dict with score, task_id, and details (score strictly between 0 and 1)
    """
    if task_id == "easy":
        score = grade_easy(actions_history, ground_truth_decision)
    elif task_id == "medium":
        score = grade_medium(actions_history, ground_truth_decision)
    elif task_id == "hard":
        score = grade_hard(actions_history, ground_truth_decision, requested_document)
    else:
        score = 0.5  # Default middle value for unknown tasks
    
    # Final safety clamp to strictly between 0.01 and 0.99
    score = max(0.01, min(0.99, score))
    
    return {
        "score": score,
        "task_id": task_id,
        "num_steps": len(actions_history),
        "details": f"Graded {task_id} task with {len(actions_history)} actions"
    }
