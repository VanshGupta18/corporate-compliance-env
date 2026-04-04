"""
Grading logic for different task difficulties.
Scores episodes based on task type (easy/medium/hard).
"""

from typing import List, Dict, Any


def grade_easy(actions_history: List[Dict[str, Any]], ground_truth_decision: str) -> float:
    """
    Easy task: Single-step classification.
    Score = 1.0 if correct decision, 0.0 otherwise.
    """
    if not actions_history:
        return 0.0
    
    final_action = actions_history[-1]
    if final_action.get("action_type") == "ResolveTicket":
        decision = final_action.get("decision")
        if decision == ground_truth_decision:
            return 1.0
    
    return 0.0


def grade_medium(actions_history: List[Dict[str, Any]], ground_truth_decision: str) -> float:
    """
    Medium task: Policy retrieval + classification.
    
    Score breakdown:
    - 1.0: Correct decision AND searched policy
    - 0.5: Correct decision but NO search (lucky guess)
    - 0.0: Wrong decision
    """
    if not actions_history:
        return 0.0
    
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
        return 1.0
    elif correct_decision and not searched_policy:
        return 0.5  # Lucky guess
    else:
        return 0.0


def grade_hard(
    actions_history: List[Dict[str, Any]],
    ground_truth_decision: str,
    requested_document: bool = False
) -> float:
    """
    Hard task: Multi-turn contextual decision.
    
    Score breakdown (component-based):
    - identified_missing_doc: 0.3 (RequestInformation was called)
    - correct_final_decision: 0.7 (ResolveTicket matches ground truth)
    
    Total: 0.0 to 1.0
    """
    if not actions_history:
        return 0.0
    
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
        score += 0.7  # Full credit for correct decision
    
    return score


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
        Dict with score, task_id, and details
    """
    if task_id == "easy":
        score = grade_easy(actions_history, ground_truth_decision)
    elif task_id == "medium":
        score = grade_medium(actions_history, ground_truth_decision)
    elif task_id == "hard":
        score = grade_hard(actions_history, ground_truth_decision, requested_document)
    else:
        score = 0.0
    
    return {
        "score": max(0.0, min(1.0, score)),  # Clamp to [0, 1]
        "task_id": task_id,
        "num_steps": len(actions_history),
        "details": f"Graded {task_id} task with {len(actions_history)} actions"
    }
