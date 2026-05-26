"""Target score bands for curriculum RL evaluation (pre-RL vs post-RL)."""

from __future__ import annotations

from typing import Any, Dict

# Pre-RL baseline bands (fraction 0-1) — base LLM with strong prompting
PRE_RL_TARGETS: Dict[str, Dict[str, float]] = {
    "easy": {"min": 0.70, "max": 0.90, "ideal_mid": 0.80},
    "medium": {"min": 0.35, "max": 0.65, "ideal_mid": 0.50},
    "hard": {"min": 0.10, "max": 0.40, "ideal_mid": 0.25},
}

# Post-RL curriculum targets after staged training
POST_RL_TARGETS: Dict[str, Dict[str, float]] = {
    "easy": {"min": 0.85, "max": 0.99, "ideal_mid": 0.92},
    "medium": {"min": 0.60, "max": 0.90, "ideal_mid": 0.75},
    "hard": {"min": 0.40, "max": 0.75, "ideal_mid": 0.55},
}

CURRICULUM_STAGES = [
    {
        "id": "stage_0_baseline",
        "name": "Base model evaluation",
        "tasks": ["easy", "medium", "hard"],
        "description": "No RL; measure pre-training grader scores on held-out test split.",
    },
    {
        "id": "stage_1_easy",
        "name": "Easy curriculum",
        "tasks": ["easy"],
        "description": "Direct ResolveTicket; JSON schema and classification.",
    },
    {
        "id": "stage_2_medium",
        "name": "Medium curriculum",
        "tasks": ["easy", "medium"],
        "task_weights": {"easy": 0.3, "medium": 0.7},
        "description": "Policy retrieval before resolve.",
    },
    {
        "id": "stage_3_hard",
        "name": "Hard curriculum",
        "tasks": ["easy", "medium", "hard"],
        "task_weights": {"easy": 0.15, "medium": 0.35, "hard": 0.50},
        "description": "Multi-turn search, document request, contextual resolve.",
    },
    {
        "id": "stage_4_mixed_eval",
        "name": "Final mixed evaluation",
        "tasks": ["easy", "medium", "hard"],
        "description": "Held-out test claims only.",
    },
]


def score_in_band(task_id: str, score: float, phase: str = "pre_rl") -> bool:
    """Return True if score falls within expected band for task difficulty."""
    bands = PRE_RL_TARGETS if phase == "pre_rl" else POST_RL_TARGETS
    band = bands.get(task_id, {"min": 0.0, "max": 1.0})
    return band["min"] <= score <= band["max"]


def curriculum_summary() -> Dict[str, Any]:
    """Export targets for dashboards and training configs."""
    return {
        "pre_rl": PRE_RL_TARGETS,
        "post_rl": POST_RL_TARGETS,
        "stages": CURRICULUM_STAGES,
    }
