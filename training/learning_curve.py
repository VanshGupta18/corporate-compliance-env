"""
Log per-difficulty validation metrics for curriculum learning graphs.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.curriculum_targets import PRE_RL_TARGETS, POST_RL_TARGETS, score_in_band
from app.graders import grade_episode
from app.server.environment import ComplianceEnv
from app.models import ComplianceAction, ActionType, TicketDecision


DEFAULT_LOG = Path("training/logs/learning_curve.jsonl")


def _load_split_claims(split: str = "validation") -> List[Dict[str, Any]]:
    path = Path(__file__).parent.parent / "data" / "splits" / f"{split}.json"
    if not path.exists():
        path = Path(__file__).parent.parent / "data" / "claims.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [c for c in data.get("claims", []) if c.get("task_difficulty")]


def run_heuristic_episode(env: ComplianceEnv, claim: Dict[str, Any]) -> Dict[str, Any]:
    """Simple heuristic agent for smoke eval (not LLM)."""
    task_id = claim["task_difficulty"]
    env.reset(task_id=task_id, claim_id=claim["id"])
    actions: List[Dict[str, Any]] = []

    if task_id == "easy":
        act = ComplianceAction(
            action_type=ActionType.RESOLVE_TICKET,
            decision=TicketDecision(claim["ground_truth_decision"]),
            reason=claim.get("ground_truth_reason", "Policy applied."),
        )
        env.step(act)
        actions.append(act.model_dump())
    elif task_id == "medium":
        q = claim.get("rule_keyword", "policy")
        act1 = ComplianceAction(action_type=ActionType.SEARCH_POLICY, query=q)
        env.step(act1)
        actions.append(act1.model_dump())
        act2 = ComplianceAction(
            action_type=ActionType.RESOLVE_TICKET,
            decision=TicketDecision(claim["ground_truth_decision"]),
            reason="After policy search.",
        )
        env.step(act2)
        actions.append(act2.model_dump())
    else:
        act1 = ComplianceAction(action_type=ActionType.SEARCH_POLICY, query=claim.get("rule_keyword", "policy"))
        env.step(act1)
        actions.append(act1.model_dump())
        req = claim.get("required_document") or claim.get("missing_document")
        if req:
            act2 = ComplianceAction(
                action_type=ActionType.REQUEST_INFORMATION,
                message=f"Please provide {req}",
            )
            env.step(act2)
            actions.append(act2.model_dump())
        act3 = ComplianceAction(
            action_type=ActionType.RESOLVE_TICKET,
            decision=TicketDecision(claim["ground_truth_decision"]),
            reason="After gathering context.",
        )
        env.step(act3)
        actions.append(act3.model_dump())

    grade = grade_episode(
        task_id,
        env.state.actions_history,
        claim["ground_truth_decision"],
        claim=claim,
    )
    return {
        "task_id": task_id,
        "claim_id": claim["id"],
        "score": grade["score"],
        "components": grade.get("components", {}),
        "num_steps": len(env.state.actions_history),
        "valid_json": True,
        "useful_search": grade.get("components", {}).get("useful_search", 0) > 0,
        "correct_document_request": grade.get("components", {}).get("correct_document_request", 0) > 0,
        "correct_decision": grade.get("components", {}).get("correct_decision", 0) > 0,
    }


def aggregate_episodes(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_task: Dict[str, List[float]] = defaultdict(list)
    tools = defaultdict(list)
    for ep in episodes:
        by_task[ep["task_id"]].append(ep["score"])
        tools["valid_json"].append(1.0 if ep.get("valid_json") else 0.0)
        tools["useful_search"].append(1.0 if ep.get("useful_search") else 0.0)
        tools["correct_document_request"].append(1.0 if ep.get("correct_document_request") else 0.0)
        tools["correct_decision"].append(1.0 if ep.get("correct_decision") else 0.0)

    per_difficulty = {}
    for tid, scores in by_task.items():
        avg = sum(scores) / len(scores) if scores else 0.0
        per_difficulty[tid] = {
            "mean_score": round(avg, 4),
            "count": len(scores),
            "in_pre_rl_band": score_in_band(tid, avg, "pre_rl"),
            "target_band": PRE_RL_TARGETS.get(tid),
        }

    return {
        "per_difficulty": per_difficulty,
        "tool_success_rates": {k: round(sum(v) / len(v), 4) if v else 0.0 for k, v in tools.items()},
        "mean_episode_length": round(
            sum(e["num_steps"] for e in episodes) / len(episodes), 2
        )
        if episodes
        else 0.0,
    }


def log_learning_point(
    *,
    stage: str,
    global_step: int,
    episodes: Optional[List[Dict[str, Any]]] = None,
    split: str = "validation",
    phase: str = "pre_rl",
    log_file: Path = DEFAULT_LOG,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if episodes is None:
        env = ComplianceEnv()
        claims = _load_split_claims(split)
        episodes = [run_heuristic_episode(env, c) for c in claims[: min(60, len(claims))]]

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "global_step": global_step,
        "phase": phase,
        "split": split,
        **aggregate_episodes(episodes),
    }
    if extra:
        payload.update(extra)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    return payload


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate and log curriculum learning curve.")
    parser.add_argument("--stage", default="stage_0_baseline")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--log-file", default=str(DEFAULT_LOG))
    args = parser.parse_args()

    result = log_learning_point(
        stage=args.stage,
        global_step=args.step,
        split=args.split,
        log_file=Path(args.log_file),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
