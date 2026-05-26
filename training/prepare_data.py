"""Generate trajectories for SFT and RL post-training (curriculum-aware)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.models import ComplianceAction
from app.server.environment import ComplianceEnv

TASKS = ["easy", "medium", "hard"]


@dataclass
class StepRecord:
    task_id: str
    step: int
    observation: Dict[str, Any]
    action: Dict[str, Any]
    reward: float
    done: bool
    strategy: str
    claim_id: str


def _clean_observation(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Prompt fields only — no reward/done leakage."""
    keys = (
        "ticket_id",
        "employee_name",
        "employee_role",
        "employee_level",
        "amount",
        "currency",
        "description",
        "has_receipt",
        "missing_document",
        "rule_keyword",
        "risk_score",
        "env_message",
        "step_count",
        "max_steps",
    )
    return {k: obs[k] for k in keys if k in obs}


def choose_action(
    observation: Dict[str, Any],
    task_id: str,
    strategy: str,
    claim: Dict[str, Any],
) -> Dict[str, Any]:
    if strategy == "random":
        action_type = random.choice(["SearchPolicy", "RequestInformation", "ResolveTicket"])
        if action_type == "SearchPolicy":
            return {"action_type": "SearchPolicy", "query": "policy"}
        if action_type == "RequestInformation":
            return {
                "action_type": "RequestInformation",
                "message": "Please share required document.",
            }
        return {
            "action_type": "ResolveTicket",
            "decision": random.choice(["Approve", "Reject", "Escalate"]),
            "reason": "Random baseline",
        }

    gt = claim.get("ground_truth_decision", "Approve")
    rule = claim.get("rule_keyword", "policy")

    if task_id == "easy":
        return {
            "action_type": "ResolveTicket",
            "decision": gt,
            "reason": claim.get("ground_truth_reason", "Policy applied."),
        }

    if task_id == "medium":
        if observation.get("rule_keyword") == "hidden" or not observation.get("env_message"):
            return {"action_type": "SearchPolicy", "query": rule}
        return {
            "action_type": "ResolveTicket",
            "decision": gt,
            "reason": "Decision after policy retrieval.",
        }

    # hard
    if observation.get("rule_keyword") == "hidden" or not observation.get("env_message"):
        return {"action_type": "SearchPolicy", "query": rule}
    req = claim.get("required_document") or claim.get("missing_document")
    if req and observation.get("missing_document"):
        return {
            "action_type": "RequestInformation",
            "message": f"Please provide {req}",
        }
    return {
        "action_type": "ResolveTicket",
        "decision": gt,
        "reason": "Decision after context gathering.",
    }


def rollout_local(
    task_id: str,
    strategy: str,
    claim: Dict[str, Any] | None = None,
    max_steps: int = 8,
) -> List[StepRecord]:
    env = ComplianceEnv()
    if claim:
        obs = env.reset(task_id=task_id, claim_id=claim["id"])
    else:
        obs = env.reset(task_id=task_id)
        claim = env._current_claim or {}

    observation = _clean_observation(obs.model_dump())
    records: List[StepRecord] = []
    done = False
    step = 0

    while not done and step < max_steps:
        step += 1
        action = choose_action(observation, task_id, strategy, claim)
        next_obs = env.step(ComplianceAction(**action))
        result = next_obs.model_dump()
        reward = float(result.get("reward") or 0.0)
        done = bool(result.get("done", False))

        records.append(
            StepRecord(
                task_id=task_id,
                step=step,
                observation=observation,
                action=action,
                reward=reward,
                done=done,
                strategy=strategy,
                claim_id=claim.get("id", ""),
            )
        )
        observation = _clean_observation(result)

    return records


def to_sft_format(records: List[StepRecord], terminal_only: bool = False) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        if terminal_only and record.action.get("action_type") != "ResolveTicket":
            continue
        prompt = (
            "You are an AI compliance officer. Return only valid action JSON.\n"
            f"Task: {record.task_id}\n"
            f"Ticket: {json.dumps(record.observation, ensure_ascii=True)}"
        )
        rows.append(
            {
                "prompt": prompt,
                "response": json.dumps(record.action, ensure_ascii=True),
                "reward": record.reward,
                "done": record.done,
                "task_id": record.task_id,
                "strategy": record.strategy,
                "claim_id": record.claim_id,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate curriculum trajectories.")
    parser.add_argument("--episodes-per-task", type=int, default=40)
    parser.add_argument("--random-ratio", type=float, default=0.2)
    parser.add_argument("--output-dir", default="training/data")
    parser.add_argument("--split", default="train", help="claims split: train|validation|test")
    args = parser.parse_args()

    env = ComplianceEnv()
    split_claims = [c for c in env.claims if c.get("split", "train") == args.split]
    if not split_claims:
        split_claims = env.claims

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records: List[StepRecord] = []
    for task_id in TASKS:
        task_claims = [c for c in split_claims if c.get("task_difficulty") == task_id]
        if not task_claims:
            continue
        for _ in range(args.episodes_per_task):
            claim = random.choice(task_claims)
            strategy = "random" if random.random() < args.random_ratio else "heuristic"
            all_records.extend(rollout_local(task_id, strategy, claim=claim))

    json_records = [asdict(record) for record in all_records]
    (out_dir / "trajectories.json").write_text(
        json.dumps(json_records, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    sft_rows = to_sft_format(all_records, terminal_only=False)
    positive = [r for r in sft_rows if r.get("reward", 0) > 0.5]
    (out_dir / "sft_dataset.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in sft_rows),
        encoding="utf-8",
    )
    (out_dir / "sft_dataset_positive.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in positive),
        encoding="utf-8",
    )
    print(f"Wrote {len(all_records)} steps, {len(positive)} positive SFT rows to {out_dir}")


if __name__ == "__main__":
    main()
