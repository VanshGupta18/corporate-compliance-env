"""Generate trajectories for SFT and RL post-training."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

from app.server.environment import ComplianceEnv
from app.models import ComplianceAction

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


def choose_action(observation: Dict[str, Any], strategy: str) -> Dict[str, Any]:
    if strategy == "random":
        action_type = random.choice(["SearchPolicy", "RequestInformation", "ResolveTicket"])
        if action_type == "SearchPolicy":
            return {"action_type": "SearchPolicy", "query": "policy rule"}
        if action_type == "RequestInformation":
            return {"action_type": "RequestInformation", "message": "Please share required document."}
        return {"action_type": "ResolveTicket", "decision": random.choice(["Approve", "Reject", "Escalate"]), "reason": "Random baseline"}

    # Heuristic strategy
    if observation.get("missing_document"):
        return {
            "action_type": "RequestInformation",
            "message": f"Please provide {observation['missing_document']}",
        }
    if observation.get("risk_score", 0.0) > 0.5:
        return {"action_type": "SearchPolicy", "query": observation.get("rule_keyword", "policy")}
    return {"action_type": "ResolveTicket", "decision": "Approve", "reason": "Low risk and complete data"}


def rollout_local(task_id: str, strategy: str, max_steps: int = 8) -> List[StepRecord]:
    env = ComplianceEnv()
    obs = env.reset(task_id=task_id)
    observation = obs.model_dump()
    records: List[StepRecord] = []
    done = False
    step = 0
    while not done and step < max_steps:
        step += 1
        action = choose_action(observation, strategy)
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
            )
        )
        observation = result
    return records


def to_sft_format(records: List[StepRecord]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
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
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate trajectories for training.")
    parser.add_argument("--episodes-per-task", type=int, default=40)
    parser.add_argument("--output-dir", default="training/data")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records: List[StepRecord] = []
    for task_id in TASKS:
        for _ in range(args.episodes_per_task):
            strategy = "heuristic" if random.random() > 0.5 else "random"
            all_records.extend(rollout_local(task_id, strategy))

    json_records = [asdict(record) for record in all_records]
    (out_dir / "trajectories.json").write_text(
        json.dumps(json_records, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    sft_rows = to_sft_format(all_records)
    (out_dir / "sft_dataset.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in sft_rows),
        encoding="utf-8",
    )
    print(f"Wrote {len(all_records)} trajectory steps to {out_dir}")


if __name__ == "__main__":
    main()
