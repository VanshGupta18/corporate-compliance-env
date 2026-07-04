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
    """Prompt fields only — no reward/done leakage, no rule_keyword hint."""
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
        "policy_retrieved",
        "risk_score",
        "env_message",
        "step_count",
        "max_steps",
    )
    return {k: obs[k] for k in keys if k in obs}


def _claim_needs_search(claim: Dict[str, Any]) -> bool:
    """Content-based: does this claim warrant a policy search before resolving?"""
    level = claim.get("employee_level", "")
    if level in ("L7", "L8"):
        return False  # Escalate immediately — rule is unambiguous

    category = claim.get("policy_category", "")
    if category in ("duplicate", "personal", "seniority"):
        return False

    amount = float(claim.get("amount", 0) or 0)
    desc = (
        claim.get("vague_description") or claim.get("description") or ""
    ).lower()

    # Meal threshold ambiguity
    if amount > 2000 and any(k in desc for k in ("meal", "dinner", "lunch", "breakfast", "food", "entertainment")):
        return True
    # GST / high-value invoice rule
    if amount > 5000:
        return True
    # Cab: day vs night rule requires policy lookup
    if any(k in desc for k in ("cab", "ride", "taxi", "auto")):
        return True
    # WFH cap rule
    if any(k in desc for k in ("wfh", "internet", "electricity", "remote", "work from home")):
        return True
    # International travel rule
    if any(k in desc for k in ("international", "flight", "hotel", "travel", "airline")):
        return True

    return False


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
    reason = claim.get("ground_truth_reason") or f"Ground truth: {gt}."
    rule = claim.get("rule_keyword", "policy")
    if not rule or rule.lower() in ("hidden", "unknown", ""):
        rule = "policy"

    # Step 1: search if needed and not yet done
    if _claim_needs_search(claim) and not observation.get("policy_retrieved"):
        return {"action_type": "SearchPolicy", "query": rule}

    # Step 2: request missing document once policy is in hand
    req = claim.get("required_document") or claim.get("missing_document")
    if req and observation.get("missing_document"):
        return {
            "action_type": "RequestInformation",
            "message": f"Please provide {req}",
        }

    # Step 3: resolve
    return {
        "action_type": "ResolveTicket",
        "decision": gt,
        "reason": reason,
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


def _response_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    try:
        payload = json.loads(row.get("response", "{}"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _is_valid_sft_row(row: Dict[str, Any]) -> bool:
    """Drop SearchPolicy rows that teach query=hidden."""
    payload = _response_payload(row)
    if payload.get("action_type") != "SearchPolicy":
        return True
    query = str(payload.get("query") or "").strip().lower()
    return query not in ("hidden", "unknown", "")


def _rebalance_terminal_decisions(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_decision: Dict[str, List[Dict[str, Any]]] = {
        "Approve": [],
        "Reject": [],
        "Escalate": [],
    }
    for row in rows:
        payload = _response_payload(row)
        if payload.get("action_type") != "ResolveTicket":
            continue
        decision = str(payload.get("decision") or "")
        if decision in by_decision:
            by_decision[decision].append(row)

    non_escalate_target = max(len(by_decision["Approve"]), len(by_decision["Reject"]), 1)
    balanced: List[Dict[str, Any]] = []
    for decision in ("Approve", "Reject"):
        bucket = by_decision[decision]
        if not bucket:
            continue
        while len(bucket) < non_escalate_target:
            bucket.append(random.choice(bucket))
        balanced.extend(bucket)
    escalate_bucket = by_decision["Escalate"][:non_escalate_target]
    balanced.extend(escalate_bucket)
    random.shuffle(balanced)
    return balanced


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate curriculum trajectories.")
    parser.add_argument("--episodes-per-task", type=int, default=40)
    parser.add_argument("--random-ratio", type=float, default=0.05)
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

    sft_rows = [row for row in to_sft_format(all_records, terminal_only=False) if _is_valid_sft_row(row)]
    positive = []
    for row in sft_rows:
        payload = _response_payload(row)
        is_terminal = payload.get("action_type") == "ResolveTicket"
        if row.get("strategy") == "heuristic" and is_terminal and row.get("done"):
            positive.append(row)
    balanced = _rebalance_terminal_decisions(positive)
    (out_dir / "sft_dataset.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in sft_rows),
        encoding="utf-8",
    )
    (out_dir / "sft_dataset_positive.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in positive),
        encoding="utf-8",
    )
    (out_dir / "sft_dataset_balanced.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in balanced),
        encoding="utf-8",
    )
    print(
        f"Wrote {len(all_records)} steps, {len(positive)} positive rows, "
        f"{len(balanced)} balanced terminal rows to {out_dir}"
    )


if __name__ == "__main__":
    main()
