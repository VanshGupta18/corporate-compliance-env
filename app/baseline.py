"""
Rule-Based Baseline Agent for Compliance Environment

Uses WebSocket client (ComplianceEnvClient) for stateful multi-step episodes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from app.client import ComplianceEnvClient
from app.graders import grade_episode
from app.models import ComplianceAction, ActionType, TicketDecision


def _infer_required_document(observation: Dict[str, Any]) -> str:
    rule_keyword = (observation.get("rule_keyword") or "").lower()
    description = (observation.get("description") or "").lower()
    text = f"{rule_keyword} {description}"
    if "international" in text or "vp" in text:
        return "vp_approval"
    if "gst" in text or (observation.get("amount", 0) or 0) > 5000:
        return "gst_invoice"
    if "wfh" in text or "internet" in text or "electricity" in text:
        return "utility_bill"
    return "manager_approval"


def _normalize_action(action: Dict[str, Any]) -> Dict[str, Any]:
    """Map legacy Escalate action_type to ResolveTicket."""
    if action.get("action_type") == "Escalate":
        return {
            "action_type": "ResolveTicket",
            "decision": "Escalate",
            "reason": action.get("reason") or "L7+ requires escalation (Rule 14)",
        }
    return action


class BaselineAgent:
    """Policy-based baseline agent implementing TechCorp India Expense Policy."""

    def __init__(self, api_url: str = "http://localhost:7860"):
        self.api_url = api_url
        self._client: Optional[ComplianceEnvClient] = None
        self._sync = None

    def _obs_dict(self, observation: Any) -> Dict[str, Any]:
        if hasattr(observation, "model_dump"):
            return observation.model_dump()
        if isinstance(observation, dict):
            return observation
        return dict(observation)

    def decide_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        amount = observation.get("amount", 0)
        has_receipt = observation.get("has_receipt", False)
        missing_doc = observation.get("missing_document")
        employee_level = observation.get("employee_level", "L3")
        description = (observation.get("description") or "").lower()
        rule_keyword = (observation.get("rule_keyword") or "").lower()

        try:
            level_num = int(str(employee_level).replace("L", ""))
            if level_num >= 7:
                return {
                    "action_type": "ResolveTicket",
                    "decision": "Escalate",
                    "reason": f"Employee {employee_level} (L7+): Senior Finance Review (Rule 14)",
                }
        except (ValueError, IndexError):
            pass

        if "alcohol" in description or "wine" in description or "beer" in description:
            return {
                "action_type": "ResolveTicket",
                "decision": "Reject",
                "reason": "Alcohol prohibited (Rule 4)",
            }

        if "personal" in rule_keyword or "gym" in description:
            return {
                "action_type": "ResolveTicket",
                "decision": "Reject",
                "reason": "Personal expense (Rule 15)",
            }

        if observation.get("rule_keyword") == "hidden" or (
            "meal" not in rule_keyword and "cab" not in rule_keyword and rule_keyword
        ):
            query = rule_keyword if rule_keyword != "hidden" else "policy"
            if "cab" in description or "ride" in description:
                query = "daytime cab" if "before" in description or "business hours" in description else "cab"
            elif "meal" in description or "dinner" in description:
                query = "meal"
            return {"action_type": "SearchPolicy", "query": query}

        if missing_doc and missing_doc not in (None, "required"):
            return {
                "action_type": "RequestInformation",
                "message": f"Please provide {missing_doc}",
            }
        if missing_doc == "required":
            return {
                "action_type": "RequestInformation",
                "message": f"Please provide {_infer_required_document(observation)}",
            }

        if "meal" in description or "lunch" in description or "dinner" in description:
            if amount < 500:
                return {
                    "action_type": "ResolveTicket",
                    "decision": "Approve",
                    "reason": "Meal under Rs500 (Rule 1)",
                }
            if 500 <= amount <= 2000:
                decision = "Approve" if has_receipt else "Reject"
                return {
                    "action_type": "ResolveTicket",
                    "decision": decision,
                    "reason": f"Meal Rs{amount} receipt={'yes' if has_receipt else 'no'} (Rule 2)",
                }
            if amount > 2000:
                if has_receipt and not missing_doc:
                    return {
                        "action_type": "ResolveTicket",
                        "decision": "Approve",
                        "reason": "Large meal with documents (Rule 3)",
                    }
                return {
                    "action_type": "ResolveTicket",
                    "decision": "Reject",
                    "reason": "Large meal missing manager approval (Rule 3)",
                }

        if "cab" in description or "ride" in description:
            if "after 10" in description or "late" in description:
                return {
                    "action_type": "ResolveTicket",
                    "decision": "Approve" if has_receipt else "Reject",
                    "reason": "Late-night cab (Rule 6)",
                }
            if missing_doc:
                return {
                    "action_type": "ResolveTicket",
                    "decision": "Reject",
                    "reason": "Daytime cab missing manager note (Rule 7)",
                }
            return {
                "action_type": "ResolveTicket",
                "decision": "Approve",
                "reason": "Daytime cab with manager note (Rule 7)",
            }

        if amount > 5000 and not has_receipt:
            return {
                "action_type": "RequestInformation",
                "message": "Please provide gst_invoice",
            }

        return {
            "action_type": "ResolveTicket",
            "decision": "Approve",
            "reason": "No policy violation detected",
        }

    def run_episode(
        self,
        task_id: str = "easy",
        claim_id: str | None = None,
        claim: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Run one full episode via WebSocket client."""
        print(
            f"[START] task={task_id.upper()} env=corporate-compliance-env model=baseline-agent",
            flush=True,
        )

        reset_kwargs: Dict[str, Any] = {"task_id": task_id}
        if claim_id:
            reset_kwargs["claim_id"] = claim_id

        with ComplianceEnvClient(base_url=self.api_url).sync() as client:
            result = client.reset(**reset_kwargs)
            observation = self._obs_dict(result.observation)
            done = result.done
            total_reward = float(result.reward or 0.0)
            step_count = 0
            max_steps = int(observation.get("max_steps", 8))
            rewards: list[float] = []

            while not done and step_count < max_steps:
                step_count += 1
                action_dict = _normalize_action(self.decide_action(observation))
                step_result = client.step(ComplianceAction(**action_dict))
                observation = self._obs_dict(step_result.observation)
                reward = float(step_result.reward or 0.0)
                done = step_result.done
                total_reward += reward
                rewards.append(reward)
                print(
                    f"[STEP] step={step_count} action={action_dict['action_type']} "
                    f"reward={reward:.2f} done={str(done).lower()}",
                    flush=True,
                )

            state = client.state()
            actions_history = getattr(state, "actions_history", []) or []

        grader = grade_episode(
            task_id=task_id,
            actions_history=actions_history,
            ground_truth_decision=(claim or {}).get("ground_truth_decision", "Approve"),
            claim=claim,
        )
        score = grader["score"]
        print(
            f"[END] steps={step_count} grader_score={score:.3f} total_reward={total_reward:.3f}",
            flush=True,
        )
        return {
            "task_id": task_id,
            "claim_id": claim_id,
            "steps": step_count,
            "total_reward": total_reward,
            "grader_score": score,
            "status": "completed" if done else "truncated",
            "actions_history": actions_history,
        }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run baseline on compliance claims.")
    parser.add_argument("--api-url", default="http://127.0.0.1:7860")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test", "all"])
    args = parser.parse_args()

    agent = BaselineAgent(api_url=args.api_url)
    split_path = Path(f"data/splits/{args.split}.json")
    claims_path = Path("data/claims.json")
    if split_path.exists() and args.split != "all":
        with open(split_path, encoding="utf-8") as f:
            claims = json.load(f).get("claims", [])
    else:
        with open(claims_path, encoding="utf-8") as f:
            claims = json.load(f).get("claims", [])

    results_by_diff: Dict[str, list] = {"easy": [], "medium": [], "hard": []}
    all_scores: list = []

    for claim in claims:
        difficulty = claim["task_difficulty"]
        try:
            result = agent.run_episode(
                task_id=difficulty,
                claim_id=claim["id"],
                claim=claim,
            )
            score = result.get("grader_score", 0.0)
            results_by_diff[difficulty].append(score)
            all_scores.append(score)
        except Exception as e:
            print(f"Error on {claim['id']}: {e}")

    metrics = {
        "metrics": {
            "overall_metrics": {
                "total_claims": len(all_scores),
                "mean_grader_score": sum(all_scores) / len(all_scores) if all_scores else 0,
            },
            "performance_by_difficulty": {},
        }
    }
    for diff, scores in results_by_diff.items():
        metrics["metrics"]["performance_by_difficulty"][diff] = {
            "mean_grader_score": sum(scores) / len(scores) if scores else 0,
            "total": len(scores),
        }

    with open("baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n[SUMMARY] Baseline Results:")
    print(
        f"  OVERALL: {metrics['metrics']['overall_metrics']['mean_grader_score']:.3f} "
        f"(n={len(all_scores)})"
    )
    for diff in ["easy", "medium", "hard"]:
        d = metrics["metrics"]["performance_by_difficulty"][diff]
        print(f"  {diff.upper()}: {d['mean_grader_score']:.3f} (n={d['total']})")


if __name__ == "__main__":
    main()
