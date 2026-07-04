"""
Inference Script for Corporate Compliance Environment
=====================================================
This script demonstrates how to run an LLM-based agent on the compliance environment.

MANDATORY
- Before running, ensure the following variables are defined in your environment:
    API_BASE_URL   The API endpoint for the LLM (e.g., https://router.huggingface.co/v1)
    MODEL_NAME     The model identifier to use for inference
    HF_TOKEN       Your Hugging Face / API key
    COMPLIANCE_API The compliance environment API URL (default: http://localhost:7860)
"""

import os
import json
import sys
import textwrap
from pathlib import Path
from typing import Dict, Optional, Any

from openai import OpenAI


def _load_dotenv() -> None:
    """Load project .env into os.environ (does not override existing vars)."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_dotenv()

from app.client import ComplianceEnvClient
from app.agent_helpers import (
    document_unavailable,
    effective_rule_keyword,
    resolve_after_missing_document,
    search_query_for_hidden_policy,
    task_prompt_prefix,
    _needs_policy_search,
)
from app.document_utils import infer_required_document
from app.graders import grade_episode
from app.models import ComplianceAction, ComplianceObservation
from app.paths import INFERENCE_LOG, INFERENCE_RESULTS
from app.run_logging import (
    format_step_log,
    log_claim_start,
    log_episode_start,
    run_with_log,
    write_results_json,
)
from training.training_utils import parse_model_action

# ===== Environment Configuration =====
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
COMPLIANCE_API = os.getenv("COMPLIANCE_API", "https://mcqueenmater-env-corporate.hf.space")


# ===== Task Configuration =====
TASKS = ["easy", "medium", "hard"]
MAX_STEPS_PER_TASK = 10
TEMPERATURE = 0.0
MAX_TOKENS = 256

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI Compliance Officer. Review employee expense claims against company policy
    and decide whether to Approve, Reject, or Escalate each ticket.

    You have three action types:
    1. SearchPolicy(query: str) — search the policy rulebook when the claim amount or
       category implies a threshold you need to verify before deciding.
       Use short topical queries: meal, large meal, gst, cab, wfh, international.
    2. RequestInformation(message: str) — ask for a missing document once you know the
       applicable policy. Name the specific document type explicitly.
    3. ResolveTicket(decision: str, reason: str) — make your final decision once you have
       enough information. Valid decisions: "Approve", "Reject", "Escalate".

    When to search: meal expenses near thresholds (₹500/₹2,000), amounts above ₹5,000
    (GST rule), cab rides (day vs night approval rule), WFH claims, international travel.
    When NOT to search: amounts clearly below ₹500, obvious alcohol/personal expenses,
    or after policy has already been retrieved (policy_retrieved = true).

    After policy is retrieved, never search again. Use RequestInformation only when
    missing_document is set (not null). If env_message confirms a document was not
    provided, Reject — do not Approve. After a document response, ResolveTicket immediately.

    Respond with ONLY a valid action in JSON format:
    {"action_type": "...", "query": "...", "message": "...", "decision": "...", "reason": "..."}
    """
).strip()


def build_user_prompt(
    observation: Dict, step: int, task_id: str = "medium", claim: Optional[Dict[str, Any]] = None
) -> str:
    """Build the user prompt from the current observation — no rule_keyword field."""
    max_steps = int(observation.get("max_steps") or MAX_STEPS_PER_TASK)
    steps_remaining = max_steps - step + 1

    policy_retrieved = bool(observation.get("policy_retrieved"))
    missing_doc = observation.get("missing_document")

    # Build the contextual hint
    if policy_retrieved and document_unavailable(observation):
        policy_note = (
            "\nEnvironment says the requested document was NOT provided. "
            "Resolve with Reject (GST/meal rules), not Approve."
        )
    elif policy_retrieved:
        if missing_doc:
            if missing_doc == "required":
                hint = infer_required_document(observation)
                policy_note = (
                    f"\nPolicy retrieved. Do NOT SearchPolicy again. "
                    f"RequestInformation must name '{hint}' (not the word 'required'). "
                    "Then ResolveTicket."
                )
            else:
                policy_note = (
                    f"\nPolicy retrieved. Request '{missing_doc}' if not yet requested, "
                    "then ResolveTicket. Do NOT SearchPolicy again."
                )
        else:
            policy_note = (
                "\nPolicy retrieved. No missing document — ResolveTicket now. "
                "Do NOT SearchPolicy or RequestInformation."
            )
    else:
        # Content-based hint: suggest search when claim implies a threshold rule
        policy_note = task_prompt_prefix(task_id, observation)

    urgency = ""
    if step >= max_steps - 1:
        urgency = (
            f"\nURGENT: {steps_remaining} step(s) left. "
            "Do NOT search policy again. Resolve the ticket now."
        )

    prompt = textwrap.dedent(
        f"""
        Step: {step}/{max_steps}

        Ticket ID: {observation.get('ticket_id')}
        Employee: {observation.get('employee_name')} ({observation.get('employee_role')})
        Level: {observation.get('employee_level')}
        Amount: ₹{observation.get('amount')}
        Description: {observation.get('description')}
        Has Receipt: {observation.get('has_receipt')}
        Missing Document: {observation.get('missing_document')}
        Policy retrieved: {policy_retrieved}
        Environment message: {observation.get('env_message', '')}{policy_note}

        Your Task:
        - Review the ticket against policy rules
        - Make a FINAL decision (Approve/Reject/Escalate) when you have sufficient information{urgency}

        What action do you take?
        """
    ).strip()
    return prompt


def rule_based_fallback(
    observation: Dict, task_id: str = "medium", claim: Optional[Dict[str, Any]] = None
) -> Dict:
    """Rule-based fallback when LLM output is invalid — content-based, no rule_keyword."""
    amount = observation.get("amount", 0)
    level = observation.get("employee_level", "L1")
    has_receipt = observation.get("has_receipt", False)
    missing_doc = observation.get("missing_document")
    description = (observation.get("description") or "").lower()
    policy_retrieved = bool(observation.get("policy_retrieved"))

    if "alcohol" in description or "gift" in description or "shopping" in description:
        return {
            "action_type": "ResolveTicket",
            "decision": "Reject",
            "reason": "Policy violation: alcohol/gift/personal items not approved",
        }

    try:
        if int(str(level).replace("L", "")) >= 7:
            return {
                "action_type": "ResolveTicket",
                "decision": "Escalate",
                "reason": "High-level employee claim requires escalation",
            }
    except ValueError:
        pass

    # Search when content implies a threshold — but only once
    if not policy_retrieved and _needs_policy_search(observation, claim):
        return {
            "action_type": "SearchPolicy",
            "query": search_query_for_hidden_policy(observation, claim),
        }

    resolved = resolve_after_missing_document(observation, claim)
    if resolved:
        return resolved

    if missing_doc:
        if missing_doc == "required":
            missing_doc = infer_required_document(observation)
        return {
            "action_type": "RequestInformation",
            "message": f"Please provide {missing_doc}",
        }

    rule = effective_rule_keyword(observation, claim)
    if rule == "gst" or (claim or {}).get("policy_category") == "gst":
        return {
            "action_type": "ResolveTicket",
            "decision": "Reject",
            "reason": "GST invoice missing (Rule 12)",
        }

    if policy_retrieved or amount < 500:
        if amount < 500:
            return {
                "action_type": "ResolveTicket",
                "decision": "Approve",
                "reason": "Amount below ₹500 threshold, no receipt required",
            }
        if amount >= 500 and not has_receipt:
            return {
                "action_type": "ResolveTicket",
                "decision": "Reject",
                "reason": "Receipt required for amounts above ₹500",
            }

    return {
        "action_type": "ResolveTicket",
        "decision": "Approve",
        "reason": "Claim meets policy requirements",
    }


def run_episode(
    client: Any,
    task_id: str,
    claim_id: str | None = None,
    claim: Optional[Dict[str, Any]] = None,
) -> Dict:
    """Run a single episode on the given task (difficulty), optionally pinned to a claim."""
    log_episode_start(task_id, model=MODEL_NAME)

    reset_kwargs: Dict[str, Any] = {"task_id": task_id}
    if claim_id:
        reset_kwargs["claim_id"] = claim_id
    reset_result = client.reset(**reset_kwargs)
    observation = reset_result.observation

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    episode_data = {
        "task_id": task_id,
        "steps": [],
        "total_reward": 0.0,
        "done": False,
    }

    max_steps = int(getattr(observation, "max_steps", MAX_STEPS_PER_TASK) or MAX_STEPS_PER_TASK)
    for step in range(1, max_steps + 1):
        # Build prompt for LLM
        obs_dict = observation.model_dump() if hasattr(observation, 'model_dump') else observation.__dict__
        user_prompt = build_user_prompt(obs_dict, step, task_id=task_id, claim=claim)

        # Call LLM
        try:
            completion = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": user_prompt}],
                    },
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception:
            obs_dict = observation.model_dump() if hasattr(observation, "model_dump") else observation.__dict__
            action_data = parse_model_action(
                json.dumps(rule_based_fallback(obs_dict, task_id=task_id, claim=claim)),
                obs_dict,
            )
            action = ComplianceAction(**action_data)
            step_result = client.step(action)
            observation = step_result.observation
            reward = step_result.reward or 0.0
            
            print(format_step_log(step, action.model_dump(mode="json"), reward, step_result.done), flush=True)

            action_data["reward"] = reward
            episode_data["steps"].append(action_data)
            episode_data["total_reward"] += reward
            if step_result.done:
                episode_data["done"] = True
                break
            continue

        obs_dict = observation.model_dump() if hasattr(observation, "model_dump") else observation.__dict__
        action_data = parse_model_action(response_text, obs_dict)
        action = ComplianceAction(**action_data)
        step_result = client.step(action)
        observation = step_result.observation
        reward = step_result.reward or 0.0

        print(format_step_log(step, action.model_dump(mode="json"), reward, step_result.done), flush=True)

        action_data["reward"] = reward
        episode_data["steps"].append(action_data)
        episode_data["total_reward"] += reward

        if step_result.done:
            episode_data["done"] = True
            break

    episode_data["final_reward"] = episode_data["total_reward"]
    
    actions_history = episode_data.get("steps", [])
    if not claim or not claim.get("ground_truth_decision"):
        raise ValueError("run_episode requires claim metadata with ground_truth_decision for grading.")

    grader_result = grade_episode(
        task_id=task_id,
        actions_history=actions_history,
        ground_truth_decision=claim["ground_truth_decision"],
        claim=claim,
    )
    normalized_score = float(grader_result["score"])
    correct_decision = bool(grader_result["components"].get("correct_decision", 0.0))
    
    # Calculate success and format rewards
    success = episode_data["done"] and correct_decision
    rewards_str = ",".join(f"{step['reward']:.2f}" for step in episode_data["steps"])
    print(f"[END] success={str(success).lower()} steps={len(episode_data['steps'])} score={normalized_score:.3f} rewards={rewards_str}", flush=True)
    episode_data["grader_score"] = normalized_score
    episode_data["actions_history"] = actions_history
    return episode_data


def main() -> None:
    """Main function to run inference on all claims and save results."""
    if not API_KEY:
        print(
            "ERROR: Missing API key. Set HF_TOKEN in .env or export it:\n"
            "  export HF_TOKEN=hf_...\n"
            "  # or: export OPENAI_API_KEY=...",
            flush=True,
        )
        sys.exit(1)
    try:
        client = ComplianceEnvClient(base_url=COMPLIANCE_API).sync()
        
        claims_path = Path("data/claims.json")
        if not claims_path.exists():
            print("Claims file not found.")
            return
            
        with open(claims_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Prefer held-out test split for curriculum eval
        test_path = Path("data/splits/test.json")
        if test_path.exists():
            with open(test_path, encoding="utf-8") as f:
                claims = json.load(f).get("claims", [])
        else:
            claims = data.get("claims", [])

        results_by_diff = {"easy": [], "medium": [], "hard": []}
        all_scores = []

        with client:
            for claim in claims:
                claim_id = claim["id"]
                difficulty = claim["task_difficulty"]

                log_claim_start("inference", claim_id, difficulty)
                episode_data = run_episode(client, difficulty, claim_id=claim_id, claim=claim)

                grader_result = grade_episode(
                    task_id=difficulty,
                    actions_history=episode_data.get("actions_history", []),
                    ground_truth_decision=claim.get("ground_truth_decision", "Approve"),
                    claim=claim,
                )
                score = float(grader_result["score"])
                results_by_diff[difficulty].append(score)
                all_scores.append(score)

        metrics = {
            "overall_metrics": {
                "total_claims": len(all_scores),
                "mean_grader_score": sum(all_scores) / len(all_scores) if all_scores else 0,
            },
            "performance_by_difficulty": {},
        }

        for diff, scores in results_by_diff.items():
            metrics["performance_by_difficulty"][diff] = {
                "mean_grader_score": sum(scores) / len(scores) if scores else 0,
                "total": len(scores),
            }

        write_results_json(INFERENCE_RESULTS, results_by_diff, all_scores=all_scores)

        print("\n[SUMMARY] Inference Results:", flush=True)
        print(
            f"  OVERALL: {metrics['overall_metrics']['mean_grader_score']:.3f} "
            f"(n={len(all_scores)})",
            flush=True,
        )
        for diff in ["easy", "medium", "hard"]:
            d = metrics["performance_by_difficulty"][diff]
            print(f"  {diff.upper()}: {d['mean_grader_score']:.3f} (n={d['total']})", flush=True)
        print("[SAVED] inference_results.json", flush=True)

    except Exception as e:
        import traceback
        print(f"ERROR: {e}", flush=True)
        traceback.print_exc()

if __name__ == "__main__":
    log_path = Path(os.getenv("INFERENCE_LOG_PATH", str(INFERENCE_LOG)))
    run_with_log(log_path, main)
