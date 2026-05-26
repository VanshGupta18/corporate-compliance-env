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
import re
import json
import textwrap
from typing import Dict, Optional, Any

from openai import OpenAI
from app.client import ComplianceEnvClient
from app.models import ComplianceAction, ComplianceObservation
from app.graders import grade_episode

# ===== Environment Configuration =====
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
COMPLIANCE_API = os.getenv("COMPLIANCE_API", "https://mcqueenmater-env-corporate.hf.space")

# ===== Task Configuration =====
TASKS = ["easy", "medium", "hard"]
MAX_STEPS_PER_TASK = 10
TEMPERATURE = 0.0
MAX_TOKENS = 256

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI Compliance Officer. Your job is to audit employee expense claims
    against company policy and decide whether to Approve, Reject, or Escalate each ticket.
    
    You have three action types available:
    1. SearchPolicy(query: str) - Search the policy rulebook when you need information
    2. RequestInformation(message: str) - Ask the employee for missing documents
    3. ResolveTicket(decision: str, reason: str) - Make your final decision
    
    Valid decisions: "Approve", "Reject", "Escalate"
    
    Do not assume hidden policy details. Use SearchPolicy when the relevant rule is
    hidden or unclear. Use RequestInformation when the ticket indicates a required
    document is missing.
    
    Respond with ONLY a valid action in JSON format:
    {"action_type": "...", "query": "...", "message": "...", "decision": "...", "reason": "..."}
    """
).strip()


def build_user_prompt(observation: Dict, task_id: str, step: int) -> str:
    """Build the user prompt from the current observation."""
    max_steps = int(observation.get("max_steps") or MAX_STEPS_PER_TASK)
    steps_remaining = (max_steps - step + 1)
    
    # Add urgency message if running low on steps
    urgency = ""
    if step >= 3:
        urgency = "\n⚠️  URGENT: You only have {} step(s) remaining. YOU MUST MAKE A FINAL DECISION NOW.\nDo NOT search policy again. Make your FINAL decision: Approve, Reject, or Escalate.".format(steps_remaining - 1)
    
    prompt = textwrap.dedent(
        f"""
        Task: {task_id} | Step: {step}/{max_steps}
        
        Ticket ID: {observation.get('ticket_id')}
        Employee: {observation.get('employee_name')} ({observation.get('employee_role')})
        Level: {observation.get('employee_level')}
        Amount: ₹{observation.get('amount')}
        Description: {observation.get('description')}
        Has Receipt: {observation.get('has_receipt')}
        Missing Document: {observation.get('missing_document')}
        Risk Score: {observation.get('risk_score')}
        
        Your Task:
        - Review the ticket against policy rules
        - Make a FINAL decision (Approve/Reject/Escalate) when you have sufficient information{urgency}
        
        What action do you take?
        """
    ).strip()
    return prompt


def rule_based_fallback(observation: Dict) -> Dict:
    """
    Rule-based fallback decision when LLM fails.
    Uses policy rules to make a reasonable decision.
    """
    amount = observation.get("amount", 0)
    level = observation.get("employee_level", "L1")
    has_receipt = observation.get("has_receipt", False)
    missing_doc = observation.get("missing_document")
    rule_keyword = (observation.get("rule_keyword") or "").lower()
    description = observation.get("description", "").lower()
    
    # Check for alcohol or personal items
    if "alcohol" in description or "gift" in description or "shopping" in description:
        return {
            "action_type": "ResolveTicket",
            "decision": "Reject",
            "reason": "Policy violation: alcohol/gift/personal items not approved",
        }
    
    # VP and above always escalate
    if level in ["L7", "L8", "L9"]:
        return {
            "action_type": "ResolveTicket",
            "decision": "Escalate",
            "reason": "High-level employee claim requires escalation",
        }
    
    # If missing documents, request them
    if missing_doc:
        if missing_doc == "required":
            text = f"{rule_keyword} {description}"
            if "international" in text or "vp" in text:
                missing_doc = "vp_approval"
            elif "gst" in text or amount > 5000:
                missing_doc = "gst_invoice"
            elif "wfh" in text or "internet" in text or "electricity" in text:
                missing_doc = "utility_bill"
            else:
                missing_doc = "manager_approval"
        return {
            "action_type": "RequestInformation",
            "message": f"Please provide {missing_doc}",
        }
    
    # Meals and travel rules
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
    
    # Default approval for compliant claims
    return {
        "action_type": "ResolveTicket",
        "decision": "Approve",
        "reason": "Claim meets policy requirements",
    }


def parse_model_response(response_text: str) -> Optional[Dict]:
    """Parse the model's response into an action."""
    try:
        # Try to extract JSON from the response
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            action_dict = json.loads(json_match.group(0))
            return action_dict
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: try to parse text-based response
    if "SearchPolicy" in response_text:
        query_match = re.search(r'query["\']?\s*[:\-]?\s*["\']([^"\']+)["\']', response_text)
        return {
            "action_type": "SearchPolicy",
            "query": query_match.group(1) if query_match else "policy",
        }
    elif "RequestInformation" in response_text:
        msg_match = re.search(r'message["\']?\s*[:\-]?\s*["\']([^"\']+)["\']', response_text)
        return {
            "action_type": "RequestInformation",
            "message": msg_match.group(1) if msg_match else "Please provide missing information",
        }
    elif "ResolveTicket" in response_text:
        decision = "Reject"
        if "Approve" in response_text:
            decision = "Approve"
        elif "Escalate" in response_text:
            decision = "Escalate"

        reason_match = re.search(r'reason["\']?\s*[:\-]?\s*["\']([^"\']+)["\']', response_text)
        return {
            "action_type": "ResolveTicket",
            "decision": decision,
            "reason": reason_match.group(1) if reason_match else "Based on policy review",
        }

    return None


def run_episode(
    client: Any,
    task_id: str,
    claim_id: str | None = None,
    claim: Optional[Dict[str, Any]] = None,
) -> Dict:
    """Run a single episode on the given task (difficulty), optionally pinned to a claim."""
    print(f"[START] task={task_id.upper()} env=corporate-compliance-env model={MODEL_NAME}", flush=True)

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
        user_prompt = build_user_prompt(obs_dict, task_id, step)

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
            obs_dict = observation.model_dump() if hasattr(observation, 'model_dump') else observation.__dict__
            action_dict = rule_based_fallback(obs_dict)
            action_data = {
                "action_type": action_dict.get("action_type", "ResolveTicket"),
                "query": action_dict.get("query"),
                "message": action_dict.get("message"),
                "decision": action_dict.get("decision"),
                "reason": action_dict.get("reason"),
            }
            action = ComplianceAction(**action_data)
            step_result = client.step(action)
            observation = step_result.observation
            reward = step_result.reward or 0.0
            
            print(f"[STEP] step={step} action={action.action_type} reward={reward:.2f} done={str(step_result.done).lower()} error=null", flush=True)

            action_data["reward"] = reward
            episode_data["steps"].append(action_data)
            episode_data["total_reward"] += reward
            if step_result.done:
                episode_data["done"] = True
                break
            continue

        action_dict = parse_model_response(response_text)
        if not action_dict:
            action_dict = {"action_type": "ResolveTicket", "decision": "Reject", "reason": "Parse error"}


        action_data = {
            "action_type": action_dict.get("action_type", "ResolveTicket"),
            "query": action_dict.get("query"),
            "message": action_dict.get("message"),
            "decision": action_dict.get("decision"),
            "reason": action_dict.get("reason"),
        }

        action = ComplianceAction(**action_data)
        step_result = client.step(action)
        observation = step_result.observation
        reward = step_result.reward or 0.0

        print(f"[STEP] step={step} action={action.action_type} reward={reward:.2f} done={str(step_result.done).lower()} error=null", flush=True)

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

                print(f"Running inference for claim {claim_id} ({difficulty})...")
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
                "total_evaluations": len(all_scores),
                "mean_grader_score": sum(all_scores) / len(all_scores) if all_scores else 0,
            },
            "performance_by_difficulty": {},
        }

        for diff, scores in results_by_diff.items():
            metrics["performance_by_difficulty"][diff] = {
                "mean_grader_score": sum(scores) / len(scores) if scores else 0,
                "total": len(scores),
            }
            
        with open("inference_results.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
            
        print("Inference complete. Results saved to inference_results.json.")

    except Exception as e:
        import traceback
        print(f"ERROR: {e}", flush=True)
        traceback.print_exc()

if __name__ == "__main__":
    from pathlib import Path
    main()
