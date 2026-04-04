"""
Inference Script for Corporate Compliance Environment
=====================================================
This script demonstrates how to run an LLM-based agent on the compliance environment.

MANDATORY
- Before running, ensure the following variables are defined in your environment:
    API_BASE_URL   The API endpoint for the LLM (e.g., https://router.huggingface.co/v1)
    MODEL_NAME     The model identifier to use for inference
    HF_TOKEN       Your Hugging Face / API key
    COMPLIANCE_API The compliance environment API URL (default: http://localhost:8000)
"""

import os
import re
import json
import textwrap
from typing import List, Dict, Optional, Any

from openai import OpenAI
from app.client import ComplianceEnvClient
from app.models import ComplianceAction, ComplianceObservation

# ===== Environment Configuration =====
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/hf-inference/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.1")
COMPLIANCE_API = os.getenv("COMPLIANCE_API", "http://localhost:8000")

# ===== Task Configuration =====
TASKS = ["easy", "medium", "hard"]
MAX_STEPS_PER_TASK = 10
TEMPERATURE = 0.2
MAX_TOKENS = 256

# ===== Action Patterns =====
ACTION_PATTERN = re.compile(
    r"action\s*[:\-]?\s*(SearchPolicy|RequestInformation|ResolveTicket)",
    re.IGNORECASE,
)


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI Compliance Officer. Your job is to audit employee expense claims
    against company policy and decide whether to Approve, Reject, or Escalate each ticket.
    
    You have three action types available:
    1. SearchPolicy(query: str) - Search the policy rulebook when you need information
    2. RequestInformation(message: str) - Ask the employee for missing documents
    3. ResolveTicket(decision: str, reason: str) - Make your final decision
    
    Valid decisions: "Approve", "Reject", "Escalate"
    
    Respond with ONLY a valid action in JSON format:
    {"action_type": "...", "query": "...", "message": "...", "decision": "...", "reason": "..."}
    """
).strip()


def build_user_prompt(observation: Dict, task_id: str, step: int) -> str:
    """Build the user prompt from the current observation."""
    prompt = textwrap.dedent(
        f"""
        Task: {task_id} | Step: {step}
        
        Ticket ID: {observation.get('ticket_id')}
        Employee: {observation.get('employee_name')} ({observation.get('employee_role')})
        Level: {observation.get('employee_level')}
        Amount: ₹{observation.get('amount')}
        Description: {observation.get('description')}
        Has Receipt: {observation.get('has_receipt')}
        Missing Document: {observation.get('missing_document')}
        Risk Score: {observation.get('risk_score')}
        
        Policy Hint: {observation.get('rule_keyword', '(no hint)')}
        Steps Remaining: {observation.get('max_steps', 0) - observation.get('step_count', 0)}
        
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
        return {
            "action_type": "RequestInformation",
            "message": f"Please provide the missing: {missing_doc}",
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


def run_episode(client: Any, task_id: str) -> Dict:
    """Run a single episode on the given task."""
    print(f"\n{'='*60}")
    print(f"Running Episode: {task_id.upper()}")
    print(f"{'='*60}")

    # Reset environment
    reset_result = client.reset(task_id=task_id)
    observation = reset_result.observation

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    episode_data = {
        "task_id": task_id,
        "steps": [],
        "total_reward": 0.0,
        "done": False,
    }

    for step in range(1, MAX_STEPS_PER_TASK + 1):
        print(f"\n--- Step {step} ---")
        print(f"Ticket: {observation.ticket_id} | Risk: {observation.risk_score:.2f}")
        print(f"Amount: ₹{observation.amount} | Missing: {observation.missing_document}")

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
        except Exception as exc:
            print(f"LLM request failed: {exc}. Using rule-based fallback.")
            # Use rule-based fallback when LLM fails
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
            print(f"Fallback Action: {action.action_type}")
            if action.decision:
                print(f"  Decision: {action.decision}")
            step_result = client.step(action)
            observation = step_result.observation
            reward = step_result.reward or 0.0
            print(f"Reward: {reward:+.2f} | Done: {step_result.done}")
            episode_data["steps"].append({"step": step, "action": action.action_type, "reward": reward})
            episode_data["total_reward"] += reward
            if step_result.done:
                episode_data["done"] = True
                print(f"\nEpisode complete!")
                break
            continue

        # Parse response
        action_dict = parse_model_response(response_text)
        if not action_dict:
            print(f"Failed to parse response: {response_text}")
            action_dict = {"action_type": "ResolveTicket", "decision": "Reject", "reason": "Parse error"}

        # Ensure only required fields for ComplianceAction
        action_data = {
            "action_type": action_dict.get("action_type", "ResolveTicket"),
            "query": action_dict.get("query"),
            "message": action_dict.get("message"),
            "decision": action_dict.get("decision"),
            "reason": action_dict.get("reason"),
        }

        # Create action
        action = ComplianceAction(**action_data)
        print(f"Action: {action.action_type}")
        if action.query:
            print(f"  Query: {action.query}")
        if action.message:
            print(f"  Message: {action.message}")
        if action.decision:
            print(f"  Decision: {action.decision}")

        # Step environment
        step_result = client.step(action)
        observation = step_result.observation
        reward = step_result.reward or 0.0

        print(f"Reward: {reward:+.2f} | Done: {step_result.done}")

        episode_data["steps"].append({
            "step": step,
            "action": action.action_type,
            "reward": reward,
        })
        episode_data["total_reward"] += reward

        if step_result.done:
            episode_data["done"] = True
            print(f"\nEpisode complete!")
            break

    episode_data["final_reward"] = episode_data["total_reward"]
    return episode_data


def main() -> None:
    """Main function to run inference on all tasks."""
    print("Corporate Compliance Environment - Inference Script")
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"Compliance API: {COMPLIANCE_API}")

    try:
        # Initialize client
        client = ComplianceEnvClient(base_url=COMPLIANCE_API).sync()

        results = {}

        with client:
            for task_id in TASKS:
                episode_data = run_episode(client, task_id)
                results[task_id] = episode_data

    except Exception as exc:
        print(f"Error: {exc}")
        return

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for task_id, data in results.items():
        print(f"\n{task_id.upper()}:")
        print(f"  Final Reward: {data['final_reward']:+.2f}")
        print(f"  Episodes Complete: {data['done']}")
        print(f"  Steps: {len(data['steps'])}")


if __name__ == "__main__":
    main()
