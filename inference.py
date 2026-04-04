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
from typing import List, Dict, Optional

from openai import OpenAI
from app.client import ComplianceEnvClient
from app.models import ComplianceAction

# ===== Environment Configuration =====
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
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


def run_episode(client: ComplianceEnvClient, task_id: str) -> Dict:
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
        user_prompt = build_user_prompt(observation.model_dump(), task_id, step)

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
            print(f"LLM request failed: {exc}. Using default action.")
            response_text = '{"action_type": "ResolveTicket", "decision": "Reject", "reason": "Error"}'

        # Parse response
        action_dict = parse_model_response(response_text)
        if not action_dict:
            print(f"Failed to parse response: {response_text}")
            action_dict = {"action_type": "ResolveTicket", "decision": "Reject", "reason": "Parse error"}

        # Create action
        action = ComplianceAction(**action_dict)
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
