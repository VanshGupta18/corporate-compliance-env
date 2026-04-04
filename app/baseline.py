"""
Baseline agent for the Compliance Environment.
A simple heuristic-based policy that demonstrates basic gameplay.
"""

import requests
from typing import Dict, Any, Optional


class BaselineAgent:
    """Simple baseline agent with hardcoded heuristic policy."""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.session = requests.Session()
    
    def reset(self, task_id: str = "easy") -> Dict[str, Any]:
        """Start a new episode."""
        response = self.session.post(f"{self.api_url}/ws", json={"method": "reset", "task_id": task_id})
        response.raise_for_status()
        return response.json()
    
    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Take a step in the environment."""
        response = self.session.post(f"{self.api_url}/ws", json={"method": "step", "action": action})
        response.raise_for_status()
        return response.json()
    
    def get_state(self) -> Dict[str, Any]:
        """Get current episode state."""
        response = self.session.get(f"{self.api_url}/ws")
        response.raise_for_status()
        return response.json()
    
    def decide_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simple heuristic policy:
        1. For easy tasks: immediately resolve
        2. For medium tasks: search policy first, then resolve
        3. For hard tasks: request info, then resolve
        """
        risk_score = observation.get("risk_score", 0.0)
        rule_keyword = observation.get("rule_keyword", "")
        missing_doc = observation.get("missing_document")
        
        # If this is clearly a hard task (missing document)
        if missing_doc:
            return {
                "action_type": "RequestInformation",
                "message": f"Please provide {missing_doc}"
            }
        
        # If risky or unknown rule, search first
        if risk_score > 0.5 or rule_keyword:
            return {
                "action_type": "SearchPolicy",
                "query": rule_keyword or "general policy"
            }
        
        # Otherwise, make a decision
        # Simple heuristic: reject if high risk, approve if low risk
        decision = "Reject" if risk_score > 0.5 else "Approve"
        return {
            "action_type": "ResolveTicket",
            "decision": decision,
            "reason": f"Risk score: {risk_score}"
        }
    
    def run_episode(self, task_id: str = "easy") -> Dict[str, Any]:
        """Run one full episode and return results."""
        print(f"\n🤖 Starting {task_id} task...")
        
        obs = self.reset(task_id=task_id)
        observation = obs.get("observation", {})
        done = obs.get("done", False)
        episode_reward = obs.get("reward", 0.0)
        
        step_count = 0
        max_steps = 10
        print(f"Initial obs: {observation.get('ticket_id', 'N/A')}, risk: {observation.get('risk_score', 0)}")
        
        while not done and step_count < max_steps:
            step_count += 1
            
            # Decide action
            action = self.decide_action(observation)
            print(f"  Step {step_count}: {action['action_type']}")
            
            # Take step
            result = self.step(action)
            observation = result.get("observation", {})
            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            episode_reward += reward
            
            print(f"  → Reward: {reward:.2f}, Done: {done}")
        
        print(f"✅ Episode finished. Total reward: {episode_reward:.2f}")
        return {
            "task_id": task_id,
            "steps": step_count,
            "total_reward": episode_reward,
            "status": "completed" if done else "truncated"
        }


def main():
    """Run baseline agent on all task difficulties."""
    agent = BaselineAgent()
    
    results = {}
    for task_id in ["easy", "medium", "hard"]:
        try:
            result = agent.run_episode(task_id=task_id)
            results[task_id] = result
        except Exception as e:
            print(f"❌ Error on {task_id}: {e}")
            results[task_id] = {"error": str(e)}
    
    print("\n📊 Baseline Results:")
    for task_id, result in results.items():
        if "error" in result:
            print(f"  {task_id}: ERROR - {result['error']}")
        else:
            print(f"  {task_id}: {result['steps']} steps, reward={result['total_reward']:.2f}")


if __name__ == "__main__":
    main()
