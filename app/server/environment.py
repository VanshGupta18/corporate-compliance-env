import json
from pathlib import Path
import random
import uuid
from openenv.core.env_server import Environment
from app.models import ComplianceAction, ComplianceObservation, ComplianceState

class ComplianceEnv(Environment):
    # ✅ Enable concurrent WebSocket sessions
    # This allows multiple clients to connect simultaneously instead of limiting to 1
    SUPPORTS_CONCURRENT_SESSIONS = True
    
    def __init__(self):
        # Load data/policy.md and data/claims.json here
        self.datadir = Path(__file__).parent.parent.parent / "data"
        with open(self.datadir / "claims.json") as f:
            self.claims = json.load(f)["claims"]
        with open(self.datadir / "policy.md") as f:
            self.policy = f.read()
        
        self._state = ComplianceState()
        self._current_claim = None  # Store current claim
        self.max_steps = 5

    def reset(self, seed=None, episode_id=None, **kwargs):
        self._state.step_count = 0
        
        task_id = kwargs.get("task_id", "easy")
        # Filter claims by task_difficulty
        filtered_claims = [claim for claim in self.claims if claim["task_difficulty"] == task_id]
        if not filtered_claims:
            # Fallback to any claim if no claim with the given difficulty is found
            self._current_claim = random.choice(self.claims)
        else:
            self._current_claim = random.choice(filtered_claims)

        self._state = ComplianceState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            is_done=False,
        )
        # Return first observation
        return self._get_observation()

    def step(self, action: ComplianceAction, timeout_s=None, **kwargs):
        self._state.step_count += 1
        reward = 0.0
        done = False

        # Action validation
        if action.action_type == "SearchPolicy" and not action.query:
            reward = -0.1
            self._state.rewards_history.append(reward)
            self._state.actions_history.append(action.model_dump())
            self._state.cumulative_reward += reward
            return self._get_observation()

        if action.action_type == "ResolveTicket":
            if not action.decision or not action.reason:
                reward = -0.1
                self._state.rewards_history.append(reward)
                self._state.actions_history.append(action.model_dump())
                self._state.cumulative_reward += reward
                return self._get_observation()

            is_correct = (action.decision == self._current_claim["ground_truth_decision"])
            reward = 1.0 if is_correct else -1.0
            done = True
            self._state.is_done = done
            
            self._state.rewards_history.append(reward)
            self._state.actions_history.append(action.model_dump())
            self._state.cumulative_reward += reward
            
            return self._get_observation()

        if self._state.step_count >= self.max_steps:
            done = True
            reward = -0.5  # Penalty for running out of steps
            self._state.is_done = done

        self._state.rewards_history.append(reward)
        self._state.actions_history.append(action.model_dump())
        self._state.cumulative_reward += reward
        self._state.is_done = done

        print(f"Step {self._state.step_count}: Action={action.action_type}, Reward={reward}, Done={done}")
        return self._get_observation()

    def _get_observation(self):
        if not self._current_claim:
            # This should not happen in a normal flow
            return None
        
        claim = self._current_claim
        obs = ComplianceObservation(
            done=self._state.is_done,
            reward=self._state.rewards_history[-1] if self._state.rewards_history else None,
            ticket_id=claim.get("id", "UNKNOWN"),  # Use "id" field from claims
            employee_name=claim["employee_name"],
            employee_role=claim["employee_role"],
            employee_level=claim["employee_level"],
            amount=claim["amount"],
            currency=claim["currency"],
            description=claim["description"],
            has_receipt=claim["has_receipt"],
            missing_document=claim.get("missing_document"),
            rule_keyword=claim["rule_keyword"],
            risk_score=claim["risk_score"],
            env_message="",
            step_count=self._state.step_count,
            max_steps=self.max_steps,
        )
        self._state.current_observation = obs
        return obs

    @property
    def state(self):
        return self._state
