import json
from pathlib import Path
import random
import uuid
from openenv.core.env_server import Environment
from app.models import ComplianceAction, ComplianceObservation, ComplianceState


class ComplianceEnv(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self.datadir = Path(__file__).parent.parent.parent / "data"
        with open(self.datadir / "claims.json") as f:
            self.claims = json.load(f)["claims"]
        with open(self.datadir / "policy.md") as f:
            self.policy = f.read()

        self._state = ComplianceState()
        self._current_claim = None
        self.max_steps = 5
        self.task_max_steps = {"easy": 3, "medium": 5, "hard": 8}

    def reset(self, seed=None, episode_id=None, **kwargs):
        task_id = kwargs.get("task_id", "easy")
        self.max_steps = self.task_max_steps.get(task_id, 5)

        filtered_claims = [c for c in self.claims if c["task_difficulty"] == task_id]
        self._current_claim = random.choice(filtered_claims if filtered_claims else self.claims)

        self._state = ComplianceState(
            episode_id=episode_id or str(uuid.uuid4()),
            task_id=task_id,
            step_count=0,
            is_done=False,
        )
        return self._get_observation()

    def step(self, action: ComplianceAction, timeout_s=None, **kwargs):
        self._state.step_count += 1
        reward = 0.05   # neutral default (strictly > 0)
        done = False

        if action.action_type == "SearchPolicy":
            # Tool use — no reward, just costs a step
            reward = 0.05

        elif action.action_type == "RequestInformation":
            # Reward if claim actually has a missing document, else small penalty
            if self._current_claim and self._current_claim.get("missing_document"):
                reward = 0.15   # useful action
            else:
                reward = 0.02   # unnecessary but not harshly penalised

        elif action.action_type == "ResolveTicket":
            if not action.decision or not action.reason:
                reward = 0.02   # incomplete resolve
            else:
                is_correct = (
                    self._current_claim and
                    action.decision == self._current_claim.get("ground_truth_decision")
                )
                reward = 0.99 if is_correct else 0.01
                done = True
                self._state.is_done = True

        # Timeout penalty (still strictly > 0)
        if not done and self._state.step_count >= self.max_steps:
            reward = 0.01
            done = True
            self._state.is_done = True

        # Clamp strictly between 0 and 1 as a safety net
        reward = max(0.01, min(0.99, reward))

        self._state.rewards_history.append(reward)
        self._state.actions_history.append(action.model_dump())
        self._state.cumulative_reward += reward
        self._state.is_done = done

        return self._get_observation()

    def _get_observation(self):
        if not self._current_claim:
            return ComplianceObservation(
                done=True,
                reward=0.01,
                ticket_id="UNKNOWN",
                employee_name="Unknown",
                employee_role="Unknown",
                employee_level="Unknown",
                amount=0.0,
                currency="INR",
                description="No claim available",
                has_receipt=False,
                missing_document=None,
                rule_keyword="",
                risk_score=0.0,
                env_message="No claim loaded",
                step_count=self._state.step_count,
                max_steps=self.max_steps,
            )

        claim = self._current_claim
        last_reward = self._state.rewards_history[-1] if self._state.rewards_history else 0.05
        obs = ComplianceObservation(
            done=self._state.is_done,
            reward=last_reward,
            ticket_id=claim.get("id", "UNKNOWN"),
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
