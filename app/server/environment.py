import json
import os
import random
import uuid
from copy import deepcopy
from pathlib import Path

from openenv.core.env_server import Environment

from app.models import ComplianceAction, ComplianceObservation, ComplianceState, TicketDecision
from app.document_utils import infer_required_document
from app.policy_snippets import document_simulation, match_policy_snippet


class ComplianceEnv(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self.datadir = Path(__file__).parent.parent.parent / "data"
        self._load_claims()
        with open(self.datadir / "policy.md", encoding="utf-8") as f:
            self.policy = f.read()

        self._state = ComplianceState()
        self._current_claim = None
        self.max_steps = 5
        self.task_max_steps = {"easy": 3, "medium": 5, "hard": 8}

        # Per-episode hidden state
        self._policy_revealed = False
        self._useful_search = False
        self._ever_useful_search = False
        self._document_requested = False
        self._document_satisfied = False
        self._document_request_resolved = False
        self._last_requested_document = None
        self._env_message = ""

    def _load_claims(self):
        """Load all claims and retain split labels for deterministic evaluation."""
        splits_dir = self.datadir / "splits"
        all_claims = []
        if splits_dir.exists():
            for split in ("train", "validation", "test"):
                path = splits_dir / f"{split}.json"
                if path.exists():
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                        for claim in data.get("claims", []):
                            claim.setdefault("split", split)
                            all_claims.append(claim)
        if not all_claims:
            with open(self.datadir / "claims.json", encoding="utf-8") as f:
                all_claims = json.load(f)["claims"]
        self.claims = all_claims
        self._claims_by_id = {c["id"]: c for c in all_claims}

    def reset(self, seed=None, episode_id=None, **kwargs):
        task_id = kwargs.get("task_id", "easy")
        split = kwargs.get("split") or os.getenv("COMPLIANCE_SPLIT", "train")
        self.max_steps = self.task_max_steps.get(task_id, 5)

        claim_id = kwargs.get("claim_id")
        if claim_id and claim_id in self._claims_by_id:
            self._current_claim = deepcopy(self._claims_by_id[claim_id])
        else:
            filtered = [
                c
                for c in self.claims
                if c.get("task_difficulty") == task_id and c.get("split", "train") == split
            ]
            if not filtered:
                filtered = [c for c in self.claims if c.get("task_difficulty") == task_id]
            if seed is not None:
                rng = random.Random(seed)
                self._current_claim = deepcopy(rng.choice(filtered if filtered else self.claims))
            else:
                self._current_claim = deepcopy(
                    random.choice(filtered if filtered else self.claims)
                )

        self._policy_revealed = False
        self._useful_search = False
        self._ever_useful_search = False
        self._document_requested = False
        self._document_satisfied = False
        self._document_request_resolved = False
        self._last_requested_document = None
        self._env_message = ""

        self._state = ComplianceState(
            episode_id=episode_id or str(uuid.uuid4()),
            task_id=task_id,
            step_count=0,
            is_done=False,
        )
        return self._get_observation()

    def _has_searched_policy(self) -> bool:
        return any(
            a.get("action_type") == "SearchPolicy" for a in self._state.actions_history
        )

    def _has_requested_info(self) -> bool:
        return any(
            a.get("action_type") == "RequestInformation"
            for a in self._state.actions_history
        )

    def _check_max_steps(self, reward: float) -> tuple[float, bool]:
        if self._state.step_count >= self.max_steps and not self._state.is_done:
            self._state.is_done = True
            self._env_message = "Maximum steps reached. Episode terminated."
            return -0.5, True
        return reward, False

    @staticmethod
    def _clamp_reward(reward: float) -> float:
        return max(-1.0, min(1.0, reward))

    @staticmethod
    def _decision_value(decision) -> str | None:
        """Normalize TicketDecision enum or string to Approve/Reject/Escalate."""
        if decision is None:
            return None
        if isinstance(decision, TicketDecision):
            return decision.value
        text = str(decision)
        if text.startswith("TicketDecision."):
            return text.rsplit(".", 1)[-1]
        return text

    def _finalize_step(self, reward: float, action: ComplianceAction) -> ComplianceObservation:
        reward = self._clamp_reward(reward)
        self._state.rewards_history.append(reward)
        self._state.actions_history.append(action.model_dump())
        self._state.cumulative_reward += reward
        reward, forced_done = self._check_max_steps(reward)
        reward = self._clamp_reward(reward)
        if forced_done:
            self._state.rewards_history[-1] = reward
            self._state.cumulative_reward = sum(self._state.rewards_history)
        return self._get_observation()

    def step(self, action: ComplianceAction, timeout_s=None, **kwargs):
        if self._state.is_done:
            raise RuntimeError("Episode already done. Call reset() first.")

        self._state.step_count += 1
        reward = 0.0
        task_id = self._state.task_id or "easy"
        claim = self._current_claim or {}

        if action.action_type == "SearchPolicy":
            if not action.query:
                reward = -0.1
                self._env_message = "SearchPolicy requires a non-empty query."
            elif task_id == "easy":
                reward = -0.15
                self._env_message = "Easy tasks do not require policy search. Resolve directly."
            elif self._has_searched_policy():
                reward = -0.35 if self._ever_useful_search else -0.25
                self._env_message = (
                    "Policy already retrieved. Do not search again — "
                    "use RequestInformation (if a document is missing) or ResolveTicket."
                )
            else:
                snippet, relevant = match_policy_snippet(
                    claim.get("rule_keyword", ""), action.query or ""
                )
                self._policy_revealed = True
                self._useful_search = relevant
                self._ever_useful_search = self._ever_useful_search or relevant
                self._env_message = snippet
                reward = 0.15 if relevant else -0.05

            return self._finalize_step(reward, action)

        if action.action_type == "RequestInformation":
            if task_id == "easy":
                reward = -0.15
                self._env_message = "Easy tasks do not require document requests."
            else:
                required = (
                    claim.get("missing_document")
                    or claim.get("required_document")
                )
                msg = (action.message or "").lower()
                if not required:
                    reward = -0.2
                    self._env_message = "No missing document on this ticket."
                elif self._document_satisfied:
                    reward = -0.2
                    self._env_message = (
                        "Required document already received. Resolve the ticket now."
                    )
                elif self._document_request_resolved:
                    reward = -0.3
                    self._env_message = (
                        "Document already requested. Resolve the ticket now "
                        "(Approve/Reject/Escalate)."
                    )
                elif (
                    required in msg
                    or required.replace("_", " ") in msg
                    or required.replace("_", "") in msg.replace("_", "").replace(" ", "")
                ):
                    duplicate = (
                        self._document_requested
                        and self._last_requested_document == required
                    )
                    self._document_requested = True
                    self._last_requested_document = required
                    doc_type = required
                    self._env_message = document_simulation(doc_type, claim)
                    if claim.get("document_outcome") == "provided":
                        self._document_satisfied = True
                        claim["missing_document"] = None
                        claim["_document_cleared"] = True
                        reward = -0.05 if duplicate else 0.2
                    else:
                        self._document_request_resolved = True
                        reward = -0.2 if duplicate else 0.1
                        self._env_message = (
                            f"{document_simulation(doc_type, claim)} "
                            "Resolve the ticket now; do not request the same document again."
                        )
                else:
                    reward = -0.1
                    hint = infer_required_document(
                        self._get_observation().model_dump(), claim
                    )
                    self._env_message = (
                        f"Requested document does not match required '{required}'. "
                        f"Request '{hint}' explicitly in your message."
                    )

            return self._finalize_step(reward, action)

        if action.action_type == "ResolveTicket":
            if not action.decision or not action.reason:
                reward = -0.1
                return self._finalize_step(reward, action)

            decision = self._decision_value(action.decision)

            if task_id == "medium":
                if not self._has_searched_policy():
                    reward -= 1.0
                elif not self._ever_useful_search:
                    reward -= 0.4

            if task_id == "hard":
                required = claim.get("missing_document") or claim.get("required_document")
                if required and not self._document_requested:
                    reward -= 0.15
                if (
                    required
                    and claim.get("document_outcome") == "provided"
                    and not self._document_satisfied
                ):
                    reward -= 0.3
                if (
                    required
                    and claim.get("document_outcome") == "not_provided"
                    and decision == TicketDecision.APPROVE.value
                ):
                    reward -= 0.3
                if not self._has_searched_policy():
                    reward -= 0.15

            gt = claim.get("ground_truth_decision")
            gt_value = self._decision_value(gt) or str(gt)
            is_correct = decision == gt_value
            reward += 1.0 if is_correct else -1.0
            self._state.is_done = True

            return self._finalize_step(reward, action)

        reward = -0.1
        return self._finalize_step(reward, action)

    def _visible_description(self, claim: dict, task_id: str) -> str:
        if task_id == "medium" and claim.get("vague_description"):
            return claim["vague_description"]
        if task_id == "hard" and claim.get("vague_description"):
            return claim["vague_description"]
        return claim.get("description", "")

    def _visible_rule_keyword(self, claim: dict, task_id: str) -> str:
        if task_id == "easy":
            return claim.get("rule_keyword", "unknown")
        if self._policy_revealed:
            return claim.get("rule_keyword", "unknown")
        return "hidden"

    def _visible_missing_document(self, claim: dict, task_id: str) -> str | None:
        md = claim.get("missing_document") or claim.get("required_document")
        if task_id == "easy":
            return claim.get("missing_document")
        if (
            self._document_satisfied
            or self._document_request_resolved
            or claim.get("_document_cleared")
        ):
            return None
        if task_id in ("medium", "hard") and md:
            if self._policy_revealed:
                return md
            return "required"
        return claim.get("missing_document")

    def _get_observation(self):
        if not self._current_claim:
            raise RuntimeError("No active claim. Call reset() before step().")

        claim = self._current_claim
        task_id = self._state.task_id or "easy"

        obs = ComplianceObservation(
            done=self._state.is_done,
            reward=self._state.rewards_history[-1] if self._state.rewards_history else None,
            ticket_id=claim.get("id", "UNKNOWN"),
            employee_name=claim["employee_name"],
            employee_role=claim["employee_role"],
            employee_level=claim["employee_level"],
            amount=claim["amount"],
            currency=claim["currency"],
            description=self._visible_description(claim, task_id),
            has_receipt=claim["has_receipt"],
            missing_document=self._visible_missing_document(claim, task_id),
            rule_keyword=self._visible_rule_keyword(claim, task_id),
            policy_retrieved=self._policy_revealed,
            risk_score=claim["risk_score"],
            env_message=self._env_message,
            step_count=self._state.step_count,
            max_steps=self.max_steps,
            ground_truth_decision=None,
        )
        self._state.current_observation = obs
        return obs

    @property
    def state(self):
        return self._state
