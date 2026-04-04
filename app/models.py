from openenv.core.env_server import Action, Observation, State
from pydantic import Field
from typing import Optional, List, Dict, Any

class ComplianceAction(Action):
    action_type: str
    query: Optional[str] = None
    message: Optional[str] = None
    decision: Optional[str] = None
    reason: Optional[str] = None

class ComplianceObservation(Observation):
    ticket_id: str
    employee_name: str
    employee_role: str
    employee_level: str
    amount: float
    currency: str
    description: str
    has_receipt: bool
    missing_document: Optional[str] = None
    rule_keyword: str
    risk_score: float = Field(..., ge=0, le=1)
    env_message: str
    step_count: int
    max_steps: int

class ComplianceState(State):
    current_observation: Optional[ComplianceObservation] = None
    actions_history: List[Dict[str, Any]] = Field(default_factory=list)
    rewards_history: List[float] = Field(default_factory=list)
    cumulative_reward: float = 0.0
    is_done: bool = False
