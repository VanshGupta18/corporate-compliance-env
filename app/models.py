from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server import Action, Observation, State
from pydantic import BaseModel, Field


class ActionType(str, Enum):
    SEARCH_POLICY = "SearchPolicy"
    REQUEST_INFORMATION = "RequestInformation"
    RESOLVE_TICKET = "ResolveTicket"


class TicketDecision(str, Enum):
    APPROVE = "Approve"
    REJECT = "Reject"
    ESCALATE = "Escalate"

class ComplianceAction(Action):
    action_type: ActionType = Field(
        ...,
        description="One of SearchPolicy, RequestInformation, ResolveTicket.",
    )
    query: Optional[str] = None
    message: Optional[str] = None
    decision: Optional[TicketDecision] = None
    reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

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
    ground_truth_decision: Optional[TicketDecision] = None  # Included only when episode is done

class ComplianceState(State):
    task_id: Optional[str] = None
    current_observation: Optional[ComplianceObservation] = None
    actions_history: List[Dict[str, Any]] = Field(default_factory=list)
    rewards_history: List[float] = Field(default_factory=list)
    cumulative_reward: float = 0.0
    is_done: bool = False


class ErrorResponse(BaseModel):
    error: str


class TaskMetadata(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    max_steps: int
    expected_steps: Optional[str] = None
    difficulty: Optional[str] = None


class TasksResponse(BaseModel):
    tasks: Dict[str, TaskMetadata]
    action_schema: Dict[str, Any]
    observation_schema: Dict[str, Any]


class GraderRequest(BaseModel):
    task_id: Literal["easy", "medium", "hard"] = "easy"
    actions_history: List[Dict[str, Any]] = Field(default_factory=list)
    ground_truth_decision: TicketDecision


class GraderResponse(BaseModel):
    task_id: str
    score: Dict[str, Any]


class BaselineTaskResult(BaseModel):
    score: float
    steps: int
    done: bool


class BaselineResponse(BaseModel):
    baseline_results: Dict[str, BaselineTaskResult]
    average_score: float


class HealthResponse(BaseModel):
    status: str


class RootResponse(BaseModel):
    status: str
    service: str
    docs_url: str


class DemoInfoResponse(BaseModel):
    message: str
    path: str
    ui_library: str
