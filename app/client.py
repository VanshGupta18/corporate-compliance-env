from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from app.models import ComplianceAction, ComplianceObservation, ComplianceState

class ComplianceEnvClient(EnvClient[ComplianceAction, ComplianceObservation, ComplianceState]):
    """
    WebSocket-based client for the Compliance Environment.
    
    Key: Keep the WebSocket connection OPEN across multiple step() calls.
    The server maintains session state internally in ComplianceEnv._state.
    Each step() message uses the existing connection.
    
    DO NOT close the connection between steps.
    Only close() when the full episode is complete.
    """
    
    def _step_payload(self, action: ComplianceAction) -> dict:
        """Convert action to WebSocket message body."""
        return action.model_dump()

    def _parse_result(self, payload: dict) -> StepResult:
        """Parse WebSocket response into StepResult with typed observation."""
        obs_data = payload.get("observation", {})
        return StepResult(
            observation=ComplianceObservation(**obs_data),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> ComplianceState:
        """Parse state if needed."""
        return ComplianceState(**payload)
