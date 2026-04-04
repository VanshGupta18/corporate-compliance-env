import pytest
from app.client import ComplianceEnvClient
from app.models import ComplianceAction

API_URL = "http://localhost:8000"

# NOTE: Using context manager (with statement) to keep WebSocket connection
# open through the full episode lifecycle, then close automatically


def test_client_can_reset_environment():
    """
    Tests that the client can successfully call the /reset endpoint.
    """
    with ComplianceEnvClient(base_url=API_URL).sync() as client:
        # reset() returns a StepResult containing the observation
        result = client.reset(task_id="easy")
        observation = result.observation
        
        # Observation contains ticket information
        assert hasattr(observation, 'ticket_id')
        assert hasattr(observation, 'employee_name')
        assert hasattr(observation, 'amount')
        # ← connection closes here in __exit__


def test_client_can_take_a_step():
    """
    Tests that the client can call the /step endpoint with a valid action.
    """
    with ComplianceEnvClient(base_url=API_URL).sync() as client:
        client.reset(task_id="easy")
        
        # Create a valid action object
        action = ComplianceAction(action_type="ResolveTicket", decision="Approve", reason="Test")
        
        # step() returns a StepResult with observation, reward, done
        result = client.step(action)
        
        assert hasattr(result, 'observation')
        assert hasattr(result, 'reward')
        assert hasattr(result, 'done')
        # ← connection closes here in __exit__
    assert hasattr(result, 'done')
    
    # Verify observation has required fields
    obs = result.observation
    assert hasattr(obs, 'ticket_id')
    assert hasattr(obs, 'employee_name')
    assert hasattr(obs, 'amount')
