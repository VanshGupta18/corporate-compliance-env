import pytest
from app.client import ComplianceEnvClient
from app.models import ComplianceAction

API_URL = "http://localhost:8000"

# NOTE: The CRITICAL FIX for WebSocket errors:
# ============================================
# WRONG: Create new client for each test, connection closes between steps
# RIGHT: Keep ONE WebSocket connection OPEN through all steps(), 
#        then close() only when episode is done
#
# Server maintains session state internally, so the connection persisting
# is what allows multi-step episodes to work.


def test_penalty_for_invalid_action_missing_query():
    """
    Tests that the environment returns a penalty when a SearchPolicy action
    is sent without a 'query'.
    
    Single-step episode: reset() → step() → close()
    """
    with ComplianceEnvClient(base_url=API_URL).sync() as client:
        client.reset(task_id="easy")
        
        action = ComplianceAction(action_type="SearchPolicy")  # Missing query
        result = client.step(action)
        
        assert result.reward == -0.1
        assert not result.done
        # ← connection stays open until __exit__


def test_penalty_for_invalid_action_missing_decision():
    """
    Tests that the environment returns a penalty when a ResolveTicket action
    is sent without a 'decision'.
    
    Single-step episode: reset() → step() → close()
    """
    with ComplianceEnvClient(base_url=API_URL).sync() as client:
        client.reset(task_id="easy")
        
        action = ComplianceAction(action_type="ResolveTicket", reason="Incomplete action")  # Missing decision
        result = client.step(action)
        
        assert result.reward == -0.1
        assert not result.done


def test_penalty_for_max_steps_reached():
    """
    Tests that the episode terminates with a penalty if the max number of steps is exceeded.
    
    Multi-step episode: reset() → step() → step() → ... → step() → close()
    
    KEY: Keep connection open through all step() calls!
    The ConnectionClosedOK error happened because we were closing between steps.
    """
    with ComplianceEnvClient(base_url=API_URL).sync() as client:
        client.reset(task_id="hard")
        
        # Multiple steps in same episode - connection stays open
        # First step to initialize result
        action = ComplianceAction(action_type="SearchPolicy", query="some policy")
        result = client.step(action)
        
        # Additional steps until max_steps exceeded
        for i in range(1, 8):  # hard task has max_steps of 8
            if result.done:
                break
            action = ComplianceAction(action_type="SearchPolicy", query="some policy")
            result = client.step(action)
        
        # After exceeding max_steps, should get penalty
        assert result.done
        assert result.reward == -0.5  # Penalty for running out of steps
        # ← connection closes only here in __exit__


@pytest.mark.skip(reason="Test needs implementation - placeholder for reward testing")
def test_reward_for_wrong_resolution():
    """
    Tests that the environment returns a negative reward for an incorrect resolution.
    """
    with ComplianceEnvClient(base_url=API_URL).sync() as client:
        client.reset(task_id="easy")
        # TODO: implement
