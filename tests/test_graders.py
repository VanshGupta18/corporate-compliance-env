import pytest
from app.client import ComplianceEnvClient
from app.models import ComplianceAction

API_URL = "http://localhost:8000"

# --- Grader Tests for 'easy' tasks ---
# NOTE: Using .sync() context manager to keep WebSocket connection open through episode

@pytest.mark.skip(reason="Test needs implementation - placeholder for grader testing")
def test_easy_grader_correct_approval():
    """Tests grader gives full score for a correct 'Approve' on an easy task."""
    with ComplianceEnvClient(base_url=API_URL).sync() as client:
        # Reset and get an easy task observation
        result = client.reset(task_id="easy")
        
        # Make a decision (in easy tasks we just decide directly)
        action = ComplianceAction(action_type="ResolveTicket", decision="Approve", reason="Looks good.")
        response = client.step(action)
        
        # Extract done - handle both dict and object formats
        done = response.get("done") if isinstance(response, dict) else response.done
        
        # Check that episode terminated
        assert done == True

@pytest.mark.skip(reason="Test needs implementation - placeholder for grader testing")
def test_easy_grader_correct_rejection():
    """Tests grader gives full score for a correct 'Reject' on an easy task."""
    with ComplianceEnvClient(base_url=API_URL).sync() as client:
        client.reset(task_id="easy")
        
        action = ComplianceAction(action_type="ResolveTicket", decision="Reject", reason="Policy violation.")
        response = client.step(action)
        
        done = response.get("done") if isinstance(response, dict) else response.done
        assert done == True

@pytest.mark.skip(reason="Test needs implementation - placeholder for grader testing")
def test_easy_grader_incorrect_decision():
    """Tests grader gives negative score for an incorrect decision on an easy task."""
    with ComplianceEnvClient(base_url=API_URL).sync() as client:
        client.reset(task_id="easy")
        
        action = ComplianceAction(action_type="ResolveTicket", decision="Approve", reason="Made a decision.")
        response = client.step(action)
        
        done = response.get("done") if isinstance(response, dict) else response.done
        assert done == True

# --- Placeholder tests for 'medium' and 'hard' tasks ---
# These are more complex as they involve multi-step interactions.
# The environment logic will need to be expanded to fully support these tests.

def test_medium_grader_searched_and_correct_placeholder():
    """
    Placeholder: Medium task where agent searches policy and is correct.
    """
    # TODO:
    # 1. Find a medium task from the dataset.
    # 2. Step 1: client.step(action_type="SearchPolicy", ...).
    # 3. Step 2: client.step(action_type="ResolveTicket", decision=correct_decision).
    # 4. Assert final reward/score is 1.0 (will require grader logic).
    pass

def test_hard_grader_full_credit_placeholder():
    """
    Placeholder: Hard task where agent requests info and is correct.
    """
    # TODO:
    # 1. Find a hard task from the dataset.
    # 2. Step 1: client.step(action_type="RequestInformation", ...).
    # 3. Step 2: client.step(action_type="ResolveTicket", decision=correct_decision).
    # 4. Assert final reward/score is 1.0 (will require grader logic).
    pass
