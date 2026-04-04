import requests
import pytest

# It's a good practice to have the API URL as a configurable fixture
# For now, we'll hardcode it as the server is expected to run locally during tests.
API_URL = "http://localhost:8000"

def test_tasks_endpoint_is_available():
    """
    Smoke test to check if the /tasks endpoint is running and returns a valid structure.
    """
    try:
        response = requests.get(f"{API_URL}/tasks")
        # Check for a successful response
        assert response.status_code == 200
        
        # Check that the response body is valid JSON
        data = response.json()
        
        # Check for the presence of top-level keys
        assert "tasks" in data
        assert "action_schema" in data
        
        # Check if the tasks list contains the expected tasks
        assert "easy" in data["tasks"]
        assert "medium" in data["tasks"]
        assert "hard" in data["tasks"]

    except requests.exceptions.ConnectionError as e:
        pytest.fail(f"Could not connect to the API server at {API_URL}. Is it running? Error: {e}")

def test_reset_endpoint_works_for_all_tasks():
    """
    Smoke test to ensure the /reset endpoint works for all task difficulties.
    """
    for task_id in ["easy", "medium", "hard"]:
        try:
            response = requests.post(f"{API_URL}/reset", params={"task_id": task_id})
            assert response.status_code == 200
            data = response.json()
            # Reset response wraps observation in 'observation' field along with done/reward
            assert isinstance(data, dict)
            assert "observation" in data
            obs = data["observation"]
            assert "ticket_id" in obs
            assert "employee_name" in obs
            assert "amount" in obs
        except requests.exceptions.ConnectionError as e:
            pytest.fail(f"Could not connect to the API server at {API_URL}. Is it running? Error: {e}")

def test_state_endpoint_is_available():
    """
    Smoke test to check if the /state endpoint is running.
    """
    try:
        response = requests.get(f"{API_URL}/state")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        # State contains episode tracking data
        assert "episode_id" in data or "step_count" in data
    except requests.exceptions.ConnectionError as e:
        pytest.fail(f"Could not connect to the API server at {API_URL}. Is it running? Error: {e}")
