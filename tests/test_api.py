from fastapi.testclient import TestClient

from app.server.app import app


client = TestClient(app)


def test_tasks_endpoint_is_available():
    response = client.get("/tasks")
    assert response.status_code == 200

    data = response.json()
    assert "tasks" in data
    assert "action_schema" in data
    assert "observation_schema" in data
    assert "easy" in data["tasks"]
    assert "medium" in data["tasks"]
    assert "hard" in data["tasks"]


def test_reset_endpoint_works_for_all_tasks():
    for task_id in ["easy", "medium", "hard"]:
        response = client.post("/reset", params={"task_id": task_id})
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert "observation" in data
        obs = data["observation"]
        assert "ticket_id" in obs
        assert "employee_name" in obs
        assert "amount" in obs


def test_state_endpoint_is_available():
    client.post("/reset", params={"task_id": "easy"})
    response = client.get("/state")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "step_count" in data


def test_openapi_contains_custom_paths():
    schema = client.get("/openapi.json").json()
    paths = schema.get("paths", {})
    assert "/tasks" in paths
    assert "/grader" in paths
    assert "/baseline" in paths
    assert "/demo/info" in paths
