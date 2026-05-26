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
        response = client.post("/reset", json={"task_id": task_id})
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert "observation" in data
        obs = data["observation"]
        assert "ticket_id" in obs
        assert "employee_name" in obs
        assert "amount" in obs


def test_state_endpoint_is_available():
    client.post("/reset", json={"task_id": "easy"})
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


def test_grader_accepts_claim_metadata():
    response = client.post(
        "/grader",
        json={
            "task_id": "medium",
            "actions_history": [
                {"action_type": "SearchPolicy", "query": "daytime cab manager"},
                {
                    "action_type": "ResolveTicket",
                    "decision": "Reject",
                    "reason": "Daytime cab requires manager note.",
                },
            ],
            "ground_truth_decision": "Reject",
            "claim": {"rule_keyword": "daytime cab"},
        },
    )
    assert response.status_code == 200
    score = response.json()["score"]
    assert score["components"]["useful_search"] > 0


def test_websocket_multi_step_episode():
    """WebSocket session preserves state across steps (HTTP does not)."""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "reset", "data": {"task_id": "easy"}})
        reset_msg = ws.receive_json()
        assert reset_msg.get("type") == "observation"
        ws.send_json(
            {
                "type": "step",
                "data": {
                    "action_type": "ResolveTicket",
                    "decision": "Approve",
                    "reason": "Test resolution.",
                },
            }
        )
        step_msg = ws.receive_json()
        assert step_msg.get("type") == "observation"
        assert step_msg.get("data", {}).get("done") is True
