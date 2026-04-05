from openenv.core.env_server import create_fastapi_app
from app.server.environment import ComplianceEnv
from app.models import ComplianceAction, ComplianceObservation
from fastapi import FastAPI
import json
from pathlib import Path

# Create the main app using OpenEnv
# This automatically creates /ws, /reset, /step, /state endpoints
app = create_fastapi_app(
    ComplianceEnv,
    ComplianceAction,
    ComplianceObservation,
)

# ============================================================================
# ADD CUSTOM /tasks ENDPOINT to the main app
# ============================================================================

@app.get("/tasks")
async def get_tasks():
    """
    Returns list of available tasks and their metadata.
    """
    # Load openenv.yaml to get task definitions
    yaml_path = Path(__file__).parent.parent.parent / "openenv.yaml"
    try:
        import yaml
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        
        tasks_list = config.get("tasks", [])
        # Convert to dict format with task id as key for compatibility
        tasks_dict = {task["id"]: task for task in tasks_list}
        
        return {
            "tasks": tasks_dict,
            "action_schema": config.get("action", {}),
            "observation_schema": config.get("observation", {})
        }
    except Exception as e:
        # Fallback if yaml loading fails
        return {
            "tasks": {
                "easy": {
                    "id": "easy",
                    "name": "single_step_classification",
                    "max_steps": 3,
                    "expected_steps": 1
                },
                "medium": {
                    "id": "medium", 
                    "name": "policy_retrieval",
                    "max_steps": 5,
                    "expected_steps": 2
                },
                "hard": {
                    "id": "hard",
                    "name": "multi_turn_contextual_decision", 
                    "max_steps": 8,
                    "expected_steps": "3-4"
                }
            },
            "action_schema": {
                "type": "object",
                "properties": {
                    "action_type": {"enum": ["SearchPolicy", "RequestInformation", "ResolveTicket"]}
                }
            }
        }

