from openenv.core.env_server import create_fastapi_app
from app.server.environment import ComplianceEnv
from app.models import ComplianceAction, ComplianceObservation
from app.graders import grade_episode
from app.baseline import BaselineAgent
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
# HEALTH CHECK ENDPOINT
# ============================================================================

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}

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

# ============================================================================
# GRADER ENDPOINT — Score completed episodes
# ============================================================================

@app.post("/grader")
async def grade_episode_endpoint(request: dict):
    """
    Score a completed episode based on task difficulty.
    
    Request body:
    {
        "task_id": "easy|medium|hard",
        "actions_history": [...],
        "ground_truth_decision": "Approve|Reject|Escalate"
    }
    """
    task_id = request.get("task_id", "easy")
    actions_history = request.get("actions_history", [])
    ground_truth_decision = request.get("ground_truth_decision")
    
    if not ground_truth_decision:
        return {"error": "ground_truth_decision is required"}
    
    score = grade_episode(task_id, actions_history, ground_truth_decision)
    return {"task_id": task_id, "score": score}

# ============================================================================
# BASELINE ENDPOINT — Run baseline agent on all tasks
# ============================================================================

@app.post("/baseline")
async def run_baseline():
    """
    Run baseline agent on all 3 task difficulties and return scores.
    Returns scores for easy, medium, and hard tasks.
    """
    baseline = BaselineAgent(api_url="http://localhost:7860")
    results = {}
    
    try:
        for task_id in ["easy", "medium", "hard"]:
            try:
                obs = baseline.reset(task_id=task_id)
                done = False
                steps = 0
                max_steps = 10
                
                while not done and steps < max_steps:
                    action = baseline.decide_action(obs)
                    obs = baseline.step(action)
                    done = obs.get("done", False)
                    steps += 1
                
                state = baseline.get_state()
                score = state.get("cumulative_reward", 0.0)
                results[task_id] = {
                    "score": score,
                    "steps": steps,
                    "done": done
                }
            except Exception as e:
                results[task_id] = {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}
    
    return {
        "baseline_results": results,
        "average_score": sum([r.get("score", 0) for r in results.values()]) / 3
    }

