from openenv.core.env_server import create_fastapi_app
from app.server.environment import ComplianceEnv
from app.models import (
    BaselineResponse,
    BaselineTaskResult,
    ComplianceAction,
    ComplianceObservation,
    DemoInfoResponse,
    ErrorResponse,
    GraderRequest,
    GraderResponse,
    HealthResponse,
    RootResponse,
    TaskMetadata,
    TasksResponse,
)
from app.graders import grade_episode
from app.baseline import BaselineAgent
from app.dashboard import build_demo
from fastapi import HTTPException, Request
from fastapi.responses import RedirectResponse
import uvicorn
from pathlib import Path
from typing import Dict

import gradio as gr


APP_TITLE = "Corporate Compliance OpenEnv API"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = (
    "OpenEnv-compatible reinforcement learning environment for auditing corporate "
    "expense claims against internal policy."
)

# Create the main app using OpenEnv
# This automatically creates /ws, /reset, /step, /state endpoints
app = create_fastapi_app(
    ComplianceEnv,
    ComplianceAction,
    ComplianceObservation,
)
app.title = APP_TITLE
app.version = APP_VERSION
app.description = APP_DESCRIPTION
app.contact = {"name": "Vansh Gupta"}
app.openapi_tags = [
    {"name": "core", "description": "Health and service metadata endpoints."},
    {"name": "tasks", "description": "Task metadata and schema endpoints."},
    {"name": "evaluation", "description": "Grading and baseline benchmark endpoints."},
    {"name": "demo", "description": "Live demo dashboard endpoint."},
]


@app.get("/", response_model=RootResponse, tags=["core"])
async def root(request: Request):
    """Root endpoint for platform readiness checks and docs discovery."""
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return RedirectResponse(url="/docs", status_code=307)

    return RootResponse(
        status="ok",
        service="corporate-compliance-env",
        docs_url="/docs",
    )


# ============================================================================
# HEALTH CHECK ENDPOINT
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["core"])
async def health_check():
    """Simple health check endpoint."""
    return HealthResponse(status="ok")

# ============================================================================
# ADD CUSTOM /tasks ENDPOINT to the main app
# ============================================================================

@app.get(
    "/tasks",
    response_model=TasksResponse,
    tags=["tasks"],
    responses={500: {"model": ErrorResponse}},
)
async def get_tasks() -> TasksResponse:
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
        tasks_dict: Dict[str, TaskMetadata] = {
            task["id"]: TaskMetadata(**task) for task in tasks_list
        }
        return TasksResponse(
            tasks=tasks_dict,
            action_schema=config.get("action", {}),
            observation_schema=config.get("observation", {}),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load tasks metadata: {exc}") from exc

# ============================================================================
# GRADER ENDPOINT — Score completed episodes
# ============================================================================

@app.post(
    "/grader",
    response_model=GraderResponse,
    tags=["evaluation"],
    responses={400: {"model": ErrorResponse}},
)
async def grade_episode_endpoint(request: GraderRequest) -> GraderResponse:
    """
    Score a completed episode based on task difficulty.
    
    Request body:
    {
        "task_id": "easy|medium|hard",
        "actions_history": [...],
        "ground_truth_decision": "Approve|Reject|Escalate"
    }
    """
    score = grade_episode(
        request.task_id,
        request.actions_history,
        request.ground_truth_decision.value,
    )
    return GraderResponse(task_id=request.task_id, score=score)

# ============================================================================
# BASELINE ENDPOINT — Run baseline agent on all tasks
# ============================================================================

@app.post(
    "/baseline",
    response_model=BaselineResponse,
    tags=["evaluation"],
    responses={500: {"model": ErrorResponse}},
)
async def run_baseline() -> BaselineResponse:
    """
    Run baseline agent on all 3 task difficulties and return scores.
    Returns scores for easy, medium, and hard tasks.
    """
    baseline = BaselineAgent(api_url="http://127.0.0.1:7860")
    env = ComplianceEnv()
    results = {}
    
    try:
        for task_id in ["easy", "medium", "hard"]:
            try:
                obs = env.reset(task_id=task_id)
                done = False
                steps = 0
                max_steps = env.task_max_steps.get(task_id, 8) + 2
                while not done and steps < max_steps:
                    action_dict = baseline.decide_action(obs.model_dump())
                    action = ComplianceAction(**action_dict)
                    obs = env.step(action)
                    done = bool(obs.done)
                    steps += 1
                score = float(env.state.cumulative_reward)
                results[task_id] = BaselineTaskResult(score=score, steps=steps, done=done)
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed baseline run for task '{task_id}': {exc}",
                ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    average_score = sum(result.score for result in results.values()) / len(results)
    return BaselineResponse(baseline_results=results, average_score=average_score)
demo = build_demo()
app = gr.mount_gradio_app(app, demo, path="/demo")


@app.get("/demo/info", response_model=DemoInfoResponse, tags=["demo"])
async def demo_info() -> DemoInfoResponse:
    return DemoInfoResponse(
        message="Gradio dashboard mounted successfully.",
        path="/demo",
        ui_library="gradio",
    )


def main():
    """Main entry point for the server."""
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()

