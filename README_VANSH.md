# Vansh - 3 Day Plan (Beginner Friendly)

## Role Focus
Project setup, backend environment flow, and final integration.

## Shared Communication Rules (Follow Every Day)
1. Post daily standup update at 09:30 in the team chat using:
   `[Day X][Vansh] Plan: ... | Blockers: ... | Need from: ...`
2. Post dependency check-in at 14:00:
   `Ready for handoff / Waiting for <name> on <item>`
3. Post end-of-day handoff at 18:00 with links to changed files and pending items.
4. If blocked for more than 30 minutes, ping both teammates immediately.

## Day 1 - Project Foundation

### Main Goal
Create a runnable project skeleton and lock API/data schemas.

### Tasks
1.  **Create project folders**: `app/`, `data/`, `tests/`.
    -   **Why it's important**: This sets up a standard, organized directory structure. Separating `app` (business logic), `data` (datasets), and `tests` (quality assurance) keeps the project clean and easy for everyone to navigate.
    -   **How to do it**: Open your terminal and run the command `mkdir app data tests`.
    -   **Reference Code**:
        ```bash
        mkdir app data tests
        ```

2.  **Add starter files**: `app/main.py`, `app/models.py`, `app/environment.py`, `app/graders.py`, `data/policy.md`, `data/claims.json`, `tests/test_graders.py`, `requirements.txt`, `openenv.yaml`.
    -   **Why it's important**: Creating empty files establishes the complete project skeleton, so all team members can see where their work will eventually go.
    -   **How to do it**: Use the `touch` command for each file. For example: `touch app/main.py app/models.py data/policy.md requirements.txt openenv.yaml` and so on for all the files listed.
    -   **Reference Code**:
        ```bash
        touch app/main.py app/models.py app/environment.py app/graders.py data/policy.md data/claims.json tests/test_graders.py requirements.txt openenv.yaml
        ```

3.  **Set up Python environment and install core packages** (`fastapi`, `uvicorn`, `pydantic`, `pytest`).
    -   **Why it's important**: This creates an isolated space for the project's software libraries. It prevents conflicts with other projects and ensures all three team members are using the exact same library versions, which is critical for avoiding "it works on my machine" problems.
    -   **How to do it**:
        1.  Create the virtual environment: `python3 -m venv venv`
        2.  Activate it: `source venv/bin/activate`
        3.  Create a `requirements.txt` file and add the package names, each on a new line.
        4.  Install the packages: `pip install -r requirements.txt`
    -   **Reference Code (`requirements.txt`)**:
        ```
        fastapi
        uvicorn[standard]
        pydantic
        pytest
        ```

4.  **Implement initial Pydantic schemas in `app/models.py`** for action, observation, and state.
    -   **Why it's important**: This is a critical task. Pydantic schemas act as a formal "contract" for all data moving through the system. By defining the exact structure of data, you eliminate guesswork. Sanya will use this contract to create her dataset, and Vedika will use it to write her tests, preventing bugs from data mismatches.
    -   **How to do it**:
        1.  In `app/models.py`, import `BaseModel` and `Optional` from `pydantic` and `typing`.
        2.  Define a class for each schema, inheriting from `BaseModel`. Example: `class ComplianceAction(BaseModel):`.
        3.  Inside each class, define the fields with their types, like `action_type: str` and `query: Optional[str] = None`. Use `Optional` for fields that are not always required.
    -   **Reference Code (`app/models.py`)**:
        ```python
        from pydantic import BaseModel
        from typing import Optional

        class ComplianceAction(BaseModel):
            action_type: str
            query: Optional[str] = None
            message: Optional[str] = None
            decision: Optional[str] = None
            reason: Optional[str] = None
        ```

5.  **Add placeholder API endpoints in `app/main.py`**: `/reset`, `/step`, `/state`, `/tasks`, `/grader`, `/baseline`.
    -   **Why it's important**: You'll create dummy functions for each API endpoint. They won't do anything yet, but they allow the server to run without crashing. This is crucial for Vedika, as it unblocks her from writing initial "smoke tests" to confirm the server is reachable.
    -   **How to do it**:
        1.  In `app/main.py`, import `FastAPI` and your Pydantic models.
        2.  Initialize the app: `app = FastAPI()`.
        3.  For each endpoint, create a function with a FastAPI decorator, like `@app.post("/step")`.
        4.  The function should accept the corresponding Pydantic model as an argument (e.g., `async def step(action: ComplianceAction):`).
        5.  For now, just return a simple dictionary: `return {"status": "ok", "action_received": action.action_type}`.
    -   **Reference Code (`app/main.py`)**:
        ```python
        from fastapi import FastAPI
        from .models import ComplianceAction

        app = FastAPI()

        @app.post("/step")
        async def step(action: ComplianceAction):
            # Placeholder logic
            return {"status": "ok", "action_received": action.action_type}
        ```

6.  **Share a sample request/response JSON in team chat** so everyone uses the same format.
    -   **Why it's important**: By sharing concrete examples of what the API expects as input and what it will send as output, you provide a clear guide for your teammates and remove any ambiguity.
    -   **How to do it**: Based on your Pydantic models, write out a JSON object in a text file. For a `ResolveTicket` action, it would look like: `{"action_type": "ResolveTicket", "decision": "Approve", "reason": "All documents are in order."}`. Post this in the team chat.
    -   **Reference Code (Sample JSON for a `curl` command)**:
        ```json
        {
            "action_type": "SearchPolicy",
            "query": "entertainment policy"
        }
        ```

### Dependency Notes
- No prerequisites for starting Day 1 tasks.
- Sanya needs your schema confirmation before finalizing dataset structure.
- Vedika needs your running server stub before writing API smoke tests.

### Required Handoffs
- To Sanya (by 14:00): final schema field names and accepted values.
- To Vedika (by EOD): working local server command (`uvicorn app.main:app --reload`) and endpoint payload examples.

## Day 2 - Core Environment Logic

### Main Goal
Implement environment transitions and reward behavior.

### Tasks
1.  **Build `ComplianceEnv` in `app/environment.py`** with `reset()`, `step()`, `state()`.
    -   **Why it's important**: This is the heart of the simulation. The `reset` method will start a new problem (episode), and the `step` method will contain the logic to process an incoming agent action, calculate the reward, and produce the next observation for the agent.
    -   **How to do it**:
        1.  Create a `ComplianceEnv` class.
        2.  The `__init__` method can load the policy and claims data from the `data/` directory.
        3.  The `reset(task_id)` method should select a claim from the dataset, set the initial state (like `step_count = 1`), and return the first observation.
        4.  The `step(action)` method will contain the core logic for how the environment changes in response to an agent's action.
    -   **Reference Code (`app/environment.py`)**:
        ```python
        class ComplianceEnv:
            def __init__(self):
                # Load data/policy.md and data/claims.json here
                self.claims = [...]
                self.policy = "..."
                self.current_claim = None
                self.steps_taken = 0

            def reset(self, task_id: str):
                self.current_claim = self.claims[0] # Placeholder
                self.steps_taken = 0
                # Return first observation
                return {"ticket_id": self.current_claim["id"], "env_message": "New ticket."}

            def step(self, action):
                # Logic will go here
                observation = {}
                reward = 0.0
                done = False
                info = {}
                return observation, reward, done, info
        ```

2.  **Add action validation logic and penalties** for malformed actions.
    -   **Why it's important**: This is where you implement the rules of the game. You'll write code that checks if an agent's action is valid (e.g., does a `SearchPolicy` action include a `query`?). This provides helpful feedback if the agent makes a mistake.
    -   **How to do it**: Inside the `step` method, use `if/elif/else` statements to check the `action.action_type`. For example: `if action.action_type == "SearchPolicy" and not action.query: return observation, -0.1, False, {"message": "SearchPolicy requires a query."}`.
    -   **Reference Code (`step` method part)**:
        ```python
        if action.action_type == "SearchPolicy" and not action.query:
            reward = -0.1
            info = {"message": "SearchPolicy action requires a 'query' field."}
            return self.current_observation, reward, False, info
        ```

3.  **Implement reward logic and episode termination conditions**.
    -   **Why it's important**: You will implement the reward function, giving positive points for correct actions and negative penalties for mistakes. This feedback signal is essential for training an effective AI agent. Termination logic defines when an episode is over.
    -   **How to do it**: After validating the action, determine the reward. For a `ResolveTicket` action, compare `action.decision` to the `ground_truth_decision` from the claim data. If they match, the reward is `+1.0` and `done` is `True`. If they don't, reward is `-1.0` and `done` is `True`.
    -   **Reference Code (`step` method part)**:
        ```python
        if action.action_type == "ResolveTicket":
            is_correct = (action.decision == self.current_claim["ground_truth_decision"])
            reward = 1.0 if is_correct else -1.0
            done = True
            info = {"message": "Episode finished."}
            return self.current_observation, reward, done, info
        ```

4.  **Integrate env methods into FastAPI endpoints**.
    -   **Why it's important**: You will connect your `ComplianceEnv` methods to the placeholder API endpoints you created on Day 1. Now, when a request hits `/step`, it will actually call your `env.step()` method and return a real result.
    -   **How to do it**:
        1.  In `app/main.py`, create a single instance of your environment: `env = ComplianceEnv()`.
        2.  In your endpoint functions (e.g., `async def step(action: ComplianceAction):`), call the corresponding environment method: `obs, reward, done, info = env.step(action)`.
        3.  Return the results in a JSON response.
    -   **Reference Code (`app/main.py`)**:
        ```python
        # ... imports
        from .environment import ComplianceEnv

        app = FastAPI()
        env = ComplianceEnv() # Create one instance

        @app.post("/step")
        async def step(action: ComplianceAction):
            obs, reward, done, info = env.step(action)
            return {"observation": obs, "reward": reward, "done": done, "info": info}
        ```

5.  **Add minimal logs/messages** to make debugging easy for beginners.
    -   **Why it's important**: Printing out the current step, action taken, and reward received helps everyone see what the environment is doing. This makes it much easier to find and fix bugs.
    -   **How to do it**: Use simple `print()` statements inside your `step` method. For example: `print(f"Step {env.step_count}: Action={action.action_type}, Reward={reward}")`.
    -   **Reference Code (`step` method part)**:
        ```python
        # ... at the end of the step method
        print(f"Step {self.steps_taken}: Action={action.action_type}, Reward={reward}")
        return observation, reward, done, info
        ```

### Dependency Notes
- Start after receiving from Sanya:
  - final `policy.md` wording,
  - stable claim field mapping,
  - key edge-case list.
- Start after receiving from Vedika:
  - smoke-test feedback from Day 1 endpoint behavior.

### Required Handoffs
- To Sanya: confirm any schema updates before she regenerates full dataset.
- To Vedika: share final expected behavior for invalid actions and max-step termination.

## Day 3 - Integration and Final Polish

### Main Goal
Complete integration and ensure the project can be validated end-to-end.

### Tasks
1.  **Finalize `app/graders.py` wiring** into `/grader` endpoint.
    -   **Why it's important**: You will integrate the final grading logic into the `/grader` endpoint so it can correctly score a completed episode based on the ground truth from Sanya's dataset.
    -   **How to do it**:
        1.  The `/grader` endpoint will receive the full episode history (actions, rewards).
        2.  Implement logic in `graders.py` that takes this history and the original `task_id` to calculate a final score from 0.0 to 1.0 based on the rules in the main `README.md`.
        3.  For example, for the 'medium' task, the grader must check if a `SearchPolicy` action was taken *and* if the final decision was correct.
    -   **Reference Code (`app/graders.py`)**:
        ```python
        def grade_episode(task_id: str, history: list, ground_truth_decision: str) -> float:
            if task_id == "medium":
                searched = any(action["action_type"] == "SearchPolicy" for action in history)
                final_action = history[-1]
                correct_decision = (final_action["decision"] == ground_truth_decision)
                if searched and correct_decision:
                    return 1.0
                # ... other logic
            return 0.0
        ```

2.  **Add/clean `app/baseline.py` integration** with `/baseline` endpoint.
    -   **Why it's important**: This ensures the simple baseline agent can run correctly against your live server, providing a benchmark score for the project.
    -   **How to do it**: The `baseline.py` script will use a library like `requests` or `httpx` to make POST calls to your running server's endpoints (`/reset`, `/step`). It will simulate a full episode and then call `/grader` to get a score.
    -   **Reference Code (`app/baseline.py`)**:
        ```python
        import requests

        API_URL = "http://localhost:7860"

        # Reset the environment for a medium task
        response = requests.post(f"{API_URL}/reset", json={"task_id": "medium"})
        obs = response.json()

        # Take a step
        action = {"action_type": "SearchPolicy", "query": "flights"}
        response = requests.post(f"{API_URL}/step", json=action)
        # ... continue until done
        ```

3.  **Verify all task IDs** (`easy`, `medium`, `hard`) match `openenv.yaml`.
    -   **Why it's important**: This is a final check to ensure the project's metadata file accurately reflects the live application. This is critical for the project to be compliant with the OpenEnv specification.
    -   **How to do it**: Manually open the `openenv.yaml` file and look at the `tasks` list. Compare the `id` of each task with the strings your `/reset` endpoint accepts (e.g., "easy", "medium", "hard"). They must match exactly.
    -   **Reference Code (`openenv.yaml`)**:
        ```yaml
        tasks:
          - id: easy # This must match the string used in your /reset endpoint
            name: single_step_classification
            ...
          - id: medium # This too
            ...
        ```

4.  **Fix issues reported by Vedika QA/testing**.
    -   **Why it's important**: Vedika will have a list of bugs found during her testing. Your job is to work through this list, fix the problems in the backend code, and confirm with her that the fixes work.
    -   **How to do it**: Follow a "reproduce, fix, verify" cycle. First, use Vedika's bug report to reproduce the error on your machine. Then, fix the code in your editor. Finally, re-run the failing test or manual step to confirm the bug is gone.
    -   **Reference Code (A hypothetical fix)**:
        ```python
        # Before fix in environment.py
        # if action.decision == self.current_claim["ground_truth_decision"]:
        #     reward = 0.5 # Bug! Should be 1.0

        # After fix
        if action.decision == self.current_claim["ground_truth_decision"]:
            reward = 1.0 # Correct reward
        ```

5.  **Run final sanity check** with local server and sample episodes.
    -   **Why it's important**: Before handing off, you'll do one last check to make sure the main flows of the application are working as expected.
    -   **How to do it**: Use a tool like `curl` from your terminal or write a very simple Python script to call `/reset` and then `/step` a few times for each task difficulty. Ensure you get valid responses and no crashes.
    -   **Reference Code (Terminal command)**:
        ```bash
        curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id": "easy"}'
        ```

### Dependency Notes
- Start Day 3 after Sanya shares final dataset and edge-case confirmations.
- Start final fixes after Vedika shares failing tests or validation findings.

### End-of-Day Final Handoff
1. Confirm all endpoints run without crashes.
2. Share final integration summary in team chat with any known limitations.
3. Mark unresolved risks clearly so teammates can communicate them in final demo.
