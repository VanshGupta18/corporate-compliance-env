# Vedika - 3 Day Plan (Beginner Friendly)

## Role Focus
Testing, validation, QA feedback, and release readiness.

## Shared Communication Rules (Follow Every Day)
1. Post daily standup update at 09:30 in the team chat using:
   `[Day X][Vedika] Plan: ... | Blockers: ... | Need from: ...`
2. Post dependency check-in at 14:00:
   `Ready for handoff / Waiting for <name> on <item>`
3. Post end-of-day handoff at 18:00 with links to changed files and pending items.
4. If blocked for more than 30 minutes, ping both teammates immediately.

## Day 1 - Testing Setup and API Smoke Checks

### Main Goal
Set up a beginner-friendly testing workflow and validate initial API behavior.

### Tasks
1.  **Set up `pytest` structure and basic test naming conventions** in `tests/`.
    -   **Why it's important**: You will configure the testing framework and establish simple conventions for naming test files and functions. This keeps the test suite organized and makes it easy for anyone on the team to run and understand the tests.
    -   **How to do it**: `pytest` works by automatically discovering test files and functions. Simply create a file in the `tests/` directory named `test_graders.py`. Inside, create functions that start with `test_`, like `def test_easy_grader_approve():`. `pytest` will find and run them automatically.
    -   **Reference Code (`tests/test_graders.py`)**:
        ```python
        import pytest

        def test_example_passing():
            assert 1 + 1 == 2

        def test_example_failing():
            assert 1 + 1 == 3
        ```

2.  **Create smoke tests for endpoint availability** (`/tasks`, `/reset`, `/state`).
    -   **Why it's important**: These are basic, preliminary tests to check for fundamental problems. You will write tests to confirm that the API endpoints exist and don't crash when called. This provides rapid feedback to Vansh that his initial server setup is working.
    -   **How to do it**:
        1.  In your test file, you will need a client to make web requests. The `requests` library is good for this.
        2.  Write a test function that makes a `GET` request to Vansh's running server (e.g., `http://localhost:7860/tasks`).
        3.  Use an `assert` statement to check that the response status code is 200 (OK). Example: `assert response.status_code == 200`.
    -   **Reference Code (`tests/test_api.py`)**:
        ```python
        import requests

        API_URL = "http://localhost:7860"

        def test_tasks_endpoint():
            response = requests.get(f"{API_URL}/tasks")
            assert response.status_code == 200
            assert "tasks" in response.json()
        ```

3.  **Document how to run tests in one command**.
    -   **Why it's important**: Making the tests easy to run encourages everyone to use them. A single, simple command removes friction and helps the team catch bugs faster.
    -   **How to do it**: Add a section to the main project `README.md` called "Run Tests". In it, add the single command: `pytest -v`. The `-v` flag stands for "verbose" and gives more detailed output.
    -   **Reference Code (in main `README.md`)**:
        ```markdown
        ### 5. Run Tests

        ```bash
        pytest -v
        ```
        ```

4.  **Build a shared issue list template for bugs**: endpoint, input, actual output, expected output.
    -   **Why it's important**: A standardized template for reporting bugs makes communication with Vansh and Sanya efficient. It ensures they have all the information they need to understand and fix the problem quickly.
    -   **How to do it**: Create a simple text file or a shared document (like a Google Doc) with a template. Post the template in the team chat:
        `**Bug Report**`
        `- **Endpoint**: (e.g., /step)`
        `- **Input Sent**: (Paste the JSON here)`
        `- **Expected Output**: (Describe what should have happened)`
        `- **Actual Output**: (Describe what actually happened)`
    -   **Reference Code**: N/A (This is a communication/documentation task).

5.  **Review Sanya's edge-case list and map each case to future tests**.
    -   **Why it's important**: You will take the list of tricky cases from Sanya and start planning how you will write specific tests for them on Day 2. This preparation ensures that your tests will be comprehensive and meaningful.
    -   **How to do it**: For each edge case Sanya provides, write a placeholder test function in your `test_graders.py` file. The function can be empty for now, just with a descriptive name. Example: `def test_rejects_claim_just_over_receipt_limit(): pass`.
    -   **Reference Code (`tests/test_graders.py`)**:
        ```python
        # ... other tests

        def test_rejects_claim_just_over_receipt_limit():
            # TODO: Implement this test on Day 2
            pass

        def test_approves_claim_just_under_receipt_limit():
            # TODO: Implement this test on Day 2
            pass
        ```

### Dependency Notes
- Wait for Vansh to provide running API server + sample payloads before endpoint smoke tests.
- Wait for Sanya edge-case handoff before building strong grader test plan.

### Required Handoffs
- To Vansh: endpoint failures or ambiguous behavior with reproducible test inputs.
- To Sanya: unclear/ambiguous data cases found during smoke-test review.

## Day 2 - Grader and Behavior Testing

### Main Goal
Write meaningful tests for decision grading and episode behavior.

### Tasks
1.  **Implement `tests/test_graders.py`** for easy/medium/hard logic.
    -   **Why it's important**: You will write specific tests that check if the grading logic is correct for all three task types. For a given episode, you'll check if the final score matches the expected score based on Sanya's ground truth data.
    -   **How to do it**: For each placeholder test you wrote, you will now fill in the logic. Your test will need to simulate a full episode by calling `/reset` and then `/step` one or more times. Finally, it will call `/grader` with the episode history and `assert` that the returned score is what you expect (e.g., `assert score == 1.0`).
    -   **Reference Code (`tests/test_graders.py`)**:
        ```python
        def test_easy_task_correct_rejection():
            # 1. Reset env to get an easy task that should be rejected
            # 2. Call /step with Resolve(decision="Reject")
            # 3. Call /grader with the history
            # 4. Assert that the score is 1.0
            pass # Placeholder for full implementation
        ```

2.  **Add tests for reward edge cases** (invalid action penalty, max-step penalty, wrong resolve).
    -   **Why it's important**: You will write tests for specific behaviors, such as ensuring the correct negative reward is applied for an invalid action or that the episode correctly terminates if the agent exceeds the maximum number of steps. This validates the core mechanics of the environment.
    -   **How to do it**: Write a test where you intentionally send a malformed action to the `/step` endpoint (e.g., a `SearchPolicy` action with no `query`). Then, `assert` that the reward in the response is `-0.1`.
    -   **Reference Code (`tests/test_rewards.py`)**:
        ```python
        def test_penalty_for_invalid_action():
            requests.post(f"{API_URL}/reset", json={"task_id": "easy"})
            
            # Send an action that is missing a required field
            invalid_action = {"action_type": "SearchPolicy"} # Missing 'query'
            response = requests.post(f"{API_URL}/step", json=invalid_action)
            
            assert response.json()["reward"] == -0.1
        ```

3.  **Add tests for dependency flow behavior** (requesting missing docs, policy search relevance).
    -   **Why it's important**: These tests check the more complex, multi-step interactions. For example, does the agent get rewarded for correctly asking for a missing document? This ensures the learning signals are correct.
    -   **How to do it**: For a 'hard' task claim that is missing a document, your test should first call `/step` with a `RequestInformation` action. You would then `assert` that the reward returned is `+0.1`.
    -   **Reference Code (`tests/test_flows.py`)**:
        ```python
        def test_reward_for_correctly_requesting_info():
            # 1. Reset env to get a hard task missing a document
            # 2. Call /step with a correct RequestInformation action
            # 3. Assert that the reward in the response is 0.1
            pass # Placeholder for full implementation
        ```

4.  **Track bugs in a simple table and assign owner** (Vansh or Sanya).
    -   **Why it's important**: You will use your template to log all failures in a simple, shared table. Assigning an owner (Vansh for backend bugs, Sanya for data bugs) ensures clear accountability and makes it easy to track progress on fixes.
    -   **How to do it**: Create a shared spreadsheet (Google Sheets) or a simple Markdown table in a file. The columns should be: `Bug ID`, `Description`, `Reported By`, `Assigned To`, `Status (Open/Closed)`. When a test fails, fill out a new row.
    -   **Reference Code (Markdown Table)**:
        ```markdown
        | Bug ID | Description | Assigned To | Status |
        |---|---|---|---|
        | 001 | `/grader` returns 0.5 instead of 1.0 for easy task | Vansh | Open |
        ```

5.  **Re-run full tests after each fix and post status in chat**.
    -   **Why it's important**: This confirms that a bug fix actually worked and didn't accidentally break something else (a "regression"). Constant feedback keeps the project healthy.
    -   **How to do it**: After Vansh or Sanya tells you they've fixed a bug, run the `pytest -v` command again. If the previously failing test now passes, update its status to "Closed" in your bug tracker and notify the team.
    -   **Reference Code**: N/A (This is a process task).

### Dependency Notes
- Start after Vansh has implemented core env and graders.
- Start after Sanya delivers final edge-case IDs + expected outcomes.

### Required Handoffs
- To Vansh: failing test logs with exact inputs and expected behavior.
- To Sanya: label mismatches or policy ambiguity found from tests.

## Day 3 - End-to-End Validation and Release Checks

### Main Goal
Ensure the whole environment is stable and demo-ready.

### Tasks
1.  **Run full test suite and summarize pass/fail** with blocker severity.
    -   **Why it's important**: You will execute every single test and provide a final summary report to the team, highlighting any remaining critical bugs that must be fixed before the demo.
    -   **How to do it**: Run `pytest -v` one last time. Post a summary in the team chat: "Final test run complete. 48/50 tests passing. 2 failing tests are low-severity. No blockers for demo."
    -   **Reference Code**: N/A (This is a communication task).

2.  **Validate OpenEnv behavior manually** across all tasks (`easy`, `medium`, `hard`).
    -   **Why it's important**: You will step through a few episodes yourself, acting as the agent, to get a feel for the user experience and catch any issues that might not be covered by automated tests.
    -   **How to do it**: Use a tool like `curl` or a simple Python script to manually call the API. For an easy task, call `/reset`, then `/step` with a `ResolveTicket` action. Check that the response makes sense. Do the same for a medium and hard task, sending the appropriate sequence of actions.
    -   **Reference Code (Manual `curl` check)**:
        ```bash
        # Reset to start an easy task
        curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id": "easy"}'
        
        # Take a step to resolve it
        curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"action_type": "ResolveTicket", "decision": "Approve", "reason": "Looks good"}'
        ```

3.  **Check Docker run flow and endpoint availability** after container start.
    -   **Why it's important**: You will verify that the project can be built and run using Docker. This is a key release check to ensure the project is portable and can be run by others easily.
    -   **How to do it**:
        1.  Run `docker build -t compliance-env .` from the terminal.
        2.  Run `docker run -p 7860:7860 compliance-env`.
        3.  In a new terminal, run a smoke test like `curl http://localhost:7860/tasks` to ensure the server inside the container is running and accessible.
    -   **Reference Code (Terminal commands)**:
        ```bash
        docker build -t compliance-env .
        docker run -d --rm -p 7860:7860 --name my-compliance-app compliance-env
        curl http://localhost:7860/tasks
        docker stop my-compliance-app
        ```

4.  **Verify README commands work in order for a beginner**.
    -   **Why it's important**: You will personally follow the `README.md` instructions from start to finish to ensure they are correct, complete, and easy for a newcomer to follow. This is a critical part of making the project accessible.
    -   **How to do it**: Open a brand new terminal window to ensure you have a clean environment. Copy and paste every single command from the main `README.md`'s "Quickstart" section, in order. If any command fails or produces an unexpected result, the README needs to be fixed.
    -   **Reference Code**: N/A (This is a manual validation process).

5.  **Publish final QA summary**: passed checks, known issues, risk level.
    -   **Why it's important**: You will give the final "go/no-go" decision. Your summary will state what has passed, what known issues still exist, and whether you assess the project as stable enough for the demo. This is the final quality gate.
    -   **How to do it**: Post a final, formal message in the team chat. "QA Sign-off: All critical functionality is verified. The `pytest` suite passes, Docker build is successful, and README commands are validated. We have 2 known low-priority bugs. The project is stable and ready for demo."
    -   **Reference Code**: N/A (This is a communication task).

### Dependency Notes
- Begin final validation only after Vansh and Sanya mark integration/data as frozen.
- If final regressions appear, coordinate quick fix loop with both teammates before closure.

### End-of-Day Final Handoff
1. Send final QA sign-off note in chat.
2. Tag unresolved issues as `must-fix` or `post-demo`.
3. Confirm final command list for demo execution order.
