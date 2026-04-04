# Sanya - 3 Day Plan (Beginner Friendly)

## Role Focus
Policy and dataset pipeline, edge-case quality, and grading support data.

## Shared Communication Rules (Follow Every Day)
1. Post daily standup update at 09:30 in the team chat using:
   `[Day X][Sanya] Plan: ... | Blockers: ... | Need from: ...`
2. Post dependency check-in at 14:00:
   `Ready for handoff / Waiting for <name> on <item>`
3. Post end-of-day handoff at 18:00 with links to changed files and pending items.
4. If blocked for more than 30 minutes, ping both teammates immediately.

## Day 1 - Policy and Data Blueprint

### Main Goal
Prepare policy text and a reliable claim schema draft.

### Tasks
1.  **Draft all 15 policy rules in `data/policy.md`** in simple, unambiguous language.
    -   **Why it's important**: This document is the "rulebook" the AI agent will eventually learn to read and interpret. Writing the rules with clarity is key to avoiding ambiguity that could confuse the agent or the grading logic.
    -   **How to do it**: Open the `data/policy.md` file in your editor. Write each of the 15 rules from the main project `README.md` as a numbered list using Markdown (e.g., `1. Meals under ₹500 — no receipt required.`).
    -   **Reference Code (`data/policy.md`)**:
        ```markdown
        1. Meals under ₹500 — no receipt required.
        2. Meals ₹500–₹2,000 — receipt required.
        3. ...and so on for all 15 rules.
        ```

2.  **Create a starter `data/claims.json`** with 10-15 sample claims covering easy/medium/hard types.
    -   **Why it's important**: This small, representative sample helps Vansh and Vedika understand the data format in a practical way. It allows them to start their work without waiting for the full dataset.
    -   **How to do it**: Open `data/claims.json`. Start by creating an empty JSON array `[ ]`. Inside the array, manually write 10-15 JSON objects `{ }`, where each object is a single claim. Make sure you create a mix of `easy`, `medium`, and `hard` examples.
    -   **Reference Code (`data/claims.json`)**:
        ```json
        [
          {
            "id": "EXP-001",
            "employee_name": "Ankit Verma",
            "employee_role": "Junior Engineer",
            "employee_level": "L3",
            "description": "Taxi ride at 2:00 PM without manager note",
            "amount": 800,
            "currency": "INR",
            "has_receipt": true,
            "missing_document": "manager_approval",
            "rule_keyword": "daytime cab",
            "risk_score": 0.65,
            "ground_truth_decision": "Reject",
            "ground_truth_reason": "Daytime cab requires manager approval per policy rule 7"
          }
        ]
        ```

3.  **Include required fields for each claim** (`id`, role, level, amount, missing docs, keyword, ground truth decision/reason).
    -   **Why it's important**: You are deciding the exact structure for every piece of training data. This structure must be reviewed with Vansh to ensure it aligns perfectly with his Pydantic schemas (the data "contract").
    -   **How to do it**: For each JSON object you created in `claims.json`, add the necessary key-value pairs. For example: `"id": "EXP-001"`, `"employee_role": "Junior Engineer"`, `"amount": 800`, etc. Ensure every object has all the fields.
    -   **Reference Code**: The JSON object in the task above serves as the reference.

4.  **Review sample claim format with Vansh** to ensure schema alignment.
    -   **Why it's important**: This quick check prevents major rework later. By confirming your data structure matches Vansh's backend schemas early, you ensure both parts of the project will fit together perfectly.
    -   **How to do it**: Copy one complete JSON claim object from your `claims.json` file, paste it into the team chat, and ask Vansh: "Does this structure match the Pydantic models and observation fields used in `app/models.py` and `app/server/environment.py`?"
    -   **Reference Code (Team chat message)**:
        ```
        @Vansh Here is a sample claim object. Can you confirm this matches your Pydantic schema for the observation space?
        ```
        ```json
        {
          "id": "EXP-001",
          "employee_name": "Ankit Verma",
          ...
        }
        ```

5.  **Share 5 tricky edge cases with Vedika** so tests can be designed early.
    -   **Why it's important**: Identifying tricky scenarios (e.g., a claim amount that is just one rupee over a limit) is vital for building a robust system. Sharing these with Vedika allows her to design powerful tests that check if the system behaves correctly in non-obvious situations.
    -   **How to do it**: Look at the "Edge Cases" table in the main `README.md`. Pick 5 of them, and for each one, write down the scenario and the expected `ground_truth_decision` in the team chat for Vedika.
    -   **Reference Code (Team chat message)**:
        ```
        @Vedika Here are 5 edge cases for test planning:
        1. Scenario: Meal just under limit (₹1,999). Expected Decision: Approve.
        2. Scenario: Meal just over limit (₹2,001). Expected Decision: Reject.
        3. ...and so on.
        ```

### Dependency Notes
- You can start policy writing immediately.
- Before finalizing claim JSON structure, wait for Vansh schema confirmation from Day 1 (`app/models.py` + `app/server/environment.py`).
- Wait for Vansh to confirm that exported imports in `app/__init__.py` are stable, so field names are locked for client/framework use.
- Vedika needs your edge-case list before she can design good grader tests.

### Required Handoffs
- To Vansh (by 14:00): final policy terminology and rule keyword list.
- To Vansh (by EOD): confirmation that dataset fields match what `app/client.py` and `/reset`/`/step` flows consume.
- To Vedika (by EOD): sample claims + expected outcomes for tricky cases.

## Day 2 - Dataset Generation and Validation

### Main Goal
Generate full synthetic dataset and verify quality.

### Tasks
1.  **Implement `data/generate_dataset.py`** with `--count`, `--seed`, `--output` options.
    -   **Why it's important**: You will write a script that can automatically generate a large number of synthetic claims. Making this a script ensures the process is repeatable and can be easily modified if the data requirements change. The `--seed` option is crucial for reproducibility.
    -   **How to do it**:
        1.  In `data/generate_dataset.py`, import Python's `json`, `random`, and `argparse` libraries.
        2.  Create templates or lists of possible values for employee roles, descriptions, amounts, etc.
        3.  Write a function that randomly combines these templates to generate a single claim object.
        4.  Use a `for` loop to call this function `count` times and append the results to a list.
        5.  Finally, use `json.dump()` to write the list of claims to the output file.
    -   **Reference Code (`data/generate_dataset.py`)**:
        ```python
        import json
        import random

        def generate_claim():
            # Logic to create one random claim dictionary
            return {"id": f"EXP-{random.randint(100, 999)}", ...}

        claims = [generate_claim() for _ in range(100)]

        with open("claims.json", "w") as f:
            json.dump(claims, f, indent=2)
        ```

2.  **Generate 100 claims and verify split intent** (easy/medium/hard distribution).
    -   **Why it's important**: You will run your script to create the full dataset. Afterwards, you must review it to ensure the distribution of easy, medium, and hard tasks is correct and that the edge cases you designed are present.
    -   **How to do it**: Run your script from the terminal: `python data/generate_dataset.py --count 100 --seed 42`. Then, open the generated `claims.json` file and visually scan through it to get a sense of the variety. You can write a small helper script to count how many claims have a `missing_document` (hard) or a hidden `rule_keyword` (medium) to verify the distribution.
    -   **Reference Code (Verification script)**:
        ```python
        import json
        with open('claims.json', 'r') as f:
            data = json.load(f)
        
        hard_tasks = [c for c in data if c.get('missing_document')]
        print(f"Found {len(hard_tasks)} hard tasks.")
        ```

3.  **Ensure edge cases are explicitly present** (threshold boundaries, duplicate claims, senior-level escalation).
    -   **Why it's important**: A good dataset includes tricky examples. Explicitly checking for these ensures the agent is trained on more than just the "easy" cases, making it more robust.
    -   **How to do it**: Modify your `generate_dataset.py` script to have specific functions that create these edge cases. For example, a function `create_threshold_claim()` could generate a claim with an amount that is exactly `1999` or `2001`. Ensure these special functions are called a few times during generation.
    -   **Reference Code (`data/generate_dataset.py`)**:
        ```python
        def create_threshold_claim():
            return {"amount": 1999, "description": "Meal just under limit", ...}

        # In main generation logic:
        all_claims.append(create_threshold_claim())
        ```

4.  **Validate every claim has correct ground truth decision and reason**.
    -   **Why it's important**: This is a critical quality check. You must ensure that for every single claim, the `ground_truth_decision` is the correct outcome according to the `policy.md` you wrote. A mistake here will confuse the agent and lead to incorrect grading.
    -   **How to do it**: This is a careful, manual, or semi-automated process. Read through a large sample of the generated claims. For each one, read the `description` and check the `policy.md` file to determine the correct outcome. Then, verify that your determination matches the `ground_truth_decision` in the JSON file.
    -   **Reference Code (Manual check process)**:
        1.  Open `claims.json` and `policy.md` side-by-side.
        2.  Read Claim `EXP-101`: "Client dinner for 2500".
        3.  Find Rule #3 in `policy.md`: "Meals above ₹2,000 — receipt + manager approval".
        4.  Check `EXP-101` in `claims.json`. If it's missing manager approval, its `ground_truth_decision` must be "Reject".

5.  **Share a short data quality report in chat** (counts, checks run, known assumptions).
    -   **Why it's important**: This keeps the team informed about the state of the data and builds confidence in the dataset's quality.
    -   **How to do it**: Post a message in the team chat like: "Data quality check complete. Generated 100 claims (40 easy, 35 medium, 25 hard). All edge cases from the README are present. Manually validated 20% of claims for ground truth accuracy."
    -   **Reference Code**: N/A (This is a communication task).

### Dependency Notes
- Start after Vansh confirms no further schema changes in `app/models.py` and `app/server/environment.py`.
- Start deeper validation after Vedika points out gaps found in early tests.

### Required Handoffs
- To Vansh: final `claims.json` format confirmation before backend lock.
- To Vedika: edge-case IDs and expected grader outcomes for unit tests.

## Day 3 - Evaluation Support and Final Consistency

### Main Goal
Support final scoring reliability and documentation consistency.

### Tasks
1.  **Cross-check all claim `ground_truth_decision` values** against policy rules.
    -   **Why it's important**: This is a final, thorough review of the dataset against the policy rules and the final grader logic that Vansh has implemented. It's the last chance to catch data errors.
    -   **How to do it**: This is your most thorough validation pass. Read through as many claims as possible, especially the complex ones, and double-check that the ground truth is correct based on the final policy document.
    -   **Reference Code**: N/A (This is a manual review process).

2.  **Validate random sample tickets against grader expectations with Vedika**.
    -   **Why it's important**: You will work with Vedika to investigate any discrepancies between the test results and the dataset labels. This collaborative debugging is key to resolving complex bugs where it's unclear if the code is wrong or the data is wrong.
    -   **How to do it**: Sit with Vedika (or share your screen). She will run a test for a specific claim ID. If the test fails, you will both look at the claim data in `claims.json`, the logic in `graders.py`, and the rules in `policy.md` to find the source of the disagreement.
    -   **Reference Code**: N/A (This is a collaborative debugging task).

3.  **Confirm policy-rule numbering and wording are consistent with README examples**.
    -   **Why it's important**: This ensures the project documentation is accurate and that the examples shown in the main project `README.md` are not misleading.
    -   **How to do it**: Open the main `README.md` and your `data/policy.md` side-by-side. Check that the rule numbers and descriptions mentioned in the README's examples match your policy file exactly.
    -   **Reference Code**: N/A (This is a manual review process).

4.  **Help Vansh investigate any data-related failures** from `/grader` or `/baseline`.
    -   **Why it's important**: If a test fails, it could be a bug in the code or an error in the data. Your role is to help determine if the data is the source of the problem.
    -   **How to do it**: When Vansh reports a failure on a specific claim ID, your first step is to find that claim in `claims.json` and verify its ground truth. If the ground truth is correct, the problem is likely in the code. If the ground truth is wrong, you need to fix it.
    -   **Reference Code**: N/A (This is a debugging task).

5.  **Freeze dataset files for final submission**.
    -   **Why it's important**: You will announce a "freeze," meaning no more changes will be made to the data files. This provides stability for Vansh and Vedika to complete their final integration and testing without worrying about the data changing underneath them.
    -   **How to do it**: Post a clear message in the team chat: "Dataset is now frozen. Please do not make any more edits to `data/policy.md` or `data/claims.json`." You can also make the files read-only if you are using a version control system like Git.
    -   **Reference Code (Git command - optional)**:
        ```bash
        # This is an advanced command to prevent local changes
        git update-index --assume-unchanged data/claims.json data/policy.md
        ```

### Dependency Notes
- Begin final consistency pass after Vansh publishes final reward/grader logic.
- Work jointly with Vedika when test outputs and dataset labels disagree.

### End-of-Day Final Handoff
1. Share final dataset stats and confidence level.
2. Report any known ambiguous policy cases still unresolved.
3. Confirm file freeze status so no one edits data accidentally afterward.
