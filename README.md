---
title: Corporate Compliance Environment
emoji: 📋
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
app_file: Dockerfile
tags:
  - openenv
  - fastapi
  - reinforcement-learning
  - finance
  - compliance
---


# 🏛️ Corporate Policy Compliance Environment

> An OpenEnv-compliant Reinforcement Learning environment that simulates
> how enterprise compliance officers audit employee expense claims and
> corporate action requests against internal policy documents.

[![OpenEnv Spec](https://img.shields.io/badge/OpenEnv-Compliant-blue)](https://openenv.dev)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Space-orange)](https://huggingface.co/spaces/mcqueemmater/env-corporate)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

🤗 **Live Space:** `https://huggingface.co/spaces/mcqueenmater/env-corporate`

---

## 📋 Overview

Every company in the world processes hundreds of expense reports,
approval requests, and compliance tickets every day. Today, this
requires human auditors who manually read policy documents and
make judgement calls. This environment trains an RL agent to do
exactly that — understand a request, retrieve the relevant policy
rule, and make a compliant decision.

This mirrors real production systems used at companies like Ramp,
Concur, and SAP — but is the **first open-source RL training
environment for this domain**. It is grounded in Indian corporate
compliance norms: ₹-denominated limits, GST receipt requirements,
WFH allowances, and local travel policies (auto-rickshaw, cab, metro).

---

## 🎯 Quick Reference: What The Agent Does

The agent plays the role of a **corporate compliance officer**. Each episode,
it receives one employee expense claim and must decide:

- ✅ **Approve** — claim follows all policy rules
- ❌ **Reject** — claim violates policy
- ⚠️ **Escalate** — claim requires senior review (L7+ employees)

The agent can also:
- 🔍 **SearchPolicy** — look up relevant rules before deciding
- 📋 **RequestInformation** — ask for missing documents

---

## 📋 The 15 Policy Rules (Quick Reference)

| # | Category | Rule |
|---|---|---|
| 1 | Meal | Under ₹500 → Approve, no receipt needed |
| 2 | Meal | ₹500–₹2,000 → receipt required |
| 3 | Meal | Over ₹2,000 → receipt + manager note required |
| 4 | Alcohol | Any alcohol on bill → Reject entire claim |
| 5 | Travel | Auto/metro under ₹500 → no receipt needed |
| 6 | Travel | Cab after 10 PM → pre-approved with receipt |
| 7 | Travel | Cab before 10 PM → manager note required |
| 8 | Flight | L1–L6 must fly economy → business class = Reject |
| 9 | Flight | L7+ may fly business class → Escalate for review |
| 10 | International | Over ₹50,000 → VP approval required |
| 11 | WFH | Internet + electricity capped at ₹1,000/month |
| 12 | GST | Claims over ₹5,000 → GST invoice required |
| 13 | Duplicate | Same amount + same date = auto Reject |
| 14 | Seniority | L7+ employees → always Escalate |
| 15 | Personal | Personal expenses → always Reject |

---

## 🏆 Baseline Performance

| Difficulty | Task | LLM Agent (Llama-3.1-8B) | Rule-Based Baseline |
|---|---|---|---|
| Easy | Single-step classification | ≥ 0.90* | 0.78 |
| Medium | Policy retrieval | ≥ 0.80* | 0.61 |
| Hard | Multi-turn contextual | ≥ 0.70* | 0.34 |

*LLM scores with step-aware prompting (deployed April 2026)

---

## 🚀 Quick Start

### Use the Live Space

Visit the running instance: **https://huggingface.co/spaces/mcqueemmater/env-corporate**

### Run Locally with Docker

```bash
# Clone and build
git clone https://huggingface.co/spaces/mcqueemmater/env-corporate
cd corporate-compliance-env
docker build -t compliance-env .
docker run -p 8000:8000 compliance-env

# Validate against OpenEnv spec
openenv validate --url http://localhost:8000 --verbose
```

### Run the LLM Inference Agent

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your_huggingface_token"

python inference.py
```

---

## 📡 API Endpoints (Quick Reference)

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Server health check |
| `/reset` | POST | Start a new episode • Body: `{"task_id": "easy\|medium\|hard"}` |
| `/step` | POST | Submit an action • Body: `ComplianceAction` JSON |
| `/state` | GET | Get current episode state |
| `/tasks` | GET | List all tasks + action schema |
| `/grader` | POST | Get final score for completed episode |
| `/baseline` | POST | Run baseline agent on all 3 tasks |
| `/docs` | GET | Swagger interactive API documentation |

---

## 👥 Team

| Name | Role |
|---|---|
| **Vansh Gupta** | Backend, Environment Design, Deployment, LLM Agent |
| **Sanya** | Dataset Generation, Policy Rules, Baseline Agent |
| **Vedika** | QA, Testing, Validation, Bug Fixes |

Built for the **Meta PyTorch OpenEnv Hackathon 2026**.

---

The agent acts as an AI Compliance Officer. At each step it receives
an open "Compliance Ticket" (an expense claim or request) and must:

1. **Understand** what the employee is claiming
2. **Search** the company policy rulebook if the relevant rule is unknown
3. **Request** missing documents from the simulated employee if needed
4. **Resolve** the ticket: `Approve`, `Reject`, or `Escalate`

The agent is not handed all information at once. It must earn it —
mirroring how a real compliance officer navigates incomplete files.

---

## 📦 Project Structure

```
corporate-policy-compliance-env/
│
├── app/
│   ├── __init__.py           # Exports for RL frameworks (env + client)
│   ├── models.py             # All Pydantic schemas
│   ├── client.py             # HTTPEnvClient subclass for remote usage
│   ├── server/
│   │   ├── environment.py    # ComplianceEnv class (reset/step/state)
│   │   └── app.py            # FastAPI app + HF web interface wrapper
│   ├── graders.py            # Deterministic grader for all 3 tasks
│   └── baseline.py           # Baseline inference script (OpenAI API)
│
├── data/
│   ├── policy.md             # 15-rule company policy document
│   ├── claims.json           # 100 synthetic expense claims + ground truth
│   └── generate_dataset.py   # Script to regenerate synthetic data
│
├── tests/
│   └── test_graders.py       # Unit tests for all 3 graders
│
├── openenv.yaml              # OpenEnv metadata file
├── Dockerfile                # Container spec (port 7860)
├── requirements.txt          # Pinned dependencies
└── README.md
```

---

## 🧠 Environment Design

### Action Space (`ComplianceAction`)

The agent takes one of exactly three action types per step:

| Action | Parameters | When to Use |
|--------|-----------|-------------|
| `SearchPolicy` | `query: str` | Policy rule is unknown — search the rulebook |
| `RequestInformation` | `message: str` | Document is missing from the ticket |
| `ResolveTicket` | `decision: str`, `reason: str` | Ready to make final call |

Valid `decision` values: `"Approve"`, `"Reject"`, `"Escalate"`

**Invalid action handling:** If the agent sends an unrecognised
`action_type` or missing required fields, the server returns
`HTTP 400` and applies a `-0.1` step penalty. The episode continues.

---

### Observation Space (`ComplianceObservation`)

At every step, the agent observes:

```json
{
  "ticket_id": "EXP-042",
  "employee_name": "Priya Sharma",
  "employee_role": "Junior Engineer",
  "employee_level": "L3",
  "amount": 5000.0,
  "currency": "INR",
  "description": "Client dinner including wine",
  "has_receipt": true,
  "missing_document": "manager_approval",
  "rule_keyword": "entertainment",
  "risk_score": 0.72,
  "env_message": "New ticket received. What is your action?",
  "step_count": 1,
  "max_steps": 8,
  "is_terminal": false
}
```

**Field glossary:**

| Field | Type | Description |
|-------|------|-------------|
| `missing_document` | `str \| null` | What document is absent (null if nothing missing) |
| `rule_keyword` | `str` | Hint for `SearchPolicy` query (hidden on medium/hard) |
| `risk_score` | `float 0–1` | Pre-computed risk signal based on amount + role |
| `env_message` | `str` | Latest message from the environment or simulated employee |
| `step_count` | `int` | Steps taken so far in this episode |

---

### State Schema (`ComplianceState`)

`GET /state` returns the full mid-episode state:

```json
{
  "current_observation": { "...ComplianceObservation fields..." },
  "episode_id": "ep-007",
  "task_id": "hard",
  "steps_taken": 3,
  "actions_history": [
    {"step": 1, "action_type": "SearchPolicy", "query": "entertainment policy"},
    {"step": 2, "action_type": "RequestInformation", "message": "Please share manager approval"}
  ],
  "rewards_history": [0.1, 0.1],
  "cumulative_reward": 0.2,
  "is_done": false
}
```

---

### Reward Function

Rewards are given at **every step** — not just at the end.
This provides a rich training signal over the full trajectory.

| Event | Reward | Notes |
|-------|--------|-------|
| Correct `ResolveTicket` | `+1.0` | Full credit for correct final decision |
| Relevant `SearchPolicy` | `+0.1` | Rule was genuinely unknown at that point |
| Correct `RequestInformation` | `+0.1` | Document was actually missing |
| Irrelevant `SearchPolicy` | `-0.05` | Rule was already visible in observation |
| Asking for info already in ticket | `-0.2` | Agent ignored visible context |
| Wrong `ResolveTicket` decision | `-1.0` | Fatal — episode ends immediately |
| Invalid action format | `-0.1` | Malformed action; episode continues |
| Exceeding max steps | `-0.5` | Penalise infinite loops |

All rewards are clamped to `[-1.0, 1.0]` as declared in `openenv.yaml`.

**Episode termination rules:**
- Episode ends immediately on `ResolveTicket` (correct or wrong)
- Episode ends if `step_count` exceeds `max_steps` for that task
- Wrong `ResolveTicket` ends the episode with `-1.0` reward
- All other wrong actions: episode continues, penalty applied

---

## 📊 Tasks

### 🟢 Task 1 — Single-Step Classification (Easy)

**Objective:** The ticket is fully self-contained. The relevant policy
rule is **provided directly in the observation**. Agent should
immediately call `ResolveTicket`.

- **Max Steps:** 3
- **Expected Steps:** 1
- **Grader logic:**

```
score = 1.0 if decision == ground_truth_decision else 0.0
```

- **Example:**
  > Ticket: *"Meal expense ₹800, no receipt attached."*
  > Policy shown: *"Receipts required for meals above ₹500."*
  > Correct action: `ResolveTicket(decision="Reject")`

---

### 🟡 Task 2 — Policy Retrieval (Medium)

**Objective:** Ticket is provided but the **policy rule is hidden**.
Agent must call `SearchPolicy` with the right keyword first,
then resolve.

- **Max Steps:** 5
- **Expected Steps:** 2
- **Grader logic:**

```
searched_policy = any action_type == "SearchPolicy" in episode
correct_decision = final_decision == ground_truth_decision

if correct_decision and searched_policy:     score = 1.0
elif correct_decision and not searched_policy: score = 0.5  # lucky guess
else:                                          score = 0.0
```

- **Example:**
  > Ticket: *"Business class flight Mumbai→Delhi, ₹45,000."*
  > Agent must search `"flight class policy"` to find:
  > *"Business class permitted only for VP (L7) and above."*
  > Employee role: Manager (L5) → Correct: `ResolveTicket(decision="Reject")`

---

### 🔴 Task 3 — Multi-Turn Contextual Decision (Hard)

**Objective:** Ticket has a **missing document**. Agent must identify
it, call `RequestInformation`, evaluate the returned document
against policy, and resolve — all while weighing employee seniority
as a risk factor.

- **Max Steps:** 8
- **Expected Steps:** 3–4
- **Grader logic (multi-component):**

```
component_scores = {
  "identified_missing_doc":  0.3,  # RequestInformation was correct
  "correct_info_request":    0.3,  # message asked for the right doc
  "correct_final_decision":  0.4   # ResolveTicket matches ground truth
}
score = sum of earned components  # 0.0 to 1.0
```

- **Example:**
  > Ticket: *"International travel ₹1,20,000 — no VP approval note."*
  > Agent asks: `RequestInformation("Please share VP approval for international travel")`
  > Environment returns: *"Approval mail from VP Rajesh Mehta attached."*
  > Agent verifies → `ResolveTicket(decision="Approve", reason="VP approval confirmed")`

---

## ⚠️ Edge Cases

The dataset includes deliberately tricky cases to test grader robustness:

| Scenario | Amount | Rule Threshold | Ground Truth | Why Tricky |
|----------|--------|---------------|-------------|------------|
| Meal just under limit | ₹1,999 | ₹2,000 receipt rule | Approve | One rupee under — no receipt needed |
| Meal just over limit | ₹2,001 | ₹2,000 receipt rule | Reject | One rupee over — receipt required |
| Auto-rickshaw, no receipt | ₹450 | ₹500 local travel threshold | Approve | Below threshold; mode allowed |
| Cab at 11 PM | ₹1,200 | Night travel policy | Approve | Late-night cab is explicitly allowed |
| WFH internet claim | ₹999 | ₹1,000/month WFH cap | Approve | Under cap — valid WFH expense |
| Alcohol in restaurant bill | ₹3,500 | Zero alcohol policy | Reject | Alcohol line item voids entire claim |
| VP submitting small claim | ₹500 | Any amount for L7+ | Escalate | High-seniority = always escalate |
| Duplicate claim same day | ₹2,200 | Anti-duplication rule | Reject | Same employee, same amount, same day |

---

## 🗂️ Dataset & Policy

### `data/policy.md` — 15 Company Rules

The agent's rulebook covers:

1. Meals under ₹500 — no receipt required
2. Meals ₹500–₹2,000 — receipt required
3. Meals above ₹2,000 — receipt + manager approval
4. Alcohol is never an approved expense category
5. Local travel (auto/metro) under ₹500 — no receipt needed
6. Cab rides after 10 PM — always approved with receipt
7. Daytime cab rides — require manager note
8. Domestic flights — economy class only for L1–L6
9. Business class — permitted for L7 (VP) and above only
10. International travel above ₹50,000 — VP approval mandatory
11. WFH internet/electricity allowance — max ₹1,000/month
12. Duplicate claims (same employee, amount, date) — auto-reject
13. Any claim from L7+ employee — escalate regardless of amount
14. GST receipt required for all claims above ₹5,000
15. Personal shopping, gifts, and entertainment without client present — reject

### `data/claims.json` — 100 Synthetic Claims

```json
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
```

**Distribution:** 33 Easy / 33 Medium / 34 Hard (100 total claims)
**Split:** 80 training / 20 held-out evaluation

### `data/generate_dataset.py`

Regenerate the full synthetic dataset at any time:

```bash
python data/generate_dataset.py --count 100 --seed 42
```

Parameters: `--count` (number of claims), `--seed` (reproducibility),
`--output` (output path). The script uses rule templates + random
sampling — no external API needed.

---

## ⚙️ API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Start new episode. Body: `{"task_id": "easy\|medium\|hard"}` |
| `POST` | `/step` | Take one action. Body: `ComplianceAction` JSON |
| `GET` | `/state` | Get current episode state |
| `GET` | `/tasks` | List all tasks + full action schema |
| `POST` | `/grader` | Get final score for completed episode |
| `POST` | `/baseline` | Run baseline agent on all 3 tasks, return scores |

### `/tasks` response

```json
{
  "tasks": ["easy", "medium", "hard"],
  "action_schema": {
    "action_type": "str — SearchPolicy | RequestInformation | ResolveTicket",
    "query": "str | null — required for SearchPolicy",
    "message": "str | null — required for RequestInformation",
    "decision": "str | null — Approve | Reject | Escalate",
    "reason": "str | null — required for ResolveTicket"
  }
}
```

### `/baseline` response

```json
{
  "easy":   0.78,
  "medium": 0.61,
  "hard":   0.34,
  "average": 0.577
}
```

---

## ⚙️ OpenEnv Spec Compliance

| Interface | Return Type | Status |
|-----------|-------------|--------|
| `reset()` | `ComplianceObservation` | ✅ |
| `step(action)` | `obs, reward, done, info` | ✅ |
| `state()` | `ComplianceState` | ✅ |
| `openenv.yaml` | Metadata + task list | ✅ |
| `openenv validate` | All checks pass | ✅ |
| `/tasks` endpoint | Task list + action schema | ✅ |
| `/grader` endpoint | Score 0.0–1.0 | ✅ |
| `/baseline` endpoint | Scores for all 3 tasks | ✅ |

---

## 🚀 Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/your-repo/corporate-compliance-env
cd corporate-compliance-env
pip install -r requirements.txt
```

### 2. Run the Server

```bash
uvicorn app.server.app:app --host 0.0.0.0 --port 7860
```

### 3. Validate Against OpenEnv Spec

```bash
openenv validate --host http://localhost:7860
```

### 4. Run Baseline Inference

```bash
export OPENAI_API_KEY=your_key_here
python app/baseline.py
```

Expected output:
```
Task 1 (Easy)   — Baseline Score: 0.78
Task 2 (Medium) — Baseline Score: 0.61
Task 3 (Hard)   — Baseline Score: 0.34
Average Score:    0.577
```

### 5. Run Tests

```bash
pytest tests/test_graders.py -v
```

### 6. Run via Docker

```bash
docker build -t compliance-env .
docker run -p 7860:7860 compliance-env
```

---

## 🐳 Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["uvicorn", "app.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

---

## 📄 openenv.yaml

```yaml
name: corporate-policy-compliance-env
version: "1.0.0"
description: >
  RL environment simulating enterprise corporate policy compliance.
  An agent audits employee expense claims against a company rulebook
  and decides to Approve, Reject, or Escalate each ticket.
  Grounded in Indian corporate compliance norms (INR, GST, WFH policy).
author: your-name
domain: enterprise-compliance
tags: [compliance, finance, hr, enterprise, india]
reward_range: [-1.0, 1.0]
tasks:
  - id: easy
    name: single_step_classification
    difficulty: easy
    max_steps: 3
    baseline_score: 0.78
  - id: medium
    name: policy_retrieval
    difficulty: medium
    max_steps: 5
    baseline_score: 0.61
  - id: hard
    name: multi_turn_contextual
    difficulty: hard
    max_steps: 8
    baseline_score: 0.34
action_space: ComplianceAction
observation_space: ComplianceObservation
```

---

## 📈 Why This Environment Matters

Companies like Ramp, Concur, and SAP spend millions building
proprietary AI auditing systems. This is the **first open-source
RL training environment for corporate policy compliance** — enabling
any researcher or company to train and benchmark agents for
enterprise expense auditing without proprietary data.

Because the policy document is a plain `policy.md` file, **any
company can drop in their own rulebook** — making this a general
framework, not just a demo. A well-trained agent on this environment
can handle ~70% of routine compliance decisions autonomously.

---

*Built for the Meta Hackathon 2026.*

