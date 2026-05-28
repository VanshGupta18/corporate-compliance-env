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
[![Python](https://img.shields.io/badge/Python-3.11+-green)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Space-orange)](https://huggingface.co/spaces/mcqueenmater/env-corporate)
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

| Difficulty | Task | Post-RL target (GRPO) | Rule-based baseline |
|---|---|---|---|
| Easy | Single-step classification | 0.85–0.99 | ~0.78 |
| Medium | Policy retrieval | 0.60–0.90 | ~0.61 |
| Hard | Multi-turn contextual | 0.40–0.75 | ~0.34 |

Curriculum bands are defined in [`app/curriculum_targets.py`](app/curriculum_targets.py) and [`openenv.yaml`](openenv.yaml).

---

## 🚀 Quick Start

### Use the Live Space

Visit the running instance: **https://huggingface.co/spaces/mcqueenmater/env-corporate**

### Run locally

```bash
git clone https://github.com/VanshGupta18/corporate-compliance-env.git
cd corporate-compliance-env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

uvicorn app.server.app:app --host 0.0.0.0 --port 7860
# API docs: http://localhost:7860/docs
# Dashboard: http://localhost:7860/demo

openenv validate --url http://localhost:7860 --verbose
pytest tests/ -q
```

### Baseline and inference

```bash
python app/baseline.py          # uses COMPLIANCE_API or the default HF Space
python inference.py             # writes inference_results.json (gitignored)
```

### Train on Google Colab (Unsloth SFT + GRPO)

**Notebook:** [`notebooks/Colab_T4_Training.ipynb`](notebooks/Colab_T4_Training.ipynb) — in Colab use *File → Open notebook → GitHub* and pick your fork, or upload the notebook. Set **Runtime → T4 GPU**, then set `GITHUB_USER` in the first code cell.

See **[`TRAINING.md`](TRAINING.md)** for the same pipeline as copy-paste cells.

### Docker

```bash
docker build -t compliance-env .
docker run -p 7860:7860 compliance-env
```

See [`Dockerfile`](Dockerfile) and [`openenv.yaml`](openenv.yaml) for container and OpenEnv metadata.

---

## 📡 API Endpoints (Quick Reference)

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Server health check |
| `/ws` | WebSocket | **Primary OpenEnv episode API**. Use for stateful `reset` / `step` / `state` loops |
| `/reset` | POST | Stateless reset smoke check • Body: `{"task_id": "easy\|medium\|hard"}` |
| `/step` | POST | Stateless single action endpoint • Body: `{"action": ComplianceAction}`. Do not use for multi-step episodes |
| `/state` | GET | Stateless state endpoint for debugging |
| `/tasks` | GET | List all tasks + action schema |
| `/grader` | POST | Get final score for completed episode |
| `/baseline` | POST | Run baseline agent on all 3 tasks |
| `/docs` | GET | Swagger interactive API documentation |
| `/demo` | GET | Gradio live demo dashboard |

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
meta-openenv/
├── app/
│   ├── models.py, client.py, graders.py, baseline.py, dashboard.py
│   ├── curriculum_targets.py, policy_snippets.py
│   └── server/               # ComplianceEnv + FastAPI app
├── server/                   # OpenEnv entrypoint (re-exports app.server.app)
├── data/
│   ├── policy.md, claims.json, splits/
│   └── generate_dataset.py
├── training/
│   ├── prepare_data.py, sft_train.py, grpo_train.py, eval_checkpoint.py
│   ├── learning_curve.py, smoke_test.py, training_utils.py
│   └── requirements-training.txt
├── tests/                    # API, graders, curriculum, training smoke
├── scripts/                  # validate-submission.sh, validate_dataset.py
├── inference.py
├── openenv.yaml, Dockerfile, TRAINING.md
└── requirements.txt
```

Generated at runtime (gitignored): `baseline_results.json`, `inference_results.json`, `training/data/`, `training/logs/`, `training/checkpoints/`.

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
| Relevant `SearchPolicy` | `+0.15` | Rule was genuinely unknown at that point |
| Correct `RequestInformation` | `+0.15` | Document was actually missing |
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

Easy scoring is component-based: valid `ResolveTicket`, correct decision,
valid reason, and no unnecessary tool calls. A wrong decision stays below
the success threshold even if the JSON and reason are valid.

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

Medium scoring requires a useful `SearchPolicy` before final decision credit.
A lucky correct `ResolveTicket` without useful search is capped below the
success threshold.

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

Hard scoring requires the intended workflow: useful policy search, correct
document request, then a correct final decision. Correct guesses that skip
search or document request are capped below the success threshold.

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

### `data/claims.json` and `data/splits/` — Curriculum Claims

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

`ground_truth_*` fields are stored in datasets for graders and offline analysis only. They are not populated in agent observations.

**Distribution:** balanced easy / medium / hard curriculum claims
**Split:** explicit `train`, `validation`, and `test` JSON files under `data/splits/`

### `data/generate_dataset.py`

Regenerate the full synthetic dataset at any time:

```bash
python data/generate_dataset.py \
  --train-per-diff 120 --val-per-diff 40 --test-per-diff 40 --seed 42
```

Writes `data/claims.json` and `data/splits/{train,validation,test}.json`. Then build SFT rows:

```bash
python training/prepare_data.py --episodes-per-task 40 --split train
```

Manual QA report: `python scripts/validate_dataset.py` (run from repo root).

---

## ⚙️ API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `WS` | `/ws` | Primary stateful OpenEnv session API. Use this for all multi-step episodes |
| `POST` | `/reset` | Stateless reset smoke check. Body: `{"task_id": "easy\|medium\|hard"}` |
| `POST` | `/step` | Stateless single action endpoint. Body: `{"action": ComplianceAction}` |
| `GET` | `/state` | Stateless debugging endpoint |
| `GET` | `/tasks` | List all tasks + full action schema |
| `POST` | `/grader` | Get final score for completed episode |
| `POST` | `/baseline` | Run baseline agent on all 3 tasks, return scores |

OpenEnv creates a fresh environment for each HTTP request. Keep a WebSocket open through `ComplianceEnvClient(...).sync()` for normal `reset()` / `step()` loops.

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

---

## 🧪 Training Pipeline (SFT + GRPO)

Colab runbook: [`TRAINING.md`](TRAINING.md). Quick checks:

```bash
python training/smoke_test.py
python -m training.sft_train --dry-run
python -m training.grpo_train --dry-run
```