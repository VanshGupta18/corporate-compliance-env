"""React-powered dashboard for Corporate Compliance RL benchmark.

Architecture: _render_dashboard() generates a full standalone HTML page (served at
/dashboard by FastAPI as HTMLResponse). Script tags execute normally in a proper HTML
page, bypassing Gradio's innerHTML limitation. React + Recharts are loaded from CDN.
build_demo() retains a minimal Gradio wrapper at /demo for backward compat.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from app.policy_snippets import POLICY_SNIPPETS, match_policy_snippet

# ── File paths ─────────────────────────────────────────────────────────────────

BASELINE_RESULTS = Path("baseline_results.json")
INFERENCE_RESULTS = Path("inference_results.json")
BASELINE_LOG = Path("baseline_run.log")
INFERENCE_LOG = Path("inference_run.log")
TRAINING_EPISODES = Path("training/logs/episodes.jsonl")
CLAIMS_DATA = Path("data/claims.json")
TEST_SPLIT = Path("data/splits/test.json")

_CLAIM_RUN_RE = re.compile(
    r"Running (?:inference|baseline) for claim\s+([A-Z0-9-]+)\s+\((easy|medium|hard)\)\.\.\.",
    re.IGNORECASE,
)

TASK_ORDER = ["easy", "medium", "hard"]

TASK_CURRICULUM = [
    {
        "id": "easy",
        "title": "Easy — Single-step classification",
        "max_steps": 3,
        "expected_steps": "~1 step",
        "agent_sees": "Full ticket upfront: amount, receipt status, description, and which policy rule applies (rule keyword visible).",
        "agent_goal": "Call ResolveTicket immediately with Approve, Reject, or Escalate — no policy search required.",
        "why_hard": "Looks simple, but wrong thresholds (e.g. ₹501 without receipt) still fail the grader.",
    },
    {
        "id": "medium",
        "title": "Medium — Policy retrieval",
        "max_steps": 5,
        "expected_steps": "~2 steps",
        "agent_sees": "Ticket details visible, but the policy rule keyword is hidden — the agent must discover the right rule.",
        "agent_goal": "SearchPolicy with a relevant query first, then ResolveTicket using what the rulebook returns.",
        "why_hard": "Wrong or vague searches earn small penalties; resolving without searching scores poorly.",
    },
    {
        "id": "hard",
        "title": "Hard — Multi-turn contextual decision",
        "max_steps": 8,
        "expected_steps": "3–4 steps",
        "agent_sees": "Vague description, missing document flagged as \"required\" (type hidden until requested), rule keyword hidden.",
        "agent_goal": "SearchPolicy → RequestInformation for the missing doc → read simulated reply → ResolveTicket.",
        "why_hard": "Request loops, off-topic searches, and early guesses before evidence arrives destroy the score.",
    },
]

POLICY_RULES = [
    "Meals under \u20b9500 \u2014 no receipt required",
    "Meals \u20b9500\u2013\u20b92,000 \u2014 receipt required",
    "Meals above \u20b92,000 \u2014 receipt + manager approval",
    "Alcohol is never an approved expense",
    "Local travel (auto/metro) under \u20b9500 \u2014 no receipt needed",
    "Cab rides after 10\u00a0PM \u2014 always approved with receipt",
    "Daytime cab rides \u2014 require manager note",
    "Domestic flights \u2014 economy only for L1\u2013L6",
    "Business class \u2014 only for L7 (VP) and above",
    "International travel above \u20b950,000 \u2014 VP approval required",
    "WFH allowance \u2014 max \u20b91,000/month",
    "Duplicate claims \u2014 auto-reject",
    "L7 employees \u2014 always escalate regardless of amount",
    "GST receipt required for all claims above \u20b95,000",
    "Personal shopping/gifts \u2014 always reject",
]

# ── Dataclasses ────────────────────────────────────────────────────────────────


@dataclass
class MethodMetrics:
    method: str
    overall: float
    total: int
    by_task: Dict[str, float]
    by_task_total: Dict[str, int]


# ── I/O helpers ────────────────────────────────────────────────────────────────


def _safe_read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


# ── Metrics loaders ────────────────────────────────────────────────────────────


def _load_metrics(path: Path, method: str) -> MethodMetrics:
    payload = _safe_read_json(path)
    if not payload:
        return MethodMetrics(method=method, overall=0.0, total=0, by_task={}, by_task_total={})
    metrics = payload.get("metrics", payload)
    overall_payload = metrics.get("overall_metrics", {})
    overall = float(overall_payload.get("mean_grader_score", 0.0) or 0.0)
    total = int(
        overall_payload.get("total_claims", overall_payload.get("total_evaluations", 0) or 0)
    )
    by_diff = metrics.get("performance_by_difficulty", metrics.get("by_difficulty", {}))
    by_task: Dict[str, float] = {}
    by_task_total: Dict[str, int] = {}
    for task in TASK_ORDER:
        node = by_diff.get(task, {})
        by_task[task] = float(node.get("mean_grader_score", 0.0) or 0.0)
        by_task_total[task] = int(node.get("total", 0) or 0)
    return MethodMetrics(method=method, overall=overall, total=total, by_task=by_task, by_task_total=by_task_total)


def _training_metrics(rows: List[Dict[str, Any]]) -> MethodMetrics:
    scored = []
    for row in rows:
        score = row.get("grader_score", row.get("score"))
        if score is None:
            continue
        scored.append({**row, "grader_score": float(score)})
    if not scored:
        return MethodMetrics(method="training", overall=0.0, total=0, by_task={}, by_task_total={})
    by_task: Dict[str, float] = {}
    by_task_total: Dict[str, int] = {}
    for task in TASK_ORDER:
        sc = [float(r["grader_score"]) for r in scored if str(r.get("task_id", "")).lower() == task]
        by_task[task] = mean(sc) if sc else 0.0
        by_task_total[task] = len(sc)
    return MethodMetrics(
        method="training",
        overall=mean(float(r["grader_score"]) for r in scored),
        total=len(scored),
        by_task=by_task,
        by_task_total=by_task_total,
    )


# ── JSONL reader ───────────────────────────────────────────────────────────────


def _read_jsonl(path: Path, last_n: int = 500) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines()[-last_n:]:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
        except json.JSONDecodeError:
            continue
    return rows


# ── Log parsers ────────────────────────────────────────────────────────────────


_STEP_PARAM_RE = re.compile(r'(\w+)="([^"]*)"|(\w+)=([^\s]+)')


def _parse_step_params(tail: str) -> Dict[str, str]:
    """Parse trailing key=value pairs on [STEP] log lines."""
    params: Dict[str, str] = {}
    if not tail:
        return params
    for match in _STEP_PARAM_RE.finditer(tail.strip()):
        if match.group(1):
            params[match.group(1)] = match.group(2)
        elif match.group(3):
            params[match.group(3)] = match.group(4)
    return params


def _expected_policy_fields(claim: Dict[str, Any]) -> Dict[str, str]:
    """Ground-truth policy category the agent should search for this ticket."""
    rule_keyword = str(claim.get("rule_keyword") or claim.get("policy_category") or "")
    entries = POLICY_SNIPPETS.get(rule_keyword, [])
    summary = entries[0][1] if entries else (
        f"Search the rulebook for policies related to: {rule_keyword}" if rule_keyword else ""
    )
    return {
        "rule_keyword": rule_keyword,
        "policy_category": str(claim.get("policy_category") or ""),
        "expected_policy_search": rule_keyword,
        "expected_policy_summary": summary,
    }


def _action_fields_from_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract dashboard-facing fields from an action dict."""
    action_type = _normalize_action(payload.get("action_type", ""))
    fields: Dict[str, Any] = {"action_type": action_type}
    for key in ("query", "message", "decision", "reason"):
        val = payload.get(key)
        if val is not None and str(val).strip():
            fields[key] = str(val)
    if action_type == "SEARCH_POLICY" and fields.get("query"):
        rk = str(payload.get("_rule_keyword") or payload.get("rule_keyword") or "")
        if rk:
            snippet, relevant = match_policy_snippet(rk, fields["query"])
            fields["matched_policy"] = snippet
            fields["search_relevant"] = relevant
    return fields


def _normalize_action(action: str) -> str:
    a = str(action or "").split(".")[-1]
    m = {
        "SearchPolicy": "SEARCH_POLICY",
        "SEARCH_POLICY": "SEARCH_POLICY",
        "RequestInformation": "REQUEST_INFORMATION",
        "REQUEST_INFORMATION": "REQUEST_INFORMATION",
        "ResolveTicket": "RESOLVE_TICKET",
        "RESOLVE_TICKET": "RESOLVE_TICKET",
    }
    return m.get(a, a.upper())


def _parse_episode_log(path: Path, method: str) -> Tuple[List[Dict], List[Dict]]:
    text = _safe_read_text(path)
    if not text:
        return [], []
    current_claim: Optional[str] = None
    current_task: Optional[str] = None
    current: Optional[Dict[str, Any]] = None
    episodes: List[Dict[str, Any]] = []
    actions: List[Dict[str, Any]] = []
    episode_idx = 0
    claim_queue: List[Dict[str, Any]] = (
        list(_load_baseline_claim_order()) if method == "baseline" else []
    )

    for raw in text.splitlines():
        line = raw.strip()
        cm = _CLAIM_RUN_RE.match(line)
        if cm:
            current_claim, current_task = cm.group(1), cm.group(2).lower()
            continue

        sm = re.match(r"\[START\]\s+task=(EASY|MEDIUM|HARD)\s+.*", line)
        if sm:
            episode_idx += 1
            task_from_start = sm.group(1).lower()
            if not current_claim and method == "baseline" and claim_queue:
                queued = claim_queue.pop(0)
                current_claim = str(queued.get("id", ""))
                current_task = str(queued.get("task_difficulty", task_from_start)).lower()
            current = {
                "method": method,
                "episode_id": f"{method}-{episode_idx:04d}",
                "claim_id": current_claim or "",
                "task_id": current_task or task_from_start,
                "steps": 0,
                "total_reward": 0.0,
                "grader_score": 0.0,
                "success": False,
            }
            continue

        if current is None:
            continue

        step_m = re.match(
            r"\[STEP\]\s+step=(\d+)\s+action=([A-Za-z0-9_\.]+)\s+reward=([-0-9.]+)\s+done=(true|false)(?:\s+(.*))?$",
            line,
        )
        if step_m:
            reward = float(step_m.group(3))
            step_num = int(step_m.group(1))
            current["steps"] = max(current["steps"], step_num)
            current["total_reward"] = round(current["total_reward"] + reward, 4)
            extras = _parse_step_params(step_m.group(5) or "")
            action_row = {
                "method": method,
                "episode_id": current["episode_id"],
                "task_id": current["task_id"],
                "step": step_num,
                "action_type": _normalize_action(step_m.group(2)),
                "reward": reward,
                "done": step_m.group(4) == "true",
                **extras,
            }
            actions.append(action_row)
            continue

        end_m = re.match(
            r"\[END\]\s+(?:success=(true|false)\s+)?steps=(\d+)\s+(?:score|grader_score)=([0-9.]+)",
            line,
        )
        if end_m:
            current["success"] = end_m.group(1) == "true" if end_m.group(1) else False
            current["steps"] = int(end_m.group(2))
            current["grader_score"] = float(end_m.group(3))
            episodes.append(current)
            current = None
            current_claim = None
            current_task = None

    return episodes, actions


def _training_frames(rows: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
    episodes: List[Dict[str, Any]] = []
    actions: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        score = row.get("grader_score", row.get("score"))
        reward = row.get("total_reward", row.get("reward", 0.0))
        task = str(row.get("task_id", "training")).lower()
        ep_id = f"training-{idx:04d}"
        steps = row.get("steps", 0)
        episodes.append({
            "method": "training",
            "episode_id": ep_id,
            "claim_id": row.get("claim_id", ""),
            "task_id": task,
            "steps": int(steps or 0),
            "total_reward": float(reward or 0.0),
            "grader_score": float(score or 0.0),
            "success": bool(row.get("success", False)),
        })
        history = row.get("actions_history", row.get("actions", []))
        if isinstance(history, list):
            for si, action in enumerate(history, start=1):
                p = action if isinstance(action, dict) else {"action_type": str(action)}
                p = {**p, "_rule_keyword": row.get("rule_keyword", "")}
                fields = _action_fields_from_dict(p)
                actions.append({
                    "method": "training",
                    "episode_id": ep_id,
                    "task_id": task,
                    "step": si,
                    "reward": float(p.get("reward", 0.0) or 0.0),
                    "done": bool(p.get("done", False)),
                    **fields,
                })
    return episodes, actions


# ── Claims enrichment ──────────────────────────────────────────────────────────


def _load_baseline_claim_order() -> List[Dict[str, Any]]:
    """Claim list in the same order baseline.py iterates (test split by default)."""
    if TEST_SPLIT.exists():
        try:
            data = json.loads(TEST_SPLIT.read_text(encoding="utf-8"))
            return list(data.get("claims", []))
        except Exception:
            pass
    if not CLAIMS_DATA.exists():
        return []
    try:
        data = json.loads(CLAIMS_DATA.read_text(encoding="utf-8"))
        return list(data.get("claims", []))
    except Exception:
        return []


def _load_claims_index() -> Dict[str, Any]:
    """Load data/claims.json and return a dict keyed by claim ID."""
    if not CLAIMS_DATA.exists():
        return {}
    try:
        data = json.loads(CLAIMS_DATA.read_text(encoding="utf-8"))
        return {c["id"]: c for c in data.get("claims", [])}
    except Exception:
        return {}


def _enrich_episodes(episodes: List[Dict], claims_idx: Dict[str, Any]) -> List[Dict]:
    """Merge full ticket fields from claims.json into each episode record."""
    enriched = []
    for ep in episodes:
        claim = claims_idx.get(str(ep.get("claim_id") or ""), {})
        policy_fields = _expected_policy_fields(claim) if claim else _expected_policy_fields({})
        enriched.append({
            **ep,
            "employee_name": claim.get("employee_name", ""),
            "employee_role": claim.get("employee_role", ""),
            "employee_level": claim.get("employee_level", ""),
            "amount": float(claim.get("amount", 0) or 0),
            "description": str(claim.get("description") or claim.get("vague_description") or ""),
            "has_receipt": bool(claim.get("has_receipt", False)),
            "missing_document": claim.get("missing_document"),
            "risk_score": float(claim.get("risk_score", 0.0) or 0.0),
            "ground_truth": claim.get("ground_truth_decision", ""),
            **policy_fields,
        })
    return enriched


def _enrich_action_policies(episodes: List[Dict[str, Any]], actions: List[Dict[str, Any]]) -> None:
    """Attach matched policy snippet + relevance to SearchPolicy steps."""
    ep_by_id = {str(e.get("episode_id")): e for e in episodes}
    for action in actions:
        if _normalize_action(action.get("action_type", "")) != "SEARCH_POLICY":
            continue
        query = str(action.get("query") or "").strip()
        if not query:
            continue
        ep = ep_by_id.get(str(action.get("episode_id")), {})
        rule_kw = str(ep.get("expected_policy_search") or ep.get("rule_keyword") or "")
        if not rule_kw:
            continue
        snippet, relevant = match_policy_snippet(rule_kw, query)
        action["matched_policy"] = snippet
        action["search_relevant"] = relevant


# ── Main data loader ───────────────────────────────────────────────────────────


def _load_dashboard_data() -> Dict[str, Any]:
    baseline = _load_metrics(BASELINE_RESULTS, "baseline")
    inference = _load_metrics(INFERENCE_RESULTS, "inference")
    b_eps, b_acts = _parse_episode_log(BASELINE_LOG, "baseline")
    i_eps, i_acts = _parse_episode_log(INFERENCE_LOG, "inference")
    training_rows = _read_jsonl(TRAINING_EPISODES, last_n=800)
    training = _training_metrics(training_rows)
    t_eps, t_acts = _training_frames(training_rows)

    claims_idx = _load_claims_index()
    all_eps = _enrich_episodes(b_eps + i_eps + t_eps, claims_idx)
    all_acts = b_acts + i_acts + t_acts
    _enrich_action_policies(all_eps, all_acts)

    learning = [
        {"episode": idx, "grader_score": float(r.get("grader_score", r.get("score", 0.0)) or 0.0)}
        for idx, r in enumerate(training_rows, 1)
        if r.get("grader_score", r.get("score")) is not None
    ]

    return {
        "metrics": {
            "baseline": {
                "overall": baseline.overall,
                "easy": baseline.by_task.get("easy", 0.0),
                "medium": baseline.by_task.get("medium", 0.0),
                "hard": baseline.by_task.get("hard", 0.0),
                "n": baseline.total,
            },
            "inference": {
                "overall": inference.overall,
                "easy": inference.by_task.get("easy", 0.0),
                "medium": inference.by_task.get("medium", 0.0),
                "hard": inference.by_task.get("hard", 0.0),
                "n": inference.total,
            },
            "training": {
                "overall": training.overall if training.total > 0 else None,
                "easy": training.by_task.get("easy") if training.total > 0 else None,
                "medium": training.by_task.get("medium") if training.total > 0 else None,
                "hard": training.by_task.get("hard") if training.total > 0 else None,
                "n": training.total,
            },
        },
        "episodes": all_eps,
        "actions": all_acts,
        "learning_data": learning,
        "policy_rules": POLICY_RULES,
        "task_curriculum": TASK_CURRICULUM,
    }


# ── React HTML template ────────────────────────────────────────────────────────
# Full standalone HTML page (no Babel, no JSX — plain React.createElement).
# Eliminates the 1.5 MB Babel CDN dependency that caused "Loading..." to hang.
# __DATA_PLACEHOLDER__ is replaced at render time with the serialised JSON payload.

_REACT_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Corporate Compliance RL Dashboard</title>
<!-- No Babel / no JSX: plain React.createElement via the h() shorthand -->
<style>
:root{
  --bg:#0f1117;--surface:#151c27;--surface2:#1d2536;
  --border:rgba(255,255,255,0.09);--text:#edf2f7;--muted:#8ea0b5;
  --blue:#378ADD;--orange:#EF9F27;--gray:#888780;
  --green:#27ae60;--red:#e74c3c;--amber:#f39c12;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{min-height:100%;background:var(--bg);color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Inter',sans-serif;line-height:1.5}
button{cursor:pointer;font-family:inherit}
a{color:var(--blue);text-decoration:none}
.dash{max-width:1300px;margin:0 auto;padding:24px 20px 64px}

/* Hero */
.hero{padding:44px 48px;border-radius:28px;margin-bottom:28px;
  background:radial-gradient(circle at 18% 38%,rgba(55,138,221,.22) 0%,transparent 48%),
  radial-gradient(circle at 78% 64%,rgba(39,174,96,.15) 0%,transparent 40%),
  linear-gradient(135deg,#111827,#0d1321);
  border:1px solid var(--border)}
.hero-pill{display:inline-block;padding:5px 14px;border-radius:999px;
  background:rgba(255,255,255,.08);color:#d8f5e2;font-size:12px;
  letter-spacing:.06em;margin-bottom:20px}
.hero h1{font-size:38px;font-weight:860;line-height:1.06;margin-bottom:14px;letter-spacing:-.02em}
.hero p{font-size:15px;color:#c7d2de;max-width:840px;line-height:1.72}

/* KPI */
.kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin-bottom:24px}
@media(max-width:800px){.kpi-grid{grid-template-columns:repeat(2,1fr)}}
.kpi-card{padding:22px 18px;border:1px solid var(--border);border-radius:20px;
  background:linear-gradient(180deg,rgba(255,255,255,.065),rgba(255,255,255,.022))}
.kpi-lbl{font-size:11px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);margin-bottom:8px}
.kpi-val{font-size:32px;font-weight:860;margin-bottom:4px}
.kpi-sub{font-size:12px;color:var(--muted)}

/* Narrative */
.narrative{padding:20px 24px;border:1px solid var(--border);border-radius:18px;
  background:linear-gradient(180deg,rgba(255,255,255,.04),rgba(255,255,255,.015));
  margin-bottom:24px}
.eyebrow{font-size:11px;text-transform:uppercase;letter-spacing:.14em;color:var(--muted);margin-bottom:12px}
.story-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
@media(max-width:800px){.story-grid{grid-template-columns:repeat(2,1fr)}}
.story-stat{padding:14px;background:rgba(0,0,0,.28);border-radius:12px}
.story-stat b{display:block;font-size:22px;font-weight:810;margin-bottom:4px}
.story-stat span{font-size:12px;color:var(--muted)}

/* Chart */
.chart-section{padding:22px 20px;border:1px solid var(--border);border-radius:20px;
  background:linear-gradient(180deg,rgba(255,255,255,.05),rgba(255,255,255,.02));
  margin-bottom:24px}
.chart-section h2{font-size:17px;font-weight:750;margin-bottom:3px}
.chart-section .sub{font-size:13px;color:var(--muted);margin-bottom:14px}

/* Task curriculum */
.curriculum-section{margin-bottom:28px;padding:20px 22px;border:1px solid var(--border);border-radius:20px;
  background:linear-gradient(180deg,rgba(255,255,255,.04),rgba(255,255,255,.015))}
.curriculum-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:14px}
@media(max-width:960px){.curriculum-grid{grid-template-columns:1fr}}
.curriculum-card{padding:16px;border:1px solid var(--border);border-radius:16px;background:var(--surface);
  border-top:3px solid var(--border)}
.curriculum-card.easy{border-top-color:#27ae60}
.curriculum-card.medium{border-top-color:#f39c12}
.curriculum-card.hard{border-top-color:#e74c3c}
.curr-hdr{display:flex;justify-content:space-between;align-items:flex-start;gap:8px;margin-bottom:10px}
.curr-title{font-size:14px;font-weight:800;line-height:1.3}
.curr-meta{font-size:11px;color:var(--muted);white-space:nowrap}
.curr-lbl{font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin:10px 0 4px}
.curr-txt{font-size:12px;color:#c7d2de;line-height:1.55}
.curr-scores{display:flex;flex-wrap:wrap;gap:6px;margin-top:12px;padding-top:10px;border-top:1px solid var(--border)}
.curr-score{font-size:11px;padding:3px 8px;border-radius:6px;background:rgba(0,0,0,.25);color:var(--muted)}
.curr-score b{color:var(--text);font-weight:700}

/* Section header */
.sec-hd{margin:28px 0 14px}
.sec-hd h2{font-size:20px;font-weight:750;margin-bottom:3px}
.sec-hd p{font-size:13px;color:var(--muted)}

/* Replay */
.replay-controls{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin-bottom:16px}
.random-btn{padding:8px 18px;border-radius:999px;border:1px solid var(--green);color:#58d68d;
  background:rgba(39,174,96,.12);font-size:13px;font-weight:700;transition:all .15s}
.random-btn:hover:not(:disabled){background:rgba(39,174,96,.28)}
.random-btn:disabled{opacity:.4;cursor:not-allowed}
.replay-meta{font-size:12px;color:var(--muted);margin-left:4px}
.replay-empty{padding:44px;text-align:center;color:var(--muted);border:1px dashed var(--border);
  border-radius:20px;font-size:15px;margin-bottom:24px}
.replay-grid{display:grid;grid-template-columns:1fr 1.4fr .76fr;gap:14px;margin-bottom:24px;align-items:start}
@media(max-width:960px){.replay-grid{grid-template-columns:1fr}}

/* Column panels */
.col-panel{border:1px solid var(--border);border-radius:20px;background:var(--surface);padding:18px;min-height:300px}
.col-title{font-size:11px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);margin-bottom:14px}
.col-title-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px}

/* Ticket */
.ticket-hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;flex-wrap:wrap;gap:6px}
.ticket-id{font-size:17px;font-weight:800}
.diff-badge{padding:3px 10px;border-radius:999px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em}
.diff-easy{background:rgba(39,174,96,.2);color:#2ecc71}
.diff-medium{background:rgba(243,156,18,.18);color:#f39c12}
.diff-hard{background:rgba(231,76,60,.18);color:#e74c3c}
.diff-training{background:rgba(136,135,128,.18);color:#aaa}
.f-lbl{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:2px}
.f-val{font-size:14px;margin-bottom:2px}
.f-sub{font-size:12px;color:var(--muted)}
.t-field{margin-bottom:11px}
.amount-big{font-size:20px;font-weight:800;color:#58d68d}
.badges-row{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0}
.receipt-y{padding:4px 10px;border-radius:999px;background:rgba(39,174,96,.18);color:#2ecc71;font-size:12px}
.receipt-n{padding:4px 10px;border-radius:999px;background:rgba(231,76,60,.18);color:#e74c3c;font-size:12px}
.miss-tag{padding:4px 10px;border-radius:999px;background:rgba(243,156,18,.18);color:#f39c12;font-size:12px}
.risk-bg{height:6px;background:rgba(255,255,255,.1);border-radius:3px;margin:5px 0 2px}
.risk-fill{height:100%;border-radius:3px;background:linear-gradient(90deg,#27ae60,#e74c3c);transition:width .4s}
.risk-val{font-size:12px;color:var(--muted)}
.gt-chip{display:inline-block;margin-top:4px;padding:4px 12px;border-radius:8px;font-size:13px;font-weight:700}
.gt-Approve{background:rgba(39,174,96,.2);color:#2ecc71}
.gt-Reject{background:rgba(231,76,60,.18);color:#e74c3c}
.gt-Escalate{background:rgba(243,156,18,.18);color:#f39c12}
.policy-expected-box{padding:12px;border-radius:12px;background:rgba(55,138,221,.1);border:1px solid rgba(55,138,221,.25)}
.policy-pill{display:inline-block;margin-top:4px;padding:5px 12px;border-radius:999px;
  background:rgba(55,138,221,.22);color:#7eb8f7;font-size:13px;font-weight:700}
.policy-summary{font-size:12px;color:#b6c2d2;margin-top:8px;line-height:1.5}
.policy-note{font-size:11px;color:var(--muted);margin-top:6px;font-style:italic}
.policy-search{color:#c7d2de}
.policy-hit{color:#a8d4a0;margin-top:4px;padding:8px;background:rgba(39,174,96,.1);border-radius:8px;font-size:12px;line-height:1.45}
.policy-expected{margin-top:4px}
.policy-ok{color:#2ecc71;font-weight:700}
.policy-miss{color:#e74c3c;font-weight:700}

/* Timeline / agent trace */
.play-btn{display:inline-flex;align-items:center;gap:7px;padding:6px 16px;border-radius:999px;
  border:1px solid var(--blue);color:var(--blue);background:rgba(55,138,221,.12);
  font-size:13px;font-weight:600;transition:all .2s}
.play-btn:hover:not(:disabled){background:rgba(55,138,221,.28)}
.play-btn:disabled{opacity:.48;cursor:not-allowed}
.play-icon{width:0;height:0;border-top:5px solid transparent;border-bottom:5px solid transparent;
  border-left:8px solid currentColor;flex-shrink:0}
.play-icon.playing{width:8px;height:8px;border:2px solid currentColor;border-radius:2px;
  border-left:2px;background:currentColor;box-sizing:border-box}
.action-icon{display:inline-flex;align-items:center;justify-content:center;width:22px;height:22px;
  border-radius:6px;font-size:9px;font-weight:800;letter-spacing:-.02em;flex-shrink:0;margin-right:6px}
.action-icon.search{background:rgba(55,138,221,.22);color:#7eb8f7;border:1px solid rgba(55,138,221,.45)}
.action-icon.request{background:rgba(243,156,18,.18);color:#f5c842;border:1px solid rgba(243,156,18,.4)}
.action-icon.resolve{background:rgba(39,174,96,.18);color:#58d68d;border:1px solid rgba(39,174,96,.4)}
.status-dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:6px;vertical-align:middle}
.status-dot.ok{background:#2ecc71}
.status-dot.no{background:#e74c3c}
.status-dot.warn{background:#f39c12}
.tag{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:700;margin-left:6px}
.tag-ok{background:rgba(39,174,96,.2);color:#2ecc71}
.tag-miss{background:rgba(231,76,60,.18);color:#e74c3c}
.timeline{display:flex;flex-direction:column;gap:9px}
.step-card{padding:11px 13px;border-radius:12px;border:1px solid var(--border);
  border-left:3px solid transparent;background:var(--surface2);
  animation:fadeUp .3s ease both}
@keyframes fadeUp{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:translateY(0)}}
.step-hdr{display:flex;align-items:center;gap:8px}
.step-dot{width:25px;height:25px;border-radius:50%;display:grid;place-items:center;
  background:#243654;color:#fff;font-size:11px;font-weight:800;flex-shrink:0}
.step-lbl{flex:1;font-size:13px;font-weight:640;display:flex;align-items:center}
.r-badge{padding:2px 8px;border-radius:999px;font-size:12px;font-weight:700}
.r-pos{background:rgba(39,174,96,.2);color:#2ecc71}
.r-neg{background:rgba(231,76,60,.18);color:#e74c3c}
.r-zero{background:rgba(255,255,255,.08);color:var(--muted)}
.step-detail{font-size:12px;color:var(--muted);margin-top:3px;margin-left:33px;line-height:1.4}
.dec-chip{display:inline-block;margin-top:5px;margin-left:33px;padding:4px 12px;
  border-radius:8px;font-size:13px;font-weight:700}
.dec-Approve{background:rgba(39,174,96,.2);color:#2ecc71}
.dec-Reject{background:rgba(231,76,60,.18);color:#e74c3c}
.dec-Escalate{background:rgba(243,156,18,.18);color:#f39c12}
.tl-empty{padding:18px;text-align:center;color:var(--muted);font-size:13px}

/* Reward panel */
.grade-box{margin:12px 0;padding:12px;background:rgba(0,0,0,.24);border-radius:14px;text-align:center}
.grade-lbl{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted)}
.grade-score{font-size:42px;font-weight:900;margin:4px 0}
.steps-txt{font-size:12px;color:var(--muted)}
.tab-grp{display:flex;gap:4px;background:rgba(255,255,255,.05);padding:3px;border-radius:10px}
.tab-btn{padding:5px 13px;border-radius:7px;border:none;background:transparent;color:var(--muted);
  font-size:12px;font-weight:640;transition:all .15s}
.tab-btn.act{background:rgba(255,255,255,.13);color:var(--text)}

/* Rulebook */
.rulebook{margin-bottom:24px;border:1px solid var(--border);border-radius:18px;overflow:hidden}
.rb-toggle{width:100%;padding:15px 20px;background:var(--surface);border:none;color:var(--text);
  font-size:14px;font-weight:720;text-align:left;display:flex;justify-content:space-between;align-items:center;
  transition:background .15s}
.rb-chevron{display:inline-block;width:8px;height:8px;border-right:2px solid var(--muted);
  border-bottom:2px solid var(--muted);transform:rotate(45deg);transition:transform .2s;margin-left:8px}
.rb-chevron.open{transform:rotate(-135deg);margin-top:4px}
.rb-toggle:hover{background:var(--surface2)}
.rb-body{padding:16px 20px;background:rgba(0,0,0,.15)}
.rb-sub{font-size:13px;color:var(--muted);margin-bottom:13px}
.rules-list{padding-left:20px;display:flex;flex-direction:column;gap:7px}
.rule-item{font-size:13px;color:#c7d2de;padding:6px 10px;border-radius:8px;transition:background .2s,color .2s}
.rule-item.lit{background:rgba(245,176,65,.25);color:#f5c842;font-weight:660}

/* CSS fallback grouped bar chart (when Recharts CDN blocked) */
.css-chart{position:relative;margin-top:12px;padding:18px 10px 8px 36px;
  background:linear-gradient(to top,rgba(255,255,255,.1) 1px,transparent 1px) 0 0/100% 25%;
  border-left:1px solid rgba(255,255,255,.16);border-bottom:1px solid rgba(255,255,255,.16);
  display:grid;grid-template-columns:repeat(4,1fr);gap:16px;min-height:260px}
.css-chart-scale{position:absolute;left:10px;top:18px;height:220px;display:flex;flex-direction:column;
  justify-content:space-between;color:#77869a;font-size:11px}
.css-chart-group{display:flex;flex-direction:column;justify-content:flex-end;min-width:0}
.css-bars{height:220px;display:flex;align-items:flex-end;justify-content:center;gap:6px}
.css-bar-wrap{height:100%;display:flex;flex-direction:column;justify-content:flex-end;align-items:center;gap:4px;min-width:36px}
.css-bar{width:26px;border-radius:7px 7px 2px 2px;transition:height .4s ease}
.css-bar.pending{opacity:.35;background:repeating-linear-gradient(45deg,#888780,#888780 4px,rgba(136,135,128,.3) 4px,rgba(136,135,128,.3) 8px)!important}
.css-bar-val{font-size:10px;color:#d9e2ec;font-weight:700;white-space:nowrap}
.css-bar-lbl{font-size:9px;color:#8ea0b5;text-align:center;max-width:52px;line-height:1.2}
.css-group-lbl{text-align:center;color:#c8d3df;font-size:12px;font-weight:700;margin-top:8px}
.css-legend{grid-column:1/-1;display:flex;gap:16px;flex-wrap:wrap;color:#aab7c7;font-size:12px;margin-top:8px;padding-left:26px}
.css-legend-dot{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:middle}
</style>
</head>
<body>
<div id="root"><div style="padding:60px;text-align:center;color:#8ea0b5;font-family:sans-serif;font-size:15px">Loading&#8230;</div></div>

<script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
<script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/prop-types@15.8.1/prop-types.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/recharts@2.12.7/umd/Recharts.min.js"></script>
<script>window.__D=__DATA_PLACEHOLDER__;</script>
<script>
/* No Babel. Pure React.createElement via h() shorthand. */
(function(){
'use strict';
if(!window.React||!window.ReactDOM){
  document.getElementById('root').innerHTML='<div style="padding:48px;text-align:center;color:#e74c3c;font-family:sans-serif">React CDN failed to load. Check your internet connection and reload.</div>';
  return;
}
const {useState,useEffect,useMemo,useCallback}=React;
const h=React.createElement;
const RC=window.Recharts||{};
const hasRecharts=!!(RC.ResponsiveContainer&&RC.BarChart);

const D=window.__D||{};
const eps=D.episodes||[];
const acts=D.actions||[];
const M=D.metrics||{};
const rules=D.policy_rules||[];
const learning=D.learning_data||[];
const curriculum=D.task_curriculum||[];

/* ── Utilities ── */
const fmt3=v=>v!=null?Number(v).toFixed(3):'pending';
const fmtRs=v=>'\u20b9'+Number(v||0).toLocaleString('en-IN');
const scoreColor=v=>v>=0.7?'#27ae60':v>=0.4?'#f39c12':'#e74c3c';
const aBorder=t=>t==='SEARCH_POLICY'?'#378ADD':t==='REQUEST_INFORMATION'?'#f39c12':'#27ae60';
const aLabel=t=>t==='SEARCH_POLICY'?'Search Policy':t==='REQUEST_INFORMATION'?'Request Information':'Resolve Ticket';
const aIconCls=t=>t==='SEARCH_POLICY'?'search':t==='REQUEST_INFORMATION'?'request':'resolve';
const aIconTxt=t=>t==='SEARCH_POLICY'?'SP':t==='REQUEST_INFORMATION'?'RI':'RV';

function ActionIcon({type}){
  return h('span',{className:'action-icon '+aIconCls(type),title:aLabel(type)},aIconTxt(type));
}

/* ═══ React components (createElement, no JSX) ═══ */

function App(){
  const [selId,setSelId]=useState(null);
  const [diffTab,setDiffTab]=useState('easy');
  const [methTab,setMethTab]=useState('baseline');
  const [playStep,setPlayStep]=useState(-1);
  const [playing,setPlaying]=useState(false);

  const selEp=useMemo(()=>eps.find(e=>e.episode_id===selId)||null,[selId]);
  const epActs=useMemo(()=>acts.filter(a=>a.episode_id===selId).sort((a,b)=>a.step-b.step),[selId]);
  const visActs=playStep>=0?epActs.slice(0,playStep+1):epActs;

  const filtEps=useMemo(()=>eps.filter(e=>e.task_id===diffTab&&e.method===methTab),[diffTab,methTab]);

  const handlePlay=useCallback(()=>{setPlayStep(0);setPlaying(true);},[]);

  const handleRandom=useCallback(()=>{
    if(!filtEps.length)return;
    const pick=filtEps[Math.floor(Math.random()*filtEps.length)];
    setSelId(pick.episode_id);
    setPlayStep(-1);
    setPlaying(false);
  },[filtEps]);

  useEffect(()=>{
    if(!playing)return;
    if(playStep>=epActs.length){setPlaying(false);return;}
    const t=setTimeout(()=>setPlayStep(s=>s+1),800);
    return()=>clearTimeout(t);
  },[playing,playStep,epActs.length]);

  useEffect(()=>{
    if(!selId)return;
    if(!filtEps.some(e=>e.episode_id===selId)){
      setSelId(null);
      setPlayStep(-1);
      setPlaying(false);
    }
  },[diffTab,methTab,filtEps,selId]);

  const highlightedRule=useMemo(()=>{
    if(selEp&&selEp.expected_policy_search)return selEp.expected_policy_search;
    const vis=playStep>=0?epActs.slice(0,playStep+1):epActs;
    const searches=vis.filter(a=>a.action_type==='SEARCH_POLICY');
    return searches.length?(searches[searches.length-1].query||null):null;
  },[playStep,epActs,selEp]);

  const methLabel={baseline:'Baseline',inference:'Inference',training:'Train'}[methTab]||methTab;

  return h('div',{className:'dash'},
    h(HeroSection,null),
    h(KPIGrid,null),
    h(NarrativePanel,null),
    h(ScoreChart,null),
    h(TaskCurriculum,null),
    h('div',{className:'sec-hd'},
      h('h2',null,'Episode Replay'),
      h('p',null,'Choose difficulty and agent, load a random episode, then press Play to step through the trace.')
    ),
    h('div',{className:'replay-controls'},
      h('div',{className:'tab-grp'},
        ...['easy','medium','hard'].map(t=>
          h('button',{key:t,className:'tab-btn'+(diffTab===t?' act':''),onClick:()=>setDiffTab(t)},
            t.charAt(0).toUpperCase()+t.slice(1)))),
      h('div',{className:'tab-grp'},
        ...[['baseline','Baseline'],['inference','Inference'],['training','Train']].map(([id,lbl])=>
          h('button',{key:id,className:'tab-btn'+(methTab===id?' act':''),onClick:()=>setMethTab(id)},lbl))),
      h('button',{className:'random-btn',onClick:handleRandom,disabled:!filtEps.length},
        'Random episode'),
      filtEps.length?h('span',{className:'replay-meta'},
        selEp?(selEp.claim_id||selEp.episode_id)+' \xb7 '+methLabel+' \xb7 '+diffTab
          :filtEps.length+' episodes available'):null
    ),
    selEp
      ?h('div',{className:'replay-grid'},
          h(TicketPanel,{ep:selEp}),
          h(AgentPanel,{ep:selEp,allActs:epActs,visActs,playing,onPlay:handlePlay,expectedSearch:selEp.expected_policy_search||''}),
          h(RewardPanel,{ep:selEp,allActs:epActs,visActs})
        )
      :h('div',{className:'replay-empty'},
        filtEps.length
          ?'Pick Easy / Medium / Hard and Baseline / Inference / Train, then click Random episode.'
          :'No episodes for this filter yet. Re-run baseline or inference logs.'),
    h(PolicyRulebook,{highlighted:highlightedRule}),
    learning.length>0?h(LearnCurve,null):null
  );
}

function HeroSection(){
  return h('section',{className:'hero'},
    h('div',{className:'hero-pill'},'Corporate Compliance RL Environment \xb7 OpenEnv'),
    h('h1',null,'A compliance officer reads policy.',h('br',null),'An RL agent learns it.'),
    h('p',null,
      'Every company processes hundreds of expense claims daily. A rule-based system ',
      'breaks on edge cases. A prompted LLM hallucinates approvals. This environment ',
      'trains an agent to navigate incomplete information\u2014searching for policy, ',
      'requesting missing documents, and resolving tickets across 3 difficulty levels\u2014',
      'using reward signals to learn when to act and when to ask.'
    )
  );
}

function KPIGrid(){
  const bN=M.baseline&&M.baseline.n||0,iN=M.inference&&M.inference.n||0,tN=M.training&&M.training.n||0;
  return h('div',{className:'kpi-grid'},
    h(KPICard,{lbl:'Rule Baseline',val:fmt3(M.baseline&&M.baseline.overall),color:'#378ADD',sub:'n='+bN+' claims'}),
    h(KPICard,{lbl:'Generic LLM',val:fmt3(M.inference&&M.inference.overall),color:'#EF9F27',sub:'n='+iN+' claims'}),
    h(KPICard,{lbl:'SFT + GRPO',val:tN>0?fmt3(M.training&&M.training.overall):'pending',color:'#888780',sub:tN>0?'n='+tN+' episodes':'training ongoing'}),
    h(KPICard,{lbl:'Policy Rules',val:'15',color:'#58d68d',sub:'rules the agent must learn'})
  );
}

function KPICard({lbl,val,color,sub}){
  return h('div',{className:'kpi-card'},
    h('div',{className:'kpi-lbl'},lbl),
    h('div',{className:'kpi-val',style:{color}},val),
    h('div',{className:'kpi-sub'},sub)
  );
}

function TaskCurriculum(){
  const fmt=v=>v!=null?Number(v).toFixed(3):'—';
  return h('div',{className:'curriculum-section'},
    h('div',{className:'sec-hd',style:{marginTop:0}},
      h('h2',null,'Task curriculum: Easy, Medium, Hard'),
      h('p',null,'Each expense claim is labeled with a difficulty. Agents face different information and step budgets — this is why benchmark scores drop from Easy to Hard.')
    ),
    h('div',{className:'curriculum-grid'},
      ...curriculum.map(task=>{
        const b=M.baseline||{},inf=M.inference||{};
        return h('div',{key:task.id,className:'curriculum-card '+task.id},
          h('div',{className:'curr-hdr'},
            h('div',{className:'curr-title'},task.title),
            h('span',{className:'curr-meta'},'max '+task.max_steps+' steps \xb7 '+task.expected_steps)
          ),
          h('div',{className:'curr-lbl'},'What the agent sees'),
          h('div',{className:'curr-txt'},task.agent_sees),
          h('div',{className:'curr-lbl'},'What the agent should do'),
          h('div',{className:'curr-txt'},task.agent_goal),
          h('div',{className:'curr-lbl'},'Why it is still tricky'),
          h('div',{className:'curr-txt'},task.why_hard),
          h('div',{className:'curr-scores'},
            h('span',{className:'curr-score'},'Baseline ',h('b',null,fmt(b[task.id]))),
            h('span',{className:'curr-score'},'LLM ',h('b',null,fmt(inf[task.id])))
          )
        );
      })
    )
  );
}

function NarrativePanel(){
  const reqN=acts.filter(a=>a.action_type==='REQUEST_INFORMATION').length;
  const srcN=acts.filter(a=>a.action_type==='SEARCH_POLICY').length;
  const bm=M.baseline||{},im=M.inference||{};
  const od=(im.overall||0)-(bm.overall||0);
  const hd=(im.hard||0)-(bm.hard||0);
  const s=v=>(v>0?'+':'')+v.toFixed(3);
  return h('div',{className:'narrative'},
    h('div',{className:'eyebrow'},'Why reinforcement learning'),
    h('div',{className:'story-grid'},
      h('div',{className:'story-stat'},h('b',{style:{color:od<0?'#e74c3c':'#58d68d'}},s(od)),h('span',null,'LLM vs baseline overall grader score')),
      h('div',{className:'story-stat'},h('b',{style:{color:hd<0?'#e74c3c':'#58d68d'}},s(hd)),h('span',null,'LLM vs baseline on hard tasks')),
      h('div',{className:'story-stat'},h('b',null,reqN),h('span',null,'RequestInformation actions in logs')),
      h('div',{className:'story-stat'},h('b',null,srcN),h('span',null,'SearchPolicy actions in logs'))
    )
  );
}

function GroupedBarFallback({data,hasT}){
  const colors={baseline:'#378ADD',llm:'#EF9F27',rl:'#888780'};
  const labels={baseline:'Rule Baseline',llm:'Generic LLM',rl:'SFT+GRPO'};
  return h('div',{className:'css-chart'},
    h('div',{className:'css-chart-scale'},'1.0','0.5','0.0'),
    ...data.map((row,i)=>h('div',{key:i,className:'css-chart-group'},
      h('div',{className:'css-bars'},
        ...['baseline','llm','rl'].map(k=>{
          const v=row[k]||0;
          const pending=k==='rl'&&!hasT;
          const ht=pending?2:Math.max(2,Math.round(v*100));
          return h('div',{key:k,className:'css-bar-wrap'},
            h('div',{className:'css-bar-val'},pending?'pending':v.toFixed(3)),
            h('div',{className:'css-bar'+(pending?' pending':''),
              style:{height:ht+'%',background:colors[k]}}),
            h('div',{className:'css-bar-lbl'},labels[k])
          );
        })
      ),
      h('div',{className:'css-group-lbl'},row.d)
    )),
    h('div',{className:'css-legend'},
      ...['baseline','llm','rl'].map(k=>h('span',{key:k},
        h('i',{className:'css-legend-dot',style:{background:colors[k]}}),labels[k]))
    )
  );
}

function ScoreChart(){
  const bm=M.baseline||{},im=M.inference||{},tm=M.training||{};
  const hasT=(tm.n||0)>0;
  const data=[
    {d:'Easy',   baseline:bm.easy||0,  llm:im.easy||0,  rl:hasT?tm.easy||0:0},
    {d:'Medium', baseline:bm.medium||0,llm:im.medium||0, rl:hasT?tm.medium||0:0},
    {d:'Hard',   baseline:bm.hard||0,  llm:im.hard||0,  rl:hasT?tm.hard||0:0},
    {d:'Overall',baseline:bm.overall||0,llm:im.overall||0,rl:hasT?tm.overall||0:0},
  ];
  const tt={background:'#1a2332',border:'1px solid rgba(255,255,255,0.12)',borderRadius:'10px',color:'#edf2f7'};
  if(!hasRecharts){
    return h('div',{className:'chart-section'},
      h('h2',null,'Mean Grader Score Comparison'),
      h('p',{className:'sub'},'Chart library failed to load \u2014 refresh the page.'),
      h(GroupedBarFallback,{data,hasT})
    );
  }
  return h('div',{className:'chart-section'},
    h('h2',null,'Mean Grader Score Comparison'),
    h('p',{className:'sub'},'Side-by-side bars \u2014 same metric across all three methods. SFT+GRPO bars are at 0 (pending) until GRPO training completes.'),
    h(RC.ResponsiveContainer,{width:'100%',height:270},
      h(RC.BarChart,{data,barCategoryGap:'24%',barGap:4,margin:{top:8,right:16,left:0,bottom:0}},
        h(RC.CartesianGrid,{strokeDasharray:'3 3',stroke:'rgba(255,255,255,0.07)',vertical:false}),
        h(RC.XAxis,{dataKey:'d',stroke:'#8ea0b5',tick:{fill:'#8ea0b5',fontSize:13}}),
        h(RC.YAxis,{domain:[0,1],stroke:'#8ea0b5',tick:{fill:'#8ea0b5',fontSize:12},tickFormatter:v=>v.toFixed(1)}),
        h(RC.Tooltip,{contentStyle:tt,formatter:v=>typeof v==='number'?v.toFixed(3):v}),
        h(RC.Legend,{wrapperStyle:{color:'#aab7c7',fontSize:'13px',paddingTop:'6px'}}),
        h(RC.Bar,{dataKey:'baseline',name:'Rule Baseline',fill:'#378ADD',radius:[4,4,0,0],maxBarSize:46}),
        h(RC.Bar,{dataKey:'llm',name:'Generic LLM',fill:'#EF9F27',radius:[4,4,0,0],maxBarSize:46}),
        h(RC.Bar,{dataKey:'rl',name:'SFT+GRPO',fill:'#888780',radius:[4,4,0,0],maxBarSize:46,opacity:hasT?1:0.32})
      )
    )
  );
}

function TicketPanel({ep}){
  const rPct=Math.min(100,Math.round((ep.risk_score||0)*100));
  const gt=ep.ground_truth||'';
  return h('div',{className:'col-panel'},
    h('div',{className:'col-title'},'Compliance Ticket'),
    h('div',{className:'ticket-hdr'},
      h('span',{className:'ticket-id'},ep.claim_id||ep.episode_id),
      h('span',{className:'diff-badge diff-'+ep.task_id},ep.task_id)
    ),
    ep.employee_name?h('div',{className:'t-field'},
      h('div',{className:'f-lbl'},'Employee'),
      h('div',{className:'f-val'},ep.employee_name),
      h('div',{className:'f-sub'},(ep.employee_role||'')+' \xb7 '+(ep.employee_level||''))
    ):null,
    ep.amount>0?h('div',{className:'t-field'},
      h('div',{className:'f-lbl'},'Claim Amount'),
      h('div',{className:'f-val amount-big'},fmtRs(ep.amount))
    ):null,
    ep.description?h('div',{className:'t-field'},
      h('div',{className:'f-lbl'},'Description'),
      h('div',{className:'f-val',style:{fontSize:'13px',lineHeight:'1.55'}},ep.description)
    ):null,
    h('div',{className:'badges-row'},
      ep.has_receipt!==undefined?h('span',{className:ep.has_receipt?'receipt-y':'receipt-n'},
        h('span',{className:'status-dot '+(ep.has_receipt?'ok':'no')}),
        ep.has_receipt?'Receipt attached':'No receipt'):null,
      ep.missing_document?h('span',{className:'miss-tag'},
        h('span',{className:'status-dot warn'}),'Missing: '+ep.missing_document):null
    ),
    ep.expected_policy_search?h('div',{className:'t-field policy-expected-box'},
      h('div',{className:'f-lbl'},'Policy to search'),
      h('div',{className:'policy-pill'},ep.expected_policy_search),
      ep.expected_policy_summary?h('div',{className:'policy-summary'},ep.expected_policy_summary):null,
      ep.task_id&&ep.task_id!=='easy'?h('div',{className:'policy-note'},
        'On '+ep.task_id+' tasks the agent does not see this keyword until SearchPolicy runs.'):null
    ):null,
    gt?h('div',{className:'t-field'},
      h('div',{className:'f-lbl'},'Ground truth'),
      h('span',{className:'gt-chip gt-'+gt},gt)
    ):null
  );
}

function AgentPanel({ep,allActs,visActs,playing,onPlay,expectedSearch}){
  return h('div',{className:'col-panel'},
    h('div',{className:'col-title-row'},
      h('div',{className:'col-title'},'Agent Reasoning'),
      h('button',{className:'play-btn',onClick:onPlay,disabled:playing},
        h('span',{className:'play-icon'+(playing?' playing':'')}),
        playing?'Playing\u2026':'Play')
    ),
    h('div',{className:'timeline'},
      ...allActs.map((a,i)=>h(StepCard,{key:i,action:a,vis:i<visActs.length,expectedSearch})),
      allActs.length===0?h('div',{className:'tl-empty'},'No step actions parsed for this episode.'):null
    )
  );
}

function StepCard({action,vis,expectedSearch}){
  if(!vis)return null;
  const t=action.action_type||'';
  const r=action.reward||0;
  const rc=r>0.05?'r-pos':r<-0.05?'r-neg':'r-zero';
  const isSearch=t==='SEARCH_POLICY';
  return h('div',{className:'step-card',style:{borderLeftColor:aBorder(t)}},
    h('div',{className:'step-hdr'},
      h('div',{className:'step-dot'},action.step),
      h('div',{className:'step-lbl'},h(ActionIcon,{type:t}),aLabel(t)),
      h('div',{className:'r-badge '+rc},(r>0?'+':'')+r.toFixed(2))
    ),
    isSearch?h('div',{className:'step-detail policy-search'},
      h('strong',null,'Searched policy: '),
      action.query?'\u201c'+action.query+'\u201d':'(not logged \u2014 re-run agent to capture query)'
    ):null,
    isSearch&&action.matched_policy?h('div',{className:'step-detail policy-hit'},action.matched_policy):null,
    isSearch&&expectedSearch?h('div',{className:'step-detail policy-expected'},
      'Should search: \u201c'+expectedSearch+'\u201d',
      action.search_relevant===true?h('span',{className:'tag tag-ok'},'on-topic'):null,
      action.search_relevant===false?h('span',{className:'tag tag-miss'},'off-topic'):null,
      action.query&&!action.search_relevant&&action.search_relevant!==false
        ?h('span',{className:'tag tag-miss'},'relevance unknown'):null
    ):null,
    !isSearch&&action.query?h('div',{className:'step-detail'},'Query: \u201c'+action.query+'\u201d'):null,
    action.decision?h('div',{className:'dec-chip dec-'+action.decision},action.decision):null,
    action.message?h('div',{className:'step-detail'},'Requested: \u201c'+action.message+'\u201d'):null,
    action.reason?h('div',{className:'step-detail'},'Reason: '+action.reason):null
  );
}

function RewardPanel({ep,allActs,visActs}){
  const score=ep&&ep.grader_score||0;
  const mkCum=arr=>arr.map((a,i)=>({
    s:''+a.step,
    cum:parseFloat(arr.slice(0,i+1).reduce((acc,x)=>acc+(x.reward||0),0).toFixed(3))
  }));
  const visCum=visActs.length?mkCum(allActs.slice(0,visActs.length)):mkCum(allActs);
  const tt={background:'#1a2332',border:'1px solid rgba(255,255,255,.1)',borderRadius:'8px',color:'#edf2f7'};
  return h('div',{className:'col-panel'},
    h('div',{className:'col-title'},'Reward Signal'),
    RC.BarChart?h(RC.ResponsiveContainer,{width:'100%',height:155},
      h(RC.BarChart,{data:visCum,barSize:18,margin:{top:0,right:6,left:-22,bottom:0}},
        h(RC.CartesianGrid,{strokeDasharray:'3 3',stroke:'rgba(255,255,255,.06)',horizontal:false}),
        h(RC.XAxis,{dataKey:'s',tick:{fill:'#8ea0b5',fontSize:11}}),
        h(RC.YAxis,{tick:{fill:'#8ea0b5',fontSize:10},tickFormatter:v=>v.toFixed(1)}),
        h(RC.Tooltip,{contentStyle:tt,formatter:v=>v.toFixed(3),labelFormatter:v=>'Step '+v}),
        h(RC.ReferenceLine,{y:0,stroke:'rgba(255,255,255,.2)',strokeDasharray:'4 2'}),
        h(RC.Bar,{dataKey:'cum',name:'Cumulative reward',fill:'#5dade2',radius:[3,3,0,0]})
      )
    ):null,
    h('div',{className:'grade-box'},
      h('div',{className:'grade-lbl'},'Final grader score'),
      h('div',{className:'grade-score',style:{color:scoreColor(score)}},score.toFixed(3)),
      h('div',{className:'steps-txt'},'Steps: '+(ep&&ep.steps||0)+' / 8 max \xb7 Total reward: '+(ep&&ep.total_reward||0).toFixed(2))
    )
  );
}

function PolicyRulebook({highlighted}){
  const [open,setOpen]=useState(false);
  const words=highlighted?highlighted.toLowerCase().split(' ').slice(0,3):[];
  return h('div',{className:'rulebook'},
    h('button',{className:'rb-toggle',onClick:()=>setOpen(o=>!o)},
      h('span',null,'Policy Rulebook \u2014 '+rules.length+' rules the agent must learn'),
      h('span',{className:'rb-chevron'+(open?' open':'')})
    ),
    open?h('div',{className:'rb-body'},
      h('p',{className:'rb-sub'},'When the agent uses SearchPolicy it is consulting one of these rules. Rules highlight during episode replay when a SearchPolicy action is playing.'),
      h('ol',{className:'rules-list'},
        ...rules.map((rule,i)=>{
          const hit=words.length>0&&words.every(w=>rule.toLowerCase().includes(w));
          return h('li',{key:i,className:'rule-item'+(hit?' lit':'')},rule);
        })
      )
    ):null
  );
}

function LearnCurve(){
  const tt={background:'#1a2332',border:'1px solid rgba(255,255,255,.12)',borderRadius:'10px',color:'#edf2f7'};
  if(!RC.LineChart){
    return h('div',{className:'chart-section'},h('h2',null,'Learning Curve'),h('p',{className:'sub'},learning.length+' training episodes loaded.'));
  }
  return h('div',{className:'chart-section'},
    h('h2',null,'Learning Curve'),
    h('p',{className:'sub'},'Training grader score over GRPO episodes (training/logs/episodes.jsonl).'),
    h(RC.ResponsiveContainer,{width:'100%',height:200},
      h(RC.LineChart,{data:learning,margin:{top:8,right:16,left:0,bottom:0}},
        h(RC.CartesianGrid,{strokeDasharray:'3 3',stroke:'rgba(255,255,255,.07)',vertical:false}),
        h(RC.XAxis,{dataKey:'episode',stroke:'#8ea0b5',tick:{fill:'#8ea0b5',fontSize:12}}),
        h(RC.YAxis,{domain:[0,1],stroke:'#8ea0b5',tick:{fill:'#8ea0b5',fontSize:12},tickFormatter:v=>v.toFixed(1)}),
        h(RC.Tooltip,{contentStyle:tt,formatter:v=>v.toFixed(3)}),
        h(RC.Line,{type:'monotone',dataKey:'grader_score',name:'Grader Score',stroke:'#58d68d',dot:false,strokeWidth:2})
      )
    )
  );
}

try{ReactDOM.createRoot(document.getElementById('root')).render(h(App,null));}
catch(e){document.getElementById('root').innerHTML='<div style="padding:40px;color:#e74c3c;font-family:sans-serif">Dashboard error: '+e.message+'</div>';}
})();
</script>
</body>
</html>
"""

# ── Cache ──────────────────────────────────────────────────────────────────────
_cache: tuple[str, float] = ("", 0.0)
_CACHE_TTL = 30.0


def _render_dashboard() -> str:
    """Generate (and cache for 30 s) the full React HTML page."""
    global _cache
    now = time.monotonic()
    if _cache[0] and now - _cache[1] < _CACHE_TTL:
        return _cache[0]
    payload = _load_dashboard_data()
    data_json = json.dumps(payload, ensure_ascii=True, default=str, separators=(",", ":"))
    # Prevent </script> inside JSON from breaking the enclosing <script> tag
    data_json = data_json.replace("</", "<\\/")
    html = _REACT_TEMPLATE.replace("__DATA_PLACEHOLDER__", data_json)
    _cache = (html, now)
    return html


# ── Gradio wrapper (kept at /demo for backward compat) ─────────────────────────


def build_demo() -> gr.Blocks:
    """Minimal Gradio wrapper at /demo — full interactive dashboard is at /dashboard."""
    with gr.Blocks(title="Corporate Compliance RL Dashboard") as demo:
        gr.HTML("""
        <div style="padding:48px 40px;text-align:center;font-family:-apple-system,sans-serif;background:#0f1117;color:#edf2f7;min-height:300px;border-radius:20px">
          <div style="font-size:13px;text-transform:uppercase;letter-spacing:.12em;color:#8ea0b5;margin-bottom:14px">Corporate Compliance RL Environment</div>
          <h1 style="font-size:28px;font-weight:860;margin-bottom:16px">Full interactive dashboard</h1>
          <p style="font-size:15px;color:#c7d2de;max-width:560px;margin:0 auto 28px;line-height:1.7">
            The React-powered dashboard with grouped bar charts, episode replay and policy rulebook is served at
            <code style="background:rgba(255,255,255,.1);padding:2px 8px;border-radius:6px">/dashboard</code>.
          </p>
          <a href="/dashboard"
             style="display:inline-block;padding:12px 32px;border-radius:999px;background:#378ADD;color:#fff;font-size:15px;font-weight:700;text-decoration:none">
            Open Dashboard &rarr;
          </a>
        </div>
        """)
    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
