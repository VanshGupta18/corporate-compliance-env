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

from app.paths import (
    ARTIFACT_ROOT,
    BASELINE_LOG,
    BASELINE_RESULTS,
    CLAIMS_DATA,
    INFERENCE_LOG,
    INFERENCE_RESULTS,
    TEST_SPLIT,
    TRAINING_EPISODES,
    TRAINING_EPISODES_LEGACY,
    TRAINING_EPISODES_ROOT,
    TRAINING_LOG,
    TRAINING_RESULTS,
)
from app.policy_snippets import POLICY_SNIPPETS, match_policy_snippet

_CLAIM_RUN_RE = re.compile(
    r"Running (?:inference|baseline|training) for claim\s+([A-Z0-9-]+)\s+\((easy|medium|hard)\)\.\.\.",
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
            fields[key] = _normalize_decision(val) if key == "decision" else str(val)
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


def _normalize_decision(decision: Any) -> str:
    """Strip TicketDecision.APPROVE-style enums to Approve/Reject/Escalate."""
    if decision is None:
        return ""
    text = str(decision).strip()
    if not text:
        return ""
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    key = text.upper()
    return {
        "APPROVE": "Approve",
        "REJECT": "Reject",
        "ESCALATE": "Escalate",
    }.get(key, text if text[:1].isupper() else text.capitalize())


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
    claim_queue: List[Dict[str, Any]] = list(_load_baseline_claim_order())

    for raw in text.splitlines():
        line = raw.strip()
        cm = _CLAIM_RUN_RE.match(line)
        if cm:
            current_claim, current_task = cm.group(1), cm.group(2).lower()
            continue

        sm = re.match(r"\[START\]\s+task=(EASY|MEDIUM|HARD)\s+.*", line)
        if sm:
            if current is not None:
                episodes.append(current)
            episode_idx += 1
            task_from_start = sm.group(1).lower()
            if not current_claim and claim_queue:
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
            if extras.get("decision"):
                extras["decision"] = _normalize_decision(extras["decision"])
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
        end_legacy = None
        if not end_m:
            end_legacy = re.match(r"\[END\]\s+steps=(\d+)\s+grader_score=([0-9.]+)", line)
        if end_m or end_legacy:
            if end_m:
                current["success"] = end_m.group(1) == "true" if end_m.group(1) else False
                current["steps"] = int(end_m.group(2))
                current["grader_score"] = float(end_m.group(3))
            else:
                current["success"] = False
                current["steps"] = int(end_legacy.group(1))
                current["grader_score"] = float(end_legacy.group(2))
            episodes.append(current)
            current = None
            current_claim = None
            current_task = None

    return episodes, actions


def _training_jsonl_candidate_paths() -> List[Path]:
    """Resolve episodes.jsonl from results/, then legacy Colab/root paths."""
    candidates = [
        TRAINING_EPISODES,
        TRAINING_EPISODES_ROOT,
        TRAINING_EPISODES_LEGACY,
    ]
    seen: set[str] = set()
    ordered: List[Path] = []
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    return ordered


def _read_training_jsonl(last_n: int = 800) -> List[Dict[str, Any]]:
    """Prefer the JSONL with claim_id fields (root Colab copy over smoke logs)."""
    best: List[Dict[str, Any]] = []
    best_key = (-1, -1)
    for path in _training_jsonl_candidate_paths():
        if not path.exists():
            continue
        rows = _read_jsonl(path, last_n=last_n)
        if not rows:
            continue
        with_claim = sum(1 for r in rows if r.get("claim_id"))
        key = (with_claim, len(rows))
        if key > best_key:
            best_key = key
            best = rows
    return best


_EPISODE_ID_IDX_RE = re.compile(r"^(?:training|baseline|inference)-(\d+)$")


def _episode_index_from_id(episode_id: str) -> int:
    match = _EPISODE_ID_IDX_RE.match(episode_id or "")
    return int(match.group(1)) if match else 0


def _jsonl_row_for_episode(
    ep: Dict[str, Any],
    by_claim: Dict[str, Dict[str, Any]],
    by_index: Dict[int, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    claim_id = str(ep.get("claim_id") or "")
    if claim_id and claim_id in by_claim:
        return by_claim[claim_id]
    idx = _episode_index_from_id(str(ep.get("episode_id", "")))
    if idx and idx in by_index:
        return by_index[idx]
    return None


def _metrics_from_episodes(episodes: List[Dict[str, Any]], method: str) -> MethodMetrics:
    if not episodes:
        return MethodMetrics(method=method, overall=0.0, total=0, by_task={}, by_task_total={})
    by_task: Dict[str, List[float]] = {t: [] for t in TASK_ORDER}
    for ep in episodes:
        task = str(ep.get("task_id", "")).lower()
        if task in by_task:
            by_task[task].append(float(ep.get("grader_score", 0.0) or 0.0))
    return MethodMetrics(
        method=method,
        overall=mean(float(ep.get("grader_score", 0.0) or 0.0) for ep in episodes),
        total=len(episodes),
        by_task={t: (mean(v) if v else 0.0) for t, v in by_task.items()},
        by_task_total={t: len(v) for t, v in by_task.items()},
    )


def _attach_claim_ids_from_order(episodes: List[Dict[str, Any]]) -> None:
    """Last resort: map training-NNNN to test-split claim order when logs omit claim lines."""
    claim_order = _load_baseline_claim_order()
    if not claim_order:
        return
    for ep in episodes:
        if ep.get("claim_id"):
            continue
        idx = _episode_index_from_id(str(ep.get("episode_id", "")))
        if not idx or idx > len(claim_order):
            continue
        claim = claim_order[idx - 1]
        ep["claim_id"] = str(claim.get("id", ""))
        ep["task_id"] = str(claim.get("task_difficulty", ep.get("task_id", "easy"))).lower()


def _merge_training_jsonl_meta(
    episodes: List[Dict[str, Any]], rows: List[Dict[str, Any]]
) -> None:
    """Align log-parsed episodes with eval JSONL (claim_id, scores) by claim or episode index."""
    by_claim = {
        str(r.get("claim_id")): r for r in rows if r.get("claim_id")
    }
    by_index = {
        int(r.get("episode_index", 0) or 0): r
        for r in rows
        if int(r.get("episode_index", 0) or 0) > 0
    }
    for ep in episodes:
        row = _jsonl_row_for_episode(ep, by_claim, by_index)
        if row:
            if not ep.get("claim_id") and row.get("claim_id"):
                ep["claim_id"] = row["claim_id"]
            if row.get("task_id"):
                ep["task_id"] = str(row["task_id"]).lower()
            if row.get("rule_keyword") and not ep.get("rule_keyword"):
                ep["rule_keyword"] = row["rule_keyword"]
        if row and row.get("success") is not None:
            ep["success"] = bool(row["success"])
        if row:
            score = row.get("grader_score", row.get("score"))
            if score is not None:
                ep["grader_score"] = float(score)
            total_reward = row.get("total_reward")
            if total_reward is not None:
                ep["total_reward"] = float(total_reward)
    _attach_claim_ids_from_order(episodes)


def _training_frames(rows: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
    """Fallback when training_run.log is missing — mirror log-shaped episode/action rows."""
    episodes: List[Dict[str, Any]] = []
    actions: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        score = row.get("grader_score", row.get("score"))
        task = str(row.get("task_id", "easy")).lower()
        ep_id = f"training-{idx:04d}"
        steps = int(row.get("steps", 0) or 0)
        total_reward = float(row.get("total_reward", row.get("reward", 0.0)) or 0.0)
        episodes.append({
            "method": "training",
            "episode_id": ep_id,
            "claim_id": row.get("claim_id", ""),
            "task_id": task,
            "steps": steps,
            "total_reward": total_reward,
            "grader_score": float(score or 0.0),
            "success": bool(row.get("success", False)),
        })
        rule_kw = str(row.get("rule_keyword") or "")
        trajectory = row.get("trajectory") or []
        history = row.get("actions_history") or row.get("actions") or []
        step_rows: List[Dict[str, Any]] = []
        if isinstance(trajectory, list) and trajectory:
            step_rows = [s for s in trajectory if isinstance(s, dict)]
        elif isinstance(history, list) and history:
            for si, action in enumerate(history, start=1):
                if not isinstance(action, dict):
                    continue
                step_rows.append(
                    {
                        "step": si,
                        "action_type": action.get("action_type", ""),
                        "decision": action.get("decision"),
                        "query": action.get("query"),
                        "message": action.get("message"),
                        "reason": action.get("reason"),
                        "reward": 0.0,
                        "done": si == steps,
                    }
                )
        for step_row in step_rows:
            step_num = int(step_row.get("step", 0) or 0)
            p = {
                "action_type": step_row.get("action_type", ""),
                "decision": step_row.get("decision"),
                "query": step_row.get("query"),
                "message": step_row.get("message"),
                "reason": step_row.get("reason"),
                "_rule_keyword": rule_kw,
            }
            fields = _action_fields_from_dict(p)
            actions.append({
                "method": "training",
                "episode_id": ep_id,
                "task_id": task,
                "step": step_num,
                "reward": float(step_row.get("reward", 0.0) or 0.0),
                "done": bool(step_row.get("done", False)),
                **fields,
            })
    return episodes, actions


def _load_training_replay(
    training_metrics_file: MethodMetrics,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], MethodMetrics, List[Dict[str, Any]]]:
    """
    Training replay uses training_run.log (same parser as baseline/inference).
    episodes.jsonl supplies optional score sync by claim_id.
    KPI cards prefer training_results.json when present.
    """
    jsonl_rows = _read_training_jsonl()
    t_log_eps, t_log_acts = _parse_episode_log(TRAINING_LOG, "training")

    if t_log_eps:
        _merge_training_jsonl_meta(t_log_eps, jsonl_rows)
        t_eps, t_acts = t_log_eps, t_log_acts
    else:
        t_eps, t_acts = _training_frames(jsonl_rows)
        _attach_claim_ids_from_order(t_eps)

    if training_metrics_file.total > 0:
        training = training_metrics_file
    elif jsonl_rows:
        training = _training_metrics(jsonl_rows)
    else:
        training = _metrics_from_episodes(t_eps, "training")

    return t_eps, t_acts, training, jsonl_rows


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


def _tool_action_counts(actions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Count SearchPolicy / RequestInformation steps in eval logs by method."""

    def _counts(method: str) -> Dict[str, int]:
        subset = [a for a in actions if a.get("method") == method]
        return {
            "request_information": sum(
                1 for a in subset if a.get("action_type") == "REQUEST_INFORMATION"
            ),
            "search_policy": sum(
                1 for a in subset if a.get("action_type") == "SEARCH_POLICY"
            ),
        }

    return {
        "before_training": {
            "label": "Inference (before training)",
            **_counts("inference"),
        },
        "after_training": {
            "label": "Trained LLM (after training)",
            **_counts("training"),
        },
    }


def _build_rl_story(
    inference: Dict[str, Any],
    training: Dict[str, Any],
    tool_actions: Dict[str, Any],
) -> Dict[str, Any]:
    """Derived deltas for dashboard copy: trained vs generic LLM (before training)."""
    if not training.get("n"):
        return {"has_training": False}

    def _f(key: str, src: Dict[str, Any]) -> float:
        val = src.get(key)
        return float(val) if val is not None else 0.0

    im_medium = _f("medium", inference)
    im_hard = _f("hard", inference)
    im_overall = _f("overall", inference)
    tr_medium = _f("medium", training)
    tr_hard = _f("hard", training)
    tr_overall = _f("overall", training)

    im_complex = (im_medium + im_hard) / 2.0
    tr_complex = (tr_medium + tr_hard) / 2.0
    complex_gain = tr_complex - im_complex
    complex_pct = (complex_gain / im_complex * 100.0) if im_complex > 0 else 0.0

    before = tool_actions.get("before_training", {})
    after = tool_actions.get("after_training", {})
    req_before = int(before.get("request_information", 0) or 0)
    req_after = int(after.get("request_information", 0) or 0)
    src_before = int(before.get("search_policy", 0) or 0)
    src_after = int(after.get("search_policy", 0) or 0)

    return {
        "has_training": True,
        "headline": "Limited training, measurable lift where compliance is hardest",
        "summary": (
            "Overall score stays nearly flat because easy tickets are already mostly solved. "
            "The useful signal is on Medium and Hard claims—policy search, missing documents, "
            "and multi-step judgment."
        ),
        "why_it_matters": (
            "With a small SFT + GRPO run, the model did not only memorize easy approvals. "
            "It shifted toward realistic compliance work: searching policy more often, "
            "requesting missing documents more often, and scoring higher on Medium and Hard "
            "tickets. That makes this environment feasible for training agents under incomplete "
            "information—not just benchmarking a prompted LLM."
        ),
        "medium_gain": round(tr_medium - im_medium, 3),
        "hard_gain": round(tr_hard - im_hard, 3),
        "complex_task_gain": round(complex_gain, 3),
        "complex_task_gain_pct": round(complex_pct, 1),
        "overall_delta": round(tr_overall - im_overall, 3),
        "inference_overall": round(im_overall, 3),
        "trained_overall": round(tr_overall, 3),
        "request_information_before": req_before,
        "request_information_after": req_after,
        "search_policy_before": src_before,
        "search_policy_after": src_after,
        "before_label": before.get("label", "Inference (before training)"),
        "after_label": after.get("label", "Trained LLM (after training)"),
    }


def _load_dashboard_data() -> Dict[str, Any]:
    baseline = _load_metrics(BASELINE_RESULTS, "baseline")
    inference = _load_metrics(INFERENCE_RESULTS, "inference")
    training_metrics_file = _load_metrics(TRAINING_RESULTS, "training")
    b_eps, b_acts = _parse_episode_log(BASELINE_LOG, "baseline")
    i_eps, i_acts = _parse_episode_log(INFERENCE_LOG, "inference")
    t_eps, t_acts, training, _ = _load_training_replay(training_metrics_file)

    claims_idx = _load_claims_index()
    all_eps = _enrich_episodes(b_eps + i_eps + t_eps, claims_idx)
    all_acts = b_acts + i_acts + t_acts
    for action in all_acts:
        action["action_type"] = _normalize_action(str(action.get("action_type", "")))
        if action.get("decision") is not None:
            action["decision"] = _normalize_decision(action["decision"])
    _enrich_action_policies(all_eps, all_acts)
    tool_actions = _tool_action_counts(all_acts)
    inf_metrics = {
        "overall": inference.overall,
        "easy": inference.by_task.get("easy", 0.0),
        "medium": inference.by_task.get("medium", 0.0),
        "hard": inference.by_task.get("hard", 0.0),
        "n": inference.total,
    }
    tr_metrics = {
        "overall": training.overall if training.total > 0 else None,
        "easy": training.by_task.get("easy") if training.total > 0 else None,
        "medium": training.by_task.get("medium") if training.total > 0 else None,
        "hard": training.by_task.get("hard") if training.total > 0 else None,
        "n": training.total,
    }
    rl_story = _build_rl_story(inf_metrics, tr_metrics, tool_actions)

    return {
        "artifact_root": str(ARTIFACT_ROOT),
        "metrics": {
            "baseline": {
                "overall": baseline.overall,
                "easy": baseline.by_task.get("easy", 0.0),
                "medium": baseline.by_task.get("medium", 0.0),
                "hard": baseline.by_task.get("hard", 0.0),
                "n": baseline.total,
            },
            "inference": inf_metrics,
            "training": tr_metrics,
        },
        "episodes": all_eps,
        "actions": all_acts,
        "tool_actions": tool_actions,
        "rl_story": rl_story,
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
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"/>
<title>Corporate Compliance Audit Console</title>
<style>
:root{
  --bg:#f8fafc;--surface:#ffffff;--surface2:#f1f5f9;
  --border:#e2e8f0;--divider:#e2e8f0;--text:#0f172a;--body:#334155;--muted:#64748b;--faint:#94a3b8;
  --blue:#2563eb;--green:#16a34a;--red:#dc2626;--amber:#d97706;
  --blue-soft:#eff6ff;--green-soft:#f0fdf4;--red-soft:#fef2f2;--amber-soft:#fffbeb;
  --shadow:0 1px 4px rgba(0,0,0,0.07);
  --shadow-md:0 4px 16px rgba(0,0,0,0.08),0 1px 4px rgba(0,0,0,0.04);
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{min-height:100%;background:var(--bg);color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Inter',sans-serif;line-height:1.55}
button{cursor:pointer;font-family:inherit}
a{color:var(--blue);text-decoration:none}
.dash{max-width:1280px;margin:0 auto;padding:28px 24px 80px}

/* ── Section rhythm: most sections are open, no box ──────────────── */
.page-section{padding:40px 0;border-top:1px solid var(--divider)}
.page-section:first-child{border-top:none;padding-top:0}
.section-eyebrow{font-size:10px;font-weight:800;letter-spacing:.16em;text-transform:uppercase;color:var(--faint);margin-bottom:10px}
.section-heading{font-size:22px;font-weight:900;color:var(--text);margin-bottom:8px;letter-spacing:-.015em}
.section-lead{font-size:14px;color:var(--body);line-height:1.75;max-width:780px}

/* ── Hero ─────────────────────────────────────────────────────────── */
.hero{padding:52px 56px;border-radius:24px;margin-bottom:0;
  background:linear-gradient(135deg,#ffffff 60%,#eff6ff 100%);
  border:1px solid var(--border);position:relative;overflow:hidden}
.hero::before{content:"";position:absolute;right:-40px;top:-40px;
  width:220px;height:220px;border-radius:50%;
  background:radial-gradient(circle,rgba(37,99,235,.07) 0%,transparent 70%)}
.hero-badge{display:inline-flex;align-items:center;gap:7px;padding:5px 14px;border-radius:999px;
  background:#dbeafe;color:#1e40af;font-size:11px;font-weight:700;
  letter-spacing:.06em;margin-bottom:20px;text-transform:uppercase}
.hero-badge-dot{width:7px;height:7px;border-radius:50%;background:#2563eb}
.hero-layout{display:grid;grid-template-columns:1fr auto;gap:40px;align-items:start}
@media(max-width:860px){.hero-layout{grid-template-columns:1fr}}
.hero h1{font-size:40px;font-weight:900;line-height:1.1;margin-bottom:16px;letter-spacing:-.025em;color:#0f172a}
.hero p{font-size:15px;color:#475569;max-width:640px;line-height:1.75}
.hero-toolkit{background:#ffffff;border:1px solid var(--border);border-radius:16px;padding:20px 22px;min-width:240px}
.hero-toolkit-title{font-size:10px;font-weight:800;letter-spacing:.14em;text-transform:uppercase;color:var(--faint);margin-bottom:14px}
.toolkit-action{display:flex;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid var(--divider)}
.toolkit-action:last-child{border-bottom:none;padding-bottom:0}
.toolkit-icon{width:32px;height:32px;border-radius:8px;display:grid;place-items:center;font-size:10px;font-weight:900;flex-shrink:0}
.toolkit-icon.search{background:var(--blue-soft);color:var(--blue)}
.toolkit-icon.request{background:var(--amber-soft);color:var(--amber)}
.toolkit-icon.resolve{background:var(--green-soft);color:var(--green)}
.toolkit-action-lbl{font-size:12px;font-weight:700;color:var(--text)}
.toolkit-action-desc{font-size:11px;color:var(--muted);line-height:1.4}

/* ── Paradigm comparison strip ────────────────────────────────────── */
.paradigm-strip{display:grid;grid-template-columns:1fr 28px 1fr 28px 1fr;gap:0;align-items:stretch;margin-top:24px}
@media(max-width:860px){.paradigm-strip{grid-template-columns:1fr;gap:12px}
  .paradigm-arrow{display:none}}
.paradigm-item{padding:22px 20px;border-radius:16px;border:1px solid var(--border);background:var(--surface)}
.paradigm-item.highlight{border-color:#2563eb;box-shadow:0 0 0 3px rgba(37,99,235,.08)}
.paradigm-arrow{display:flex;align-items:center;justify-content:center;color:var(--faint);font-size:18px;padding:0 4px}
.paradigm-tag{display:inline-block;font-size:10px;font-weight:800;letter-spacing:.1em;text-transform:uppercase;
  padding:3px 10px;border-radius:999px;margin-bottom:12px}
.paradigm-tag.rule{background:#eff6ff;color:var(--blue)}
.paradigm-tag.llm{background:#fff7ed;color:#c2410c}
.paradigm-tag.rl{background:#f0fdf4;color:#15803d}
.paradigm-name{font-size:16px;font-weight:800;color:var(--text);margin-bottom:6px}
.paradigm-desc{font-size:12px;color:var(--muted);line-height:1.6;margin-bottom:14px}
.paradigm-score-row{display:flex;align-items:baseline;gap:8px}
.paradigm-score{font-size:32px;font-weight:900;line-height:1}
.paradigm-score-lbl{font-size:11px;color:var(--faint);font-weight:600}
.paradigm-meta{font-size:11px;color:var(--faint);margin-top:6px}
.paradigm-limits{font-size:11px;color:var(--muted);margin-top:10px;padding-top:10px;border-top:1px solid var(--divider);line-height:1.5}

/* ── Audit lifecycle band ─────────────────────────────────────────── */
.lifecycle-band{margin-top:24px;padding:28px 28px;background:var(--surface);border:1px solid var(--border);border-radius:18px}
.lifecycle-steps{display:flex;align-items:stretch;gap:0}
@media(max-width:900px){.lifecycle-steps{flex-direction:column;gap:8px}
  .lc-arrow{transform:rotate(90deg)}}
.lc-step{flex:1;display:flex;flex-direction:column;gap:6px;padding:18px 16px;border-radius:12px}
.lc-step.claim{background:var(--blue-soft)}
.lc-step.search{background:rgba(37,99,235,.06)}
.lc-step.request{background:var(--amber-soft)}
.lc-step.resolve{background:var(--green-soft)}
.lc-step.grade{background:#f5f3ff;border:1px dashed #c4b5fd}
.lc-num{width:24px;height:24px;border-radius:50%;background:rgba(0,0,0,.08);
  font-size:10px;font-weight:900;display:grid;place-items:center;margin-bottom:4px;color:var(--text)}
.lc-title{font-size:13px;font-weight:800;color:var(--text)}
.lc-body{font-size:11px;color:var(--muted);line-height:1.55}
.lc-badge{display:inline-block;margin-top:6px;padding:3px 9px;border-radius:999px;font-size:10px;font-weight:800}
.lc-badge.search{background:rgba(37,99,235,.15);color:var(--blue)}
.lc-badge.request{background:rgba(217,119,6,.15);color:var(--amber)}
.lc-badge.resolve{background:rgba(22,163,74,.15);color:var(--green)}
.lc-arrow{display:flex;align-items:center;padding:0 6px;color:var(--faint);font-size:16px;flex-shrink:0}

/* ── RL outcome narrative (open, no box) ──────────────────────────── */
.narrative-layout{display:grid;grid-template-columns:1fr 1fr;gap:36px;margin-top:20px;align-items:start}
@media(max-width:860px){.narrative-layout{grid-template-columns:1fr}}
.narrative-text .narrative-title{font-size:17px;font-weight:800;color:var(--text);margin-bottom:10px;line-height:1.4}
.narrative-text .narrative-lead{font-size:14px;color:var(--body);line-height:1.75;margin-bottom:12px}
.narrative-text .narrative-foot{font-size:13px;color:var(--muted);line-height:1.7}
.narrative-text .narrative-note{font-size:12px;color:var(--faint);margin-top:10px;font-style:italic}
.outcome-stats{display:flex;flex-direction:column;gap:10px}
.outcome-row{display:flex;align-items:center;justify-content:space-between;gap:12px;
  padding:12px 14px;border-radius:10px;background:var(--surface2)}
.outcome-row-val{font-size:20px;font-weight:900;white-space:nowrap}
.outcome-row-lbl{font-size:12px;color:var(--muted);line-height:1.4}
.outcome-row-sub{font-size:11px;color:var(--faint)}

/* ── Chart section (raised card — data viz justifies it) ──────────── */
.chart-section{padding:24px 28px;border:1px solid var(--border);border-radius:18px;
  background:var(--surface);box-shadow:var(--shadow);margin-top:24px}
.chart-section h2{font-size:17px;font-weight:800;margin-bottom:4px;color:var(--text)}
.chart-section .sub{font-size:13px;color:var(--faint);margin-bottom:16px}

/* ── Curriculum (open, column layout) ────────────────────────────── */
.curriculum-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:20px}
@media(max-width:900px){.curriculum-grid{grid-template-columns:1fr}}
.curriculum-card{padding:20px 18px;border-radius:14px;border-left:4px solid var(--border)}
.curriculum-card.easy{border-left-color:var(--green);background:rgba(22,163,74,.04)}
.curriculum-card.medium{border-left-color:var(--amber);background:rgba(217,119,6,.04)}
.curriculum-card.hard{border-left-color:var(--red);background:rgba(220,38,38,.04)}
.curr-hdr{display:flex;justify-content:space-between;align-items:flex-start;gap:8px;margin-bottom:10px}
.curr-title{font-size:13px;font-weight:800;line-height:1.3;color:var(--text)}
.curr-meta{font-size:11px;color:var(--faint);white-space:nowrap;font-weight:600}
.curr-lbl{font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--faint);margin:10px 0 5px;font-weight:700}
.curr-txt{font-size:12px;color:var(--body);line-height:1.6}

/* ── Audit Workspace (interactive — raised card) ─────────────────── */
.audit-workspace{padding:28px;border:1px solid var(--border);border-radius:22px;
  background:var(--surface);box-shadow:var(--shadow-md);margin-top:24px}
.workspace-header{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;margin-bottom:22px;flex-wrap:wrap}
.workspace-title{font-size:19px;font-weight:900;color:var(--text);margin-bottom:4px}
.workspace-sub{font-size:13px;color:var(--muted)}
.replay-controls{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin-bottom:20px;
  padding:14px 16px;background:var(--surface2);border-radius:12px}
.random-btn{padding:8px 18px;border-radius:8px;border:none;
  background:var(--green);color:#fff;font-size:12px;font-weight:800;transition:opacity .15s}
.random-btn:hover:not(:disabled){opacity:.88}
.random-btn:disabled{opacity:.4;cursor:not-allowed}
.replay-meta{font-size:12px;color:var(--muted);margin-left:4px}
.replay-empty{padding:44px;text-align:center;color:var(--faint);border:1px dashed var(--border);
  border-radius:16px;font-size:14px;background:var(--surface2)}
/* ── Case File (unified audit surface) ───────────────────────────── */
.case-file{border:1px solid var(--border);border-radius:18px;overflow:hidden;background:var(--surface)}
.case-header{padding:16px 22px;background:var(--surface2);border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.case-id-row{display:flex;align-items:center;gap:10px;margin-bottom:3px}
.case-id{font-size:17px;font-weight:900;color:var(--text)}
.case-submitter{font-size:12px;color:var(--muted)}
.case-id-group{flex:1;min-width:0}
.case-header-right{display:flex;align-items:center;gap:10px;flex-shrink:0;flex-wrap:wrap}
.case-amount-badge{font-size:18px;font-weight:900;color:var(--green)}
.case-verdict-pill{padding:5px 14px;border-radius:999px;font-size:11px;font-weight:900;
  letter-spacing:.08em;text-transform:uppercase}
.cvp-Approve{background:rgba(22,163,74,.12);color:var(--green)}
.cvp-Reject{background:rgba(220,38,38,.12);color:var(--red)}
.cvp-Escalate{background:rgba(217,119,6,.12);color:var(--amber)}
.cvp-pending{background:var(--surface2);color:var(--faint)}
.case-score-badge{font-size:14px;font-weight:900}
.case-body{display:grid;grid-template-columns:300px 1fr}
@media(max-width:860px){.case-body{grid-template-columns:1fr}}
.case-docket{padding:20px 20px;border-right:1px solid var(--border)}
@media(max-width:860px){.case-docket{border-right:none;border-bottom:1px solid var(--border)}}
.docket-section-lbl{font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:.12em;
  color:var(--faint);margin-bottom:14px}
.investigation-rail{padding:20px 22px;display:flex;flex-direction:column}
.rail-verdict{margin-top:20px;padding-top:18px;border-top:1px solid var(--border)}
.rail-verdict-inner{display:flex;align-items:flex-start;gap:20px;margin-bottom:14px;flex-wrap:wrap}
.rail-score-block{flex:1;min-width:120px}
.ticket-id{font-size:17px;font-weight:900;color:var(--text)}
.diff-badge{padding:4px 10px;border-radius:6px;font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:.06em}
.diff-easy{background:rgba(22,163,74,.14);color:var(--green)}
.diff-medium{background:rgba(217,119,6,.14);color:var(--amber)}
.diff-hard{background:rgba(220,38,38,.14);color:var(--red)}
.diff-training{background:var(--surface2);color:var(--muted)}
.f-lbl{font-size:10px;color:var(--faint);text-transform:uppercase;letter-spacing:.09em;margin-bottom:3px;font-weight:700}
.f-val{font-size:13px;margin-bottom:2px;color:var(--text);font-weight:600}
.f-sub{font-size:11px;color:var(--faint)}
.t-field{margin-bottom:12px}
.field-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}
@media(max-width:560px){.field-grid{grid-template-columns:1fr}}
.amount-big{font-size:26px;font-weight:900;color:var(--green)}
.badges-row{display:flex;flex-wrap:wrap;gap:7px;margin:10px 0}
.receipt-y{padding:4px 10px;border-radius:6px;background:rgba(22,163,74,.1);color:var(--green);font-size:11px;font-weight:700}
.receipt-n{padding:4px 10px;border-radius:6px;background:rgba(220,38,38,.1);color:var(--red);font-size:11px;font-weight:700}
.miss-tag{padding:4px 10px;border-radius:6px;background:rgba(217,119,6,.1);color:var(--amber);font-size:11px;font-weight:700}
.status-dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:5px;vertical-align:middle}
.status-dot.ok{background:var(--green)}.status-dot.no{background:var(--red)}.status-dot.warn{background:var(--amber)}
.risk-bg{height:5px;background:#e2e8f0;border-radius:3px;margin:6px 0 3px}
.risk-fill{height:100%;border-radius:3px;background:linear-gradient(90deg,var(--green),var(--red));transition:width .4s}
.risk-val{font-size:11px;color:var(--faint);font-weight:600}
.gt-chip{display:inline-block;margin-top:5px;padding:5px 12px;border-radius:6px;font-size:12px;font-weight:700}
.gt-Approve{background:rgba(22,163,74,.12);color:var(--green)}
.gt-Reject{background:rgba(220,38,38,.12);color:var(--red)}
.gt-Escalate{background:rgba(217,119,6,.12);color:var(--amber)}
.policy-expected-box{padding:12px;border-radius:10px;background:var(--blue-soft);border:1px solid rgba(37,99,235,.2);margin-top:4px}
.policy-pill{display:inline-block;margin-top:4px;padding:5px 12px;border-radius:6px;
  background:var(--blue);color:#ffffff;font-size:11px;font-weight:700}
.policy-summary{font-size:12px;color:var(--text);margin-top:7px;line-height:1.6;font-weight:500}
.policy-note{font-size:11px;color:var(--muted);margin-top:7px;font-style:italic}

/* ── Investigation timeline ───────────────────────────────────────── */
.timeline{display:flex;flex-direction:column;gap:8px}
.step-card{padding:11px 13px;border-radius:10px;border:1px solid var(--border);
  border-left:3px solid transparent;background:var(--surface);animation:fadeUp .25s ease both}
@keyframes fadeUp{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:translateY(0)}}
.step-hdr{display:flex;align-items:center;gap:8px}
.step-dot{width:26px;height:26px;border-radius:50%;display:grid;place-items:center;
  background:var(--surface2);color:var(--text);font-size:11px;font-weight:900;flex-shrink:0}
.step-lbl{flex:1;font-size:13px;font-weight:700;display:flex;align-items:center;color:var(--text)}
.r-badge{padding:3px 9px;border-radius:5px;font-size:11px;font-weight:700}
.r-pos{background:rgba(22,163,74,.12);color:var(--green)}
.r-neg{background:rgba(220,38,38,.12);color:var(--red)}
.r-zero{background:var(--surface2);color:var(--faint)}
.step-detail{font-size:12px;color:var(--muted);margin-top:4px;margin-left:34px;line-height:1.5}
.policy-search{color:var(--blue);font-weight:600}
.policy-hit{color:var(--green);margin-top:5px;padding:9px;background:rgba(22,163,74,.07);border-radius:6px;font-size:12px;line-height:1.5;font-weight:600}
.policy-expected{margin-top:5px}
.dec-chip{display:inline-block;margin-top:5px;margin-left:34px;padding:4px 11px;border-radius:6px;font-size:12px;font-weight:700}
.dec-inline{margin-top:0;margin-left:7px;vertical-align:middle}
.dec-Approve{background:rgba(22,163,74,.12);color:var(--green)}
.dec-Reject{background:rgba(220,38,38,.12);color:var(--red)}
.dec-Escalate{background:rgba(217,119,6,.12);color:var(--amber)}
.tl-empty{padding:18px;text-align:center;color:var(--faint);font-size:13px}
.action-icon{display:inline-flex;align-items:center;justify-content:center;width:22px;height:22px;
  border-radius:5px;font-size:9px;font-weight:900;flex-shrink:0;margin-right:7px}
.action-icon.search{background:rgba(37,99,235,.12);color:var(--blue)}
.action-icon.request{background:rgba(217,119,6,.12);color:var(--amber)}
.action-icon.resolve{background:rgba(22,163,74,.12);color:var(--green)}
.tag{display:inline-block;padding:2px 7px;border-radius:5px;font-size:10px;font-weight:700;margin-left:5px}
.tag-ok{background:rgba(22,163,74,.12);color:var(--green)}
.tag-miss{background:rgba(220,38,38,.12);color:var(--red)}

/* ── Verdict panel ────────────────────────────────────────────────── */
.verdict-stamp{display:block;text-align:center;margin:0 auto 14px;padding:8px 14px;
  border:2px solid currentColor;border-radius:8px;font-size:12px;font-weight:900;
  letter-spacing:.16em;text-transform:uppercase;transform:rotate(-2deg);width:fit-content}
.stamp-Approve{color:var(--green);background:rgba(22,163,74,.06)}
.stamp-Reject{color:var(--red);background:rgba(220,38,38,.06)}
.stamp-Escalate{color:var(--amber);background:rgba(217,119,6,.06)}
.grade-box{margin:12px 0;padding:14px;background:var(--surface);border:1px solid var(--border);border-radius:10px;text-align:center}
.grade-lbl{font-size:10px;text-transform:uppercase;letter-spacing:.12em;color:var(--faint);font-weight:700}
.grade-score{font-size:48px;font-weight:900;margin:7px 0;color:var(--text)}
.steps-txt{font-size:11px;color:var(--faint);font-weight:600}

/* ── Tabs & controls ──────────────────────────────────────────────── */
.tab-grp{display:flex;gap:3px;background:var(--surface2);padding:3px;border-radius:8px}
.tab-btn{padding:6px 13px;border-radius:6px;border:none;background:transparent;color:var(--faint);
  font-size:12px;font-weight:700;transition:all .14s}
.tab-btn.act{background:var(--surface);color:var(--text);box-shadow:var(--shadow)}

/* ── Policy Rulebook accordion ────────────────────────────────────── */
.rulebook{margin-top:24px;border:1px solid var(--border);border-radius:14px;overflow:hidden;background:var(--surface)}
.rb-toggle{width:100%;padding:15px 20px;background:var(--surface);border:none;color:var(--text);
  font-size:14px;font-weight:800;text-align:left;display:flex;justify-content:space-between;align-items:center;
  transition:background .14s}
.rb-toggle:hover{background:var(--surface2)}
.rb-chevron{display:inline-block;width:8px;height:8px;border-right:2px solid var(--muted);
  border-bottom:2px solid var(--muted);transform:rotate(45deg);transition:transform .2s;margin-left:8px}
.rb-chevron.open{transform:rotate(-135deg);margin-top:4px}
.rb-body{padding:18px 20px;background:var(--surface2);border-top:1px solid var(--border)}
.rb-sub{font-size:13px;color:var(--muted);margin-bottom:12px;line-height:1.6}
.rules-list{padding-left:18px;display:flex;flex-direction:column;gap:7px}
.rule-item{font-size:13px;color:var(--body);padding:6px 10px;border-radius:6px;transition:background .2s;line-height:1.5}
.rule-item.lit{background:rgba(37,99,235,.1);color:var(--blue);font-weight:700}

/* ── CSS fallback chart ───────────────────────────────────────────── */
.css-chart{position:relative;margin-top:14px;padding:18px 10px 8px 40px;
  border-left:1px solid #e2e8f0;border-bottom:1px solid #e2e8f0;
  display:grid;grid-template-columns:repeat(4,1fr);gap:16px;min-height:260px}
.css-chart-scale{position:absolute;left:8px;top:18px;height:200px;display:flex;flex-direction:column;
  justify-content:space-between;color:var(--faint);font-size:11px;font-weight:600}
.css-chart-group{display:flex;flex-direction:column;justify-content:flex-end;min-width:0}
.css-bars{height:200px;display:flex;align-items:flex-end;justify-content:center;gap:5px}
.css-bar-wrap{height:100%;display:flex;flex-direction:column;justify-content:flex-end;align-items:center;gap:4px;min-width:32px}
.css-bar{width:24px;border-radius:5px 5px 2px 2px;transition:height .4s ease}
.css-bar.pending{opacity:.3;background:repeating-linear-gradient(45deg,#cbd5e1,#cbd5e1 4px,rgba(203,213,225,.3) 4px,rgba(203,213,225,.3) 8px)!important}
.css-bar-val{font-size:10px;color:var(--text);font-weight:800;white-space:nowrap}
.css-bar-lbl{font-size:9px;color:var(--faint);text-align:center;max-width:50px;line-height:1.2;font-weight:600}
.css-group-lbl{text-align:center;color:var(--body);font-size:12px;font-weight:800;margin-top:8px}
.css-legend{grid-column:1/-1;display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin-top:8px;padding-left:24px;font-weight:700}
.css-legend-dot{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px;vertical-align:middle}

/* ── Mobile / phone layout ────────────────────────────────────────── */
@media(max-width:640px){
  .dash{padding:16px 14px 48px;
    padding-left:max(14px,env(safe-area-inset-left));padding-right:max(14px,env(safe-area-inset-right))}
  .page-section{padding:28px 0}
  .section-heading{font-size:18px}
  .section-lead{font-size:13px}
  .hero{padding:28px 20px;border-radius:18px}
  .hero h1{font-size:26px;margin-bottom:12px}
  .hero p{font-size:14px}
  .hero-toolkit{min-width:0;width:100%}
  .paradigm-item{padding:18px 16px}
  .paradigm-score{font-size:26px}
  .lifecycle-band{padding:18px 16px;border-radius:14px}
  .lc-step{padding:14px 12px}
  .chart-section{padding:18px 16px;border-radius:14px}
  .curriculum-card{padding:16px 14px}
  .curr-hdr{flex-direction:column;align-items:flex-start}
  .curr-meta{white-space:normal}
  .audit-workspace{padding:18px 16px;border-radius:16px}
  .workspace-title{font-size:17px}
  .replay-controls{flex-direction:column;align-items:stretch;gap:12px;padding:12px}
  .replay-controls .tab-grp{width:100%}
  .replay-controls .tab-btn{flex:1;min-height:44px;padding:10px 8px;font-size:11px;
    white-space:normal;line-height:1.25;text-align:center}
  .random-btn{width:100%;min-height:44px;padding:12px 18px;font-size:13px}
  .replay-meta{margin-left:0;width:100%;text-align:center;font-size:11px;word-break:break-word}
  .replay-empty{padding:28px 16px;font-size:13px}
  .case-header{padding:14px 16px;flex-direction:column;align-items:flex-start;gap:10px}
  .case-header-right{width:100%;justify-content:space-between}
  .case-id{font-size:15px}
  .case-amount-badge{font-size:16px}
  .case-docket,.investigation-rail{padding:16px}
  .amount-big{font-size:22px}
  .rail-verdict-inner{flex-direction:column;gap:12px}
  .grade-score{font-size:36px}
  .verdict-stamp{font-size:11px;padding:6px 12px}
  .step-hdr{flex-wrap:wrap;gap:6px}
  .step-detail{margin-left:0;padding-left:0}
  .css-chart{padding:14px 8px 8px 32px;gap:10px;min-height:220px}
  .css-chart-scale{left:4px;font-size:10px;height:160px}
  .css-bars{height:160px}
  .css-bar{width:18px}
  .css-bar-lbl{font-size:8px;max-width:40px}
  .css-legend{padding-left:0;gap:10px;font-size:11px}
  .rb-toggle{padding:14px 16px;font-size:13px}
  .rb-body{padding:14px 16px}
  .outcome-row{flex-wrap:wrap;gap:8px}
  .outcome-row-val{font-size:17px}
  .narrative-text .narrative-title{font-size:15px}
  button,.tab-btn,.random-btn,.rb-toggle{-webkit-tap-highlight-color:transparent;touch-action:manipulation}
}
@media(max-width:400px){
  .hero h1{font-size:23px}
  .replay-controls .tab-grp{flex-direction:column}
  .replay-controls .tab-btn{flex:none;width:100%}
}
</style>
</head>
<body>
<div id="root"><div style="padding:60px;text-align:center;color:#94a3b8;font-family:sans-serif;font-size:15px">Loading&#8230;</div></div>

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
  document.getElementById('root').innerHTML='<div style="padding:48px;text-align:center;color:#dc2626;font-family:sans-serif">React CDN failed to load. Check your internet connection and reload.</div>';
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
const curriculum=D.task_curriculum||[];
const toolActs=D.tool_actions||{};
const rlStory=D.rl_story||{};

/* ── Utilities ── */
const fmt3=v=>v!=null?Number(v).toFixed(3):'—';
const fmtRs=v=>'\u20b9'+Number(v||0).toLocaleString('en-IN');
const scoreColor=v=>v>=0.7?'#16a34a':v>=0.4?'#d97706':'#dc2626';
const aBorder=t=>t==='SEARCH_POLICY'?'#2563eb':t==='REQUEST_INFORMATION'?'#d97706':'#16a34a';
const aLabel=t=>t==='SEARCH_POLICY'?'Search Policy':t==='REQUEST_INFORMATION'?'Request Information':'Resolve Ticket';
function fmtDecision(d){
  if(d==null||d===undefined)return '';
  let s=String(d).trim();
  if(!s)return '';
  if(s.indexOf('TicketDecision.')===0)s=s.slice(15);
  else if(s.indexOf('.')>=0)s=s.split('.').pop();
  const m={APPROVE:'Approve',REJECT:'Reject',ESCALATE:'Escalate'};
  const u=s.toUpperCase();
  return m[u]||m[s]||(s.charAt(0).toUpperCase()+s.slice(1).toLowerCase());
}
function DecisionChip({decision,inline}){
  const d=fmtDecision(decision);
  if(!d)return null;
  return h('span',{className:'dec-chip dec-'+d+(inline?' dec-inline':'')},d);
}
const aIconCls=t=>t==='SEARCH_POLICY'?'search':t==='REQUEST_INFORMATION'?'request':'resolve';
const aIconTxt=t=>t==='SEARCH_POLICY'?'SP':t==='REQUEST_INFORMATION'?'RI':'RV';
function ActionIcon({type}){
  return h('span',{className:'action-icon '+aIconCls(type),title:aLabel(type)},aIconTxt(type));
}

/* ═══ React components ═══ */

function App(){
  const [selId,setSelId]=useState(null);
  const [diffTab,setDiffTab]=useState('easy');
  const [methTab,setMethTab]=useState('baseline');
  const selEp=useMemo(()=>eps.find(e=>e.episode_id===selId)||null,[selId]);
  const epActs=useMemo(()=>acts.filter(a=>a.episode_id===selId).sort((a,b)=>a.step-b.step),[selId]);
  const filtEps=useMemo(()=>eps.filter(e=>e.task_id===diffTab&&e.method===methTab),[diffTab,methTab]);

  const handleRandom=useCallback(()=>{
    if(!filtEps.length)return;
    const pick=filtEps[Math.floor(Math.random()*filtEps.length)];
    setSelId(pick.episode_id);
  },[filtEps]);

  useEffect(()=>{
    if(!selId)return;
    if(!filtEps.some(e=>e.episode_id===selId))setSelId(null);
  },[diffTab,methTab,filtEps,selId]);

  const highlightedRule=useMemo(()=>{
    if(selEp&&selEp.expected_policy_search)return selEp.expected_policy_search;
    const searches=epActs.filter(a=>a.action_type==='SEARCH_POLICY');
    return searches.length?(searches[searches.length-1].query||null):null;
  },[epActs,selEp]);

  const methLabel={baseline:'Rule-Based Engine',inference:'Generic LLM',training:'RL-Trained Agent'}[methTab]||methTab;

  return h('div',{className:'dash'},
    /* 1. PROBLEM STATEMENT */
    h(HeroSection,null),
    /* 2. THREE APPROACHES COMPARED */
    h('section',{className:'page-section'},
      h('div',{className:'section-eyebrow'},'Three approaches to the same problem'),
      h('div',{className:'section-heading'},'How do we benchmark compliance automation?'),
      h('p',{className:'section-lead'},'We compare a rule-based engine, a generic prompted LLM, and an RL-trained agent on the same 120 test claims. Each paradigm represents a different philosophy toward compliance judgment.'),
      h(ParadigmStrip,null)
    ),
    /* 3. AUDIT LIFECYCLE */
    h('section',{className:'page-section'},
      h('div',{className:'section-eyebrow'},'How the RL agent works'),
      h('div',{className:'section-heading'},'The compliance audit lifecycle'),
      h('p',{className:'section-lead'},'The agent does not get a single prompt and guess. It navigates the audit like a compliance officer: reading the claim, searching policy, requesting missing documents, and resolving with evidence.'),
      h(LifecycleBand,null),
      h(TaskCurriculum,null)
    ),
    /* 4. WHAT TRAINING CHANGED */
    h('section',{className:'page-section'},
      h('div',{className:'section-eyebrow'},'Evidence of learning'),
      h('div',{className:'section-heading'},'What the training run changed'),
      h('p',{className:'section-lead'},'A focused SFT + GRPO training run on the same environment. The result: not a dramatic overall jump, but a measurable shift in behaviour on the cases that require real judgment.'),
      h(NarrativePanel,null),
      h(ScoreChart,null)
    ),
    /* 5. LIVE AUDIT REPLAY */
    h('section',{className:'page-section'},
      h('div',{className:'section-eyebrow'},'Interactive audit replay'),
      h(AuditWorkspace,{selEp,epActs,filtEps,diffTab,methTab,methLabel,handleRandom,setDiffTab,setMethTab})
    ),
    /* 6. POLICY CONTEXT */
    h(PolicyRulebook,{highlighted:highlightedRule})
  );
}

/* ── 1. Hero: Problem Statement ────────────────────────────────────── */
function HeroSection(){
  return h('section',{className:'hero'},
    h('div',{className:'hero-badge'},h('span',{className:'hero-badge-dot'}),'\u00a0Compliance Audit Console \xb7 OpenEnv RL Environment'),
    h('div',{className:'hero-layout'},
      h('div',null,
        h('h1',null,'Training an agent to audit expense claims like a compliance officer.'),
        h('p',null,
          'Every enterprise processes hundreds of expense reports daily. Today, human auditors read policy documents, handle incomplete information, and make judgment calls. A generic LLM can guess; a rule-based system breaks on edge cases. This environment trains RL agents to navigate the real process: retrieve policy, request missing documents, and resolve with evidence across three difficulty levels.'
        )
      ),
      h('div',{className:'hero-toolkit'},
        h('div',{className:'hero-toolkit-title'},'Agent action space'),
        h('div',{className:'toolkit-action'},
          h('div',{className:'toolkit-icon search'},'SP'),
          h('div',null,
            h('div',{className:'toolkit-action-lbl'},'SearchPolicy'),
            h('div',{className:'toolkit-action-desc'},'Query the policy rulebook to find the applicable rule')
          )
        ),
        h('div',{className:'toolkit-action'},
          h('div',{className:'toolkit-icon request'},'RI'),
          h('div',null,
            h('div',{className:'toolkit-action-lbl'},'RequestInformation'),
            h('div',{className:'toolkit-action-desc'},'Ask the employee for a missing receipt or document')
          )
        ),
        h('div',{className:'toolkit-action'},
          h('div',{className:'toolkit-icon resolve'},'RV'),
          h('div',null,
            h('div',{className:'toolkit-action-lbl'},'ResolveTicket'),
            h('div',{className:'toolkit-action-desc'},'Issue the final Approve / Reject / Escalate decision')
          )
        )
      )
    )
  );
}

/* ── 2. Paradigm comparison strip ─────────────────────────────────── */
function ParadigmStrip(){
  const bm=M.baseline||{},im=M.inference||{},tm=M.training||{};
  const bN=bm.n||0,iN=im.n||0,tN=tm.n||0;
  return h('div',{className:'paradigm-strip'},
    h('div',{className:'paradigm-item'},
      h('span',{className:'paradigm-tag rule'},'Rule-Based'),
      h('div',{className:'paradigm-name'},'Rule-Based Engine'),
      h('div',{className:'paradigm-desc'},'Deterministic: match claim attributes to a fixed decision tree. Fast, auditable, zero hallucination.'),
      h('div',{className:'paradigm-score-row'},
        h('span',{className:'paradigm-score',style:{color:'#2563eb'}},fmt3(bm.overall)),
        h('span',{className:'paradigm-score-lbl'},'overall grader score')
      ),
      h('div',{className:'paradigm-meta'},'n='+bN+' claims evaluated'),
      h('div',{className:'paradigm-limits'},'Breaks on edge cases. Cannot retrieve policy dynamically. Fails when context is missing.')
    ),
    h('div',{className:'paradigm-arrow'},'\u2192'),
    h('div',{className:'paradigm-item'},
      h('span',{className:'paradigm-tag llm'},'Generic LLM'),
      h('div',{className:'paradigm-name'},'Generic Prompted LLM'),
      h('div',{className:'paradigm-desc'},'Language model prompted with the claim. Understands natural language but has no policy grounding.'),
      h('div',{className:'paradigm-score-row'},
        h('span',{className:'paradigm-score',style:{color:'#f97316'}},fmt3(im.overall)),
        h('span',{className:'paradigm-score-lbl'},'overall grader score')
      ),
      h('div',{className:'paradigm-meta'},'n='+iN+' claims evaluated'),
      h('div',{className:'paradigm-limits'},'Guesses without searching policy. Ignores missing documents. Inconsistent on hard multi-step cases.')
    ),
    h('div',{className:'paradigm-arrow'},'\u2192'),
    h('div',{className:'paradigm-item highlight'},
      h('span',{className:'paradigm-tag rl'},'RL-Trained'),
      h('div',{className:'paradigm-name'},'RL-Trained Agent'),
      h('div',{className:'paradigm-desc'},'SFT + GRPO trained on the environment reward signal. Learns when to search, when to request, when to resolve.'),
      h('div',{className:'paradigm-score-row'},
        tN>0
          ?h('span',{className:'paradigm-score',style:{color:'#16a34a'}},fmt3(tm.overall))
          :h('span',{className:'paradigm-score',style:{color:'#94a3b8'}},'pending'),
        h('span',{className:'paradigm-score-lbl'},'overall grader score')
      ),
      h('div',{className:'paradigm-meta'},tN>0?'n='+tN+' claims evaluated':'training eval pending'),
      h('div',{className:'paradigm-limits'},'Lifts Medium and Hard scores. Uses policy search and document requests more deliberately after training.')
    )
  );
}

/* ── 3. Audit lifecycle band ──────────────────────────────────────── */
function LifecycleBand(){
  return h('div',{className:'lifecycle-band'},
    h('div',{className:'lifecycle-steps'},
      h('div',{className:'lc-step claim'},
        h('div',{className:'lc-num'},'1'),
        h('div',{className:'lc-title'},'Claim arrives'),
        h('div',{className:'lc-body'},'Employee submits an expense. Amount, description, receipt status, and employee level are visible. Policy rule keyword is hidden on Medium/Hard.')
      ),
      h('div',{className:'lc-arrow'},'\u2192'),
      h('div',{className:'lc-step search'},
        h('div',{className:'lc-num'},'2'),
        h('div',{className:'lc-title'},'Search Policy'),
        h('div',{className:'lc-body'},'Agent queries the rulebook to find the applicable rule. Relevant search earns reward. Irrelevant or redundant search incurs a penalty.'),
        h('span',{className:'lc-badge search'},'SearchPolicy')
      ),
      h('div',{className:'lc-arrow'},'\u2192'),
      h('div',{className:'lc-step request'},
        h('div',{className:'lc-num'},'3'),
        h('div',{className:'lc-title'},'Request document'),
        h('div',{className:'lc-body'},'If a required document is missing, the agent requests it. The environment simulates an employee response. Requesting what is already present is penalised.'),
        h('span',{className:'lc-badge request'},'RequestInformation')
      ),
      h('div',{className:'lc-arrow'},'\u2192'),
      h('div',{className:'lc-step resolve'},
        h('div',{className:'lc-num'},'4'),
        h('div',{className:'lc-title'},'Resolve ticket'),
        h('div',{className:'lc-body'},'Agent issues Approve, Reject, or Escalate based on policy and evidence gathered. Correct decision + valid reason earns the largest reward.'),
        h('span',{className:'lc-badge resolve'},'ResolveTicket')
      ),
      h('div',{className:'lc-arrow'},'\u2192'),
      h('div',{className:'lc-step grade'},
        h('div',{className:'lc-num'},'5'),
        h('div',{className:'lc-title'},'Grader scores'),
        h('div',{className:'lc-body'},'Multi-component grader evaluates: valid resolve, correct decision, valid reason, useful search, correct document request, and step efficiency.')
      )
    )
  );
}

/* ── Task curriculum (under lifecycle) ───────────────────────────── */
function TaskCurriculum(){
  return h('div',{className:'curriculum-grid'},
    ...curriculum.map(task=>
      h('div',{key:task.id,className:'curriculum-card '+task.id},
        h('div',{className:'curr-hdr'},
          h('div',{className:'curr-title'},task.title),
          h('span',{className:'curr-meta'},'max '+task.max_steps+' steps \xb7 '+task.expected_steps)
        ),
        h('div',{className:'curr-lbl'},'What the agent sees'),
        h('div',{className:'curr-txt'},task.agent_sees),
        h('div',{className:'curr-lbl'},'Agent goal'),
        h('div',{className:'curr-txt'},task.agent_goal),
        h('div',{className:'curr-lbl'},'Why it is tricky'),
        h('div',{className:'curr-txt'},task.why_hard)
      )
    )
  );
}

/* ── 4. RL outcome narrative ─────────────────────────────────────── */
function NarrativePanel(){
  if(!rlStory.has_training){
    return h('div',{style:{color:'#94a3b8',fontSize:'13px',padding:'16px 0'}},
      'Run checkpoint eval to populate Trained LLM metrics and show the training outcome here.'
    );
  }
  const s=v=>(v>0?'+':'')+Number(v).toFixed(3);
  const pct=v=>(v>0?'+':'')+Number(v).toFixed(1)+'%';
  const arrow=(a,b)=>a+' \u2192 '+b;
  const gainColor=v=>v>0?'#16a34a':v<0?'#dc2626':'#64748b';
  const mg=rlStory.medium_gain||0,hg=rlStory.hard_gain||0;
  const cg=rlStory.complex_task_gain||0,cp=rlStory.complex_task_gain_pct||0;
  const od=rlStory.overall_delta||0;
  const reqB=rlStory.request_information_before||0,reqA=rlStory.request_information_after||0;
  const srcB=rlStory.search_policy_before||0,srcA=rlStory.search_policy_after||0;
  return h('div',{className:'narrative-layout'},
    h('div',{className:'narrative-text'},
      h('div',{className:'narrative-title'},rlStory.headline||'Training outcome'),
      h('p',{className:'narrative-lead'},rlStory.summary||''),
      h('p',{className:'narrative-note'},
        'Overall nearly flat ('+Number(rlStory.inference_overall||0).toFixed(3)+
        ' \u2192 '+Number(rlStory.trained_overall||0).toFixed(3)+', '+s(od)+
        ') because easy tickets were already high. RL value shows on harder cases.'
      ),
      h('p',{className:'narrative-foot'},rlStory.why_it_matters||'')
    ),
    h('div',{className:'outcome-stats'},
      h('div',{className:'outcome-row'},
        h('div',null,h('div',{className:'outcome-row-lbl'},'Medium grader score lift'),h('div',{className:'outcome-row-sub'},'Policy retrieval tasks')),
        h('div',{className:'outcome-row-val',style:{color:gainColor(mg)}},s(mg))
      ),
      h('div',{className:'outcome-row'},
        h('div',null,h('div',{className:'outcome-row-lbl'},'Hard grader score lift'),h('div',{className:'outcome-row-sub'},'Multi-step evidence tasks')),
        h('div',{className:'outcome-row-val',style:{color:gainColor(hg)}},s(hg))
      ),
      h('div',{className:'outcome-row'},
        h('div',null,h('div',{className:'outcome-row-lbl'},'Medium + Hard average lift'),h('div',{className:'outcome-row-sub'},pct(cp)+' on harder claims')),
        h('div',{className:'outcome-row-val',style:{color:gainColor(cg)}},s(cg))
      ),
      h('div',{className:'outcome-row'},
        h('div',null,h('div',{className:'outcome-row-lbl'},'SearchPolicy actions in eval'),h('div',{className:'outcome-row-sub'},'Inference \u2192 Trained LLM')),
        h('div',{className:'outcome-row-val'},arrow(srcB,srcA))
      ),
      h('div',{className:'outcome-row'},
        h('div',null,h('div',{className:'outcome-row-lbl'},'RequestInformation actions'),h('div',{className:'outcome-row-sub'},'Inference \u2192 Trained LLM')),
        h('div',{className:'outcome-row-val'},arrow(reqB,reqA))
      )
    )
  );
}

/* ── Score chart ─────────────────────────────────────────────────── */
function GroupedBarFallback({data,hasT}){
  const colors={baseline:'#2563eb',llm:'#f97316',rl:'#6b7280'};
  const labels={baseline:'Rule-Based',llm:'Generic LLM',rl:'RL-Trained'};
  return h('div',{className:'css-chart'},
    h('div',{className:'css-chart-scale'},'1.0','0.5','0.0'),
    ...data.map((row,i)=>h('div',{key:i,className:'css-chart-group'},
      h('div',{className:'css-bars'},
        ...['baseline','llm','rl'].map(k=>{
          const v=row[k]||0,pending=k==='rl'&&!hasT;
          const ht=pending?2:Math.max(2,Math.round(v*100));
          return h('div',{key:k,className:'css-bar-wrap'},
            h('div',{className:'css-bar-val'},pending?'—':v.toFixed(3)),
            h('div',{className:'css-bar'+(pending?' pending':''),style:{height:ht+'%',background:colors[k]}}),
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
    {d:'Easy',baseline:bm.easy||0,llm:im.easy||0,rl:hasT?tm.easy||0:0},
    {d:'Medium',baseline:bm.medium||0,llm:im.medium||0,rl:hasT?tm.medium||0:0},
    {d:'Hard',baseline:bm.hard||0,llm:im.hard||0,rl:hasT?tm.hard||0:0},
    {d:'Overall',baseline:bm.overall||0,llm:im.overall||0,rl:hasT?tm.overall||0:0},
  ];
  const tt={background:'#ffffff',border:'1px solid #e2e8f0',borderRadius:'8px',color:'#0f172a'};
  if(!hasRecharts){
    return h('div',{className:'chart-section'},
      h('h2',null,'Mean Grader Score by Difficulty'),
      h('p',{className:'sub'},'Comparing the three approaches across claim difficulty. Refresh if chart fails.'),
      h(GroupedBarFallback,{data,hasT})
    );
  }
  return h('div',{className:'chart-section'},
    h('h2',null,'Mean Grader Score by Difficulty'),
    h('p',{className:'sub'},'Rule-Based vs Generic LLM vs RL-Trained Agent, across all difficulty tiers.'),
    h(RC.ResponsiveContainer,{width:'100%',height:300},
      h(RC.BarChart,{data,barCategoryGap:'20%',barGap:4,margin:{top:8,right:16,left:0,bottom:0}},
        h(RC.CartesianGrid,{strokeDasharray:'3 3',stroke:'#e2e8f0',vertical:false}),
        h(RC.XAxis,{dataKey:'d',stroke:'#94a3b8',tick:{fill:'#94a3b8',fontSize:13,fontWeight:600}}),
        h(RC.YAxis,{domain:[0,1],stroke:'#94a3b8',tick:{fill:'#94a3b8',fontSize:12},tickFormatter:v=>v.toFixed(1)}),
        h(RC.Tooltip,{contentStyle:tt,formatter:v=>typeof v==='number'?v.toFixed(3):v}),
        h(RC.Legend,{wrapperStyle:{color:'#0f172a',fontSize:'12px',paddingTop:'8px',fontWeight:'700'}}),
        h(RC.Bar,{dataKey:'baseline',name:'Rule-Based Engine',fill:'#2563eb',radius:[4,4,0,0],maxBarSize:48}),
        h(RC.Bar,{dataKey:'llm',name:'Generic LLM',fill:'#f97316',radius:[4,4,0,0],maxBarSize:48}),
        h(RC.Bar,{dataKey:'rl',name:'RL-Trained Agent',fill:'#6b7280',radius:[4,4,0,0],maxBarSize:48,opacity:hasT?1:0.32})
      )
    )
  );
}

/* ── 5. Audit Workspace ──────────────────────────────────────────── */
function AuditWorkspace({selEp,epActs,filtEps,diffTab,methTab,methLabel,handleRandom,setDiffTab,setMethTab}){
  return h('div',{className:'audit-workspace'},
    h('div',{className:'workspace-header'},
      h('div',null,
        h('div',{className:'workspace-title'},'Audit Workspace'),
        h('div',{className:'workspace-sub'},'Select difficulty and agent type, then load a sample claim to walk through the audit steps.')
      )
    ),
    h('div',{className:'replay-controls'},
      h('div',{className:'tab-grp'},
        ...['easy','medium','hard'].map(t=>
          h('button',{key:t,className:'tab-btn'+(diffTab===t?' act':''),onClick:()=>setDiffTab(t)},
            t.charAt(0).toUpperCase()+t.slice(1)))),
      h('div',{className:'tab-grp'},
        ...[['baseline','Rule-Based Engine'],['inference','Generic LLM'],['training','RL-Trained Agent']].map(([id,lbl])=>
          h('button',{key:id,className:'tab-btn'+(methTab===id?' act':''),onClick:()=>setMethTab(id)},lbl))),
      h('button',{className:'random-btn',onClick:handleRandom,disabled:!filtEps.length},'Load sample'),
      filtEps.length?h('span',{className:'replay-meta'},
        selEp?(selEp.claim_id||selEp.episode_id)+' \xb7 '+methLabel:filtEps.length+' samples available'):null
    ),
    selEp
      ?h(CaseFile,{ep:selEp,allActs:epActs,expectedSearch:selEp.expected_policy_search||''})
      :h('div',{className:'replay-empty'},
          filtEps.length
            ?'Pick a difficulty and agent above, then click Load sample to see a real audit episode.'
            :'No episodes for this filter. Run eval locally and serve /dashboard from the same machine (artifact root: '+(D.artifact_root||'?')+').')
  );
}

/* ── Unified case file ───────────────────────────────────────────── */
function CaseFile({ep,allActs,expectedSearch}){
  const finalAction=allActs.filter(a=>a.action_type==='RESOLVE_TICKET').slice(-1)[0]||{};
  const finalDecision=fmtDecision(finalAction.decision||'');
  const score=ep.grader_score||0;
  return h('div',{className:'case-file'},
    h(CaseHeader,{ep,finalDecision,score}),
    h('div',{className:'case-body'},
      h(CaseDocket,{ep}),
      h(InvestigationRail,{ep,allActs,expectedSearch,finalDecision,score})
    )
  );
}

function CaseHeader({ep,finalDecision,score}){
  const pillCls='case-verdict-pill '+(finalDecision?'cvp-'+finalDecision:'cvp-pending');
  return h('div',{className:'case-header'},
    h('div',{className:'case-id-group'},
      h('div',{className:'case-id-row'},
        h('span',{className:'case-id'},ep.claim_id||ep.episode_id),
        h('span',{className:'diff-badge diff-'+ep.task_id,style:{marginLeft:'10px'}},ep.task_id)
      ),
      ep.employee_name?h('div',{className:'case-submitter'},
        ep.employee_name+(ep.employee_role?' \xb7 '+ep.employee_role:'')+(ep.employee_level?' \xb7 '+ep.employee_level:'')
      ):null
    ),
    h('div',{className:'case-header-right'},
      ep.amount>0?h('span',{className:'case-amount-badge'},fmtRs(ep.amount)):null,
      h('span',{className:pillCls},finalDecision||'Pending'),
      score>0?h('span',{className:'case-score-badge',style:{color:scoreColor(score)}},score.toFixed(3)):null
    )
  );
}

function CaseDocket({ep}){
  const rPct=Math.min(100,Math.round((ep.risk_score||0)*100));
  const gt=ep.ground_truth||'';
  return h('div',{className:'case-docket'},
    h('div',{className:'docket-section-lbl'},'Claim facts'),
    h('div',{className:'t-field'},
      h('div',{className:'f-lbl'},'Pre-audit risk'),
      h('div',{className:'risk-bg'},h('div',{className:'risk-fill',style:{width:rPct+'%'}})),
      h('div',{className:'risk-val'},rPct+' / 100')
    ),
    ep.description?h('div',{className:'t-field'},
      h('div',{className:'f-lbl'},'Claim description'),
      h('div',{style:{fontSize:'13px',lineHeight:'1.6',color:'#334155'}},ep.description)
    ):null,
    h('div',{className:'badges-row'},
      ep.has_receipt!==undefined?h('span',{className:ep.has_receipt?'receipt-y':'receipt-n'},
        h('span',{className:'status-dot '+(ep.has_receipt?'ok':'no')}),
        ep.has_receipt?'Receipt attached':'No receipt'):null,
      ep.missing_document?h('span',{className:'miss-tag'},
        h('span',{className:'status-dot warn'}),'Missing: '+ep.missing_document):null
    ),
    ep.expected_policy_search?h('div',{className:'t-field policy-expected-box',style:{marginTop:'14px'}},
      h('div',{className:'f-lbl'},'Applicable policy rule'),
      h('div',{className:'policy-pill'},ep.expected_policy_search),
      ep.expected_policy_summary?h('div',{className:'policy-summary'},ep.expected_policy_summary):null,
      ep.task_id&&ep.task_id!=='easy'?h('div',{className:'policy-note'},
        ep.task_id+' difficulty: rule keyword hidden upfront. Agent must discover via SearchPolicy.'):null
    ):null,
    gt?h('div',{style:{marginTop:'14px'}},
      h('div',{className:'f-lbl'},'Ground truth verdict'),
      h('span',{className:'gt-chip gt-'+gt},gt)
    ):null
  );
}

function InvestigationRail({ep,allActs,expectedSearch,finalDecision,score}){
  const mkCum=arr=>arr.map((a,i)=>({
    s:''+a.step,
    cum:parseFloat(arr.slice(0,i+1).reduce((acc,x)=>acc+(x.reward||0),0).toFixed(3))
  }));
  const tt={background:'#ffffff',border:'1px solid #e2e8f0',borderRadius:'8px',color:'#0f172a'};
  return h('div',{className:'investigation-rail'},
    h('div',{className:'docket-section-lbl'},'Agent investigation'),
    h('div',{className:'timeline'},
      ...allActs.map((a,i)=>h(StepCard,{key:i,action:a,expectedSearch})),
      allActs.length===0?h('div',{className:'tl-empty'},'No step actions logged for this episode.'):null
    ),
    h('div',{className:'rail-verdict'},
      h('div',{className:'rail-verdict-inner'},
        finalDecision?h('div',{className:'verdict-stamp stamp-'+finalDecision},finalDecision):null,
        h('div',{className:'rail-score-block'},
          h('div',{className:'grade-lbl'},'Grader score'),
          h('div',{className:'grade-score',style:{color:scoreColor(score),fontSize:'34px',margin:'4px 0'}},score.toFixed(3)),
          h('div',{className:'steps-txt'},'Steps: '+(ep.steps||0)+' / 8 max \xb7 Reward: '+(ep.total_reward||0).toFixed(2))
        )
      ),
      RC.BarChart?h(RC.ResponsiveContainer,{width:'100%',height:110},
        h(RC.BarChart,{data:mkCum(allActs),barSize:16,margin:{top:0,right:4,left:-22,bottom:0}},
          h(RC.CartesianGrid,{strokeDasharray:'3 3',stroke:'#e2e8f0',horizontal:false}),
          h(RC.XAxis,{dataKey:'s',tick:{fill:'#94a3b8',fontSize:11}}),
          h(RC.YAxis,{tick:{fill:'#94a3b8',fontSize:10},tickFormatter:v=>v.toFixed(1)}),
          h(RC.Tooltip,{contentStyle:tt,formatter:v=>v.toFixed(3),labelFormatter:v=>'Step '+v}),
          h(RC.ReferenceLine,{y:0,stroke:'#cbd5e1',strokeDasharray:'4 2'}),
          h(RC.Bar,{dataKey:'cum',name:'Cumulative Reward',fill:'#2563eb',radius:[3,3,0,0]})
        )
      ):null
    )
  );
}

function StepCard({action,expectedSearch}){
  const t=action.action_type||'';
  const r=action.reward||0;
  const rc=r>0.05?'r-pos':r<-0.05?'r-neg':'r-zero';
  const isSearch=t==='SEARCH_POLICY',isResolve=t==='RESOLVE_TICKET';
  return h('div',{className:'step-card',style:{borderLeftColor:aBorder(t)}},
    h('div',{className:'step-hdr'},
      h('div',{className:'step-dot'},action.step),
      h('div',{className:'step-lbl'},
        h(ActionIcon,{type:t}),aLabel(t),
        isResolve?h(DecisionChip,{decision:action.decision,inline:true}):null),
      h('div',{className:'r-badge '+rc},(r>0?'+':'')+r.toFixed(2))
    ),
    isSearch?h('div',{className:'step-detail policy-search'},
      h('strong',null,'Searched: '),
      action.query?'\u201c'+action.query+'\u201d':'(query not captured)'
    ):null,
    isSearch&&action.matched_policy?h('div',{className:'step-detail policy-hit'},action.matched_policy):null,
    isSearch&&expectedSearch?h('div',{className:'step-detail policy-expected'},
      'Should search: \u201c'+expectedSearch+'\u201d',
      action.search_relevant===true?h('span',{className:'tag tag-ok'},'on-topic'):null,
      action.search_relevant===false?h('span',{className:'tag tag-miss'},'off-topic'):null
    ):null,
    !isSearch&&action.query?h('div',{className:'step-detail'},'Query: \u201c'+action.query+'\u201d'):null,
    action.message?h('div',{className:'step-detail'},'Requested: \u201c'+action.message+'\u201d'):null,
    action.reason?h('div',{className:'step-detail'},'Reason: '+action.reason):null
  );
}

/* ── 6. Policy Rulebook ──────────────────────────────────────────── */
function PolicyRulebook({highlighted}){
  const [open,setOpen]=useState(false);
  const words=highlighted?highlighted.toLowerCase().split(' ').slice(0,3):[];
  return h('div',{className:'rulebook'},
    h('button',{className:'rb-toggle',onClick:()=>setOpen(o=>!o)},
      h('span',null,'Policy Rulebook \u2014 '+rules.length+' rules the agent must learn'),
      h('span',{className:'rb-chevron'+(open?' open':'')})
    ),
    open?h('div',{className:'rb-body'},
      h('p',{className:'rb-sub'},'When the agent runs SearchPolicy it is querying one of these rules. Rules highlight when a matching search appears in the audit trace above.'),
      h('ol',{className:'rules-list'},
        ...rules.map((rule,i)=>{
          const hit=words.length>0&&words.every(w=>rule.toLowerCase().includes(w));
          return h('li',{key:i,className:'rule-item'+(hit?' lit':'')},rule);
        })
      )
    ):null
  );
}

try{ReactDOM.createRoot(document.getElementById('root')).render(h(App,null));}
catch(e){document.getElementById('root').innerHTML='<div style="padding:40px;color:#dc2626;font-family:sans-serif">Dashboard error: '+e.message+'</div>';}
})();
</script>
</body>
</html>
"""

# ── Cache ──────────────────────────────────────────────────────────────────────
_cache: tuple[str, float] = ("", 0.0)
_CACHE_TTL = 2.0


def _render_dashboard() -> str:
    """Generate and briefly cache the full React HTML page."""
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
        <div style="padding:48px 40px;text-align:center;font-family:-apple-system,sans-serif;background:#f8fafc;color:#0f172a;min-height:300px;border-radius:20px;border:1px solid #e2e8f0">
          <div style="font-size:12px;text-transform:uppercase;letter-spacing:.14em;color:#64748b;margin-bottom:14px;font-weight:700">Compliance Audit Console</div>
          <h1 style="font-size:28px;font-weight:860;margin-bottom:16px">Full interactive dashboard</h1>
          <p style="font-size:15px;color:#334155;max-width:560px;margin:0 auto 28px;line-height:1.7">
            The React-powered audit console with grouped score charts, claim replay, and policy rulebook is served at
            <code style="background:#e2e8f0;color:#0f172a;padding:2px 8px;border-radius:6px">/dashboard</code>.
          </p>
          <a href="/dashboard"
             style="display:inline-block;padding:12px 32px;border-radius:999px;background:#2563eb;color:#fff;font-size:15px;font-weight:700;text-decoration:none">
            Open Dashboard &rarr;
          </a>
        </div>
        """)
    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
