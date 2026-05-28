"""Interactive Gradio dashboard for compliance benchmark storytelling."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd

BASELINE_RESULTS = Path("baseline_results.json")
INFERENCE_RESULTS = Path("inference_results.json")
BASELINE_LOG = Path("baseline_run.log")
INFERENCE_LOG = Path("inference_run.log")
TRAINING_EPISODES = Path("training/logs/episodes.jsonl")

TASK_ORDER = ["easy", "medium", "hard"]


@dataclass
class MethodMetrics:
    method: str
    overall: float
    total: int
    by_task: Dict[str, float]
    by_task_total: Dict[str, int]


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


def _parse_episode_log(path: Path, method: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    text = _safe_read_text(path)
    if not text:
        return pd.DataFrame(), pd.DataFrame()

    current_claim: Optional[str] = None
    current_task: Optional[str] = None
    current = None
    episodes: List[Dict[str, Any]] = []
    actions: List[Dict[str, Any]] = []
    episode_idx = 0

    for raw in text.splitlines():
        line = raw.strip()
        claim_match = re.match(r"Running inference for claim\s+([A-Z0-9-]+)\s+\((easy|medium|hard)\)\.\.\.", line)
        if claim_match:
            current_claim = claim_match.group(1)
            current_task = claim_match.group(2).lower()
            continue

        start_match = re.match(r"\[START\]\s+task=(EASY|MEDIUM|HARD)\s+.*", line)
        if start_match:
            episode_idx += 1
            current = {
                "method": method,
                "episode_id": f"{method}-{episode_idx:04d}",
                "claim_id": current_claim or "",
                "task_id": (current_task or start_match.group(1).lower()),
                "steps": 0,
                "total_reward": 0.0,
                "grader_score": 0.0,
                "success": False,
            }
            continue

        if current is None:
            continue

        step_match = re.match(
            r"\[STEP\]\s+step=(\d+)\s+action=([A-Za-z0-9_\.]+)\s+reward=([-0-9.]+)\s+done=(true|false)",
            line,
        )
        if step_match:
            reward = float(step_match.group(3))
            step_num = int(step_match.group(1))
            current["steps"] = max(current["steps"], step_num)
            current["total_reward"] += reward
            action_type = step_match.group(2).split(".")[-1]
            actions.append(
                {
                    "method": method,
                    "episode_id": current["episode_id"],
                    "task_id": current["task_id"],
                    "step": step_num,
                    "action_type": action_type,
                    "reward": reward,
                    "done": step_match.group(4) == "true",
                }
            )
            continue

        end_match = re.match(
            r"\[END\]\s+(?:success=(true|false)\s+)?steps=(\d+)\s+(?:score|grader_score)=([0-9.]+)",
            line,
        )
        if end_match:
            current["success"] = end_match.group(1) == "true" if end_match.group(1) is not None else False
            current["steps"] = int(end_match.group(2))
            current["grader_score"] = float(end_match.group(3))
            episodes.append(current)
            current = None

    return pd.DataFrame(episodes), pd.DataFrame(actions)


def _build_metrics_df(baseline: MethodMetrics, inference: MethodMetrics) -> pd.DataFrame:
    rows = [
        {"method": "baseline", "difficulty": "overall", "score": baseline.overall},
        {"method": "inference", "difficulty": "overall", "score": inference.overall},
    ]
    for task in TASK_ORDER:
        rows.append({"method": "baseline", "difficulty": task, "score": baseline.by_task.get(task, 0.0)})
        rows.append({"method": "inference", "difficulty": task, "score": inference.by_task.get(task, 0.0)})
    return pd.DataFrame(rows)


def _build_need_rl_html(baseline: MethodMetrics, inference: MethodMetrics, actions_df: pd.DataFrame) -> str:
    overall_delta = inference.overall - baseline.overall
    hard_delta = inference.by_task.get("hard", 0.0) - baseline.by_task.get("hard", 0.0)
    request_loops = 0
    search_loops = 0
    if not actions_df.empty:
        by_episode = actions_df.groupby("episode_id")["action_type"].apply(list)
        request_loops = int(sum(sum(a == "REQUEST_INFORMATION" for a in seq) >= 3 for seq in by_episode))
        search_loops = int(sum(sum(a == "SEARCH_POLICY" for a in seq) >= 3 for seq in by_episode))
    return f"""
    <div style="padding:16px;border:1px solid #2a2a2a;border-radius:12px;background:#111;">
      <div style="font-size:18px;font-weight:700;margin-bottom:10px;">Why RL Is Needed</div>
      <ul style="margin:0 0 0 16px;padding:0;line-height:1.6;">
        <li>Rule baseline is strong on deterministic patterns but brittle on hidden-rule and multi-step workflows.</li>
        <li>Generic LLM prompting still struggles with medium/hard trajectories and loop control.</li>
        <li>Current overall delta (inference - baseline): <b>{overall_delta:+.3f}</b>; hard-task delta: <b>{hard_delta:+.3f}</b>.</li>
        <li>Observed loop-like behavior in logs: <b>{request_loops}</b> request-heavy episodes, <b>{search_loops}</b> search-heavy episodes.</li>
        <li>RL objective: improve trajectory quality (fewer loops, correct final decisions, better hard-task scores).</li>
      </ul>
    </div>
    """


def _kpi_html(baseline: MethodMetrics, inference: MethodMetrics, training_rows: int) -> str:
    return f"""
    <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;">
      <div style="padding:14px;border:1px solid #2a2a2a;border-radius:12px;background:#111;">
        <div style="font-size:12px;color:#aaa;">Baseline Mean Grader Score</div>
        <div style="font-size:28px;font-weight:800;color:#5dade2;">{baseline.overall:.3f}</div>
        <div style="font-size:12px;color:#888;">n={baseline.total}</div>
      </div>
      <div style="padding:14px;border:1px solid #2a2a2a;border-radius:12px;background:#111;">
        <div style="font-size:12px;color:#aaa;">Generic LLM Mean Grader Score</div>
        <div style="font-size:28px;font-weight:800;color:#58d68d;">{inference.overall:.3f}</div>
        <div style="font-size:12px;color:#888;">n={inference.total}</div>
      </div>
      <div style="padding:14px;border:1px solid #2a2a2a;border-radius:12px;background:#111;">
        <div style="font-size:12px;color:#aaa;">Training Episodes Available</div>
        <div style="font-size:28px;font-weight:800;color:#f5b041;">{training_rows}</div>
        <div style="font-size:12px;color:#888;">from training/logs/episodes.jsonl</div>
      </div>
    </div>
    """


def _load_dashboard_data() -> Dict[str, Any]:
    baseline = _load_metrics(BASELINE_RESULTS, "baseline")
    inference = _load_metrics(INFERENCE_RESULTS, "inference")
    baseline_episodes, baseline_actions = _parse_episode_log(BASELINE_LOG, "baseline")
    inference_episodes, inference_actions = _parse_episode_log(INFERENCE_LOG, "inference")
    training_rows = _read_jsonl(TRAINING_EPISODES, last_n=800)
    training_df = pd.DataFrame(training_rows)

    episodes_df = pd.concat([baseline_episodes, inference_episodes], ignore_index=True)
    actions_df = pd.concat([baseline_actions, inference_actions], ignore_index=True)
    metrics_df = _build_metrics_df(baseline, inference)
    episode_ids = episodes_df["episode_id"].tolist() if not episodes_df.empty else []
    episode_choices = episode_ids or ["No episodes parsed"]

    return {
        "kpis_html": _kpi_html(baseline, inference, len(training_rows)),
        "need_rl_html": _build_need_rl_html(baseline, inference, actions_df),
        "metrics_df": metrics_df,
        "episodes_df": episodes_df,
        "actions_df": actions_df,
        "baseline_log_text": _safe_read_text(BASELINE_LOG) or "No baseline_run.log found.",
        "inference_log_text": _safe_read_text(INFERENCE_LOG) or "No inference_run.log found.",
        "training_log_text": _safe_read_text(TRAINING_EPISODES) or "No training logs found at training/logs/episodes.jsonl.",
        "training_df": training_df,
        "episode_choices": episode_choices,
    }


def _filter_episodes(
    episodes_df: pd.DataFrame,
    method: str,
    difficulty: str,
    min_score: float,
    action_type: str,
) -> pd.DataFrame:
    if episodes_df.empty:
        return pd.DataFrame()
    df = episodes_df.copy()
    if method != "all":
        df = df[df["method"] == method]
    if difficulty != "all":
        df = df[df["task_id"] == difficulty]
    df = df[df["grader_score"] >= float(min_score)]
    if action_type != "all":
        episodes_with_action = set()
        # action filter applied at callback level using global actions DF
        # this function keeps signature simple and score/task/method filters consistent
        return df[df["episode_id"].isin(episodes_with_action)]
    return df.sort_values(["method", "task_id", "grader_score"], ascending=[True, True, False])


def build_demo() -> gr.Blocks:
    """Create interactive benchmark dashboard mounted in HF Space."""
    with gr.Blocks(title="Corporate Compliance RL Dashboard") as demo:
        store = gr.State({})

        gr.Markdown(
            """
            # Corporate Compliance RL Dashboard
            Compare rule baseline, generic LLM inference, and RL training traces across episode-level behavior.
            This dashboard is designed to show **why RL is needed** in multi-step compliance decision-making.
            """
        )

        with gr.Row():
            refresh_btn = gr.Button("Refresh Dashboard", variant="primary")
            gr.Markdown("Data sources: `baseline_results.json`, `inference_results.json`, `baseline_run.log`, `inference_run.log`, optional `training/logs/episodes.jsonl`.")

        kpis_html = gr.HTML()
        need_rl_html = gr.HTML()
        compare_plot = gr.BarPlot(
            x="difficulty",
            y="score",
            color="method",
            title="Mean Grader Score Comparison",
            tooltip=["method", "difficulty", "score"],
        )

        gr.Markdown("## Episode Explorer")
        with gr.Row():
            filter_method = gr.Dropdown(choices=["all", "baseline", "inference"], value="all", label="Method")
            filter_task = gr.Dropdown(choices=["all", "easy", "medium", "hard"], value="all", label="Difficulty")
            filter_min = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.01, label="Min Grader Score")
            filter_action = gr.Dropdown(
                choices=["all", "SEARCH_POLICY", "REQUEST_INFORMATION", "RESOLVE_TICKET"],
                value="all",
                label="Contains Action",
            )
        episodes_table = gr.Dataframe(label="Episodes", wrap=True, max_height=320)

        with gr.Row():
            episode_selector = gr.Dropdown(choices=["No episodes parsed"], value="No episodes parsed", label="Select Episode")
            episode_summary = gr.Markdown("Pick an episode to inspect action-by-action trace.")
        action_timeline = gr.Dataframe(label="Action Timeline", wrap=True, max_height=320)

        gr.Markdown("## Training View")
        training_plot = gr.LinePlot(
            x="episode",
            y="total_reward",
            title="Training Reward Curve (if logs available)",
            tooltip=["episode", "task_id", "total_reward"],
        )
        training_table = gr.Dataframe(label="Recent Training Episodes", wrap=True, max_height=280)

        gr.Markdown("## Raw Logs")
        with gr.Tabs():
            with gr.Tab("Baseline Log"):
                baseline_log_box = gr.Code(language="shell", lines=18, label="baseline_run.log")
            with gr.Tab("Inference Log"):
                inference_log_box = gr.Code(language="shell", lines=18, label="inference_run.log")
            with gr.Tab("Training Log"):
                training_log_box = gr.Code(language="shell", lines=18, label="training/logs/episodes.jsonl")

        def refresh_dashboard() -> Tuple[Any, ...]:
            payload = _load_dashboard_data()
            training_df = payload["training_df"].copy()
            if not training_df.empty and "episode" not in training_df.columns:
                training_df["episode"] = range(1, len(training_df) + 1)
            episodes_df = payload["episodes_df"].copy()
            actions_df = payload["actions_df"].copy()
            payload["episodes_json"] = episodes_df.to_json(orient="records")
            payload["actions_json"] = actions_df.to_json(orient="records")
            return (
                payload,
                payload["kpis_html"],
                payload["need_rl_html"],
                payload["metrics_df"],
                episodes_df,
                gr.Dropdown(choices=payload["episode_choices"], value=payload["episode_choices"][0]),
                training_df,
                training_df.tail(30),
                payload["baseline_log_text"],
                payload["inference_log_text"],
                payload["training_log_text"],
            )

        def filter_rows(
            payload: Dict[str, Any], method: str, difficulty: str, min_score: float, action_type: str
        ) -> pd.DataFrame:
            episodes_df = pd.read_json(payload.get("episodes_json", "[]"))
            actions_df = pd.read_json(payload.get("actions_json", "[]"))
            if episodes_df.empty:
                return episodes_df
            df = episodes_df.copy()
            if method != "all":
                df = df[df["method"] == method]
            if difficulty != "all":
                df = df[df["task_id"] == difficulty]
            df = df[df["grader_score"] >= min_score]
            if action_type != "all" and not actions_df.empty:
                include_ids = set(actions_df[actions_df["action_type"] == action_type]["episode_id"])
                df = df[df["episode_id"].isin(include_ids)]
            return df.sort_values(["method", "task_id", "grader_score"], ascending=[True, True, False])

        def show_episode(payload: Dict[str, Any], episode_id: str) -> Tuple[str, pd.DataFrame]:
            episodes_df = pd.read_json(payload.get("episodes_json", "[]"))
            actions_df = pd.read_json(payload.get("actions_json", "[]"))
            if episodes_df.empty or episode_id not in set(episodes_df["episode_id"]):
                return "No episode selected.", pd.DataFrame()
            ep = episodes_df[episodes_df["episode_id"] == episode_id].iloc[0]
            step_df = actions_df[actions_df["episode_id"] == episode_id].sort_values("step")
            summary = (
                f"**Episode:** `{episode_id}` | **Method:** `{ep['method']}` | "
                f"**Task:** `{ep['task_id']}` | **Score:** `{ep['grader_score']:.3f}` | "
                f"**Steps:** `{int(ep['steps'])}` | **Total reward:** `{ep['total_reward']:.3f}`"
            )
            return summary, step_df

        demo.load(
            refresh_dashboard,
            outputs=[
                store,
                kpis_html,
                need_rl_html,
                compare_plot,
                episodes_table,
                episode_selector,
                training_plot,
                training_table,
                baseline_log_box,
                inference_log_box,
                training_log_box,
            ],
        )
        refresh_btn.click(
            refresh_dashboard,
            outputs=[
                store,
                kpis_html,
                need_rl_html,
                compare_plot,
                episodes_table,
                episode_selector,
                training_plot,
                training_table,
                baseline_log_box,
                inference_log_box,
                training_log_box,
            ],
        )

        for trigger in [filter_method.change, filter_task.change, filter_min.change, filter_action.change]:
            trigger(
                filter_rows,
                inputs=[store, filter_method, filter_task, filter_min, filter_action],
                outputs=[episodes_table],
            )

        episode_selector.change(show_episode, inputs=[store, episode_selector], outputs=[episode_summary, action_timeline])

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
