"""Read-only Gradio monitor for training and episode logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr


def _read_jsonl(path: str, last_n: int = 200) -> List[Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    lines = file_path.read_text(encoding="utf-8").splitlines()
    selected = lines[-last_n:] if last_n > 0 else lines
    rows: List[Dict[str, Any]] = []
    for line in selected:
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _summary(episodes: List[Dict[str, Any]], metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not episodes and not metrics:
        return {"status": "waiting_for_logs", "message": "No logs yet. Start training/evaluation first."}

    episode_rewards = [float(row.get("total_reward", 0.0)) for row in episodes if "total_reward" in row]
    task_counts: Dict[str, int] = {}
    for row in episodes:
        task = row.get("task_id", "unknown")
        task_counts[task] = task_counts.get(task, 0) + 1

    latest_metric = metrics[-1] if metrics else {}
    return {
        "episodes_seen": len(episodes),
        "avg_episode_reward": round(sum(episode_rewards) / len(episode_rewards), 4) if episode_rewards else None,
        "tasks_seen": task_counts,
        "latest_training_metric": latest_metric,
    }


def _kpi_html(summary: Dict[str, Any], latest_episode: Dict[str, Any]) -> str:
    episodes_seen = summary.get("episodes_seen", 0)
    avg_reward = summary.get("avg_episode_reward")
    avg_reward_txt = f"{avg_reward:.3f}" if isinstance(avg_reward, float) else "n/a"
    latest_task = latest_episode.get("task_id", "n/a")
    latest_reward = latest_episode.get("total_reward")
    latest_reward_txt = f"{latest_reward:.3f}" if isinstance(latest_reward, (int, float)) else "n/a"

    return f"""
<div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;">
  <div style="padding:14px;border:1px solid #2a2a2a;border-radius:12px;background:#111;">
    <div style="font-size:12px;color:#aaa;">Episodes Logged</div>
    <div style="font-size:22px;font-weight:700;">{episodes_seen}</div>
  </div>
  <div style="padding:14px;border:1px solid #2a2a2a;border-radius:12px;background:#111;">
    <div style="font-size:12px;color:#aaa;">Average Episode Reward</div>
    <div style="font-size:22px;font-weight:700;">{avg_reward_txt}</div>
  </div>
  <div style="padding:14px;border:1px solid #2a2a2a;border-radius:12px;background:#111;">
    <div style="font-size:12px;color:#aaa;">Latest Task</div>
    <div style="font-size:22px;font-weight:700;">{latest_task}</div>
  </div>
  <div style="padding:14px;border:1px solid #2a2a2a;border-radius:12px;background:#111;">
    <div style="font-size:12px;color:#aaa;">Latest Episode Reward</div>
    <div style="font-size:22px;font-weight:700;">{latest_reward_txt}</div>
  </div>
</div>
"""


def refresh_monitor(
    episode_log_path: str,
    metrics_log_path: str,
    max_rows: int,
) -> Tuple[str, Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    episodes = _read_jsonl(episode_log_path, last_n=max_rows)
    metrics = _read_jsonl(metrics_log_path, last_n=max_rows)
    latest_episode = episodes[-1] if episodes else {}
    summary = _summary(episodes, metrics)
    return _kpi_html(summary, latest_episode), summary, latest_episode, episodes, metrics


def build_demo() -> gr.Blocks:
    """Create a read-only training monitor dashboard."""
    with gr.Blocks(title="Corporate Compliance Training Monitor") as demo:
        gr.Markdown("""
# Corporate Compliance AI - OpenEnv Training Dashboard
### RL-style LLM Decision System for Enterprise Expense Auditing

This product demo shows how a language model improves in a custom OpenEnv environment:
- **Episode-level actions** (SearchPolicy, RequestInformation, ResolveTicket)
- **Reward trajectory** and completion quality
- **Training metrics** from GRPO fine-tuning
""")

        gr.HTML(
            "<div style='padding:10px 12px;border:1px solid #2a2a2a;border-radius:10px;background:#111;'>"
            "<b>Built with:</b> FastAPI • OpenEnv • Gradio • TRL/Unsloth • Hugging Face"
            "</div>"
        )

        episode_log_path = gr.Textbox(
            label="Episode log file",
            value="training/logs/episodes.jsonl",
            interactive=False,
        )
        metrics_log_path = gr.Textbox(
            label="Training metrics log file",
            value="training/logs/grpo_metrics.jsonl",
            interactive=False,
        )
        max_rows = gr.Slider(20, 1000, step=20, value=200, label="Rows to display", interactive=True)
        refresh = gr.Button("Refresh Monitor")

        kpis = gr.HTML(label="KPI Overview")
        summary = gr.JSON(label="Run Summary")
        latest_episode = gr.JSON(label="Latest Episode")
        episodes_table = gr.Dataframe(label="Episode Timeline", wrap=True)
        metrics_table = gr.Dataframe(label="Training Metrics", wrap=True)

        with gr.Tab("Docs"):
            gr.Markdown(
                "- Swagger: `/docs`\n"
                "- OpenAPI schema: `/openapi.json`\n"
                "- OpenEnv tasks: `/tasks`\n"
                "- This monitor is read-only."
            )

        refresh.click(
            refresh_monitor,
            inputs=[episode_log_path, metrics_log_path, max_rows],
            outputs=[kpis, summary, latest_episode, episodes_table, metrics_table],
        )

        timer = gr.Timer(5)
        timer.tick(
            refresh_monitor,
            inputs=[episode_log_path, metrics_log_path, max_rows],
            outputs=[kpis, summary, latest_episode, episodes_table, metrics_table],
        )
    return demo
