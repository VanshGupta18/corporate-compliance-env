"""Interactive Gradio Command Center for the Corporate Compliance Environment."""

from __future__ import annotations

import json
import subprocess
import threading
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr
import pandas as pd


def _read_jsonl(path: str, last_n: int = 200) -> List[Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
        selected = lines[-last_n:] if last_n > 0 else lines
        rows = []
        for line in selected:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows
    except Exception:
        return []


def stream_command(command: List[str]):
    """Generator to stream the output of a subprocess."""
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True
    )
    log_content = ""
    for line in iter(process.stdout.readline, ""):
        log_content += line
        yield log_content
    process.stdout.close()
    process.wait()
    yield log_content + f"\n--- Process finished with exit code {process.returncode} ---"


def parse_baseline() -> Tuple[str, pd.DataFrame]:
    """Parse baseline_results.json and format it for the dashboard."""
    baseline_path = Path("baseline_results.json")
    if not baseline_path.exists():
        return "Baseline has not been run or results cannot be found.", pd.DataFrame()

    try:
        data = json.loads(baseline_path.read_text(encoding="utf-8"))
        
        # Support both old and new baseline_results formats
        metrics = data.get("metrics", data)
        overall = metrics.get("overall_metrics", metrics)
        
        acc = overall.get("accuracy", 0) * 100
        total = overall.get("total_claims", overall.get("total_evaluations", 0))
        
        diffs = metrics.get("by_difficulty", metrics.get("performance_by_difficulty", {}))
        
        easy_acc = diffs.get("easy", {}).get("accuracy", 0) * 100
        med_acc = diffs.get("medium", {}).get("accuracy", 0) * 100
        hard_acc = diffs.get("hard", {}).get("accuracy", 0) * 100
        
        stats_html = f"""
        <div style="display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin-bottom:20px;">
          <div style="padding:14px;border:1px solid #2a2a2a;border-radius:12px;background:#111;">
            <div style="font-size:12px;color:#aaa;">Overall Score</div>
            <div style="font-size:22px;font-weight:700;color:#2ecc71;">{acc:.1f}%</div>
          </div>
          <div style="padding:14px;border:1px solid #2a2a2a;border-radius:12px;background:#111;">
            <div style="font-size:12px;color:#aaa;">Total Claims</div>
            <div style="font-size:22px;font-weight:700;">{total}</div>
          </div>
          <div style="padding:14px;border:1px solid #2a2a2a;border-radius:12px;background:#111;">
            <div style="font-size:12px;color:#aaa;">Easy Score</div>
            <div style="font-size:22px;font-weight:700;color:#3498db;">{easy_acc:.1f}%</div>
          </div>
          <div style="padding:14px;border:1px solid #2a2a2a;border-radius:12px;background:#111;">
            <div style="font-size:12px;color:#aaa;">Medium Score</div>
            <div style="font-size:22px;font-weight:700;color:#f39c12;">{med_acc:.1f}%</div>
          </div>
          <div style="padding:14px;border:1px solid #2a2a2a;border-radius:12px;background:#111;">
            <div style="font-size:12px;color:#aaa;">Hard Score</div>
            <div style="font-size:22px;font-weight:700;color:#e74c3c;">{hard_acc:.1f}%</div>
          </div>
        </div>
        """
        
        # Build confusion matrix dataframe
        cm = metrics.get("confusion_matrix", {})
        cm_data = []
        for true_label, preds in cm.items():
            row = {"Ground Truth": true_label}
            row.update(preds)
            cm_data.append(row)
            
        df = pd.DataFrame(cm_data) if cm_data else pd.DataFrame()
        return stats_html, df

    except Exception as e:
        return f"Error parsing baseline results: {str(e)}", pd.DataFrame()


def parse_inference() -> Tuple[str, pd.DataFrame]:
    """Parse inference_results.json and format it for the dashboard."""
    inference_path = Path("inference_results.json")
    if not inference_path.exists():
        return "Inference has not been run or results cannot be found.", pd.DataFrame()

    try:
        data = json.loads(inference_path.read_text(encoding="utf-8"))
        
        metrics = data.get("metrics", data)
        overall = metrics.get("overall_metrics", metrics)
        
        acc = overall.get("accuracy", 0) * 100
        total = overall.get("total_evaluations", overall.get("total_claims", 0))
        
        diffs = metrics.get("performance_by_difficulty", metrics.get("by_difficulty", {}))
        
        easy_acc = diffs.get("easy", {}).get("accuracy", 0) * 100
        med_acc = diffs.get("medium", {}).get("accuracy", 0) * 100
        hard_acc = diffs.get("hard", {}).get("accuracy", 0) * 100
        
        stats_html = f"""
        <div style="display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin-bottom:20px;">
          <div style="padding:14px;border:1px solid #2a2a2a;border-radius:12px;background:#111;">
            <div style="font-size:12px;color:#aaa;">Overall Score</div>
            <div style="font-size:22px;font-weight:700;color:#2ecc71;">{acc:.1f}%</div>
          </div>
          <div style="padding:14px;border:1px solid #2a2a2a;border-radius:12px;background:#111;">
            <div style="font-size:12px;color:#aaa;">Total Claims</div>
            <div style="font-size:22px;font-weight:700;">{total}</div>
          </div>
          <div style="padding:14px;border:1px solid #2a2a2a;border-radius:12px;background:#111;">
            <div style="font-size:12px;color:#aaa;">Easy Score</div>
            <div style="font-size:22px;font-weight:700;color:#3498db;">{easy_acc:.1f}%</div>
          </div>
          <div style="padding:14px;border:1px solid #2a2a2a;border-radius:12px;background:#111;">
            <div style="font-size:12px;color:#aaa;">Medium Score</div>
            <div style="font-size:22px;font-weight:700;color:#f39c12;">{med_acc:.1f}%</div>
          </div>
          <div style="padding:14px;border:1px solid #2a2a2a;border-radius:12px;background:#111;">
            <div style="font-size:12px;color:#aaa;">Hard Score</div>
            <div style="font-size:22px;font-weight:700;color:#e74c3c;">{hard_acc:.1f}%</div>
          </div>
        </div>
        """
        
        cm = metrics.get("confusion_matrix", {})
        cm_data = []
        for true_label, preds in cm.items():
            row = {"Ground Truth": true_label}
            row.update(preds)
            cm_data.append(row)
            
        df = pd.DataFrame(cm_data) if cm_data else pd.DataFrame()
        return stats_html, df

    except Exception as e:
        return f"Error parsing inference results: {str(e)}", pd.DataFrame()


def update_training_plots() -> Tuple[gr.Plot, pd.DataFrame]:
    episodes = _read_jsonl("training/logs/episodes.jsonl", last_n=500)
    df = pd.DataFrame(episodes)
    if df.empty or "total_reward" not in df.columns:
        return None, pd.DataFrame()
        
    df["episode"] = range(1, len(df) + 1)
    
    # We will use gr.LinePlot on the frontend, we just need to return the dataframe
    return df, df


def run_baseline_stream():
    for log in stream_command(["bash", "-c", "source .venv/bin/activate && [ -f .env ] && set -a && source .env && set +a && PYTHONUNBUFFERED=1 python app/baseline.py"]):
        yield log

def run_train_stream():
    for log in stream_command(["bash", "-c", "source .venv/bin/activate && [ -f .env ] && set -a && source .env && set +a && PYTHONUNBUFFERED=1 python training/grpo_train.py"]):
        yield log

def run_inference_sync():
    import subprocess
    subprocess.run(["bash", "-c", "source .venv/bin/activate && [ -f .env ] && set -a && source .env && set +a && python inference.py"], check=True)
    return parse_inference()

def build_demo() -> gr.Blocks:
    """Create the interactive execution Command Center dashboard."""
    with gr.Blocks(title="Corporate Compliance Command Center") as demo:
        gr.Markdown("# 🏛️ Corporate Policy Compliance Environment: Command Center")
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("""
                ### The Challenge: Automating Expense Policy Auditing
                Every company processes hundreds of expense reports, approval requests, and compliance tickets every day. Today, this requires human auditors who manually read policy documents and make judgement calls. 
                
                This **OpenEnv-compliant Reinforcement Learning environment** trains an RL agent to do exactly that — understand a request, retrieve the relevant policy rule, and make a compliant decision.
                """)
            with gr.Column(scale=1):
                gr.HTML(
                    "<div style='padding:10px 12px;border:1px solid #2a2a2a;border-radius:10px;background:#111;'>"
                    "<b>Built with:</b> FastAPI • OpenEnv • Gradio • TRL/Unsloth<br/>"
                    "<b>Domain:</b> Finance & Compliance<br/>"
                    "</div>"
                )
                
        gr.Markdown("### Execution Pipeline")
        with gr.Row():
            btn_baseline = gr.Button("🏃 Run Baseline System", variant="secondary")
            btn_train = gr.Button("🧠 Train RLHF LLM Model", variant="primary")
            btn_inference = gr.Button("🤖 Run LLM Inference", variant="secondary")

        gr.Markdown("---")
        
        gr.Markdown("### Baseline Agent Performance")
        btn_refresh_baseline = gr.Button("🔄 Refresh Metrics")
        baseline_kpis = gr.HTML()
        baseline_cm = gr.Dataframe(label="Confusion Matrix")
        
        # Hidden init loading
        demo.load(parse_baseline, outputs=[baseline_kpis, baseline_cm])
        btn_refresh_baseline.click(parse_baseline, outputs=[baseline_kpis, baseline_cm])

        gr.Markdown("---")
        
        gr.Markdown("### Reinforcement Learning Pipeline")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### Training Logs (Live)")
                train_log_output = gr.Code(language="shell", lines=20, label="GRPO Training stdout")
            
            with gr.Column(scale=1):
                gr.Markdown("#### Rewards (Live)")
                reward_plot = gr.LinePlot(
                    x="episode",
                    y="total_reward",
                    title="Total Reward per Episode",
                    tooltip=["episode", "task_id", "total_reward"]
                )
                episodes_table = gr.Dataframe(label="Recent Episodes", wrap=True, max_height=300)
        
        gr.Timer(3).tick(update_training_plots, outputs=[reward_plot, episodes_table])

        gr.Markdown("---")
        
        gr.Markdown("### LLM Inference Performance")
        btn_refresh_inference = gr.Button("🔄 Refresh Inference Metrics")
        inference_kpis = gr.HTML()
        inference_cm = gr.Dataframe(label="Confusion Matrix")
        
        # Hidden init loading
        demo.load(parse_inference, outputs=[inference_kpis, inference_cm])
        btn_refresh_inference.click(parse_inference, outputs=[inference_kpis, inference_cm])


        # Wiring executions
        btn_baseline.click(
            fn=run_baseline_stream, 
            outputs=[train_log_output]
        ).then(parse_baseline, outputs=[baseline_kpis, baseline_cm])

        btn_train.click(
            fn=run_train_stream,
            outputs=[train_log_output]
        )

        btn_inference.click(
            fn=run_inference_sync,
            outputs=[inference_kpis, inference_cm]
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
