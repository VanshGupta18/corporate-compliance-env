"""Shared stdout log format for baseline, inference, and training eval runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO


class Tee:
    """Write console output to both terminal and a log file."""

    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def format_step_log(step: int, action: Dict[str, Any], reward: float, done: bool) -> str:
    """Single-line [STEP] record consumed by the dashboard log parser."""
    action_type = action.get("action_type", "")
    parts = [
        f"[STEP] step={step}",
        f"action={action_type}",
        f"reward={reward:.2f}",
        f"done={str(done).lower()}",
    ]
    if action.get("query"):
        q = str(action["query"]).replace('"', "'")
        parts.append(f'query="{q}"')
    if action.get("message"):
        m = str(action["message"]).replace('"', "'")
        parts.append(f'message="{m}"')
    if action.get("decision") is not None:
        parts.append(f"decision={action['decision']}")
    if action.get("reason"):
        r = str(action["reason"]).replace('"', "'")[:120]
        parts.append(f'reason="{r}"')
    return " ".join(parts)


def log_claim_start(method: str, claim_id: str, difficulty: str) -> None:
    print(f"Running {method} for claim {claim_id} ({difficulty})...", flush=True)


def log_episode_start(task_id: str, *, model: str = "training-agent") -> None:
    print(
        f"[START] task={task_id.upper()} env=corporate-compliance-env model={model}",
        flush=True,
    )


def log_episode_end(
    *,
    steps: int,
    grader_score: float,
    rewards: List[float],
    success: bool,
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={grader_score:.3f} rewards={rewards_str}",
        flush=True,
    )


def write_results_json(
    path: Path,
    scores_by_diff: Dict[str, List[float]],
    *,
    all_scores: Optional[List[float]] = None,
) -> None:
    """Write baseline/inference/training-compatible results JSON."""
    combined = all_scores if all_scores is not None else [
        s for diff in ("easy", "medium", "hard") for s in scores_by_diff.get(diff, [])
    ]
    payload = {
        "metrics": {
            "overall_metrics": {
                "total_claims": len(combined),
                "mean_grader_score": sum(combined) / len(combined) if combined else 0.0,
            },
            "performance_by_difficulty": {},
        }
    }
    for diff in ("easy", "medium", "hard"):
        scores = scores_by_diff.get(diff, [])
        payload["metrics"]["performance_by_difficulty"][diff] = {
            "mean_grader_score": sum(scores) / len(scores) if scores else 0.0,
            "total": len(scores),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_episode_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append one training episode record for the dashboard JSONL loader."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def run_with_log(log_path: Path, main_fn) -> None:
    """Run main() with stdout/stderr tee'd to log_path (same pattern as baseline/inference)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        tee = Tee(sys.stdout, log_file)
        with redirect_stdout_stderr(tee):
            print(f"[LOG] Writing run log to {log_path}", flush=True)
            main_fn()


class redirect_stdout_stderr:
    def __init__(self, tee: Tee):
        self.tee = tee

    def __enter__(self):
        import contextlib

        self._out = contextlib.redirect_stdout(self.tee)
        self._err = contextlib.redirect_stderr(self.tee)
        self._out.__enter__()
        self._err.__enter__()
        return self.tee

    def __exit__(self, *args):
        self._err.__exit__(*args)
        self._out.__exit__(*args)
