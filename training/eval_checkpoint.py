"""Evaluate a checkpoint with in-process env (Colab) or WebSocket server."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from statistics import mean
from typing import Any, Dict, List, Optional, Protocol

import torch

from app.graders import grade_episode
from app.models import ComplianceAction
from app.server.environment import ComplianceEnv
from training.training_utils import (
    normalize_compliance_action,
    parse_model_action,
    render_compliance_prompt,
)

TASKS = ["easy", "medium", "hard"]


class EnvRunner(Protocol):
    def reset(self, task_id: str, claim_id: str | None = None) -> tuple[Dict[str, Any], bool]:
        ...

    def step(
        self,
        action: Dict[str, Any],
        observation: Optional[Dict[str, Any]] = None,
    ) -> tuple[Dict[str, Any], float, bool]:
        ...

    def actions_history(self) -> List[Dict[str, Any]]:
        ...


class LocalEnvRunner:
    def __init__(self, split: str = "validation"):
        self.env = ComplianceEnv()
        self.split = split
        self._claim: Dict[str, Any] = {}

    def reset(self, task_id: str, claim_id: str | None = None) -> tuple[Dict[str, Any], bool]:
        self.env.reset(task_id=task_id, claim_id=claim_id, split=self.split)
        self._claim = dict(getattr(self.env, "_current_claim", {}) or {})
        obs = self.env._get_observation().model_dump()
        return obs, False

    def step(
        self,
        action: Dict[str, Any],
        observation: Optional[Dict[str, Any]] = None,
    ) -> tuple[Dict[str, Any], float, bool]:
        payload = normalize_compliance_action(action, observation)
        result = self.env.step(ComplianceAction(**payload))
        obs = result.model_dump()
        return obs, float(result.reward or 0.0), bool(result.done)

    def actions_history(self) -> List[Dict[str, Any]]:
        return list(self.env.state.actions_history)

    @property
    def claim(self) -> Dict[str, Any]:
        return self._claim


class WebSocketEnvRunner:
    def __init__(self, api_url: str, split: str = "validation"):
        from app.client import ComplianceEnvClient

        self._client_ctx = ComplianceEnvClient(base_url=api_url)
        self.client = self._client_ctx.__enter__()
        self.split = split
        self._claim: Dict[str, Any] = {}
        self._claims_by_id = {c["id"]: c for c in load_split_claims(split)}

    def reset(self, task_id: str, claim_id: str | None = None) -> tuple[Dict[str, Any], bool]:
        kwargs: Dict[str, Any] = {"task_id": task_id, "split": self.split}
        if claim_id:
            kwargs["claim_id"] = claim_id
        result = self.client.reset(**kwargs)
        obs = result.observation.model_dump()
        self._claim = dict(self._claims_by_id.get(claim_id, {})) if claim_id else {}
        return obs, result.done

    def step(
        self,
        action: Dict[str, Any],
        observation: Optional[Dict[str, Any]] = None,
    ) -> tuple[Dict[str, Any], float, bool]:
        payload = normalize_compliance_action(action, observation)
        result = self.client.step(ComplianceAction(**payload))
        obs = result.observation.model_dump()
        return obs, float(result.reward or 0.0), result.done

    def actions_history(self) -> List[Dict[str, Any]]:
        state = self.client.state()
        return list(getattr(state, "actions_history", []))

    @property
    def claim(self) -> Dict[str, Any]:
        return self._claim

    def close(self) -> None:
        self._client_ctx.__exit__(None, None, None)


def load_model_and_tokenizer(checkpoint: str):
    try:
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer

        model = AutoPeftModelForCausalLM.from_pretrained(checkpoint, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
        return model, tokenizer
    except Exception:
        from training.training_utils import load_unsloth_model

        model, tokenizer = load_unsloth_model(
            checkpoint,
            max_seq_length=512,
            load_in_4bit=True,
            for_training=False,
        )
        return model, tokenizer


def generate_action(
    model,
    tokenizer,
    task_id: str,
    observation: Dict[str, Any],
) -> Dict[str, Any]:
    prompt = render_compliance_prompt(tokenizer, task_id, observation)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=pad_id,
        )
    new_ids = output[0][input_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return parse_model_action(text, observation)


def run_episode(
    runner: EnvRunner,
    model,
    tokenizer,
    task_id: str,
    claim: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    claim_id = claim["id"] if claim else None
    observation, done = runner.reset(task_id=task_id, claim_id=claim_id)
    steps = 0
    max_steps = int(observation.get("max_steps", 8))
    trajectory: List[Dict[str, Any]] = []

    while not done and steps < max_steps:
        steps += 1
        action = generate_action(model, tokenizer, task_id, observation)
        observation, reward, done = runner.step(action, observation)
        trajectory.append(
            {
                "step": steps,
                "action_type": action.get("action_type"),
                "decision": action.get("decision"),
                "reward": reward,
                "done": done,
            }
        )

    actions_history = runner.actions_history()
    episode_claim = claim or getattr(runner, "claim", {}) or {}
    gt = episode_claim.get("ground_truth_decision", "Approve")
    grader = grade_episode(
        task_id,
        actions_history,
        gt,
        claim=episode_claim,
    )
    action_counts = {
        "SearchPolicy": sum(
            1 for action in actions_history if action.get("action_type") == "SearchPolicy"
        ),
        "RequestInformation": sum(
            1
            for action in actions_history
            if action.get("action_type") == "RequestInformation"
        ),
        "ResolveTicket": sum(
            1 for action in actions_history if action.get("action_type") == "ResolveTicket"
        ),
    }
    final_resolve = next(
        (a for a in reversed(actions_history) if a.get("action_type") == "ResolveTicket"),
        None,
    )
    final_decision = final_resolve.get("decision") if isinstance(final_resolve, dict) else None
    loop_flag = action_counts["RequestInformation"] >= 2 and action_counts["ResolveTicket"] == 0
    required_document = episode_claim.get("missing_document") or episode_claim.get(
        "required_document"
    )
    components = grader.get("components", {})

    return {
        "task_id": task_id,
        "claim_id": claim_id,
        "steps": steps,
        "grader_score": grader["score"],
        "total_reward": sum(t["reward"] for t in trajectory),
        "done": done,
        "trajectory": trajectory,
        "ground_truth_decision": gt,
        "grader_components": components,
        "action_counts": action_counts,
        "final_decision": final_decision,
        "decision_correct": bool(final_decision and str(final_decision) == str(gt)),
        "request_loop": loop_flag,
        "truncated_max_steps": bool(steps >= max_steps and not final_resolve),
        "required_document": required_document,
        "document_request_matched": bool(components.get("correct_document_request", 0) > 0),
    }


def load_split_claims(split: str) -> List[Dict[str, Any]]:
    split_path = Path(f"data/splits/{split}.json")
    if split_path.exists():
        return json.loads(split_path.read_text(encoding="utf-8")).get("claims", [])
    return json.loads(Path("data/claims.json").read_text(encoding="utf-8")).get("claims", [])


def load_baseline_scores(path: str) -> Dict[str, float]:
    baseline_path = Path(path)
    if not baseline_path.exists():
        return {}
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    by_task = payload.get("metrics", {}).get("performance_by_difficulty", {})
    return {
        task_id: float(stats.get("mean_grader_score", 0.0))
        for task_id, stats in by_task.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LoRA checkpoint.")
    parser.add_argument("--checkpoint", default="training/checkpoints/grpo")
    parser.add_argument("--api-url", default="http://127.0.0.1:7860")
    parser.add_argument(
        "--local-env",
        action="store_true",
        default=True,
        help="Use in-process ComplianceEnv (default; recommended for Colab).",
    )
    parser.add_argument(
        "--use-remote-env",
        action="store_true",
        help="Evaluate via WebSocket server at --api-url.",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--episode-log-file", default="training/logs/episodes.jsonl")
    parser.add_argument("--baseline-file", default="baseline_results.json")
    parser.add_argument("--clear-log", action="store_true")
    args = parser.parse_args()

    log_path = Path(args.episode_log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if args.clear_log and log_path.exists():
        log_path.unlink()

    model, tokenizer = load_model_and_tokenizer(args.checkpoint)
    claims = load_split_claims(args.split)
    use_local = not args.use_remote_env

    report: Dict[str, List[float]] = {t: [] for t in TASKS}
    episode_metrics: Dict[str, List[Dict[str, Any]]] = {t: [] for t in TASKS}
    episode_idx = 0

    if use_local:
        runner: EnvRunner = LocalEnvRunner(split=args.split)
        runners = [runner]
    else:
        ws = WebSocketEnvRunner(api_url=args.api_url, split=args.split)
        runners = [ws]

    try:
        for runner in runners:
            for task_id in TASKS:
                task_claims = [c for c in claims if c.get("task_difficulty") == task_id]
                n = min(args.episodes, len(task_claims) or args.episodes)
                for i in range(n):
                    claim = task_claims[i % len(task_claims)] if task_claims else None
                    episode_idx += 1
                    episode = run_episode(runner, model, tokenizer, task_id, claim=claim)
                    report[task_id].append(episode["grader_score"])
                    episode_metrics[task_id].append(episode)
                    episode["episode_index"] = episode_idx
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(episode, ensure_ascii=True) + "\n")
    finally:
        if not use_local and isinstance(runners[0], WebSocketEnvRunner):
            runners[0].close()

    mode = "local" if use_local else "remote"
    print(f"eval_mode={mode} split={args.split}")
    baseline_scores = load_baseline_scores(args.baseline_file)
    for task_id, scores in report.items():
        if scores:
            episodes = episode_metrics[task_id]
            decision_acc = mean(
                1.0 if episode.get("decision_correct") else 0.0 for episode in episodes
            )
            loop_rate = mean(
                1.0 if episode.get("request_loop") else 0.0 for episode in episodes
            )
            floor_rate = mean(
                1.0 if float(episode.get("grader_score", 0.0)) <= 0.02 else 0.0
                for episode in episodes
            )
            escalate_rate = mean(
                1.0 if str(episode.get("final_decision")) == "Escalate" else 0.0
                for episode in episodes
            )
            print(
                f"{task_id}: grader_mean={mean(scores):.3f} "
                f"min={min(scores):.3f} max={max(scores):.3f} n={len(scores)}"
            )
            print(
                f"{task_id}: decision_acc={decision_acc:.3f} "
                f"loop_rate={loop_rate:.3f} floor_rate={floor_rate:.3f} "
                f"escalate_rate={escalate_rate:.3f}"
            )
            if task_id in baseline_scores:
                baseline = baseline_scores[task_id]
                delta = mean(scores) - baseline
                gate_pass = delta >= -0.05 and loop_rate <= 0.25
                gate = "PASS" if gate_pass else "FAIL"
                print(
                    f"{task_id}: baseline_mean={baseline:.3f} delta={delta:+.3f} gate={gate}"
                )
    flat = [s for scores in report.values() for s in scores]
    if flat:
        print(f"overall_grader_mean={mean(flat):.3f}")


if __name__ == "__main__":
    main()
