"""Evaluate a checkpoint with in-process env (Colab) or WebSocket server."""

from __future__ import annotations

import argparse
import json
import re
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
from training.training_utils import build_step_prompt

TASKS = ["easy", "medium", "hard"]


class EnvRunner(Protocol):
    def reset(self, task_id: str, claim_id: str | None = None) -> tuple[Dict[str, Any], bool]:
        ...

    def step(self, action: Dict[str, Any]) -> tuple[Dict[str, Any], float, bool]:
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

    def step(self, action: Dict[str, Any]) -> tuple[Dict[str, Any], float, bool]:
        result = self.env.step(ComplianceAction(**action))
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

    def reset(self, task_id: str, claim_id: str | None = None) -> tuple[Dict[str, Any], bool]:
        kwargs: Dict[str, Any] = {"task_id": task_id, "split": self.split}
        if claim_id:
            kwargs["claim_id"] = claim_id
        result = self.client.reset(**kwargs)
        obs = result.observation.model_dump()
        self._claim = {}
        return obs, result.done

    def step(self, action: Dict[str, Any]) -> tuple[Dict[str, Any], float, bool]:
        result = self.client.step(ComplianceAction(**action))
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


def parse_action(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {
            "action_type": "ResolveTicket",
            "decision": "Reject",
            "reason": "Parse error",
        }
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {
            "action_type": "ResolveTicket",
            "decision": "Reject",
            "reason": "Parse error",
        }


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


def generate_action(model, tokenizer, prompt: str) -> Dict[str, Any]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    if prompt in text:
        text = text.split(prompt, 1)[-1]
    return parse_action(text)


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
        prompt = build_step_prompt(task_id, observation)
        action = generate_action(model, tokenizer, prompt)
        observation, reward, done = runner.step(action)
        trajectory.append(
            {
                "step": steps,
                "action_type": action.get("action_type"),
                "decision": action.get("decision"),
                "reward": reward,
                "done": done,
            }
        )

    episode_claim = claim or getattr(runner, "claim", {}) or {}
    gt = episode_claim.get("ground_truth_decision", "Approve")
    grader = grade_episode(
        task_id,
        runner.actions_history(),
        gt,
        claim=episode_claim,
    )

    return {
        "task_id": task_id,
        "claim_id": claim_id,
        "steps": steps,
        "grader_score": grader["score"],
        "total_reward": sum(t["reward"] for t in trajectory),
        "done": done,
        "trajectory": trajectory,
        "ground_truth_decision": gt,
    }


def load_split_claims(split: str) -> List[Dict[str, Any]]:
    split_path = Path(f"data/splits/{split}.json")
    if split_path.exists():
        return json.loads(split_path.read_text(encoding="utf-8")).get("claims", [])
    return json.loads(Path("data/claims.json").read_text(encoding="utf-8")).get("claims", [])


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
                    episode["episode_index"] = episode_idx
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(episode, ensure_ascii=True) + "\n")
    finally:
        if not use_local and isinstance(runners[0], WebSocketEnvRunner):
            runners[0].close()

    mode = "local" if use_local else "remote"
    print(f"eval_mode={mode} split={args.split}")
    for task_id, scores in report.items():
        if scores:
            print(
                f"{task_id}: grader_mean={mean(scores):.3f} "
                f"min={min(scores):.3f} max={max(scores):.3f} n={len(scores)}"
            )
    flat = [s for scores in report.values() for s in scores]
    if flat:
        print(f"overall_grader_mean={mean(flat):.3f}")


if __name__ == "__main__":
    main()
