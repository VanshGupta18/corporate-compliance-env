"""Evaluate a checkpoint against compliance tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

import requests
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

TASKS = ["easy", "medium", "hard"]


def prompt_from_observation(task_id: str, observation: Dict[str, Any]) -> str:
    return (
        "You are an AI compliance officer. Return only valid action JSON.\n"
        f"Task: {task_id}\n"
        f"Observation: {json.dumps(observation, ensure_ascii=True)}"
    )


def generate_action(model, tokenizer, prompt: str) -> Dict[str, Any]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    candidate = text.split("{", 1)
    if len(candidate) < 2:
        return {"action_type": "ResolveTicket", "decision": "Reject", "reason": "Fallback parse"}
    payload_text = "{" + candidate[1]
    payload_text = payload_text.split("}", 1)[0] + "}"
    try:
        return json.loads(payload_text)
    except Exception:
        return {"action_type": "ResolveTicket", "decision": "Reject", "reason": "Fallback parse"}


def run_episode(api_url: str, model, tokenizer, task_id: str) -> Dict[str, Any]:
    reset = requests.post(f"{api_url}/reset", params={"task_id": task_id}, timeout=30)
    reset.raise_for_status()
    observation = reset.json().get("observation", {})
    done = False
    total_reward = 0.0
    max_steps = int(observation.get("max_steps", 8))
    steps = 0
    trajectory: List[Dict[str, Any]] = []

    while not done and steps < max_steps:
        steps += 1
        prompt = prompt_from_observation(task_id, observation)
        action = generate_action(model, tokenizer, prompt)
        step = requests.post(f"{api_url}/step", json={"action": action}, timeout=30)
        step.raise_for_status()
        result = step.json()
        reward = float(result.get("reward", 0.0))
        total_reward += reward
        done = bool(result.get("done", False))
        trajectory.append(
            {
                "step": steps,
                "action_type": action.get("action_type"),
                "decision": action.get("decision"),
                "reward": reward,
                "done": done,
            }
        )
        observation = result.get("observation", {})
    return {
        "task_id": task_id,
        "steps": steps,
        "total_reward": total_reward,
        "done": done,
        "trajectory": trajectory,
        "ground_truth_decision": observation.get("ground_truth_decision"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a LoRA/PEFT checkpoint.")
    parser.add_argument("--checkpoint", default="training/checkpoints/grpo")
    parser.add_argument("--api-url", default="http://127.0.0.1:7860")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--episode-log-file", default="training/logs/episodes.jsonl")
    parser.add_argument("--clear-log", action="store_true")
    args = parser.parse_args()
    log_path = Path(args.episode_log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if args.clear_log and log_path.exists():
        log_path.unlink()

    model = AutoPeftModelForCausalLM.from_pretrained(args.checkpoint, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)

    report: Dict[str, List[float]] = {task: [] for task in TASKS}
    episode_idx = 0
    for task_id in TASKS:
        for _ in range(args.episodes):
            episode_idx += 1
            episode = run_episode(args.api_url, model, tokenizer, task_id)
            report[task_id].append(episode["total_reward"])
            episode["episode_index"] = episode_idx
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(episode, ensure_ascii=True) + "\n")

    for task_id, scores in report.items():
        print(f"{task_id}: mean={mean(scores):.3f} min={min(scores):.3f} max={max(scores):.3f}")
    print(f"overall_mean={mean([score for scores in report.values() for score in scores]):.3f}")


if __name__ == "__main__":
    main()
