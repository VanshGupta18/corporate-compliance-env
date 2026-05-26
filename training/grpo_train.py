"""GRPO post-training with multi-turn compliance rollouts (Unsloth + in-process env)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from typing import Any, Dict, List, Optional

from datasets import load_dataset

from app.graders import grade_episode
from app.models import ComplianceAction
from app.server.environment import ComplianceEnv

from training.training_utils import (
    CURRICULUM_STAGES,
    build_step_prompt,
    extract_task_id,
    filter_dataset_by_curriculum,
    grpo_supports_rollout_func,
    load_unsloth_model,
    parse_json_payload,
    require_rollout_dependencies,
    resolve_precision,
)

try:
    from training.learning_curve import log_learning_point
except ImportError:
    log_learning_point = None  # type: ignore

try:
    from app.client import ComplianceEnvClient
except ImportError:
    ComplianceEnvClient = None  # type: ignore


class JsonlMetricsCallback:
    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        payload = {"step": int(state.global_step), **logs}
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        content = completion.get("content", completion.get("text", ""))
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            if parts:
                return "".join(parts)
        return json.dumps(completion, ensure_ascii=True)
    if isinstance(completion, (list, tuple)) and completion:
        return completion_to_text(completion[0])
    return str(completion)


def action_json_reward(completions, **kwargs) -> List[float]:
    """Small format reward; task quality comes from grader_reward."""
    rewards = []
    for completion in completions:
        payload = parse_json_payload(completion_to_text(completion))
        valid = payload and payload.get("action_type") in {
            "SearchPolicy",
            "RequestInformation",
            "ResolveTicket",
        }
        rewards.append(0.10 if valid else -0.10)
    return rewards


def grader_reward(completions, **kwargs) -> List[float]:
    scores = kwargs.get("grader_score", [])
    if not scores:
        raise RuntimeError(
            "grader_score missing from rollout kwargs — rollout_func was not wired. "
            "Upgrade TRL and pass rollout_func to GRPOTrainer."
        )
    if len(scores) < len(completions):
        scores = list(scores) + [scores[-1]] * (len(completions) - len(scores))
    return [float(s) for s in scores[: len(completions)]]


def _step_local_env(env: ComplianceEnv, text: str) -> tuple[float, bool, Dict[str, Any]]:
    payload = parse_json_payload(text)
    if not payload:
        return -0.5, True, {}
    try:
        obs = env.step(ComplianceAction(**payload))
        return float(obs.reward or 0.0), bool(obs.done), obs.model_dump()
    except Exception:
        return -0.5, True, {}


def _grade_local_env(env: ComplianceEnv, task_id: str) -> float:
    claim = getattr(env, "_current_claim", {}) or {}
    grader = grade_episode(
        task_id=task_id,
        actions_history=env.state.actions_history,
        ground_truth_decision=claim.get("ground_truth_decision", "Approve"),
        claim=claim,
    )
    return float(grader["score"])


def _generate_step(trainer, step_prompt: str) -> Dict[str, Any]:
    try:
        from trl.experimental.openenv import generate_rollout_completions
    except ImportError as exc:
        raise RuntimeError(
            "generate_rollout_completions unavailable — upgrade trl for OpenEnv rollouts."
        ) from exc
    return generate_rollout_completions(trainer, [step_prompt])[0]


def rollout_local(prompts: List[str], trainer, task_ids: List[str]) -> List[Dict[str, Any]]:
    """Multi-turn rollouts using in-process ComplianceEnv (default for Colab)."""
    env = ComplianceEnv()
    outputs: List[Dict[str, Any]] = []

    for _prompt, task_id in zip(prompts, task_ids):
        env.reset(task_id=task_id, split="train")
        all_prompt_ids: List[int] = []
        all_completion_ids: List[int] = []
        all_logprobs: List[float] = []
        done = False
        steps = 0
        max_steps = env.task_max_steps.get(task_id, 8)

        obs = env._get_observation().model_dump()
        while not done and steps < max_steps:
            steps += 1
            step_prompt = build_step_prompt(task_id, obs)
            gen = _generate_step(trainer, step_prompt)
            text = completion_to_text(gen.get("completion", gen))
            if gen.get("prompt_ids"):
                all_prompt_ids.extend(gen["prompt_ids"])
            if gen.get("completion_ids"):
                all_completion_ids.extend(gen["completion_ids"])
            if gen.get("logprobs"):
                all_logprobs.extend(gen["logprobs"])

            _reward, done, obs = _step_local_env(env, text)

        outputs.append(
            {
                "prompt_ids": all_prompt_ids or [0],
                "completion_ids": all_completion_ids or [0],
                "logprobs": all_logprobs or [0.0],
                "grader_score": _grade_local_env(env, task_id),
                "env_reward": env.state.cumulative_reward,
            }
        )
    return outputs


def _load_claim_lookup() -> Dict[str, Dict[str, Any]]:
    root = Path(__file__).parent.parent
    lookup: Dict[str, Dict[str, Any]] = {}
    for path in (root / "data" / "claims.json", root / "data" / "splits" / "train.json"):
        if not path.exists():
            continue
        for claim in json.loads(path.read_text(encoding="utf-8")).get("claims", []):
            lookup[claim["id"]] = claim
            if claim.get("ticket_id"):
                lookup[str(claim["ticket_id"])] = claim
    return lookup


def rollout_remote(
    prompts: List[str],
    trainer,
    task_ids: List[str],
    api_url: str,
) -> List[Dict[str, Any]]:
    if ComplianceEnvClient is None:
        raise RuntimeError("ComplianceEnvClient unavailable; use --use-local-env for Colab.")

    claim_lookup = _load_claim_lookup()
    results: List[Dict[str, Any]] = []

    with ComplianceEnvClient(base_url=api_url).sync() as client:
        for _prompt, task_id in zip(prompts, task_ids):
            reset_r = client.reset(task_id=task_id, split="train")
            obs = reset_r.observation.model_dump()
            ticket_id = obs.get("ticket_id")
            claim = claim_lookup.get(ticket_id, {})

            all_prompt_ids: List[int] = []
            all_completion_ids: List[int] = []
            all_logprobs: List[float] = []
            done = reset_r.done
            steps = 0
            max_steps = int(obs.get("max_steps", 8))

            while not done and steps < max_steps:
                steps += 1
                step_prompt = build_step_prompt(task_id, obs)
                gen = _generate_step(trainer, step_prompt)
                text = completion_to_text(gen.get("completion", gen))
                if gen.get("prompt_ids"):
                    all_prompt_ids.extend(gen["prompt_ids"])
                if gen.get("completion_ids"):
                    all_completion_ids.extend(gen["completion_ids"])
                if gen.get("logprobs"):
                    all_logprobs.extend(gen["logprobs"])

                payload = parse_json_payload(text)
                if not payload:
                    break
                step_r = client.step(ComplianceAction(**payload))
                obs = step_r.observation.model_dump()
                done = step_r.done

            state = client.state()
            actions_history = getattr(state, "actions_history", [])
            grader = grade_episode(
                task_id=task_id,
                actions_history=actions_history,
                ground_truth_decision=claim.get("ground_truth_decision", "Approve"),
                claim=claim,
            )
            results.append(
                {
                    "prompt_ids": all_prompt_ids or [0],
                    "completion_ids": all_completion_ids or [0],
                    "logprobs": all_logprobs or [0.0],
                    "grader_score": float(grader["score"]),
                }
            )
    return results


def make_rollout_func(use_local_env: bool, api_url: str):
    def rollout_func(prompts: List[str], trainer) -> List[Dict[str, Any]]:
        task_ids = [extract_task_id(p) for p in prompts]
        if use_local_env:
            return rollout_local(prompts, trainer, task_ids)
        return rollout_remote(prompts, trainer, task_ids, api_url)

    return rollout_func


def check_server_health(api_url: str) -> bool:
    try:
        import requests

        r = requests.get(f"{api_url.rstrip('/')}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GRPO on compliance environment.")
    parser.add_argument("--model-id", default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
    parser.add_argument(
        "--sft-checkpoint",
        default=None,
        help="Optional SFT adapter directory (same layout as --output-dir from sft_train.py).",
    )
    parser.add_argument("--dataset-path", default="training/data/sft_dataset.jsonl")
    parser.add_argument("--output-dir", default="training/checkpoints/grpo")
    parser.add_argument(
        "--curriculum-stage",
        default="stage_3_hard",
        choices=list(CURRICULUM_STAGES.keys()),
        help="Filter/weight GRPO prompts by curriculum stage.",
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:7860")
    parser.add_argument("--use-local-env", action="store_true", default=True)
    parser.add_argument("--use-remote-env", action="store_true", help="Use WebSocket server at api-url")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--log-file", default="training/logs/grpo_metrics.jsonl")
    parser.add_argument(
        "--learning-curve-file",
        default="training/logs/learning_curve.jsonl",
    )
    parser.add_argument("--precision", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate curriculum + TRL rollout contract without GPU training.",
    )
    args = parser.parse_args()

    use_local = not args.use_remote_env
    if args.use_remote_env and not check_server_health(args.api_url):
        raise RuntimeError(f"Server not healthy at {args.api_url}")

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    if "task_id" not in dataset.column_names:
        dataset = dataset.map(
            lambda ex: {
                "prompt": ex["prompt"],
                "task_id": extract_task_id(ex["prompt"]),
            },
            remove_columns=dataset.column_names,
        )
    else:
        dataset = dataset.map(
            lambda ex: {"prompt": ex["prompt"], "task_id": ex.get("task_id", "easy")},
            remove_columns=[c for c in dataset.column_names if c not in ("prompt", "task_id")],
        )

    dataset = filter_dataset_by_curriculum(dataset, args.curriculum_stage)
    print(
        f"GRPO curriculum={args.curriculum_stage} examples={len(dataset)} "
        f"local_env={use_local}"
    )

    if args.dry_run:
        if grpo_supports_rollout_func():
            print("TRL rollout_func: supported")
        else:
            print(
                "WARN: TRL GRPOTrainer missing rollout_func — "
                "pip install -U 'trl>=0.14.0' before GPU training."
            )
        print("Dry run OK — dataset and curriculum wiring ready.")
        return

    require_rollout_dependencies(require_generate=True)
    if not grpo_supports_rollout_func():
        raise RuntimeError(
            "TRL GRPOTrainer does not accept rollout_func. "
            "pip install -U 'trl>=0.14.0' before Colab training."
        )

    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    class _JsonlMetricsCallback(TrainerCallback, JsonlMetricsCallback):
        pass

    use_bf16, use_fp16 = resolve_precision(args.precision)
    model_id = args.sft_checkpoint or args.model_id
    model, tokenizer = load_unsloth_model(
        model_id,
        args.max_seq_length,
        load_in_4bit=True,
        for_training=True,
    )

    config_kwargs = dict(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=128,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to="none",
    )
    if args.max_train_steps is not None:
        config_kwargs["max_steps"] = args.max_train_steps

    config = GRPOConfig(**config_kwargs)
    rollout_fn = make_rollout_func(use_local_env=use_local, api_url=args.api_url)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[action_json_reward, grader_reward],
        args=config,
        train_dataset=dataset,
        rollout_func=rollout_fn,
        callbacks=[_JsonlMetricsCallback(args.log_file)],
    )
    trainer.train()

    if log_learning_point is not None:
        log_learning_point(
            stage=args.curriculum_stage,
            global_step=int(getattr(trainer.state, "global_step", 0)),
            split="validation",
            phase="post_rl",
            log_file=Path(args.learning_curve_file),
        )
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved GRPO adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
