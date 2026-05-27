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
    native_grpo_supports_rollout_func,
    parse_json_payload,
    require_rollout_dependencies,
    resolve_grpo_trainer,
    resolve_precision,
)

COMPLIANCE_SYSTEM_PROMPT = """\
You are an AI Compliance Officer. Audit employee expense claims against company policy.

Use the available action JSON types:
- SearchPolicy when policy details are hidden or unclear.
- RequestInformation when a required document is missing.
- ResolveTicket when you are ready to decide Approve, Reject, or Escalate.

Return only one valid JSON action for the next step.
"""

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
    scores = kwargs.get("format_reward")
    if scores is not None:
        if len(scores) < len(completions):
            scores = list(scores) + [scores[-1]] * (len(completions) - len(scores))
        return [float(s) for s in scores[: len(completions)]]

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


def _trainer_tokenizer(trainer):
    return (
        getattr(trainer, "processing_class", None)
        or getattr(trainer, "tokenizer", None)
    )


def _render_step_prompt(trainer, task_id: str, observation: Dict[str, Any]) -> str:
    """Render the next-step prompt in the chat-template style used by the tutorial."""
    user_prompt = build_step_prompt(task_id, observation)
    tokenizer = _trainer_tokenizer(trainer)
    messages = [
        {"role": "system", "content": COMPLIANCE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
    return f"{COMPLIANCE_SYSTEM_PROMPT}\n\n{user_prompt}\n"


def _generated_text(trainer, generation: Dict[str, Any]) -> str:
    text = generation.get("text")
    if isinstance(text, str) and text.strip():
        return text
    completion = generation.get("completion")
    if completion is not None:
        return completion_to_text(completion)
    completion_ids = generation.get("completion_ids")
    tokenizer = _trainer_tokenizer(trainer)
    if completion_ids and tokenizer is not None and hasattr(tokenizer, "decode"):
        return tokenizer.decode(completion_ids, skip_special_tokens=True)
    return completion_to_text(generation)


def _batch_rollouts(episodes: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """TRL rollout_func contract: return a dict of lists, not list of dicts."""
    keys = (
        "prompt_ids",
        "completion_ids",
        "logprobs",
        "grader_score",
        "env_reward",
        "format_reward",
    )
    return {key: [episode[key] for episode in episodes] for key in keys}


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
    from training.rollout_generation import generate_rollout_completions

    return generate_rollout_completions(trainer, [step_prompt])[0]


def rollout_once_local(trainer, task_id: str) -> Dict[str, Any]:
    """Run one full ComplianceEnv episode using tutorial-style generation."""
    env = ComplianceEnv()
    env.reset(task_id=task_id, split="train")
    all_prompt_ids: List[int] = []
    all_completion_ids: List[int] = []
    all_logprobs: List[float] = []
    format_scores: List[float] = []
    done = False
    steps = 0
    max_steps = env.task_max_steps.get(task_id, 8)

    obs = env._get_observation().model_dump()
    while not done and steps < max_steps:
        steps += 1
        step_prompt = _render_step_prompt(trainer, task_id, obs)
        gen = _generate_step(trainer, step_prompt)
        text = _generated_text(trainer, gen)
        payload = parse_json_payload(text)
        valid_action = payload and payload.get("action_type") in {
            "SearchPolicy",
            "RequestInformation",
            "ResolveTicket",
        }
        format_scores.append(0.10 if valid_action else -0.10)

        if gen.get("prompt_ids"):
            all_prompt_ids.extend(gen["prompt_ids"])
        if gen.get("completion_ids"):
            all_completion_ids.extend(gen["completion_ids"])
        if gen.get("logprobs"):
            all_logprobs.extend(gen["logprobs"])

        _reward, done, obs = _step_local_env(env, text)

    return {
        "prompt_ids": all_prompt_ids or [0],
        "completion_ids": all_completion_ids or [0],
        "logprobs": all_logprobs or [0.0],
        "grader_score": _grade_local_env(env, task_id),
        "env_reward": env.state.cumulative_reward,
        "format_reward": format_scores[-1] if format_scores else -0.10,
    }


def rollout_local(prompts: List[str], trainer, task_ids: List[str]) -> Dict[str, List[Any]]:
    """Multi-turn rollouts using in-process ComplianceEnv (default for Colab)."""
    episodes = [
        rollout_once_local(trainer, task_id)
        for _prompt, task_id in zip(prompts, task_ids)
    ]
    return _batch_rollouts(episodes)


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
) -> Dict[str, List[Any]]:
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
                step_prompt = _render_step_prompt(trainer, task_id, obs)
                gen = _generate_step(trainer, step_prompt)
                text = _generated_text(trainer, gen)
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
                    "env_reward": sum(getattr(state, "rewards_history", []) or []),
                    "format_reward": 0.10,
                }
            )
    return _batch_rollouts(results)


def make_rollout_func(use_local_env: bool, api_url: str):
    def rollout_func(prompts: List[str], trainer=None) -> Dict[str, List[Any]]:
        if trainer is None:
            raise RuntimeError("rollout_func requires the active GRPOTrainer instance.")
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
    parser.add_argument("--max-prompt-length", type=int, default=1400)
    parser.add_argument("--max-completion-length", type=int, default=128)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--use-vllm", action="store_true", help="Enable TRL vLLM colocate mode.")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.1)
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
            if native_grpo_supports_rollout_func():
                print("TRL rollout_func: supported (native)")
            else:
                print("TRL rollout_func: supported (compat shim)")
        else:
            print(
                "WARN: TRL rollout_func unavailable — "
                "install training/requirements-training.txt and retry."
            )
        print("Dry run OK — dataset and curriculum wiring ready.")
        return

    require_rollout_dependencies(require_generate=True)
    if not grpo_supports_rollout_func():
        raise RuntimeError(
            "TRL GRPOTrainer does not accept rollout_func. "
            "Install training/requirements-training.txt and retry."
        )

    from transformers import TrainerCallback
    from trl import GRPOConfig

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
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        warmup_steps=args.warmup_steps,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
    )
    if args.use_vllm:
        config_kwargs.update(
            {
                "use_vllm": True,
                "vllm_mode": "colocate",
                "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            }
        )
    if args.max_train_steps is not None:
        config_kwargs["max_steps"] = args.max_train_steps

    config = GRPOConfig(**config_kwargs)
    rollout_fn = make_rollout_func(use_local_env=use_local, api_url=args.api_url)
    trainer_cls = resolve_grpo_trainer()

    trainer = trainer_cls(
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
