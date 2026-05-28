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
    extract_task_id,
    filter_dataset_by_curriculum,
    grpo_supports_rollout_func,
    load_unsloth_model,
    native_grpo_supports_rollout_func,
    parse_json_payload,
    parse_model_action,
    require_rollout_dependencies,
    render_compliance_prompt,
    resolve_grpo_trainer,
    resolve_precision,
    validate_training_checkpoint,
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
    loop_penalties = kwargs.get("loop_penalty", [0.0] * len(scores))
    unresolved_penalties = kwargs.get("unresolved_penalty", [0.0] * len(scores))
    if len(loop_penalties) < len(completions):
        loop_penalties = list(loop_penalties) + [loop_penalties[-1]] * (
            len(completions) - len(loop_penalties)
        )
    if len(unresolved_penalties) < len(completions):
        unresolved_penalties = list(unresolved_penalties) + [unresolved_penalties[-1]] * (
            len(completions) - len(unresolved_penalties)
        )
    combined: List[float] = []
    for i in range(len(completions)):
        score = (
            float(scores[i]) + float(loop_penalties[i]) + float(unresolved_penalties[i])
        )
        combined.append(max(-1.0, min(1.0, score)))
    return combined


def _step_local_env(
    env: ComplianceEnv,
    text: str,
    observation: Dict[str, Any],
) -> tuple[float, bool, Dict[str, Any]]:
    payload = parse_model_action(text, observation)
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
    return render_compliance_prompt(_trainer_tokenizer(trainer), observation)


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
        "loop_penalty",
        "unresolved_penalty",
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


def _rollout_length_limits(trainer) -> tuple[int, int]:
    args = getattr(trainer, "args", None)
    max_prompt = int(getattr(args, "max_prompt_length", 384) or 384)
    max_completion = int(getattr(args, "max_completion_length", 96) or 96)
    return max_prompt, max_completion


def _tokens_from_generation(trainer, gen: Dict[str, Any]) -> tuple[List[int], List[int], List[float]]:
    from training.rollout_generation import cap_rollout_tokens

    max_prompt, max_completion = _rollout_length_limits(trainer)
    return cap_rollout_tokens(
        list(gen.get("prompt_ids") or []),
        list(gen.get("completion_ids") or []),
        list(gen.get("logprobs") or []),
        max_prompt=max_prompt,
        max_completion=max_completion,
    )


def _generate_step(trainer, step_prompt: str) -> Dict[str, Any]:
    from training.rollout_generation import generate_rollout_completions

    _max_prompt, max_completion = _rollout_length_limits(trainer)
    generation_overrides = {
        "max_new_tokens": max_completion,
        "do_sample": True,
        "temperature": 1.0,
        "pad_token_id": getattr(
            getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None),
            "pad_token_id",
            None,
        ),
    }
    return generate_rollout_completions(
        trainer,
        [step_prompt],
        generation_overrides=generation_overrides,
        max_prompt_length=_max_prompt,
    )[0]


def rollout_once_local(trainer, task_id: str) -> Dict[str, Any]:
    """Run one full ComplianceEnv episode using tutorial-style generation."""
    env = ComplianceEnv()
    env.reset(task_id=task_id, split="train")
    last_prompt_ids: List[int] = [0]
    last_completion_ids: List[int] = [0]
    last_logprobs: List[float] = [0.0]
    format_scores: List[float] = []
    raw_json_valid = True
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
        raw_json_valid = raw_json_valid and bool(valid_action)
        format_scores.append(0.10 if valid_action else -0.10)

        last_prompt_ids, last_completion_ids, last_logprobs = _tokens_from_generation(
            trainer, gen
        )

        _reward, done, obs = _step_local_env(env, text, obs)

    actions = env.state.actions_history
    request_count = sum(1 for action in actions if action.get("action_type") == "RequestInformation")
    resolve_count = sum(1 for action in actions if action.get("action_type") == "ResolveTicket")
    loop_penalty = -0.15 * max(0, request_count - 1)
    unresolved_penalty = -0.25 if resolve_count == 0 else 0.0

    return {
        "prompt_ids": last_prompt_ids,
        "completion_ids": last_completion_ids,
        "logprobs": last_logprobs,
        "grader_score": _grade_local_env(env, task_id),
        "env_reward": env.state.cumulative_reward,
        "format_reward": 0.05 if (raw_json_valid and resolve_count > 0) else -0.15,
        "loop_penalty": loop_penalty,
        "unresolved_penalty": unresolved_penalty,
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

            last_prompt_ids: List[int] = [0]
            last_completion_ids: List[int] = [0]
            last_logprobs: List[float] = [0.0]
            raw_json_valid = True
            done = reset_r.done
            steps = 0
            max_steps = int(obs.get("max_steps", 8))

            while not done and steps < max_steps:
                steps += 1
                step_prompt = _render_step_prompt(trainer, task_id, obs)
                gen = _generate_step(trainer, step_prompt)
                text = _generated_text(trainer, gen)
                last_prompt_ids, last_completion_ids, last_logprobs = _tokens_from_generation(
                    trainer, gen
                )

                raw_payload = parse_json_payload(text)
                raw_json_valid = raw_json_valid and bool(raw_payload)
                payload = parse_model_action(text, obs)
                step_r = client.step(ComplianceAction(**payload))
                obs = step_r.observation.model_dump()
                done = step_r.done

            state = client.state()
            actions_history = getattr(state, "actions_history", [])
            request_count = sum(
                1 for action in actions_history if action.get("action_type") == "RequestInformation"
            )
            resolve_count = sum(
                1 for action in actions_history if action.get("action_type") == "ResolveTicket"
            )
            grader = grade_episode(
                task_id=task_id,
                actions_history=actions_history,
                ground_truth_decision=claim.get("ground_truth_decision", "Approve"),
                claim=claim,
            )
            results.append(
                {
                    "prompt_ids": last_prompt_ids,
                    "completion_ids": last_completion_ids,
                    "logprobs": last_logprobs,
                    "grader_score": float(grader["score"]),
                    "env_reward": sum(getattr(state, "rewards_history", []) or []),
                    "format_reward": 0.05 if (raw_json_valid and resolve_count > 0) else -0.15,
                    "loop_penalty": -0.15 * max(0, request_count - 1),
                    "unresolved_penalty": -0.25 if resolve_count == 0 else 0.0,
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
    parser.add_argument("--dataset-path", default="training/data/sft_dataset_balanced.jsonl")
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
    parser.add_argument("--max-prompt-length", type=int, default=384)
    parser.add_argument("--max-completion-length", type=int, default=96)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
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
    if args.sft_checkpoint:
        validate_training_checkpoint(args.sft_checkpoint)
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

    total_cap = args.max_prompt_length + args.max_completion_length
    if total_cap > args.max_seq_length:
        raise ValueError(
            f"max_prompt_length ({args.max_prompt_length}) + max_completion_length "
            f"({args.max_completion_length}) must be <= max_seq_length ({args.max_seq_length})."
        )

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
