"""Compatibility rollout generation for TRL stacks without experimental OpenEnv APIs."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch


def cap_rollout_tokens(
    prompt_ids: List[int],
    completion_ids: List[int],
    logprobs: List[float],
    *,
    max_prompt: int,
    max_completion: int,
) -> Tuple[List[int], List[int], List[float]]:
    """Keep prompt/completion within GRPOConfig limits (model max_seq_length is 512 on T4)."""
    capped_prompt = prompt_ids[-max_prompt:] if len(prompt_ids) > max_prompt else list(prompt_ids)
    capped_completion = (
        completion_ids[:max_completion]
        if len(completion_ids) > max_completion
        else list(completion_ids)
    )
    capped_logprobs = logprobs[: len(capped_completion)]
    if len(capped_logprobs) < len(capped_completion):
        capped_logprobs = capped_logprobs + [0.0] * (len(capped_completion) - len(capped_logprobs))
    return capped_prompt or [0], capped_completion or [0], capped_logprobs or [0.0]


def _resolve_model(trainer):
    model = getattr(trainer, "model_wrapped", None) or getattr(trainer, "model", None)
    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is not None and model is not None and hasattr(accelerator, "unwrap_model"):
        try:
            return accelerator.unwrap_model(model)
        except Exception:
            return model
    return model


def _resolve_tokenizer(trainer):
    return getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)


def _completion_logprobs(
    scores: List[torch.Tensor],
    sequences: torch.Tensor,
    prompt_lengths: List[int],
) -> List[List[float]]:
    if not scores:
        return [[] for _ in range(sequences.shape[0])]

    batch_logprobs: List[List[float]] = [[] for _ in range(sequences.shape[0])]
    for step_idx, step_scores in enumerate(scores):
        step_logprobs = torch.log_softmax(step_scores.float(), dim=-1)
        for row_idx in range(sequences.shape[0]):
            token_pos = prompt_lengths[row_idx] + step_idx
            if token_pos >= sequences.shape[1]:
                continue
            token_id = int(sequences[row_idx, token_pos].item())
            batch_logprobs[row_idx].append(float(step_logprobs[row_idx, token_id].item()))
    return batch_logprobs


def generate_rollout_completions(
    trainer,
    prompts: List[str],
    *,
    generation_overrides: Dict[str, Any] | None = None,
    max_prompt_length: int | None = None,
) -> List[Dict[str, Any]]:
    """Generate completion payloads matching the OpenEnv helper schema."""
    model = _resolve_model(trainer)
    tokenizer = _resolve_tokenizer(trainer)
    if model is None or tokenizer is None:
        raise RuntimeError("Trainer is missing model/tokenizer for rollout generation.")

    model_device = next(model.parameters()).device
    args = getattr(trainer, "args", None)
    prompt_cap = max_prompt_length or int(getattr(args, "max_prompt_length", 384) or 384)
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=prompt_cap,
    )
    input_ids = encoded["input_ids"].to(model_device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model_device)

    generate_kwargs: Dict[str, Any] = {
        "input_ids": input_ids,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    if attention_mask is not None:
        generate_kwargs["attention_mask"] = attention_mask

    if isinstance(generation_overrides, dict):
        generate_kwargs.update(
            {key: value for key, value in generation_overrides.items() if value is not None}
        )
    else:
        generation_config = generation_overrides or getattr(trainer, "generation_config", None)
        if generation_config is not None:
            generate_kwargs["generation_config"] = generation_config
        else:
            # Safe fallback for dry-runs / minimally configured trainers.
            generate_kwargs.update(
                {"max_new_tokens": 96, "do_sample": True, "temperature": 1.0}
            )

    model.eval()
    with torch.no_grad():
        output = model.generate(**generate_kwargs)

    sequences = output.sequences
    prompt_lengths = (
        attention_mask.sum(dim=1).tolist()
        if attention_mask is not None
        else [input_ids.shape[1]] * input_ids.shape[0]
    )
    scores = list(getattr(output, "scores", ()) or ())
    all_logprobs = _completion_logprobs(scores, sequences, [int(p) for p in prompt_lengths])

    rows: List[Dict[str, Any]] = []
    for row_idx in range(sequences.shape[0]):
        prompt_len = int(prompt_lengths[row_idx])
        prompt_ids = sequences[row_idx, :prompt_len].tolist()
        completion_ids = sequences[row_idx, prompt_len:].tolist()
        text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        logprobs = all_logprobs[row_idx]
        if not logprobs:
            logprobs = [0.0 for _ in completion_ids]
        rows.append(
            {
                "prompt_ids": prompt_ids,
                "completion_ids": completion_ids,
                "logprobs": logprobs,
                "text": text,
            }
        )
    return rows
