"""Compatibility GRPOTrainer for TRL builds without native rollout_func support."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from trl import GRPOTrainer


class OpenEnvGRPOTrainer(GRPOTrainer):
    """Backport rollout_func wiring for TRL<=0.24 style trainers."""

    def __init__(self, *args, rollout_func=None, **kwargs):
        self.rollout_func = rollout_func
        self._openenv_extra_fields: List[Dict[str, Any]] = []
        super().__init__(*args, **kwargs)

    def _build_rollout_payload(self, prompts: List[str]) -> Dict[str, List[Any]]:
        if self.rollout_func is None:
            raise RuntimeError("rollout_func is required for OpenEnvGRPOTrainer.")
        try:
            payload = self.rollout_func(prompts, trainer=self)
        except TypeError:
            payload = self.rollout_func(prompts, self)

        required = ("prompt_ids", "completion_ids", "logprobs")
        if not isinstance(payload, dict) or any(key not in payload for key in required):
            raise RuntimeError("rollout_func must return dict with prompt_ids/completion_ids/logprobs.")
        expected = len(prompts)
        for key in required:
            if not isinstance(payload[key], list) or len(payload[key]) != expected:
                raise RuntimeError(f"rollout_func returned invalid `{key}` shape.")
        return payload

    def _generate(self, prompts: List[str], images):
        if self.rollout_func is None:
            return super()._generate(prompts, images)

        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        payload = self._build_rollout_payload(prompts)
        prompt_ids = [list(ids) for ids in payload["prompt_ids"]]
        completion_ids = [list(ids) for ids in payload["completion_ids"]]
        logprobs = [list(lp) for lp in payload["logprobs"]]

        completion_lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        total_completion_tokens = agg_completion_lengths.sum()

        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
        agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
        total_prompt_tokens = agg_prompt_lengths.sum()
        if mode == "train":
            self.state.num_input_tokens_seen += (total_prompt_tokens + total_completion_tokens).item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids], device=device)
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
        term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
        if len(term_completion_lengths) == 0:
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        base_keys = {"prompt_ids", "completion_ids", "logprobs"}
        self._openenv_extra_fields = [
            {key: payload[key][idx] for key in payload.keys() if key not in base_keys}
            for idx in range(len(prompts))
        ]
        return prompt_ids, completion_ids, total_completion_tokens, logprobs, {}

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        if self._openenv_extra_fields and len(self._openenv_extra_fields) == len(inputs):
            merged_inputs = []
            for idx, row in enumerate(inputs):
                merged = dict(row)
                merged.update(self._openenv_extra_fields[idx])
                merged_inputs.append(merged)
            inputs = merged_inputs
        return super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
