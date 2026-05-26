# Colab Training Runbook (Unsloth SFT + GRPO)

**Notebook:** open [`notebooks/Colab_T4_Training.ipynb`](notebooks/Colab_T4_Training.ipynb) in Colab (Runtime → T4 GPU) for a cell-by-cell run of this guide.

This guide targets **Google Colab** with a T4 GPU. You do **not** need a local GPU or a running WebSocket server for training or evaluation — rollouts use the in-process `ComplianceEnv`.

Default model: `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`  
Default T4 settings: `max_seq_length=512`, `batch_size=1`, `grad_accum=8`, `num_generations=2`

## 0) Colab setup

```python
# Cell 1 — clone (adjust URL to your fork)
!git clone https://github.com/YOUR_USER/meta-openenv.git
%cd meta-openenv

# Cell 2 — env runtime deps, then Unsloth stack (avoid pinning torch/accelerate on Colab)
!pip install -q -r requirements.txt
!pip install -q "unsloth==2026.4.6"
!pip install -q "trl==0.24.0" "datasets>=3.4.1,<4.4.0"
# Runtime → Restart session (required — do not import unsloth in the same cell as pip)

# Cell 3 — verify stack + smoke test (after restart)
# Run the notebook verify cell, then:
!python training/smoke_test.py
```

Optional: regenerate claims if missing splits:

```bash
!python data/generate_dataset.py --train-per-diff 120 --val-per-diff 40 --test-per-diff 40
```

## 1) Prepare SFT data

`generate_dataset.py` creates **tickets** (claims). `prepare_data.py` turns heuristic rollouts into **prompt/response** rows for SFT (and GRPO prompts).

```bash
!python training/prepare_data.py --episodes-per-task 40 --split train
```

Outputs:
- `training/data/trajectories.json`
- `training/data/sft_dataset.jsonl`

## 2) Baseline learning curve (optional)

```bash
!python training/learning_curve.py --stage stage_0_baseline --step 0
```

Logs to `training/logs/learning_curve.jsonl` with per-difficulty scores.

## 3) SFT warm start (Unsloth QLoRA)

```bash
!python training/sft_train.py \
  --model-id unsloth/Qwen2.5-3B-Instruct-bnb-4bit \
  --dataset-path training/data/sft_dataset.jsonl \
  --output-dir training/checkpoints/sft \
  --max-length 512 \
  --batch-size 1 \
  --grad-accum 8
```

Dry-run (dataset only):

```bash
!python training/sft_train.py --dry-run
```

The SFT checkpoint directory is loaded directly by GRPO via `--sft-checkpoint`.

## 4) GRPO with curriculum stages

Curriculum stages (easy → hard):

| Stage | Tasks trained |
|-------|----------------|
| `stage_1_easy` | easy only |
| `stage_2_medium` | easy + medium (weighted) |
| `stage_3_hard` | easy + medium + hard (weighted) |

**Stage 1 — easy only**

```bash
!python training/grpo_train.py \
  --sft-checkpoint training/checkpoints/sft \
  --curriculum-stage stage_1_easy \
  --output-dir training/checkpoints/grpo_stage1 \
  --max-seq-length 512 \
  --num-generations 2 \
  --batch-size 1 \
  --grad-accum 8 \
  --max-train-steps 100
```

**Stage 2 — add medium**

```bash
!python training/grpo_train.py \
  --sft-checkpoint training/checkpoints/grpo_stage1 \
  --curriculum-stage stage_2_medium \
  --output-dir training/checkpoints/grpo_stage2 \
  --max-train-steps 100
```

**Stage 3 — full curriculum**

```bash
!python training/grpo_train.py \
  --sft-checkpoint training/checkpoints/grpo_stage2 \
  --curriculum-stage stage_3_hard \
  --output-dir training/checkpoints/grpo \
  --max-train-steps 200
```

GRPO uses **in-process** env by default (`--use-local-env`). No `uvicorn` required on Colab.

Metrics:
- `training/logs/grpo_metrics.jsonl` — trainer logs
- `training/logs/learning_curve.jsonl` — validation curve after each stage

Dry-run:

```bash
!python training/grpo_train.py --dry-run --curriculum-stage stage_1_easy
```

If dry-run fails with `rollout_func` missing, upgrade TRL:

```bash
!pip install -U "trl>=0.14.0"
```

## 5) Evaluate checkpoint (local env)

```bash
!python training/eval_checkpoint.py \
  --checkpoint training/checkpoints/grpo \
  --split validation \
  --episodes 10 \
  --episode-log-file training/logs/episodes.jsonl \
  --clear-log
```

Compare eval `overall_grader_mean` to baseline `mean_grader_score` from `python app/baseline.py` or the dashboard.

## 6) Memory tips (T4)

- Keep `--max-seq-length 512` (raise only if you have headroom).
- Use `--num-generations 2` (not 4) on T4.
- Run curriculum **stages sequentially** and save adapters between stages.
- If OOM: lower `max_completion_length` in `grpo_train.py` config or reduce `num-generations` to 1.

## 7) Optional: remote server + demo monitor

Only needed if you want the live WebSocket demo at `/demo`:

```bash
!uvicorn app.server.app:app --host 0.0.0.0 --port 7860
```

Then evaluate with:

```bash
!python training/eval_checkpoint.py --use-remote-env --api-url http://127.0.0.1:7860
```

## 8) Publish adapter

```bash
!huggingface-cli login
!python training/publish_adapter.py \
  --checkpoint training/checkpoints/grpo \
  --repo-id YOUR_USERNAME/compliance-grpo-adapter
```

## Local (non-Colab) quick reference

```bash
pip install -e ".[training]"
python training/smoke_test.py
python training/prepare_data.py --episodes-per-task 40
python training/sft_train.py --dry-run
python -m pytest tests/test_training_smoke.py -q
```

Training still requires a CUDA GPU; use Colab if you have no local GPU.
