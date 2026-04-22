# Training and Publishing Guide

This project now includes a low-cost training pipeline for SFT + GRPO with Unsloth/TRL.

Recommended base model for Colab T4: `Qwen/Qwen2.5-3B-Instruct`.

## 1) Setup

```bash
pip install -r training/requirements-training.txt
```

Start the environment server:

```bash
uvicorn app.server.app:app --host 0.0.0.0 --port 7860
```

Open the read-only monitor:
- `http://localhost:7860/demo`
- It auto-refreshes episode logs from `training/logs/episodes.jsonl`
- It auto-refreshes training metrics from `training/logs/grpo_metrics.jsonl`

## 2) Prepare Trajectories

```bash
python training/prepare_data.py --episodes-per-task 40
```

This writes:
- `training/data/trajectories.json`
- `training/data/sft_dataset.jsonl`

## 3) SFT Warm Start (QLoRA-ready)

```bash
python training/sft_train.py \
  --model-id Qwen/Qwen2.5-3B-Instruct \
  --dataset-path training/data/sft_dataset.jsonl \
  --output-dir training/checkpoints/sft
```

## 4) GRPO Post-Training

```bash
python training/grpo_train.py \
  --model-id unsloth/Qwen2.5-3B-Instruct-bnb-4bit \
  --dataset-path training/data/sft_dataset.jsonl \
  --api-url http://127.0.0.1:7860 \
  --output-dir training/checkpoints/grpo \
  --max-seq-length 768 \
  --batch-size 1 \
  --grad-accum 8 \
  --num-generations 4 \
  --log-file training/logs/grpo_metrics.jsonl
```

Colab T4 note:
- Keep `max-seq-length` low (`512-768`) if you hit OOM.
- Start with `num-generations=4`; reduce to `2` for memory pressure.

## 5) Evaluate Checkpoint

```bash
python training/eval_checkpoint.py \
  --checkpoint training/checkpoints/grpo \
  --api-url http://127.0.0.1:7860 \
  --episodes 10 \
  --episode-log-file training/logs/episodes.jsonl \
  --clear-log
```

Use this report against baseline `/baseline` output before publishing.

## Colab T4 quick run order

1. Launch API server.
2. Generate dataset (`prepare_data.py`).
3. Start GRPO (`grpo_train.py`) so `training/logs/grpo_metrics.jsonl` updates.
4. Run `eval_checkpoint.py` periodically to append episode-by-episode logs.
5. Watch `/demo` read-only monitor.

## 6) Publish Adapter to Hugging Face

```bash
huggingface-cli login
```

```bash
python training/publish_adapter.py \
  --checkpoint training/checkpoints/grpo \
  --repo-id YOUR_USERNAME/compliance-grpo-adapter
```

## 7) Use the Adapter in Demo/Inference

- Point inference to your adapter/base model combination.
- Add adapter model ID in dashboard auto-run options if you want side-by-side baseline vs trained model.
