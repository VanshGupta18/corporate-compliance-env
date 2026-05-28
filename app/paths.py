"""Repository-root paths for runtime artifacts (logs, results, training outputs)."""

from __future__ import annotations

import os
from pathlib import Path

# meta-openenv/ (parent of app/)
REPO_ROOT = Path(__file__).resolve().parent.parent

# Override for HF / custom deploy: point dashboard at a directory containing log files
_DATA_DIR = os.getenv("DASHBOARD_DATA_DIR", "").strip()
ARTIFACT_ROOT = Path(_DATA_DIR).expanduser().resolve() if _DATA_DIR else REPO_ROOT

BASELINE_RESULTS = ARTIFACT_ROOT / "baseline_results.json"
INFERENCE_RESULTS = ARTIFACT_ROOT / "inference_results.json"
TRAINING_RESULTS = ARTIFACT_ROOT / "training_results.json"
BASELINE_LOG = ARTIFACT_ROOT / "baseline_run.log"
INFERENCE_LOG = ARTIFACT_ROOT / "inference_run.log"
TRAINING_LOG = ARTIFACT_ROOT / "training_run.log"
# Colab eval may write episodes.jsonl at repo root; canonical path is under training/logs/
TRAINING_EPISODES = ARTIFACT_ROOT / "training/logs/episodes.jsonl"
TRAINING_EPISODES_ROOT = ARTIFACT_ROOT / "episodes.jsonl"
TRAINING_LEARNING_CURVE = ARTIFACT_ROOT / "training/logs/learning_curve.jsonl"
CLAIMS_DATA = REPO_ROOT / "data/claims.json"
TEST_SPLIT = REPO_ROOT / "data/splits/test.json"
