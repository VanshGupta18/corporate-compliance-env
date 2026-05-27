"""Dry-run checks for Colab training wiring (no GPU model load)."""

from __future__ import annotations

import json
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ok(msg: str) -> None:
    print(f"  OK  {msg}")


def _fail(msg: str) -> None:
    print(f"  FAIL {msg}")
    raise SystemExit(1)


def _pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _mm(ver: str) -> tuple[int, int]:
    core = ver.split("+", 1)[0]
    parts = core.split(".")
    major = int(parts[0]) if parts and parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return major, minor


def check_dependency_versions() -> None:
    trl_ver = _pkg_version("trl")
    ds_ver = _pkg_version("datasets")
    if trl_ver is None:
        print("  WARN trl not installed in current environment")
    else:
        major, minor = _mm(trl_ver)
        if (major, minor) > (0, 24):
            print(f"  WARN trl={trl_ver} (expected <=0.24.x for Unsloth stack)")
        else:
            _ok(f"trl version={trl_ver}")
    if ds_ver is None:
        print("  WARN datasets not installed in current environment")
    else:
        major, minor = _mm(ds_ver)
        if (major, minor) >= (4, 4):
            print(f"  WARN datasets={ds_ver} (expected <4.4.0 for Unsloth stack)")
        else:
            _ok(f"datasets version={ds_ver}")


def check_splits_and_sft_dataset() -> None:
    for name in ("train", "validation", "test"):
        path = ROOT / "data" / "splits" / f"{name}.json"
        if not path.exists():
            _fail(f"missing split file {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        claims = data.get("claims", [])
        if not claims:
            _fail(f"empty claims in {path}")
    _ok("data/splits/*.json present")

    sft = ROOT / "training" / "data" / "sft_dataset.jsonl"
    if not sft.exists():
        _fail(f"missing {sft} — run: python training/prepare_data.py")
    lines = [ln for ln in sft.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        _fail("sft_dataset.jsonl is empty")
    sample = json.loads(lines[0])
    for key in ("prompt", "response", "task_id"):
        if key not in sample:
            _fail(f"sft row missing '{key}'")
    _ok(f"sft_dataset.jsonl rows={len(lines)}")


def check_curriculum_filter() -> None:
    from datasets import Dataset

    from training.training_utils import CURRICULUM_STAGES, filter_dataset_by_curriculum

    rows = [
        {"prompt": "Task: easy\n", "task_id": "easy"},
        {"prompt": "Task: medium\n", "task_id": "medium"},
        {"prompt": "Task: hard\n", "task_id": "hard"},
    ]
    ds = Dataset.from_list(rows)
    filtered = filter_dataset_by_curriculum(ds, "stage_1_easy")
    tasks = {r["task_id"] for r in filtered}
    if tasks != {"easy"}:
        _fail(f"stage_1_easy filter wrong: {tasks}")
    _ok(f"curriculum stages={list(CURRICULUM_STAGES.keys())}")


def check_learning_curve() -> None:
    from training.learning_curve import log_learning_point

    log_file = ROOT / "training" / "logs" / "_smoke_learning_curve.jsonl"
    if log_file.exists():
        log_file.unlink()
    payload = log_learning_point(
        stage="stage_0_baseline",
        global_step=0,
        split="validation",
        log_file=log_file,
    )
    if "per_difficulty" not in payload:
        _fail("learning_curve payload missing per_difficulty")
    _ok("learning_curve.jsonl logging")


def check_rollout_contract() -> None:
    from training.training_utils import (
        grpo_supports_rollout_func,
        native_grpo_supports_rollout_func,
    )

    try:
        supported = grpo_supports_rollout_func()
    except Exception as exc:
        print(f"  WARN TRL import check skipped: {exc}")
        return

    try:
        from training.rollout_generation import generate_rollout_completions  # noqa: F401
    except Exception as exc:
        _fail(f"rollout_generation import failed: {exc}")

    if supported:
        if native_grpo_supports_rollout_func():
            _ok("TRL GRPOTrainer supports rollout_func (native)")
        else:
            _ok("TRL rollout_func supported via compat shim")
    else:
        _fail("TRL rollout_func unavailable (native + compat)")


def check_script_dry_runs() -> None:
    py = sys.executable
    proc = subprocess.run(
        [py, "-m", "training.sft_train", "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        _fail(f"sft_train dry-run:\n{proc.stderr or proc.stdout}")
    proc = subprocess.run(
        [py, "-m", "training.grpo_train", "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        out = proc.stderr or proc.stdout
        _fail(f"grpo_train dry-run:\n{out}")
    else:
        _ok("grpo_train.py --dry-run")
    _ok("sft_train.py --dry-run")


def main() -> None:
    print("Training smoke test")
    check_dependency_versions()
    check_splits_and_sft_dataset()
    check_curriculum_filter()
    check_learning_curve()
    check_rollout_contract()
    check_script_dry_runs()
    print("All smoke checks passed.")


if __name__ == "__main__":
    main()
