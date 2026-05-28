"""Recompute ground_truth_decision/reason from claim fields (fixes label drift)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.generate_dataset import get_ground_truth  # noqa: E402


def sync_file(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claims = payload.get("claims", payload if isinstance(payload, list) else [])
    updated = 0
    for claim in claims:
        decision, reason = get_ground_truth(claim)
        if claim.get("ground_truth_decision") != decision:
            claim["ground_truth_decision"] = decision
            claim["ground_truth_reason"] = reason
            updated += 1
        elif claim.get("ground_truth_reason") != reason:
            claim["ground_truth_reason"] = reason
            updated += 1
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return updated


def main() -> None:
    targets = [
        _ROOT / "data" / "claims.json",
        _ROOT / "data" / "splits" / "train.json",
        _ROOT / "data" / "splits" / "validation.json",
        _ROOT / "data" / "splits" / "test.json",
    ]
    total = 0
    for path in targets:
        if not path.exists():
            continue
        n = sync_file(path)
        print(f"{path.relative_to(_ROOT)}: updated {n} claims")
        total += n
    print(f"Total updates: {total}")


if __name__ == "__main__":
    main()
