"""
Generate curriculum-hard compliance claims with train/validation/test splits.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

EMPLOYEES = [
    {"name": "Rohan Mehta", "role": "Software Engineer", "level": "L3"},
    {"name": "Sneha Iyer", "role": "HR Coordinator", "level": "L2"},
    {"name": "Arjun Nair", "role": "Sales Executive", "level": "L4"},
    {"name": "Divya Krishnan", "role": "Analyst", "level": "L3"},
    {"name": "Priya Sharma", "role": "Marketing Manager", "level": "L5"},
    {"name": "Vikram Singh", "role": "Senior Engineer", "level": "L6"},
    {"name": "Ananya Rao", "role": "VP Engineering", "level": "L7"},
    {"name": "Karthik Menon", "role": "Director Sales", "level": "L8"},
]

PLACES = ["Cafe Coffee Day", "Barbeque Nation", "Pizza Hut", "Subway"]
CLIENTS = ["ABC Corp", "Innovate LLC", "Solutions Inc."]


def get_ground_truth(claim: Dict[str, Any]) -> Tuple[str, str]:
    amount = claim["amount"]
    desc = (claim.get("description") or claim.get("vague_description") or "").lower()
    level = claim.get("employee_level", "L3")

    if claim.get("force_decision"):
        return claim["force_decision"], claim.get("force_reason", "Curriculum case.")

    if "alcohol" in desc or "wine" in desc or "beer" in desc:
        return "Reject", "Rule 4: Alcohol prohibited."

    if level in ("L7", "L8"):
        return "Escalate", "Rule 14: L7+ must escalate."

    if claim.get("policy_category") == "duplicate":
        return "Reject", "Rule 13: Duplicate claim."

    if claim.get("policy_category") == "personal":
        return "Reject", "Rule 15: Personal expense."

    if any(k in desc for k in ("lunch", "dinner", "meal", "breakfast", "snacks")):
        if amount < 500:
            return "Approve", "Rule 1: Under Rs500."
        if 500 <= amount <= 2000:
            return ("Approve", "Rule 2: Receipt present.") if claim["has_receipt"] else (
                "Reject",
                "Rule 2: Receipt required.",
            )
        if amount > 2000:
            if claim.get("document_outcome") == "provided" or claim.get("missing_document") is None:
                if claim["has_receipt"]:
                    return "Approve", "Rule 3: Large meal with docs."
            return "Reject", "Rule 3: Manager note required."

    if "cab" in desc or "ride" in desc:
        if "late" in desc or "after 10" in desc:
            return "Approve" if claim["has_receipt"] else "Reject", "Rule 6: Night cab."
        if claim.get("document_outcome") == "provided" or claim.get("missing_document") is None:
            return "Approve", "Rule 7: Day cab with manager note."
        return "Reject", "Rule 7: Manager note missing."

    if claim.get("policy_category") == "gst":
        if claim.get("document_outcome") == "provided":
            return "Approve", "Rule 12: GST invoice provided."
        return "Reject", "Rule 12: GST invoice missing."

    if claim.get("policy_category") == "wfh":
        if amount <= 1000:
            return "Approve", "Rule 11: Within WFH cap."
        return "Reject", "Rule 11: Exceeds WFH cap."

    return "Approve", "No violation."


def _assign_complexity(claim: Dict[str, Any]) -> str:
    """
    Content-driven complexity label (independent of task_difficulty).
    direct   — resolve immediately, no search or doc hunt needed
    search   — policy search needed before deciding
    dochunt  — document request needed after search
    full     — both search + document request required
    """
    needs_doc = bool(claim.get("required_document") or claim.get("missing_document"))
    amount = float(claim.get("amount", 0) or 0)
    # Use the concrete description for complexity labeling (internal — not the vague version the model sees)
    desc = (claim.get("description") or claim.get("vague_description") or "").lower()
    level = claim.get("employee_level", "")
    category = claim.get("policy_category", "")

    if level in ("L7", "L8") or category in ("duplicate", "personal", "seniority"):
        return "direct"

    needs_search = (
        (amount > 2000 and any(k in desc for k in ("meal", "dinner", "lunch", "breakfast", "entertainment")))
        or amount > 5000
        or any(k in desc for k in ("cab", "ride", "taxi"))
        or any(k in desc for k in ("wfh", "internet", "electricity"))
        or any(k in desc for k in ("international", "flight", "hotel", "travel"))
    )

    if needs_search and needs_doc:
        return "full"
    if needs_search:
        return "search"
    if needs_doc:
        return "dochunt"
    return "direct"


def base_claim(cid: int) -> Dict[str, Any]:
    emp = random.choice(EMPLOYEES)
    return {
        "id": f"EXP-{cid:04d}",
        "employee_name": emp["name"],
        "employee_role": emp["role"],
        "employee_level": emp["level"],
        "currency": "INR",
        "has_receipt": True,
        "missing_document": None,
        "required_document": None,
        "risk_score": round(random.uniform(0.05, 0.85), 2),
        "expected_steps": 1,
        "policy_category": "meal",
        "document_outcome": None,
        "vague_description": None,
        "split": "train",
    }


def generate_easy_claim(cid: int, label: str | None = None) -> Dict[str, Any]:
    c = base_claim(cid)
    c["task_difficulty"] = "easy"
    category = label or random.choice(
        ["approve_small", "reject_no_receipt", "reject_alcohol", "escalate_l7", "boundary_500"]
    )

    if category == "approve_small":
        c["amount"] = random.choice([230, 399, 499])
        c["description"] = f"Team lunch at {random.choice(PLACES)}"
        c["rule_keyword"] = "meal"
        c["policy_category"] = "meal"
    elif category == "reject_no_receipt":
        c["amount"] = random.choice([501, 750, 1200])
        c["has_receipt"] = False
        c["description"] = f"Client dinner at {random.choice(PLACES)}"
        c["rule_keyword"] = "meal receipt"
        c["policy_category"] = "meal"
    elif category == "reject_alcohol":
        c["amount"] = 1800
        c["description"] = "Team dinner with wine on bill"
        c["rule_keyword"] = "meal"
        c["policy_category"] = "meal"
    elif category == "escalate_l7":
        c["employee_level"] = random.choice(["L7", "L8"])
        c["employee_role"] = "VP Engineering"
        c["amount"] = 1200
        c["description"] = "Executive team offsite meal"
        c["rule_keyword"] = "seniority"
        c["policy_category"] = "seniority"
        c["force_decision"] = "Escalate"
    elif category == "boundary_500":
        c["amount"] = random.choice([499, 500, 501])
        c["description"] = "Working lunch"
        c["rule_keyword"] = "meal"
        c["has_receipt"] = c["amount"] >= 500
    else:
        c["amount"] = 300
        c["description"] = "Snacks for team"
        c["rule_keyword"] = "meal"

    d, r = get_ground_truth(c)
    c["ground_truth_decision"] = d
    c["ground_truth_reason"] = r
    c["complexity"] = _assign_complexity(c)
    c["notes"] = f"Easy: {category}"
    return c


def generate_medium_claim(cid: int, category: str | None = None) -> Dict[str, Any]:
    c = base_claim(cid)
    c["task_difficulty"] = "medium"
    c["expected_steps"] = 2
    cat = category or random.choice(["daytime_cab", "large_meal", "night_cab", "wfh", "gst"])

    if cat == "daytime_cab":
        c["amount"] = random.randint(650, 1100)
        c["description"] = "Cab ride to client office before 10:00 PM"
        c["vague_description"] = "Cab ride to client site during business hours"
        c["missing_document"] = "manager_approval"
        c["required_document"] = "manager_approval"
        c["rule_keyword"] = "daytime cab"
        c["policy_category"] = "travel"
    elif cat == "large_meal":
        c["amount"] = random.choice([1999, 2001, 3200])
        c["description"] = f"Client entertainment dinner Rs{c['amount']}"
        c["vague_description"] = "Client entertainment dinner"
        c["rule_keyword"] = "large meal"
        c["policy_category"] = "meal"
    elif cat == "night_cab":
        c["amount"] = random.randint(400, 900)
        c["description"] = "Cab ride home after 10:30 PM"
        c["vague_description"] = "Late night return cab from office"
        c["rule_keyword"] = "night cab"
        c["policy_category"] = "travel"
    elif cat == "wfh":
        c["amount"] = random.choice([800, 1100, 1500])
        c["description"] = "Monthly WFH internet reimbursement"
        c["vague_description"] = "Remote work utility claim"
        c["rule_keyword"] = "wfh"
        c["policy_category"] = "wfh"
    else:
        c["amount"] = random.choice([5200, 8000])
        c["description"] = "Vendor software license"
        c["vague_description"] = "Software procurement expense"
        c["missing_document"] = "gst_invoice"
        c["required_document"] = "gst_invoice"
        c["rule_keyword"] = "gst"
        c["policy_category"] = "gst"

    d, r = get_ground_truth(c)
    c["ground_truth_decision"] = d
    c["ground_truth_reason"] = r
    c["complexity"] = _assign_complexity(c)
    c["notes"] = f"Medium: {cat}"
    return c


def generate_hard_claim(cid: int, scenario: str | None = None) -> Dict[str, Any]:
    c = base_claim(cid)
    c["task_difficulty"] = "hard"
    c["expected_steps"] = random.choice([3, 4])
    sc = scenario or random.choice(
        [
            "doc_provided_approve",
            "doc_missing_reject",
            "escalate_l7",
            "gst_missing",
            "conflict_personal_vs_meal",
        ]
    )

    if sc == "doc_provided_approve":
        c["amount"] = random.randint(2500, 4000)
        c["description"] = "Major client entertainment dinner"
        c["vague_description"] = "High-value client entertainment event"
        c["missing_document"] = "manager_approval"
        c["required_document"] = "manager_approval"
        c["document_outcome"] = "provided"
        c["rule_keyword"] = "large meal"
        c["policy_category"] = "meal"
        c["has_receipt"] = True
    elif sc == "doc_missing_reject":
        c["amount"] = random.randint(2800, 4500)
        c["description"] = "Team celebration dinner over policy cap"
        c["vague_description"] = "Team celebration dinner"
        c["missing_document"] = "manager_approval"
        c["required_document"] = "manager_approval"
        c["document_outcome"] = "not_provided"
        c["rule_keyword"] = "large meal"
        c["policy_category"] = "meal"
    elif sc == "escalate_l7":
        c["employee_level"] = random.choice(["L7", "L8"])
        c["amount"] = 3500
        c["vague_description"] = "Executive travel meal claim"
        c["rule_keyword"] = "seniority"
        c["policy_category"] = "seniority"
        c["force_decision"] = "Escalate"
    elif sc == "gst_missing":
        c["amount"] = 12000
        c["vague_description"] = "Annual software subscription"
        c["missing_document"] = "gst_invoice"
        c["required_document"] = "gst_invoice"
        c["document_outcome"] = "not_provided"
        c["rule_keyword"] = "gst"
        c["policy_category"] = "gst"
    else:
        c["amount"] = 450
        c["vague_description"] = "Personal gym membership charged as team wellness"
        c["rule_keyword"] = "personal"
        c["policy_category"] = "personal"
        c["force_decision"] = "Reject"

    d, r = get_ground_truth(c)
    c["ground_truth_decision"] = d
    c["ground_truth_reason"] = r
    c["complexity"] = _assign_complexity(c)
    c["notes"] = f"Hard: {sc}"
    return c


def generate_split(
    per_difficulty: int, seed: int, split_name: str
) -> List[Dict[str, Any]]:
    random.seed(seed + hash(split_name) % 10000)
    claims: List[Dict[str, Any]] = []
    split_offsets = {"train": 1, "validation": 10001, "test": 20001}
    cid = split_offsets.get(split_name, 30001)

    easy_labels = ["approve_small", "reject_no_receipt", "reject_alcohol", "escalate_l7", "boundary_500"]
    medium_cats = ["daytime_cab", "large_meal", "night_cab", "wfh", "gst"]
    hard_scenarios = [
        "doc_provided_approve",
        "doc_missing_reject",
        "escalate_l7",
        "gst_missing",
        "conflict_personal_vs_meal",
    ]

    per_label = max(1, per_difficulty // len(easy_labels))

    for i in range(per_difficulty):
        label = easy_labels[i % len(easy_labels)]
        c = generate_easy_claim(cid, label)
        c["split"] = split_name
        claims.append(c)
        cid += 1

    for i in range(per_difficulty):
        cat = medium_cats[i % len(medium_cats)]
        c = generate_medium_claim(cid, cat)
        c["split"] = split_name
        claims.append(c)
        cid += 1

    for i in range(per_difficulty):
        sc = hard_scenarios[i % len(hard_scenarios)]
        c = generate_hard_claim(cid, sc)
        c["split"] = split_name
        claims.append(c)
        cid += 1

    random.shuffle(claims)
    return claims


def write_split(path: Path, claims: List[Dict[str, Any]], split_name: str) -> None:
    dist = Counter(c["task_difficulty"] for c in claims)
    dec = Counter(c["ground_truth_decision"] for c in claims)
    data = {
        "metadata": {
            "split": split_name,
            "total_claims": len(claims),
            "distribution": dict(dist),
            "decision_distribution": dict(dec),
        },
        "claims": claims,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-per-diff", type=int, default=120)
    parser.add_argument("--val-per-diff", type=int, default=40)
    parser.add_argument("--test-per-diff", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(__file__).parent
    splits_dir = root / "splits"

    train = generate_split(args.train_per_diff, args.seed, "train")
    val = generate_split(args.val_per_diff, args.seed + 1, "validation")
    test = generate_split(args.test_per_diff, args.seed + 2, "test")

    write_split(splits_dir / "train.json", train, "train")
    write_split(splits_dir / "validation.json", val, "validation")
    write_split(splits_dir / "test.json", test, "test")

    combined = train + val + test
    legacy = {
        "metadata": {
            "total_claims": len(combined),
            "distribution": dict(Counter(c["task_difficulty"] for c in combined)),
            "seed": args.seed,
            "version": "2.0.0-curriculum",
            "splits": {
                "train": len(train),
                "validation": len(val),
                "test": len(test),
            },
        },
        "claims": combined,
    }
    with open(root / "claims.json", "w", encoding="utf-8") as f:
        json.dump(legacy, f, indent=2)

    print(f"Wrote {len(train)} train, {len(val)} val, {len(test)} test claims")
    print(f"Legacy claims.json: {len(combined)} total")


if __name__ == "__main__":
    main()
