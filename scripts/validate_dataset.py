"""Manual dataset validation report (run from repo root)."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
policy_file = ROOT / "data" / "policy.md"
claims_file = ROOT / "data" / "claims.json"

with open(policy_file, encoding="utf-8") as f:
    policy_text = f.read()

with open(claims_file, encoding="utf-8") as f:
    claims = json.load(f)["claims"]

print("=" * 80)
print("DATASET VALIDATION REPORT")
print("=" * 80)

print("\n\n### TASK 1: CROSS-CHECK GROUND TRUTH DECISIONS ###\n")

valid_decisions = {"Approve", "Reject", "Escalate"}
decision_counts: dict[str, int] = defaultdict(int)
difficulty_counts: dict[str, int] = defaultdict(int)
validation_issues: list[str] = []

for i, claim in enumerate(claims):
    claim_id = claim.get("id", f"UNKNOWN-{i}")
    decision = claim.get("ground_truth_decision")
    difficulty = claim.get("task_difficulty")

    if decision not in valid_decisions:
        validation_issues.append(
            f"  ✗ Claim {claim_id}: Invalid decision '{decision}' (valid: {valid_decisions})"
        )

    decision_counts[decision] += 1
    difficulty_counts[difficulty] += 1

print(f"✓ Total claims validated: {len(claims)}")
print(f"  - Approve: {decision_counts['Approve']}")
print(f"  - Reject: {decision_counts['Reject']}")
print(f"  - Escalate: {decision_counts['Escalate']}")
print("\n✓ Difficulty distribution:")
print(f"  - Easy: {difficulty_counts['easy']}")
print(f"  - Medium: {difficulty_counts['medium']}")
print(f"  - Hard: {difficulty_counts['hard']}")

if validation_issues:
    print(f"\n⚠ Found {len(validation_issues)} validation issues:")
    for issue in validation_issues[:10]:
        print(issue)
    if len(validation_issues) > 10:
        print(f"  ... and {len(validation_issues) - 10} more")
else:
    print("\n✓ All decisions are valid (no format errors)")

print("\n\n### TASK 2: SPOT-CHECK SAMPLE CLAIMS ###\n")

random.seed(42)
sample_claims = random.sample(claims, min(10, len(claims)))
print(f"Validating {len(sample_claims)} sample claims:\n")

for i, claim in enumerate(sample_claims, 1):
    print(f"Sample {i}: ID={claim.get('id')}")
    print(f"  Amount: ₹{claim.get('amount')} {claim.get('currency')}")
    print(f"  Description: {claim.get('description', '')[:60]}...")
    print(f"  Role/Level: {claim.get('employee_role')}/{claim.get('employee_level')}")
    print(f"  Receipt: {claim.get('has_receipt')}")
    print(f"  Missing Doc: {claim.get('missing_document') or 'None'}")
    print(f"  Rule Keyword: {claim.get('rule_keyword')}")
    print(f"  Ground Truth: {claim.get('ground_truth_decision')}")
    print(f"  Reason: {claim.get('ground_truth_reason', '')[:70]}...")
    print(f"  Difficulty: {claim.get('task_difficulty')}")
    print()

print("\n### TASK 3: POLICY-README CONSISTENCY CHECK ###\n")

rule_patterns = {
    1: "Small Meal",
    2: "Standard Meal",
    3: "Large Meal",
    4: "Alcohol",
    5: "Auto-Rickshaw",
    6: "10:00 PM",
    7: "Daytime",
    8: "Economy Class",
    9: "Business Class",
    10: "International Travel",
    11: "WFH",
    12: "GST Receipt",
    13: "Duplicate",
}

for rule_num, pattern in rule_patterns.items():
    if pattern.lower() in policy_text.lower():
        print(f"✓ Rule {rule_num}: '{pattern}' found in policy.md")
    else:
        print(f"✗ Rule {rule_num}: '{pattern}' NOT found in policy.md")

print("\n\n### TASK 4: EDGE CASES DETECTION ###\n")

edge_cases: dict[str, list] = {
    "threshold_boundaries": [],
    "missing_documents": [],
    "high_risk": [],
    "duplicates": [],
}

for claim in claims:
    amount = claim.get("amount", 0)
    if amount in [499, 500, 501, 1999, 2000, 2001, 5000, 5001, 50000, 50001]:
        edge_cases["threshold_boundaries"].append(claim.get("id"))
    if claim.get("missing_document"):
        edge_cases["missing_documents"].append(claim.get("id"))
    if claim.get("risk_score", 0) >= 0.8:
        edge_cases["high_risk"].append(claim.get("id"))

amounts_by_emp: dict[tuple, list] = defaultdict(list)
for claim in claims:
    key = (claim.get("employee_name"), claim.get("amount"))
    amounts_by_emp[key].append(claim.get("id"))

for _key, ids in amounts_by_emp.items():
    if len(ids) > 1:
        edge_cases["duplicates"].extend(ids)

print(f"✓ Threshold boundary claims: {len(edge_cases['threshold_boundaries'])}")
print(f"✓ Claims with missing documents: {len(edge_cases['missing_documents'])}")
print(f"✓ High-risk claims (score >= 0.8): {len(edge_cases['high_risk'])}")
print(f"✓ Potential duplicate scenarios: {len(set(edge_cases['duplicates']))}")

print("\n\n### TASK 5: DATA QUALITY REPORT ###\n")

required_fields = [
    "id",
    "employee_name",
    "employee_role",
    "employee_level",
    "amount",
    "currency",
    "has_receipt",
    "ground_truth_decision",
    "ground_truth_reason",
    "rule_keyword",
    "task_difficulty",
]

print("Field Completeness:")
for field in required_fields:
    present = sum(1 for c in claims if field in c and c.get(field) is not None)
    pct = present / len(claims) * 100
    status = "✓" if pct == 100 else "✗"
    print(f"  {status} {field}: {pct:.1f}%")

total = len(claims)
print("\nDecision Distribution:")
print(f"  Approve: {decision_counts['Approve'] / total * 100:.1f}%")
print(f"  Reject: {decision_counts['Reject'] / total * 100:.1f}%")
print(f"  Escalate: {decision_counts['Escalate'] / total * 100:.1f}%")

print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)
if not validation_issues:
    print("\n✓ ALL CHECKS PASSED")
else:
    print(f"\n⚠ Found {len(validation_issues)} issues that need resolution")
print("\n" + "=" * 80)
