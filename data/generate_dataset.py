import json
import random
import argparse
from pathlib import Path

# --- Data Templates ---

EMPLOYEES = [
    {"name": "Rohan Mehta", "role": "Software Engineer", "level": "L3"},
    {"name": "Sneha Iyer", "role": "HR Coordinator", "level": "L2"},
    {"name": "Arjun Nair", "role": "Sales Executive", "level": "L4"},
    {"name": "Divya Krishnan", "role": "Analyst", "level": "L3"},
    {"name": "Priya Sharma", "role": "Marketing Manager", "level": "L5"},
    {"name": "Vikram Singh", "role": "Senior Engineer", "level": "L6"},
]

DESCRIPTIONS = {
    "meal": ["Team lunch at {place}", "Dinner with client {client}", "Breakfast meeting", "Snacks for team event"],
    "travel": ["Cab ride home after late work", "Auto to client meeting", "Metro fare for office commute"],
    "supplies": ["New keyboard for workstation", "Stationery for team", "Whiteboard markers"],
}

PLACES = ["Cafe Coffee Day", "Barbeque Nation", "Pizza Hut", "Subway"]
CLIENTS = ["ABC Corp", "Innovate LLC", "Solutions Inc."]

# --- Ground Truth Logic ---

def get_ground_truth(claim):
    """Determines the correct decision and reason based on policy.md rules."""
    amount = claim["amount"]
    description = claim["description"].lower()

    # Rule 4: Alcohol
    if "alcohol" in description or "wine" in description or "beer" in description:
        return "Reject", "Rule 4: Alcohol is a prohibited expense."

    # Meal Rules
    if any(keyword in description for keyword in ["lunch", "dinner", "meal", "breakfast", "snacks"]):
        if amount < 500:
            return "Approve", "Rule 1: Meal under ₹500 is auto-approved."
        if 500 <= amount <= 2000:
            if claim["has_receipt"]:
                return "Approve", "Rule 2: Meal with receipt is approved."
            else:
                return "Reject", "Rule 2: Meal over ₹500 requires a receipt."
        if amount > 2000:
            if claim["has_receipt"] and claim["missing_document"] is None:
                return "Approve", "Rule 3: Large meal with all documents is approved."
            else:
                return "Reject", "Rule 3: Large meal requires receipt and manager approval."

    # Travel Rules
    if any(keyword in description for keyword in ["cab", "auto", "metro"]):
        if "after 10:00 pm" in description:
            return "Approve", "Rule 6: Late-night cab is pre-approved."
        if "before 10:00 pm" in description:
            if claim["missing_document"] is None:
                 return "Approve", "Rule 7: Daytime cab with manager note is approved."
            else:
                return "Reject", "Rule 7: Daytime cab requires a manager note."

    # Default to approve if no specific rule is violated
    return "Approve", "No specific policy violation found."


# --- Claim Generation Functions ---

def generate_base_claim(claim_id):
    """Generates the common fields for any claim."""
    employee = random.choice(EMPLOYEES)
    return {
        "id": f"EXP-{claim_id:03d}",
        "employee_name": employee["name"],
        "employee_role": employee["role"],
        "employee_level": employee["level"],
        "currency": "INR",
        "has_receipt": True,
        "missing_document": None,
        "risk_score": round(random.uniform(0.05, 0.8), 2),
        "expected_steps": 1,
    }

def generate_easy_claim(claim_id):
    """Generates a simple, clear-cut claim."""
    claim = generate_base_claim(claim_id)
    claim["task_difficulty"] = "easy"
    
    if random.random() > 0.5: # Compliant meal
        claim["amount"] = random.randint(200, 499)
        claim["description"] = random.choice(DESCRIPTIONS["meal"]).format(place=random.choice(PLACES), client=random.choice(CLIENTS))
        claim["rule_keyword"] = "meal"
    else: # Non-compliant meal (no receipt)
        claim["amount"] = random.randint(501, 800)
        claim["description"] = random.choice(DESCRIPTIONS["meal"]).format(place=random.choice(PLACES), client=random.choice(CLIENTS))
        claim["has_receipt"] = False
        claim["rule_keyword"] = "meal receipt"
        
    decision, reason = get_ground_truth(claim)
    claim["ground_truth_decision"] = decision
    claim["ground_truth_reason"] = reason
    claim["notes"] = "Straightforward case."
    return claim

def generate_medium_claim(claim_id):
    """Generates a claim requiring policy search."""
    claim = generate_base_claim(claim_id)
    claim["task_difficulty"] = "medium"
    
    # Threshold boundary cases
    if random.random() > 0.7:
        amount = random.choice([1999, 2001])
        claim["amount"] = amount
        claim["description"] = f"Client dinner for team, amount is ₹{amount}"
        claim["rule_keyword"] = "large meal"
    else: # Daytime cab
        claim["amount"] = random.randint(600, 1200)
        claim["description"] = "Cab ride to client office before 10:00 PM"
        claim["missing_document"] = "manager_approval"
        claim["rule_keyword"] = "daytime cab"

    decision, reason = get_ground_truth(claim)
    claim["ground_truth_decision"] = decision
    claim["ground_truth_reason"] = reason
    claim["notes"] = "Requires checking specific policy rules."
    claim["expected_steps"] = 2
    return claim

def generate_hard_claim(claim_id):
    """Generates a claim requiring information requests."""
    claim = generate_base_claim(claim_id)
    claim["task_difficulty"] = "hard"
    
    claim["amount"] = random.randint(2500, 5000)
    claim["description"] = "Major client entertainment dinner"
    claim["missing_document"] = "manager_approval"
    claim["rule_keyword"] = "large meal"
    
    decision, reason = get_ground_truth(claim)
    claim["ground_truth_decision"] = decision
    claim["ground_truth_reason"] = reason
    claim["notes"] = "Requires requesting a missing document before resolving."
    claim["expected_steps"] = 3
    return claim


# --- Main Script Logic ---

def generate_dataset(count, seed):
    random.seed(seed)
    claims = []
    
    # Ensure a good mix
    num_easy = count // 3
    num_medium = count // 3
    num_hard = count - num_easy - num_medium

    for i in range(num_easy):
        claims.append(generate_easy_claim(len(claims) + 1))
    for i in range(num_medium):
        claims.append(generate_medium_claim(len(claims) + 1))
    for i in range(num_hard):
        claims.append(generate_hard_claim(len(claims) + 1))
        
    random.shuffle(claims)

    # Add metadata
    dataset = {
        "metadata": {
            "total_claims": len(claims),
            "distribution": {
                "easy": num_easy,
                "medium": num_medium,
                "hard": num_hard,
            },
            "seed": seed,
            "version": "1.1.0",
        },
        "claims": claims
    }
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic expense claims dataset.")
    parser.add_argument("--count", type=int, default=100, help="Number of claims to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output", type=str, default="claims.json", help="Output JSON file name.")
    args = parser.parse_args()

    output_path = Path(__file__).parent / args.output
    
    print(f"Generating {args.count} claims with seed {args.seed}...")
    dataset = generate_dataset(args.count, args.seed)
    
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
        
    print(f"Successfully generated dataset and saved to {output_path}")
    print(f"Distribution: {dataset['metadata']['distribution']}")
