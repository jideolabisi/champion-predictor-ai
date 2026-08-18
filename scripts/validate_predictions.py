#!/usr/bin/env python3
"""Deterministic validation of a predictor JSON output.

Per the checkpoint design, these checks must be real code, not LLM judgment:
  1. Probabilities must sum to ~100% (auto-renormalize within tolerance,
     otherwise flag as a hard failure requiring regeneration).
  2. Top-3 predicted teams are compared against the predictor's own
     `consensus_snapshot.top3` (fan/analyst reasonableness check) via simple
     overlap count.

Usage:
  python scripts/validate_predictions.py --input path/to/prediction.json
  cat prediction.json | python scripts/validate_predictions.py

Exits 0 and prints a JSON verdict on stdout. The orchestrating skill parses
this verdict to decide whether to accept the prediction or trigger
regeneration.
"""

import argparse
import json
import sys

SUM_TARGET = 100.0
SUM_RENORMALIZE_TOLERANCE = 5.0  # percentage points; beyond this is a real bug
MIN_CONSENSUS_OVERLAP = 1        # of top-3; below this triggers regeneration


def check_probability_sum(probs: dict) -> dict:
    total = sum(float(v) for v in probs.values())
    diff = abs(total - SUM_TARGET)
    if diff <= 0.5:
        return {"passed": True, "total": total, "action": "none"}
    if diff <= SUM_RENORMALIZE_TOLERANCE:
        renormalized = {k: float(v) * SUM_TARGET / total for k, v in probs.items()}
        return {
            "passed": True,
            "total": total,
            "action": "renormalized",
            "renormalized_probabilities": renormalized,
        }
    return {
        "passed": False,
        "total": total,
        "action": "regenerate",
        "reason": f"Probabilities sum to {total:.2f}, which is off by {diff:.2f} "
                  f"points — too large to auto-correct, likely a reasoning error.",
    }


def check_consensus_variance(probs: dict, consensus_top3: list) -> dict:
    predicted_top3 = [
        team for team, _ in sorted(probs.items(), key=lambda kv: -float(kv[1]))[:3]
    ]
    overlap = len(set(predicted_top3) & set(consensus_top3 or []))
    passed = overlap >= MIN_CONSENSUS_OVERLAP
    return {
        "passed": passed,
        "predicted_top3": predicted_top3,
        "consensus_top3": consensus_top3,
        "overlap": overlap,
        "reason": None if passed else (
            f"Only {overlap} of top-3 predicted teams overlap with fan/analyst "
            f"consensus ({consensus_top3}) — variance exceeds threshold."
        ),
    }


def validate(prediction: dict) -> dict:
    probs = prediction.get("probabilities", {})
    consensus_top3 = (prediction.get("consensus_snapshot") or {}).get("top3", [])

    sum_result = check_probability_sum(probs)
    effective_probs = sum_result.get("renormalized_probabilities", probs)
    variance_result = check_consensus_variance(effective_probs, consensus_top3)

    overall_passed = sum_result["passed"] and variance_result["passed"]
    return {
        "passed": overall_passed,
        "probability_sum_check": sum_result,
        "consensus_variance_check": variance_result,
        "verdict": "accept" if overall_passed else "regenerate",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to predictor JSON output; reads stdin if omitted.")
    args = parser.parse_args()

    raw = open(args.input, encoding="utf-8").read() if args.input else sys.stdin.read()
    prediction = json.loads(raw)

    result = validate(prediction)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
