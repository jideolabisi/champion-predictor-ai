#!/usr/bin/env python3
"""Deterministic validation of a predictor JSON output.

Per the checkpoint design, these checks must be real code, not LLM judgment:
  1. Probabilities must sum to ~100% (auto-renormalize within tolerance,
     otherwise flag as a hard failure requiring regeneration). Applied
     independently to `probabilities` and, when present (what-if mode),
     `adjusted_probabilities` — both sets must each sum to ~100 on their own.
  2. Top-3 predicted teams are compared against the predictor's own
     `consensus_snapshot.top3` (fan/analyst reasonableness check) via simple
     overlap count. Applied to baseline `probabilities` only — consensus
     data describes the real season, not a hypothetical.
  3. Data-integrity sanity check on the raw seasonal-stats CSV for the
     predicted season, independent of what the predictor claimed: flags
     any numeric column that is exactly 0 for every team. This catches a
     real incident class (a stats-source merge/schema mismatch silently
     zero-filling a whole column — see scripts/fetch/fetch_nflverse.py's
     docstring) that lets a metric derived from that column (e.g. turnover
     margin from interceptions) be cited as data-grounded when it isn't,
     without any single sentence looking overconfident enough for the DiD
     guardrail to catch by reading prose alone. Informational only — it
     never flips the accept/regenerate verdict, since re-running the
     predictor against the same broken CSV would fix nothing.

Usage:
  python scripts/validate_predictions.py --input path/to/prediction.json
  cat prediction.json | python scripts/validate_predictions.py

Exits 0 and prints a JSON verdict on stdout. The orchestrating skill parses
this verdict to decide whether to accept the prediction or trigger
regeneration.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

SUM_TARGET = 100.0
SUM_RENORMALIZE_TOLERANCE = 5.0  # percentage points; beyond this is a real bug
MIN_CONSENSUS_OVERLAP = 1        # of top-3; below this triggers regeneration

REPO_ROOT = Path(__file__).resolve().parents[1]
TEAM_STATS_CSV = REPO_ROOT / "data" / "raw" / "team_stats_2006_2025.csv"
# Identifier/context columns, never a stat signal — excluded from the
# "is this column degenerate" scan in check_data_integrity below.
_NON_STAT_COLUMNS = {"Team", "Season", "week"}
# A single all-zero column is often a legitimately rare event (e.g.
# "fg_missed_0_19" — nobody misses a 0-19 yard field goal in a given
# season; confirmed empirically on real 2025 data, 1 of 135 numeric
# columns). The actual incident this check exists for zeroed out ~16
# columns at once (every def_* column plus interceptions, from a stats-
# source schema mismatch — see fetch_nflverse.py). Require several
# columns to go zero simultaneously so the check catches that signature
# without crying wolf over ordinary sparse stats.
_MIN_SUSPICIOUS_ZERO_COLUMNS = 5


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


def check_data_integrity(season) -> dict:
    """Flags the raw seasonal-stats CSV for the predicted season when
    several numeric columns are simultaneously 0 for every team — the
    signature of a stats-source merge/schema bug, not ordinary sparse
    stats (see `_MIN_SUSPICIOUS_ZERO_COLUMNS` above and the module
    docstring for the incident this is meant to catch). Never fails to
    "regenerate": if the CSV or season is missing, treat as unchecked
    rather than crashing the whole validation run.
    """
    if season is None or not TEAM_STATS_CSV.exists():
        return {"passed": True, "checked": False, "flagged_columns": [], "reason": None}

    with TEAM_STATS_CSV.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if str(r.get("Season")) == str(season)]
    if not rows:
        return {"passed": True, "checked": False, "flagged_columns": [], "reason": None}

    all_zero_columns = []
    for col in rows[0]:
        if col in _NON_STAT_COLUMNS:
            continue
        try:
            values = [float(r[col]) for r in rows]
        except (TypeError, ValueError):
            continue  # not a numeric column
        if values and all(v == 0.0 for v in values):
            all_zero_columns.append(col)

    flagged = all_zero_columns if len(all_zero_columns) >= _MIN_SUSPICIOUS_ZERO_COLUMNS else []

    reason = None
    if flagged:
        reason = (
            f"Column(s) {flagged} are exactly 0 for every team in season "
            f"{season} in {TEAM_STATS_CSV.name} — likely a data-source "
            f"merge/schema bug, not a real football signal. Any explanation "
            f"claim derived from these columns (e.g. a turnover-margin "
            f"figure computed from interceptions) is not actually grounded "
            f"in retrieved data."
        )
    return {"passed": not flagged, "checked": True, "flagged_columns": flagged, "reason": reason}


def validate(prediction: dict) -> dict:
    probs = prediction.get("probabilities", {})
    consensus_top3 = (prediction.get("consensus_snapshot") or {}).get("top3", [])
    adjusted_probs = prediction.get("adjusted_probabilities")

    sum_result = check_probability_sum(probs)
    effective_probs = sum_result.get("renormalized_probabilities", probs)
    variance_result = check_consensus_variance(effective_probs, consensus_top3)
    data_integrity_result = check_data_integrity(prediction.get("season"))

    result = {
        "probability_sum_check": sum_result,
        "consensus_variance_check": variance_result,
        # Informational only — deliberately excluded from overall_passed
        # below. A degenerate data column isn't something a predictor
        # regeneration can fix; the orchestrator surfaces this as a report
        # caveat instead (see SKILL.md step 7).
        "data_integrity_check": data_integrity_result,
    }
    overall_passed = sum_result["passed"] and variance_result["passed"]

    if adjusted_probs:
        adjusted_sum_result = check_probability_sum(adjusted_probs)
        result["adjusted_probability_sum_check"] = adjusted_sum_result
        overall_passed = overall_passed and adjusted_sum_result["passed"]

    result["passed"] = overall_passed
    result["verdict"] = "accept" if overall_passed else "regenerate"
    return result


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
