#!/usr/bin/env python3
"""Historical backtest: run the predictor with only pre-season data and
score it against the actual NFC champion.

This is the ONLY script in the repo permitted to read
data/validation/nfc_champions_2006_2025.csv. It never exposes that file to
the predictor/critic — it reads it after the fact, purely to score results.
Grep-check for this invariant: `grep -r "nfc_champions" mcp_server/ .claude/`
should return nothing.

Mechanics: sets CHAMPION_PREDICTOR_AS_OF_SEASON=<season> and
CHAMPION_PREDICTOR_BACKTEST_MODE=1 in the environment, then runs the
champion-predictor skill headlessly via `claude -p`. The MCP server
(mcp_server/safe_server.py) reads those env vars at startup and (a) filters
team_seasonal_stats to seasons strictly before the cutoff, and (b) makes the
current-season-only tools (roster, transactions, coaching, management,
injuries) report "not available" instead of silently leaking present-day
data into a historical run.

KNOWN LIMITATION (documented, not hidden): only team_seasonal_stats has real
multi-year depth. The other structured sources are current-season snapshots
only, so backtests before the current season are necessarily evidence-poor
compared to a live run. Fan/analyst consensus (WebSearch) is also skipped
during backtests since it can only retrieve today's commentary, not
period-accurate sentiment — the consensus-variance check in
validate_predictions.py is not meaningful here and is not applied.

Usage:
  python scripts/backtest.py --season 2015
  python scripts/backtest.py --seasons 2010-2020
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHAMPIONS_CSV = REPO_ROOT / "data" / "validation" / "nfc_champions_2006_2025.csv"
OUTPUT_DIR = REPO_ROOT / "outputs"

sys.path.insert(0, str(REPO_ROOT / "mcp_server"))
from lib.loaders import team_name_to_code  # noqa: E402


def load_actual_champions() -> dict[int, str]:
    with open(CHAMPIONS_CSV, newline="", encoding="utf-8") as f:
        return {int(row["Season"]): row["NFC_Champion"] for row in csv.DictReader(f)}


def run_predictor_for_season(season: int) -> dict | None:
    env = os.environ.copy()
    env["CHAMPION_PREDICTOR_AS_OF_SEASON"] = str(season)
    env["CHAMPION_PREDICTOR_BACKTEST_MODE"] = "1"

    prompt = (
        f"Run the champion-predictor skill's predictor step for NFC season "
        f"{season}, using only data available before that season concluded. "
        f"Output only the predictor's final JSON block, nothing else."
    )
    result = subprocess.run(
        ["claude", "-p", prompt],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        print(f"[season {season}] predictor run failed: {result.stderr}", file=sys.stderr)
        return None

    match = re.search(r"```json\s*(\{.*?\})\s*```", result.stdout, re.DOTALL)
    if not match:
        print(f"[season {season}] no JSON block found in output", file=sys.stderr)
        return None
    return json.loads(match.group(1))


def score_season(season: int, prediction: dict, actual_champion_name: str) -> dict:
    probs = prediction.get("probabilities", {})
    ranked = [team for team, _ in sorted(probs.items(), key=lambda kv: -float(kv[1]))]
    predicted_top1 = ranked[0] if ranked else None
    predicted_top3 = ranked[:3]

    actual_code = team_name_to_code(actual_champion_name)
    return {
        "season": season,
        "actual_champion": actual_champion_name,
        "actual_champion_code": actual_code,
        "predicted_top1": predicted_top1,
        "predicted_top3": predicted_top3,
        "top1_correct": actual_code is not None and predicted_top1 == actual_code,
        "top3_hit": actual_code is not None and actual_code in predicted_top3,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, help="Single season to backtest.")
    parser.add_argument("--seasons", help="Range, e.g. 2010-2020.")
    args = parser.parse_args()

    if args.season:
        seasons = [args.season]
    elif args.seasons:
        start, end = (int(x) for x in args.seasons.split("-"))
        seasons = list(range(start, end + 1))
    else:
        parser.error("Provide --season or --seasons")
        return 1

    actual = load_actual_champions()
    results = []
    for season in seasons:
        if season not in actual:
            print(f"[season {season}] no validation record, skipping", file=sys.stderr)
            continue
        prediction = run_predictor_for_season(season)
        if prediction is None:
            continue
        results.append(score_season(season, prediction, actual[season]))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    range_label = f"{seasons[0]}-{seasons[-1]}" if len(seasons) > 1 else str(seasons[0])
    out_path = OUTPUT_DIR / f"backtest_results_{range_label}.md"

    n = len(results)
    top1_accuracy = sum(r["top1_correct"] for r in results) / n if n else 0.0
    top3_hit_rate = sum(r["top3_hit"] for r in results) / n if n else 0.0

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Backtest results: {range_label}\n\n")
        f.write(f"Top-1 accuracy: {top1_accuracy:.1%} ({sum(r['top1_correct'] for r in results)}/{n})\n\n")
        f.write(f"Top-3 hit rate: {top3_hit_rate:.1%} ({sum(r['top3_hit'] for r in results)}/{n})\n\n")
        f.write("| Season | Actual Champion | Predicted Top-1 | Predicted Top-3 | Top-1 Correct | Top-3 Hit |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in results:
            f.write(
                f"| {r['season']} | {r['actual_champion']} | "
                f"{r['predicted_top1']} | {', '.join(r['predicted_top3'])} | "
                f"{'✓' if r['top1_correct'] else ''} | {'✓' if r['top3_hit'] else ''} |\n"
            )
        f.write(
            "\n_Known limitation: current-season-only sources (roster, "
            "transactions, coaching, management, injuries) are unavailable "
            "in backtest mode — see module docstring._\n"
        )
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
