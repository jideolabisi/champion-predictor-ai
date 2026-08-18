---
name: champion-predictor
description: Generate an NFC championship win-probability prediction with critic review and deterministic validation. Use this when the user asks to run/generate/update a Champion Predictor AI prediction for a season.
---

You are orchestrating the Champion Predictor AI pipeline for a given NFC
season (default: the current season, unless the user or an env var
`CHAMPION_PREDICTOR_AS_OF_SEASON`/backtest context specifies otherwise).
Follow this control flow exactly — the deterministic checks are never your
own judgment call, they are the output of real scripts.

## Steps

1. **Predict.** Invoke the `predictor` subagent (via Task) with the target
   season. Capture its fenced JSON output as `prediction`.

2. **Critique.** Invoke the `critic` subagent (via Task), passing only
   `prediction.probabilities` and `prediction.explanation` (nothing else —
   the critic must not see raw data or tool results). Capture `{rating,
   feedback}`.

3. **Branch on rating.**
   - `Low` → re-invoke the `predictor` subagent, appending the prior
     `prediction` and the critic's `feedback` to the prompt, and asking it to
     revise. Go back to step 2. Track a regeneration counter; **cap at 2
     regenerations** — if still not High/Medium after that, proceed anyway
     with a note that the final result carries a low-confidence flag.
   - `Medium` or `High` → continue to step 4.

4. **Validate.** Write `prediction` to a temp JSON file and run:
   ```
   .venv/Scripts/python.exe scripts/validate_predictions.py --input <temp-file>
   ```
   Parse the resulting verdict JSON.

5. **Branch on validation verdict.**
   - `verdict: "regenerate"` → re-invoke the `predictor` subagent with the
     validation failure reason (`probability_sum_check.reason` and/or
     `consensus_variance_check.reason`) appended as feedback. Go back to
     step 2. This also counts against the regeneration cap from step 3.
   - `verdict: "accept"` → continue to step 6. If
     `probability_sum_check.action == "renormalized"`, use
     `renormalized_probabilities` as the final probabilities instead of the
     predictor's raw output.

6. **Finalize.** Write the result to
   `outputs/predictions/prediction_<season>_<timestamp>.md` containing:
   - The 16 team probabilities as a table, sorted descending.
   - The explanation.
   - The critic's final rating.
   - The validation verdict (including whether renormalization happened).
   - How many predictor rounds and regenerations it took (for auditability).

   Then present a concise summary to the user: top-3 teams with
   probabilities, one-paragraph rationale, critic rating, and the output file
   path.

## Notes

- Never fabricate or skip the critic/validation steps even if the prediction
  "looks fine" — the checkpoint design requires both to actually run.
- If any subagent invocation fails or returns unparseable output, report
  that clearly rather than guessing at a prediction.
