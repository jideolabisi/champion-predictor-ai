---
name: champion-predictor
description: Generate an NFC championship win-probability prediction (or evaluate a user-named what-if scenario) with critic review, DiD guardrails, and deterministic validation. Use this when the user asks to run/generate/update a Champion Predictor AI prediction, or asks how a hypothetical change (trade, coaching hire, etc.) would affect a team's odds.
---

You are orchestrating the Champion Predictor AI pipeline for a given NFC
season (default: the current season, unless the user or an env var
`CHAMPION_PREDICTOR_AS_OF_SEASON`/backtest context specifies otherwise).
Follow this control flow exactly — the deterministic checks and the DiD
guardrail are never your own judgment call in place of theirs; always
actually invoke them.

There is exactly **one shared regeneration counter** for the whole run,
capped at **2**. It is incremented by *any* of: a Critic rating of `Low`,
the Predictor's `confidence` falling below 50, a DiD `overconfidence_verdict`
of `fail`, a deterministic validator verdict of `regenerate`, or a DiD
`groundedness_verdict` of `fail`. These are not independent loops — track
one number.

## Steps

0. **Integrity check.** Run:
   ```
   .venv/Scripts/python.exe scripts/verify_data_integrity.py --verify
   ```
   If `passed: false`, tell the user exactly which files changed and ask
   them to confirm this is expected (e.g. a legitimate re-fetch) before
   continuing. Do not proceed on your own judgment — this is a
   human-in-the-loop (HITL) pause, not an auto-fail, since a mismatch could
   be entirely legitimate.

1. **Detect mode.** Read the user's request.
   - If it names a hypothetical change to evaluate (a trade, a coaching
     hire, a front-office change, etc.) and a target team, this is
     **WHAT_IF** mode — extract the scenario text and target team.
   - Otherwise it's **PREDICT** mode (the default).

2. **Pre-generation guardrail.** Invoke the `did` subagent (via Task) with
   `MODE: pre-generation` and the user's raw request. If `verdict:
   "clarify"`, ask the user directly for clarification or confirmation and
   wait for their reply — do not proceed to Predict until resolved.

3. **Predict.** Invoke the `predictor` subagent (via Task) with the target
   season and, in WHAT_IF mode, a `SCENARIO` line naming the hypothetical
   and target team. Capture its fenced JSON output as `prediction`. (The
   `SubagentStop` hook fires automatically here and writes
   `outputs/.trace/predictor_latest.json`.)

4. **During-generation guardrail.** Read `outputs/.trace/predictor_latest.json`
   **yourself** and paste its actual contents (truncate the middle if it's
   very large, but keep enough real tool calls/observations to check
   against) directly into the `did` subagent's prompt text. **`did` has no
   tools (`tools: []`) and cannot read any file itself — never instruct it
   to "read the trace file" or hand it a path in place of the contents;
   that produces a false "file does not exist" failure and silently skips
   the check.** Invoke `did` with `MODE: during-generation`, passing
   `prediction`'s probabilities/explanation (or
   adjusted_probabilities/delta_explanation in WHAT_IF mode) plus those
   inlined trace contents (if the file is missing, note that and proceed —
   treat this as reduced guardrail coverage, not a crash). Capture
   `{overconfidence_verdict, suspicion_score, flagged}`.
   - `suspicion_score >= 80` → escalate to HITL immediately regardless of
     the regeneration counter: show the user the flagged items and the
     trace, and get their explicit go-ahead before continuing.
   - `overconfidence_verdict: "fail"` → counts against the shared
     regeneration counter (see step 6).

5. **Critique.** Invoke the `critic` subagent (via Task), passing only
   `prediction.probabilities`/`explanation` (or
   `adjusted_probabilities`/`delta_explanation` in WHAT_IF mode) — nothing
   else, the critic must not see raw data or tool results. Capture
   `{rating, feedback}`.

6. **Branch on triggers.** Increment the shared regeneration counter if
   *any* of these are true this round:
   - Critic `rating` is `Low`.
   - `prediction.confidence < 50`.
   - DiD `overconfidence_verdict` is `fail`.

   If the counter is now ≤ 2: re-invoke the `predictor` subagent, appending
   the prior `prediction`, the critic's `feedback`, and any DiD `flagged`
   items, asking it to revise. Go back to step 4.

   If the counter has exceeded 2 (cap reached): stop retrying and proceed
   to step 7, but:
   - If none of this round's triggers coincided with a suspicion-score
     flag or a `clarify` prompt-score from step 2, proceed with an explicit
     low-confidence flag in the final output.
   - If they did coincide, escalate to HITL instead — describe what's
     still unresolved and get the user's explicit go-ahead before
     finalizing.

7. **Validate.** Write `prediction` to a temp JSON file and run:
   ```
   .venv/Scripts/python.exe scripts/validate_predictions.py --input <temp-file>
   ```
   Parse the resulting verdict JSON (checks `probabilities` and, in
   WHAT_IF mode, `adjusted_probabilities` independently).
   - `verdict: "regenerate"` → increment the shared counter, re-invoke the
     `predictor` subagent with the validation failure reason
     (`probability_sum_check.reason`, `adjusted_probability_sum_check.reason`,
     and/or `consensus_variance_check.reason`) appended as feedback, and go
     back to step 4 (subject to the same cap-2 / escalation logic as step 6).
   - `verdict: "accept"` → continue to step 8. If a sum check's
     `action == "renormalized"`, use its `renormalized_probabilities` as the
     final values for that set instead of the predictor's raw output.

8. **Post-generation guardrail.** Invoke `did` with `MODE: post-generation`,
   passing the final explanation(s) plus the trace file's actual contents,
   inlined into the prompt the same way as step 4 (again: `did` cannot read
   the file itself — it has no tools).
   `groundedness_verdict: "fail"` → increment the shared regeneration
   counter and go back to step 4 with `unsupported_claims` appended as
   feedback (subject to the same cap-2 / escalation logic).

9. **Finalize.** Write the result as a **pair of files sharing the same
   stem** — `outputs/predictions/prediction_<season>_<timestamp>.{md,json}`
   (PREDICT mode) or `outputs/predictions/whatif_<team>_<season>_<timestamp>.{md,json}`
   (WHAT_IF mode). The UI's "Real Agentic Predictor" tab reads the `.json`
   file to populate its Probabilities/Chart/Guardrails panels — without it,
   those panels have nothing structured to show, only the `.md` narrative.
   Always write both files, in this order:

   a. **The `.md` file** — human-readable report containing:
      - PREDICT mode: the 16 team probabilities as a table, sorted
        descending.
      - WHAT_IF mode: baseline vs. adjusted probabilities side by side for
        all 16 teams, plus the delta explanation.
      - The explanation (or delta explanation).
      - The critic's final rating, and the predictor's confidence score.
      - The validation verdict (including whether renormalization
        happened).
      - DiD's three verdicts (prompt score, overconfidence verdict +
        suspicion score, groundedness verdict) and any flagged/unsupported
        items.
      - How many predictor rounds and regenerations it took, and which
        trigger(s) fired, if any — this is the full audit trail; an
        escalation-rate calculation later is just scanning these files.

   b. **The `.json` file** — the same information as machine-readable data,
      with exactly these top-level keys (`null` any that don't apply to the
      mode; use the *final* values — i.e. post-renormalization
      probabilities if that happened in step 7):
      ```json
      {
        "mode": "predict",
        "season": 2026,
        "probabilities": {"ARI": 4.5, "...": "...all 16 team codes..."},
        "confidence": 72,
        "explanation": "...",
        "adjusted_probabilities": null,
        "delta_explanation": null,
        "target_team": null,
        "scenario": null,
        "validation": { "...": "the full verdict JSON from step 7, verbatim" },
        "did": {
          "pre_generation": { "...": "the full step-2 output, verbatim" },
          "during_generation": { "...": "the full step-4 output, verbatim" },
          "post_generation": { "...": "the full step-8 output, verbatim" }
        },
        "critic": { "...": "the full step-5 output, verbatim" }
      }
      ```

   Then present a concise summary to the user: top-3 teams with
   probabilities (or the scenario's before/after for the target team in
   WHAT_IF mode), one-paragraph rationale, critic rating, confidence score,
   and the output file path(s).

## Notes

- Never fabricate or skip the DiD, critic, or validation steps even if the
  prediction "looks fine" — the checkpoint design requires all of them to
  actually run.
- If any subagent invocation fails or returns unparseable output, report
  that clearly rather than guessing at a prediction.
- If `outputs/.trace/predictor_latest.json` is stale (from a previous run)
  or missing, say so when reporting DiD's during/post-generation results —
  don't present guardrail coverage as complete when it wasn't.
