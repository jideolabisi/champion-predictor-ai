# Champion Predictor AI

An agentic system that predicts each of the 16 NFC teams' probability of
winning the NFC championship for a given season, explains its reasoning, and
supports what-if decision-support scenarios (player acquisitions, coaching
changes, etc.). Built as a capstone project.

Full design rationale (problem framing, ReAct/tool/memory design, RAG
decisions, ToT-vs-CoT decision) is documented in the capstone checkpoint
submissions; this README covers what was actually built.

## Architecture

This is a Claude Code project, not a standalone application — Claude Code's
own runtime _is_ the agent harness. There is no `anthropic` SDK dependency
and no API key in this repo: all model invocation happens through Claude
Code itself.

| Generic harness component | This project                                                                                                                                                                                                                                                               |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Skills                    | `.claude/skills/champion-predictor/SKILL.md` — orchestrates integrity check → DiD pre-gen → predict/what-if → DiD during-gen → critique → validate → DiD post-gen → finalize, with one shared regeneration counter (cap 2)                                                 |
| Subagents                 | `.claude/agents/predictor.md` (primary, ReAct + tools; also evaluates user-named what-if scenarios), `.claude/agents/critic.md` (explanation-quality review, no tools), `.claude/agents/did.md` (defense-in-depth guardrail, no tools, invoked pre/during/post-generation) |
| Memory                    | Predictor's short-term session context during a single prediction run; deliberately no long-term memory across runs (each prediction should be independent, per the checkpoint 2.1 design)                                                                                 |
| MCP / Tools               | `mcp_server/safe_server.py`, registered in `.mcp.json`                                                                                                                                                                                                                     |
| Resources                 | `data/raw/*.csv`, `data/raw/unstructured/*.txt`                                                                                                                                                                                                                            |
| Hooks                     | `scripts/hooks/enforce_iteration_cap.py` (`PreToolUse`), `scripts/hooks/capture_predictor_trace.py` (`SubagentStop`)                                                                                                                                                       |

Still no database and no `anthropic` SDK dependency — the CLI-driven
`/champion-predictor` flow in Claude Code works identically with or without
a UI running. A Gradio web UI (`app.py`, `ui/`) has since been added on top
of it, offering two ways to drive the pipeline: a free, instant local
heuristic simulator that runs in-process, and a "Real Agentic Predictor"
tab that shells out to the Claude Code CLI in headless mode
(`ui/cli_runner.py`, via `claude -p`) to run the actual skill/subagent/hook
pipeline and stream its output back into the page.

## Retrieval design

Hybrid, not pure vector RAG:

- **Structured data** (team stats, roster, transactions, and — once sourced —
  coaching/management/injury data) is served via exact-filter MCP tools over
  local CSVs. No chunking, no embeddings — a semantic-search layer over
  discrete structured records risks the "split record across chunks" failure
  mode called out in the design docs, and adds no value when you can just
  filter by team.
- **Unstructured text** (financial narratives, and anything else that has no
  clean structured source) goes through a local FAISS index
  (`sentence-transformers`/`all-MiniLM-L6-v2`, no paid API) via the
  `search_unstructured` tool.
- **Fan/analyst predictions** are fetched live via Claude Code's built-in
  `WebSearch` tool inside the predictor subagent — never stored/indexed
  locally, since they're used only as a reasonableness check, not ground
  truth.

Note on checkpoint 3.1 vs. this design: 3.1's original plan was chunked
vector-RAG over _all six_ structured sources (team stats, roster,
transactions, coaching, management, injuries). This repo deliberately
diverges from that — those six are served via exact-filter MCP tools
instead, with FAISS reserved for the one genuinely unstructured source
(financial narratives). This is an intentional refinement, not an
oversight: it structurally eliminates the exact failure mode 3.1 itself
flags (a record's relevant fields getting split across chunks) rather than
just mitigating it statistically. Nothing later in the capstone revisits
retrieval design, so this later, more specific decision stands.

## Validation-data isolation

`data/validation/nfc_champions_2006_2025.csv` (the actual historical NFC
champions) must never be reachable by the predictor or critic, to prevent
information leakage during evaluation. Enforced architecturally, not just by
convention:

- It is read by exactly one file: `scripts/backtest.py`, a plain Python
  script — not an MCP tool, not on any subagent's tool list.
- `mcp_server/safe_server.py` never imports or opens it.
- Inspectable by anyone reviewing the repo:
  ```
  grep -r "nfc_champions" mcp_server/ .claude/   # must return nothing
  ```
- `.claude/agents/predictor.md`'s `tools:` list contains only `champion-data`
  MCP tools + `WebSearch`. `.claude/agents/critic.md`'s `tools:` list is
  empty.

## Safety guardrails (checkpoint 6.1)

A single defense-in-depth (DiD) subagent (`.claude/agents/did.md`) is
invoked three times per run by `SKILL.md`, always as an LLM judgment call —
not a regex/keyword script — per the finalized 6.1 design:

- **Pre-generation**: scores the user's raw request for completeness and
  fit to the agent's objective, and flags suspicious language (attempts to
  get the system to skip its own checks, etc.). A low score or flagged
  language pauses for human-in-the-loop (HITL) clarification before the
  Predictor ever runs.
- **During-generation**: reviews the Predictor's output plus its _actual_
  captured tool-call/reasoning trace (see the new hook below — not a
  self-report, since a compromised or hallucinating Predictor could vouch
  for itself in a self-report) for unsupported certainty and suspicious
  activity. A suspicion score ≥ 80 escalates to HITL immediately,
  regardless of the regeneration cap.
- **Post-generation**: checks the final explanation's factual claims
  against that same trace (used as the retrieved tool/RAG evidence) to
  catch unsupported claims.

All three verdicts, plus the Critic's rating, the Predictor's self-reported
`confidence` (a _separate_ number from its outcome probability — how
reliable it believes its own estimate is, following Kadavath et al. (2022),
not how likely a team is to win), and the deterministic validator's
verdict, feed **one shared regeneration counter, capped at 2** — not four
independent retry loops. Every prediction's output `.md` file records which
trigger(s) fired, so a later escalation-rate calculation is just scanning
`outputs/predictions/*.md`, no separate metrics store needed.

**Read-only + checksums**: `predictor.md` and `critic.md` already carry no
`Write`/`Edit`/`Bash` in their `tools:` frontmatter, so they are
structurally unable to modify input files — that's documentation, not new
enforcement. `scripts/verify_data_integrity.py` adds what wasn't already
true: a SHA-256 manifest (`data/.checksums.json`) checked before every run,
which catches _at-rest_ modification of a data file between fetch and use.
It does **not** detect poisoning that happened upstream, before a fetch
script wrote the file in the first place — that would need source-level
trust verification, which is out of scope here. A mismatch pauses for HITL
confirmation rather than auto-failing, since a legitimate re-fetch looks
identical to tampering from the checksum's point of view.

**New hook**: `scripts/hooks/capture_predictor_trace.py` (`SubagentStop`,
matcher `predictor`) writes `outputs/.trace/predictor_latest.json` from the
Predictor's real transcript, reusing the JSONL-walking approach
`enforce_iteration_cap.py` already established. Same caveat as that hook:
the underlying assumption (a project-level `SubagentStop` hook fires for a
Task-invoked subagent) is not yet empirically verified — smoke-test before
relying on it (see the script's own docstring).

## Decision support (what-if mode)

Per checkpoints 1.1/4.1: the Predictor also evaluates a **user-named**
hypothetical change (a trade, a coaching hire, a front-office change) —
never an agent-invented branch, per 4.1's CoT-not-ToT decision. Ask, e.g.:

```
/champion-predictor how would trading for a top-tier pass rusher affect
the Eagles' NFC championship odds this season?
```

`SKILL.md` detects this as `WHAT_IF` mode, and the Predictor produces both
a real baseline `probabilities` set and a hypothetical
`adjusted_probabilities` set (renormalized across all 16 teams), plus a
`delta_explanation`. Both sets are validated independently. Output goes to
`outputs/predictions/whatif_<team>_<season>_<timestamp>.md`.

## Data sources

Collected (see `data/raw/` and `data/validation/`):

- Team seasonal stats, 2006–2025 (`nfl_data_py` / nflverse)
- Current (2026) roster
- Current-season transactions (already structured, including a free-text
  `DESCRIPTION` field per record — no FAISS needed for this source)
- Historical NFC champions, 2006–2025 (validation-only)

Not yet collected as structured data — `safe_server.py` degrades gracefully
with an explicit "not available yet" message rather than erroring if these
files don't exist:

- **Player injury reports**: `scripts/fetch/fetch_injuries.py` is written
  (uses `nfl_data_py.import_injuries`, requires `pyarrow`) but the 2026
  season's weekly injury reports don't exist yet as of this writing — nflverse
  only publishes them once real games are played. Re-run once the season
  starts.
- **Coaching changes, team management changes**: deliberately _not_ built as
  a hand-authored structured file. There is no clean structured API for
  this (checked: `nfl_data_py` has nothing; the ESPN transactions feed
  behind `get_transactions` is player-moves only). A first pass at compiling
  this from web search results produced contradicting claims across sources
  (e.g. conflicting reports of who was hired where) — encoding that into a
  CSV presented as ground truth would be worse than not having the tool.
  Instead, `predictor.md` instructs the predictor to fall back to
  `WebSearch` for this specific gap, cross-check multiple sources, and
  flag uncertainty in its explanation rather than stating an unverified hire
  as fact.
- **Financial information**: done — `data/raw/unstructured/financial/*.txt`,
  one file per NFC team, sourced from Sportico's 2026 franchise valuations
  (via CBS Sports' summary, a single non-contradictory ranked list — unlike
  the coaching-changes search above). FAISS-indexed; retrieval spot-checked
  and correctly surfaces the Seattle ownership-sale and Green Bay
  non-profit-structure files for matching queries. Known limitation: local
  MiniLM embeddings are weak at numeric-superlative queries (e.g. "highest
  valuation" didn't surface the actual #1 team, Dallas) — semantic search
  isn't a substitute for reading exact numbers when precision matters.

## Known limitations

- **Backtest data depth**: only team-seasonal-stats has real historical
  depth. Roster/transactions/coaching/management/injuries are current-season
  snapshots only, so `scripts/backtest.py` runs are evidence-poorer for older
  seasons than a live current-season run — those tools explicitly return
  "not available for historical evaluation" in backtest mode rather than
  silently leaking present-day data.
- **Consensus check in backtests**: skipped — `WebSearch` can only retrieve
  today's commentary, not period-accurate fan/analyst sentiment for a past
  season.
- **Iteration-cap hook** (`.claude/settings.json` →
  `scripts/hooks/enforce_iteration_cap.py`): the hook script's own logic is
  verified (pipe-tested: allows below the cap, denies with the correct
  `hookSpecificOutput` schema exactly at the cap). What's _not_ yet verified
  is whether Claude Code actually routes tool calls made inside a
  subagent's internal loop through a project-level `PreToolUse` hook, vs.
  only top-level session calls — that can only be confirmed by watching a
  live `predictor` subagent run hit the cap. Smoke-test before relying on
  it (see script docstring); the prompt-level budget in `predictor.md` is
  the fallback either way.

## Setup

```
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
# nfl_data_py (only needed to re-run scripts/fetch/*.py) has a numpy<2 pin
# that conflicts with the numpy>=2 wheels used here — install separately:
.venv/Scripts/python.exe -m pip install --no-deps nfl_data_py
.venv/Scripts/python.exe -m pip install appdirs requests requests-cache
```

If any `.txt` files are added under `data/raw/unstructured/<source>/`,
(re)build the FAISS index:

```
.venv/Scripts/python.exe scripts/build_faiss_index.py
```

Whenever any file under `data/raw/` or `data/validation/` legitimately
changes (a re-fetch, a new source added), regenerate the checksum baseline
so the next run's integrity check doesn't flag it as a mismatch:

```
.venv/Scripts/python.exe scripts/verify_data_integrity.py --generate
```

## Running

### Gradio Interactive Web UI

Launch the interactive Gradio dashboard:

```
.venv/Scripts/python.exe app.py
```

Open your browser at `http://127.0.0.1:7860`. The UI provides:

- **🔮 Live Predictions & Rankings**: Probability bar charts with official franchise colors, division sunburst breakdown, leaderboard table, and multi-agent DiD audit summaries.
- **⚡ What-If Decision Support**: Interactive scenario simulator (trades, free-agent acquisitions, coaching changes, injuries) with side-by-side odds comparison charts.
- **📊 Conference Data Explorer**: Historical team stat trends (2006–2025), active 2026 rosters, and transactions.
- **🔍 FAISS Vector RAG Search**: Semantic search over unstructured franchise valuation narratives.
- **🛡️ Audit Traces & Integrity**: Real-time SHA-256 data integrity checks, manifest regeneration, and saved prediction reports viewer.

### CLI & Agent Execution

From Claude Code, in this repo:

```
/champion-predictor
```

(or just ask Claude Code to "run the champion-predictor skill for the
current season"). Output is written to `outputs/predictions/`.

Backtest a past season:

```
.venv/Scripts/python.exe scripts/backtest.py --season 2015
.venv/Scripts/python.exe scripts/backtest.py --seasons 2010-2020
```

Output is written to `outputs/backtest_results_<range>.md`.
