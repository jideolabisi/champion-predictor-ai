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
own runtime *is* the agent harness. There is no `anthropic` SDK dependency
and no API key in this repo: all model invocation happens through Claude
Code itself.

| Generic harness component | This project |
|---|---|
| Skills | `.claude/skills/champion-predictor/SKILL.md` — orchestrates the full predict → critique → validate → regenerate loop |
| Subagents | `.claude/agents/predictor.md` (primary, ReAct + tools), `.claude/agents/critic.md` (explanation-quality review, no tools) |
| Memory | Predictor's short-term session context during a single prediction run; deliberately no long-term memory across runs (each prediction should be independent, per the checkpoint 2.1 design) |
| MCP / Tools | `mcp_server/safe_server.py`, registered in `.mcp.json` |
| Resources | `data/raw/*.csv`, `data/raw/unstructured/*.txt` |

No GUI/web server/database — this is a CLI-driven deliverable, invoked via
Claude Code. (A simple GUI may be added later; out of scope for now.)

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

## Data sources

Collected (see `data/raw/` and `data/validation/`):
- Team seasonal stats, 2006–2025 (`nfl_data_py` / nflverse)
- Current (2026) roster
- Current-season transactions (already structured, including a free-text
  `DESCRIPTION` field per record — no FAISS needed for this source)
- Historical NFC champions, 2006–2025 (validation-only)

Not yet collected (open, tackled opportunistically — `safe_server.py`
degrades gracefully with an explicit "not available yet" message rather than
erroring if these files don't exist):
- Coaching changes (`data/raw/coaching_changes_2026.csv`)
- Team management changes (`data/raw/management_changes_2026.csv`)
- Player injury reports (`data/raw/injury_reports_2026.csv`)
- Financial information (`data/raw/unstructured/financial/*.txt`)

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
  `hookSpecificOutput` schema exactly at the cap). What's *not* yet verified
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

## Running

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
