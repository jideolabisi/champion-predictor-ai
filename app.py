"""Champion Predictor AI — Interactive Gradio UI.

Provides an interactive dashboard for:
- NFC Championship win-probability predictions & visualizations
- What-If scenario simulation & decision support
- Conference structured data explorer (stats, rosters, transactions)
- FAISS vector RAG semantic search over unstructured narratives
- Audit traces, Defense-in-Depth guardrail inspector, and saved reports
"""

from __future__ import annotations

import datetime
import functools
import json
import re
import traceback
from typing import Any
import pandas as pd
import gradio as gr
from gradio.themes.utils import sizes

from ui.constants import NFC_TEAMS, TEAM_CHOICES, TEAM_METADATA, DIVISIONS
from ui.data_service import (
    check_integrity,
    generate_checksum_manifest,
    get_available_seasons,
    get_historical_champions,
    get_latest_session_trace,
    get_team_roster,
    get_team_stats,
    get_transactions,
    search_unstructured_financial,
)
from ui.charts import (
    create_probability_chart,
    create_stats_trend_chart,
    create_whatif_comparison_chart,
)
from ui.prediction_service import (
    execute_prediction_pipeline,
    list_saved_predictions,
    load_prediction_file,
)
from ui.cli_runner import DEFAULT_MAX_BUDGET_USD, run_real_predictor_stream

# Season choices span the historical dataset plus the current forward-looking
# season (which has no completed stats yet, only roster/transaction data).
_HISTORICAL_SEASONS = get_available_seasons()
_CURRENT_SEASON = 2026
SEASON_CHOICES = sorted({_CURRENT_SEASON, *_HISTORICAL_SEASONS}, reverse=True)

# Custom CSS for dark sports analytics aesthetic
CUSTOM_CSS = """
.gradio-container {
    max-width: 1350px !important;
    margin: 0 auto !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.header-box {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #004c54 100%);
    color: white;
    padding: 24px 30px;
    border-radius: 12px;
    margin-bottom: 20px;
    border: 1px solid #334155;
    box-shadow: 0 4px 14px rgba(0,0,0,0.3);
}
.metric-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.metric-val {
    font-size: 28px;
    font-weight: 700;
    color: #38bdf8;
}
.metric-label {
    font-size: 14px;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.status-line {
    font-size: 15px;
    font-weight: 600;
    margin: 4px 0 8px 0;
}
.audit-badge-pass {
    display: inline-block;
    background: #065f46;
    color: #34d399;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 600;
}
.tab-content {
    padding-top: 10px;
}
.confirm-box {
    background: #fef3c7 !important;
    border: 2px solid #d97706 !important;
    border-radius: 10px;
    padding: 12px 14px !important;
    margin-bottom: 10px;
}
.confirm-box label {
    color: #78350f !important;
    font-weight: 600;
}
"""


def _handle_errors(fn):
    """Turn an unhandled exception in a Gradio callback into a clean toast
    instead of a raw traceback (or a silent 500) reaching the user."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            traceback.print_exc()
            raise gr.Error(f"{fn.__name__.replace('_', ' ')} failed: {exc}") from exc

    return wrapper


def _build_leaderboard_table(probs: dict[str, float]) -> pd.DataFrame:
    """Rank/Code/Team/Division/Win% table, shared by the Predict-mode results
    of both the local simulator and the real agentic run."""
    sorted_probs = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    rows = []
    for idx, (t, p) in enumerate(sorted_probs, 1):
        meta = TEAM_METADATA.get(t, {})
        rows.append([idx, t, meta.get("name", t), meta.get("division", ""), f"{p:.1f}%"])
    return pd.DataFrame(rows, columns=["Rank", "Code", "Team Name", "Division", "Win Probability"])


def _build_whatif_table(probs: dict[str, float], adj_probs: dict[str, float], target_team: str) -> pd.DataFrame:
    """Rank/Code/Team/Baseline/Scenario/Net Shift table, shared by the
    What-If-mode results of both the local simulator and the real agentic run."""
    sorted_teams = sorted(probs.keys(), key=lambda t: probs[t], reverse=True)
    rows = []
    for idx, t in enumerate(sorted_teams, 1):
        meta = TEAM_METADATA.get(t, {})
        base_p = probs.get(t, 0.0)
        adj_p = adj_probs.get(t, 0.0)
        delta = adj_p - base_p
        delta_str = f"+{delta:.1f}%" if delta > 0 else (f"{delta:.1f}%" if delta < 0 else "0.0%")
        marker = "🎯 TARGET" if t == target_team else ""
        rows.append([idx, t, meta.get("name", t), f"{base_p:.1f}%", f"{adj_p:.1f}%", delta_str, marker])
    return pd.DataFrame(rows, columns=["Rank", "Code", "Team Name", "Baseline", "Scenario Odds", "Net Shift", "Status"])


def _build_audit_markdown(
    validation: dict[str, Any],
    did: dict[str, Any],
    critic: dict[str, Any],
    filename: str,
) -> str:
    """Guardrails & Verification panel, shared by both tabs."""
    integrity = check_integrity()
    if integrity.get("passed"):
        checksum_line = "- **Data Checksum (SHA-256 manifest)**: `PASSED` — all watched files match the baseline\n"
    else:
        changed = integrity.get("changed", [])
        checksum_line = (
            f"- **Data Checksum (SHA-256 manifest)**: `MISMATCH` — {len(changed)} file(s) changed since baseline\n"
        )
    return (
        f"### 🛡️ Guardrails & Verification\n"
        f"- **Deterministic Validation**: `{validation.get('verdict', '?').upper()}` "
        f"(Sum: `{validation.get('probability_sum_check', {}).get('total', 0):.1f}%`)\n"
        f"- **Consensus Variance**: `{validation.get('consensus_variance_check', {}).get('overlap', '?')}/3` "
        f"overlap with Top-3 consensus\n"
        f"{checksum_line}"
        f"- **DiD Pre-gen Prompt**: `{did.get('pre_generation', {}).get('prompt_score', '?')}/100` "
        f"(`{did.get('pre_generation', {}).get('verdict', '?')}`)\n"
        f"- **DiD During-gen Overconfidence**: `{did.get('during_generation', {}).get('overconfidence_verdict', '?')}` "
        f"(Suspicion: `{did.get('during_generation', {}).get('suspicion_score', '?')}/100`)\n"
        f"- **DiD Post-gen Groundedness**: `{did.get('post_generation', {}).get('groundedness_verdict', '?')}`\n"
        f"- **Critic Rating**: `{critic.get('rating', '?')}`\n"
        f"- **Report Saved**: `{filename}`"
    )


_TEAM_CODE_RE = re.compile(r"\*\*([A-Z]{2,4})\*\*")
_PERCENT_RE = re.compile(r"([+-]?\d{1,3}(?:\.\d+)?)\s*%")
_CONFIDENCE_RE = re.compile(r"Confidence[^\d]{0,20}(\d{1,3})\s*%", re.IGNORECASE)


def _parse_probabilities_from_markdown(md: str) -> tuple[dict[str, float], dict[str, float] | None]:
    """Best-effort fallback: recover a probabilities table (and, if present,
    an adjusted-probabilities table) from a freeform markdown report when no
    structured JSON companion file exists alongside it — e.g. real agentic
    runs made before the champion-predictor skill was updated to also write
    one. Relies only on the report's table shape being roughly what
    SKILL.md's step 9 asks for: a bolded team code per row, followed by one
    percentage (PREDICT mode) or two — baseline then adjusted — (WHAT_IF
    mode) in column order. Never raises; returns empty dicts if it can't
    find anything that looks like a probability table.
    """
    probs: dict[str, float] = {}
    adjusted: dict[str, float] = {}
    for line in md.splitlines():
        if "|" not in line:
            continue
        team_match = _TEAM_CODE_RE.search(line)
        if not team_match or team_match.group(1) not in TEAM_METADATA:
            continue
        team = team_match.group(1)
        percents = _PERCENT_RE.findall(line)
        if not percents:
            continue
        try:
            probs[team] = float(percents[0])
            if len(percents) >= 2:
                adjusted[team] = float(percents[1])
        except ValueError:
            continue
    return probs, (adjusted if adjusted else None)


def _parse_confidence_from_markdown(md: str) -> int | None:
    match = _CONFIDENCE_RE.search(md)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


@_handle_errors
def run_simulator_ui(mode_choice: str, season_val: int, target_team: str, scenario_text: str):
    """Execute the local, non-agentic heuristic pipeline (same predict/what-if
    modes as the Real Agentic Predictor tab, same report format) but computed
    instantly in-process — no Claude Code CLI call, no usage cost. Useful for
    exercising the guardrail/report pipeline without spending a real run.
    """
    mode = "what_if" if mode_choice == "What-If" else "predict"
    if mode == "what_if" and not scenario_text.strip():
        scenario_text = "Trade for an All-Pro Defensive Pass Rusher"

    result = execute_prediction_pipeline(
        season=int(season_val),
        mode=mode,
        target_team=target_team if mode == "what_if" else None,
        scenario_description=scenario_text if mode == "what_if" else None,
    )

    # Render the ReAct-style trace into the log panel to mirror the shape of
    # the real agentic run's live log, even though this executes synchronously.
    log_lines = [f"Running local {mode} simulation for season {season_val}...\n"]
    for step in result["trace"]:
        log_lines.append(
            f"[Round {step['round']}] Thought: {step['thought']}\n"
            f"  Action: {step['action']}\n"
            f"  Observation: {step['observation']}\n"
        )
    did = result["did"]
    log_lines.append(
        f"Guardrails — DiD pre-gen: {did['pre_generation']['verdict']} | "
        f"during-gen: {did['during_generation']['overconfidence_verdict']} | "
        f"post-gen: {did['post_generation']['groundedness_verdict']}"
    )
    log_lines.append(
        f"Validation: {result['validation']['verdict'].upper()} | Critic: {result['critic']['rating']}"
    )
    log_lines.append(f"\nDone. Report saved: {result['filename']}")
    log_text = "\n".join(log_lines)

    probs = result["prediction"]["probabilities"]
    confidence = result["prediction"]["confidence"]

    if mode == "what_if":
        adj_probs = result["prediction"]["adjusted_probabilities"]
        chart = create_whatif_comparison_chart(probs, adj_probs, target_team=target_team)
        report_md = f"### 🔄 Scenario Delta Analysis\n{result['prediction']['delta_explanation']}"
        prob_table = _build_whatif_table(probs, adj_probs, target_team)
    else:
        chart = create_probability_chart(probs, f"{season_val} NFC Championship Odds (Simulated)")
        report_md = f"### 📝 Simulated Reasoning & Narrative\n{result['prediction']['explanation']}"
        prob_table = _build_leaderboard_table(probs)

    prob_md = f"### 🏆 Win Probability Leaderboard\n**Predictor Confidence**: `{confidence}%`"
    audit_md = _build_audit_markdown(result["validation"], result["did"], result["critic"], result["filename"])

    return log_text, report_md, prob_md, prob_table, chart, audit_md


def run_real_predictor_ui(
    mode_choice: str,
    season_val: int,
    target_team: str,
    scenario_text: str,
    max_budget: float,
    confirmed: bool,
):
    """Stream a real Claude Code run of the champion-predictor skill (the
    actual predictor/critic/did subagents, MCP tools, and hooks) and render
    its output as it completes. Unlike the other tabs, this is a genuine
    multi-minute agentic run, not a local heuristic — gated behind an
    explicit confirmation checkbox and a hard per-run budget cap.
    """
    if not confirmed:
        yield (
            "Not started. In the panel on the left, check the highlighted "
            "confirmation checkbox (below Max Budget) to launch a real Claude "
            "Code agent run (this consumes real usage against your Claude "
            "account, up to the budget cap), then click Run Real Predictor again.",
            "*Awaiting confirmation.*",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            "⚪ **Not started** — check the confirmation box first.",
            gr.update(),
        )
        return

    mode = "what_if" if mode_choice == "What-If" else "predict"
    log_lines = ["Starting real predictor run — this can take several minutes...\n"]
    yield (
        "\n".join(log_lines), "*Running...*", gr.update(), gr.update(), gr.update(), gr.update(),
        "🟠 **Running...** This can take several minutes — the button is disabled until it finishes.",
        gr.update(interactive=False),
    )

    try:
        for event in run_real_predictor_stream(
            mode=mode,
            season=int(season_val),
            target_team=target_team if mode == "what_if" else None,
            scenario=scenario_text if mode == "what_if" else None,
            max_budget_usd=float(max_budget),
        ):
            if event["type"] == "log":
                log_lines.append(event["text"])
                yield (
                    "\n".join(log_lines), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(),
                )
                continue

            # event["type"] == "done"
            result_event = event.get("result_event") or {}
            cost = result_event.get("total_cost_usd")
            cost_note = f" (cost: ${cost:.4f})" if cost is not None else ""

            if event["report_stem"]:
                data = load_prediction_file(event["report_stem"]) or {}
                md = data.get("markdown_content") or "*Report written but could not be read back.*"
                probs = data.get("probabilities") or data.get("prediction", {}).get("probabilities", {})
                adj_probs = data.get("adjusted_probabilities") or data.get("prediction", {}).get("adjusted_probabilities")
                confidence = data.get("confidence") or data.get("prediction", {}).get("confidence")

                # The champion-predictor skill now writes a JSON companion
                # file with structured probabilities/guardrail data (see
                # SKILL.md step 9). For reports saved before that change
                # (or if the JSON write is ever skipped), fall back to
                # best-effort parsing of the markdown table instead of
                # showing nothing.
                if not probs and md:
                    parsed_probs, parsed_adj = _parse_probabilities_from_markdown(md)
                    if parsed_probs:
                        probs, adj_probs = parsed_probs, parsed_adj
                if confidence is None and md:
                    confidence = _parse_confidence_from_markdown(md)

                if adj_probs:
                    chart = create_whatif_comparison_chart(probs, adj_probs, target_team=target_team)
                elif probs:
                    chart = create_probability_chart(probs, f"Real Agentic Prediction — {season_val}")
                else:
                    chart = gr.update()

                if probs:
                    prob_table = _build_whatif_table(probs, adj_probs, target_team) if adj_probs else _build_leaderboard_table(probs)
                    conf_line = f"**Predictor Confidence**: `{confidence}%`" if confidence is not None else "*Confidence score not reported as structured data.*"
                    prob_md = f"### 🏆 Win Probability Leaderboard\n{conf_line}"
                else:
                    prob_table = gr.update()
                    prob_md = "*Structured probability data isn't available for this report — the full narrative (including the probability table) is in the Result - Analysis tab.*"

                did = data.get("did") or {}
                validation = data.get("validation") or {}
                critic = data.get("critic") or {}
                if did or validation or critic:
                    audit_md = _build_audit_markdown(validation, did, critic, event["report_stem"])
                else:
                    audit_md = "*Structured guardrail verdicts aren't available for this report — the skill reports them in prose in the Result - Analysis tab instead.*"

                log_lines.append(f"\nDone. Report saved: {event['report_stem']}{cost_note}")
                yield (
                    "\n".join(log_lines), md, prob_md, prob_table, chart, audit_md,
                    f"✅ **Done.** Report saved: `{event['report_stem']}`{cost_note}",
                    gr.update(interactive=True),
                )
            else:
                status = "completed without writing a report" if event["ok"] else "failed"
                explanation = event.get("error") or "No further detail was returned."
                log_lines.append(f"\nRun {status}{cost_note}.")
                icon = "⚠️" if event["ok"] else "❌"
                yield (
                    "\n".join(log_lines), f"### Run {status}\n\n{explanation}",
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    f"{icon} **Run {status}**{cost_note}.",
                    gr.update(interactive=True),
                )
    except Exception as exc:
        traceback.print_exc()
        log_lines.append(f"\nError: {exc}")
        yield (
            "\n".join(log_lines), "", gr.update(), gr.update(), gr.update(), gr.update(),
            f"❌ **Error**: {exc}",
            gr.update(interactive=True),
        )
        raise gr.Error(f"Real predictor run failed: {exc}") from exc


@_handle_errors
def load_stats_ui(selected_teams: list[str], metric_name: str):
    """Filter team stats and generate historical trend chart."""
    metric_map = {
        "Passing EPA (passing_epa)": ("passing_epa", "Passing EPA"),
        "Rushing EPA (rushing_epa)": ("rushing_epa", "Rushing EPA"),
        "Passing Touchdowns (passing_tds)": ("passing_tds", "Passing TDs"),
        "Rushing Yards (rushing_yards)": ("rushing_yards", "Rushing Yards"),
        "Defensive Sacks (def_sacks)": ("def_sacks", "Defensive Sacks"),
        "Interceptions Thrown (interceptions)": ("interceptions", "Interceptions"),
        "Fantasy Points PPR (fantasy_points_ppr)": ("fantasy_points_ppr", "Fantasy Points PPR"),
    }
    col, label = metric_map.get(metric_name, ("passing_epa", "Passing EPA"))

    all_rows = []
    if not selected_teams:
        selected_teams = ["PHI", "SF", "DET", "DAL"]

    for t in selected_teams:
        rows = get_team_stats(team_code=t)
        all_rows.extend(rows)

    chart = create_stats_trend_chart(all_rows, col, label)

    # DataFrame view
    df = pd.DataFrame(all_rows)
    cols_to_show = ["Team", "Season", "passing_yards", "passing_tds", "interceptions", "passing_epa", "rushing_yards", "rushing_tds", "def_sacks"]
    cols_exist = [c for c in cols_to_show if c in df.columns]
    df_view = df[cols_exist].sort_values(["Season", "Team"], ascending=[False, True]) if not df.empty else pd.DataFrame()

    return chart, df_view


@_handle_errors
def load_roster_ui(team_val: str, pos_val: str):
    """Filter 2026 roster."""
    team = None if team_val == "ALL" else team_val
    pos = None if pos_val == "ALL" else pos_val
    rows = get_team_roster(team_code=team, position=pos)
    df = pd.DataFrame(rows)
    cols = ["team", "jersey_number", "player_name", "position", "depth_chart_position", "years_exp", "age", "college", "status"]
    cols_exist = [c for c in cols if c in df.columns]
    return df[cols_exist] if not df.empty else pd.DataFrame()


@_handle_errors
def load_transactions_ui(team_val: str, tx_type_val: str, kw_val: str):
    """Filter 2026 transactions."""
    rows = get_transactions(team_code=team_val, transaction_type=tx_type_val, search_keyword=kw_val)
    df = pd.DataFrame(rows)
    cols = ["SNAPSHOT_DATE", "TEAM", "TRANSACTION_TYPE", "PLAYER", "DESCRIPTION"]
    cols_exist = [c for c in cols if c in df.columns]
    return df[cols_exist] if not df.empty else pd.DataFrame()


@_handle_errors
def load_faiss_search_ui(query_text: str, top_k_val: int):
    """Execute FAISS vector search and format results."""
    if not query_text.strip():
        query_text = "Seattle Seahawks ownership sale franchise valuation"
    results = search_unstructured_financial(query_text, top_k=int(top_k_val))

    formatted_cards = []
    for idx, r in enumerate(results, 1):
        score = r.get("score", 0.0)
        source_file = r.get("source_file", "financial")
        text = r.get("text", "")
        formatted_cards.append(
            f"### Result {idx} — Score: `{score:.4f}`\n"
            f"**Source**: `{source_file}`\n\n"
            f"> {text}\n"
            f"---\n"
        )
    return "\n".join(formatted_cards)


@_handle_errors
def run_integrity_ui():
    """Run data integrity check."""
    res = check_integrity()
    passed = res.get("passed", False)
    changed = res.get("changed", [])
    if passed:
        return f"✅ **Data Integrity Verified**: All raw and validation data match the SHA-256 baseline manifest.\n\n```json\n{json.dumps(res, indent=2)}\n```"
    else:
        return f"⚠️ **Integrity Alert / Mismatch Detected**:\n\n```json\n{json.dumps(res, indent=2)}\n```\nChanged files: {len(changed)}"


@_handle_errors
def regenerate_manifest_ui(confirmed: bool):
    """Regenerate SHA-256 manifest.

    Regenerating overwrites the tamper-detection baseline with whatever is
    currently on disk — if a file was tampered with since the last fetch,
    this makes that tampering look "verified". Requires an explicit
    confirmation checkbox so it can't be triggered by a stray click.
    """
    if not confirmed:
        return (
            "⚠️ **Not regenerated.** Regenerating the manifest resets the tamper-detection "
            "baseline to whatever is on disk right now — if any watched file was modified "
            "since the last legitimate fetch, that change becomes silently \"verified\". "
            "Check the confirmation box only if you intentionally re-fetched or changed the source data."
        )
    res = generate_checksum_manifest()
    return f"✅ **New Baseline Manifest Generated**:\n\n```json\n{json.dumps(res, indent=2)}\n```"


@_handle_errors
def load_reports_archive_ui():
    """List saved prediction files."""
    items = list_saved_predictions()
    if not items:
        return gr.Dropdown(choices=[]), "No saved reports found in `outputs/predictions/`."
    choices = [item["stem"] for item in items]
    return gr.Dropdown(choices=choices, value=choices[0]), f"Found {len(choices)} saved prediction reports."


@_handle_errors
def load_latest_trace_ui():
    """Load the Predictor's most recent short-term session trace, if any.

    Two different producers write this same file with two different shapes:
    - The local simulator writes a *list* of `{round, thought, action,
      observation}` dicts.
    - The real agent's `SubagentStop` hook (capture_predictor_trace.py)
      writes a single *dict* with separate `tool_calls` and `reasoning_text`
      lists (plus an optional `note`/`error`) — it has no per-round grouping
      because it's reading the raw transcript, not a self-report. Treating
      that dict as if it were the simulator's list of round-dicts (iterating
      it yields its string keys) is what previously crashed with
      `'str' object has no attribute 'get'`.
    """
    trace = get_latest_session_trace()
    if not trace:
        return "*No session trace found yet — run a prediction (simulator or real agent) first.*"

    mtime_str = datetime.datetime.fromtimestamp(trace["mtime"]).strftime("%Y-%m-%d %H:%M:%S")
    steps = trace["steps"]
    lines = [f"**Last updated**: `{mtime_str}` — from `outputs/.trace/predictor_latest.json`\n"]

    if isinstance(steps, list):
        for step in steps:
            lines.append(
                f"**Round {step.get('round', '?')}**\n"
                f"- Thought: {step.get('thought', '')}\n"
                f"- Action: `{step.get('action', '')}`\n"
                f"- Observation: {step.get('observation', '')}\n"
            )
    elif isinstance(steps, dict):
        note = steps.get("note") or steps.get("error")
        if note:
            lines.append(f"> ⚠️ {note}\n")
        for text in steps.get("reasoning_text", []):
            lines.append(f"- Thought: {text}\n")
        for call in steps.get("tool_calls", []):
            if "tool" in call:
                lines.append(f"- Action: `{call.get('tool', '')}({json.dumps(call.get('input', {}))})`\n")
            elif "observation" in call:
                lines.append(f"- Observation: {call.get('observation', '')}\n")
        if not steps.get("reasoning_text") and not steps.get("tool_calls") and not note:
            lines.append("*Trace file present but empty.*")
    else:
        lines.append(f"*Unrecognized trace format: `{type(steps).__name__}`.*")

    return "\n".join(lines)


@_handle_errors
def view_report_detail_ui(selected_stem: str):
    """Load detail of a specific saved prediction report."""
    if not selected_stem:
        return "Select a report to view.", gr.update()
    data = load_prediction_file(selected_stem)
    if not data:
        return "Report file could not be read.", gr.update()

    md_content = data.get("markdown_content", "")
    probs = data.get("probabilities", {})
    if not probs and "prediction" in data:
        probs = data["prediction"].get("probabilities", {})

    chart = create_probability_chart(probs, f"Report Odds: {selected_stem}") if probs else gr.update()
    return md_content, chart


# Build the Gradio App Interface
def build_app() -> gr.Blocks:
    with gr.Blocks(title="Champion Predictor AI") as demo:
        # Header Box
        with gr.Row():
            gr.HTML(
                """
                <div class="header-box">
                    <h1 style="margin: 0; font-size: 28px; font-weight: 800; color: #ffffff;">🏈 Champion Predictor AI</h1>
                    <p style="margin: 6px 0 0 0; font-size: 15px; color: #94a3b8;">
                        NFC Championship Win Probability Engine &amp; What-If Decision Support System
                    </p>
                </div>
                """
            )

        with gr.Tabs() as main_tabs:
            # ----------------------------------------------------
            # TAB 1: Non-Agentic Simulator (local heuristic, no LLM)
            # ----------------------------------------------------
            with gr.TabItem("🧪 Non-Agentic Simulator (For Test)", id="tab_simulator"):
                gr.Markdown(
                    """
                    ### 🧪 Fast Local Heuristic Simulator
                    Runs the same Predict / What-If pipeline and report format as the **Real
                    Agentic Predictor** tab, but computed instantly in-process from a deterministic
                    local heuristic over the CSVs — no Claude Code CLI call, no usage cost. Useful
                    for exercising the guardrail/report pipeline (or demoing the UI) for free.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        sim_mode = gr.Radio(
                            choices=["Predict", "What-If"],
                            value="Predict",
                            label="Mode",
                        )
                        sim_season = gr.Dropdown(
                            choices=SEASON_CHOICES,
                            value=_CURRENT_SEASON,
                            label="Season",
                            info=(
                                "Historical stats are restricted to seasons before the one "
                                "selected. Roster/transaction/financial inputs always reflect "
                                f"the current {_CURRENT_SEASON} snapshot."
                            ),
                        )
                        sim_team = gr.Dropdown(
                            choices=[(f"{c} — {TEAM_METADATA[c]['name']}", c) for c in NFC_TEAMS],
                            value="PHI",
                            label="Target NFC Franchise",
                            visible=False,
                        )
                        sim_preset = gr.Dropdown(
                            choices=[
                                "Trade for an All-Pro Defensive Pass Rusher",
                                "Starting Quarterback Suffers Season-Ending Knee Injury",
                                "Sign Elite Pro-Bowl Wide Receiver in Free Agency",
                                "Hire High-Powered Super Bowl-Winning Offensive Coordinator",
                                "Lose Multiple Starting Offensive Linemen to Injury",
                            ],
                            value="Trade for an All-Pro Defensive Pass Rusher",
                            label="Quick Presets",
                            visible=False,
                        )
                        sim_scenario = gr.Textbox(
                            lines=3,
                            value="Trade for an All-Pro Defensive Pass Rusher to anchor the defensive front.",
                            label="Hypothetical Scenario Description",
                            placeholder="Describe the trade, signing, coaching change, or injury in detail...",
                            visible=False,
                        )
                        sim_preset.change(
                            fn=lambda p: p,
                            inputs=[sim_preset],
                            outputs=[sim_scenario],
                        )

                        def _toggle_sim_mode(choice):
                            is_whatif = choice == "What-If"
                            return (
                                gr.update(visible=is_whatif),
                                gr.update(visible=is_whatif),
                                gr.update(visible=is_whatif),
                            )

                        sim_mode.change(
                            fn=_toggle_sim_mode,
                            inputs=[sim_mode],
                            outputs=[sim_team, sim_preset, sim_scenario],
                        )

                        sim_run_btn = gr.Button("⚡ Run Local Simulation", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.TabItem("Reasoning Trace"):
                                sim_log = gr.Textbox(
                                    label="Simulated Reasoning Trace",
                                    lines=20,
                                    max_lines=20,
                                    interactive=False,
                                    autoscroll=True,
                                    show_label=False,
                                )
                            with gr.TabItem("Result - Analysis"):
                                sim_report_md = gr.Markdown("*Run the simulator to see the report here.*")
                            with gr.TabItem("Result - Probabilities"):
                                sim_prob_md = gr.Markdown("*Run the simulator to see the leaderboard here.*")
                                sim_prob_table = gr.Dataframe(interactive=False)
                            with gr.TabItem("Result - Chart"):
                                sim_chart = gr.Plot(label="Probability Outcome", show_label=False)
                            with gr.TabItem("Guardrails & Verification"):
                                sim_audit_md = gr.Markdown("*Run the simulator to see the guardrail verdicts here.*")

                sim_run_btn.click(
                    fn=run_simulator_ui,
                    inputs=[sim_mode, sim_season, sim_team, sim_scenario],
                    outputs=[sim_log, sim_report_md, sim_prob_md, sim_prob_table, sim_chart, sim_audit_md],
                )

            # ----------------------------------------------------
            # TAB 2: Real Agentic Predictor (Claude Code CLI)
            # ----------------------------------------------------
            with gr.TabItem("🤖 Real Agentic Predictor", id="tab_real_agent"):
                gr.Markdown(
                    f"""
                    ### 🤖 Run the Real ReAct Predictor Pipeline
                    **This tab launches the actual `champion-predictor` skill in Claude Code** — the
                    `predictor` subagent doing real ReAct tool-calling reasoning over the MCP data
                    tools (and WebSearch), reviewed by the `critic` subagent, checked by the
                    three-stage `did` guardrail, and validated deterministically — exactly as if
                    you'd typed `/champion-predictor` yourself.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        agent_mode = gr.Radio(
                            choices=["Predict", "What-If"],
                            value="Predict",
                            label="Mode",
                        )
                        agent_season = gr.Dropdown(
                            choices=SEASON_CHOICES,
                            value=_CURRENT_SEASON,
                            label="Season",
                        )
                        agent_team = gr.Dropdown(
                            choices=[(f"{c} — {TEAM_METADATA[c]['name']}", c) for c in NFC_TEAMS],
                            value="PHI",
                            label="Target NFC Franchise",
                            visible=False,
                        )
                        agent_scenario = gr.Textbox(
                            lines=3,
                            label="Hypothetical Scenario Description",
                            placeholder="Describe the trade, signing, coaching change, or injury in detail...",
                            visible=False,
                        )

                        def _toggle_agent_mode(choice):
                            is_whatif = choice == "What-If"
                            return gr.update(visible=is_whatif), gr.update(visible=is_whatif)

                        agent_mode.change(
                            fn=_toggle_agent_mode,
                            inputs=[agent_mode],
                            outputs=[agent_team, agent_scenario],
                        )

                        agent_budget = gr.Number(
                            value=DEFAULT_MAX_BUDGET_USD,
                            minimum=0.10,
                            maximum=25.0,
                            label="Max Budget (USD) — hard cap via --max-budget-usd",
                        )
                        with gr.Group(elem_classes=["confirm-box"]):
                            agent_confirm = gr.Checkbox(
                                label="I understand this launches a real Claude Code agent run using my account's usage, up to the budget cap above",
                                value=False,
                            )
                        agent_run_btn = gr.Button("▶ Run Real Predictor", variant="stop", size="lg")
                        agent_status = gr.Markdown("⚪ **Idle**", elem_classes=["status-line"])

                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.TabItem("Reasoning Trace"):
                                agent_log = gr.Textbox(
                                    label="Live Agent Log",
                                    lines=20,
                                    max_lines=20,
                                    interactive=False,
                                    autoscroll=True,
                                    show_label=False,
                                )
                            with gr.TabItem("Result - Analysis"):
                                agent_report_md = gr.Markdown("*Run the predictor to see the report here.*")
                            with gr.TabItem("Result - Probabilities"):
                                agent_prob_md = gr.Markdown("*Run the predictor to see the leaderboard here.*")
                                agent_prob_table = gr.Dataframe(interactive=False)
                            with gr.TabItem("Result - Chart"):
                                agent_chart = gr.Plot(label="Probability Outcome", show_label=False)
                            with gr.TabItem("Guardrails & Verification"):
                                agent_audit_md = gr.Markdown("*Run the predictor to see the guardrail verdicts here.*")

                agent_run_btn.click(
                    fn=run_real_predictor_ui,
                    inputs=[agent_mode, agent_season, agent_team, agent_scenario, agent_budget, agent_confirm],
                    outputs=[
                        agent_log, agent_report_md, agent_prob_md, agent_prob_table, agent_chart, agent_audit_md,
                        agent_status, agent_run_btn,
                    ],
                )

            # ----------------------------------------------------
            # TAB 3: Data Explorer
            # ----------------------------------------------------
            with gr.TabItem("📊 Data Explorer", id="tab_data"):
                with gr.Tabs():
                    # Sub-tab: Seasonal Stats
                    with gr.TabItem("📈 Historical Team Stats (2006–2025)"):
                        with gr.Row():
                            stats_teams = gr.CheckboxGroup(
                                choices=[(f"{c} ({TEAM_METADATA[c]['name'].split()[-1]})", c) for c in NFC_TEAMS],
                                value=["PHI", "SF", "DET", "DAL", "GB"],
                                label="Select Teams to Compare",
                            )
                            stats_metric = gr.Dropdown(
                                choices=[
                                    "Passing EPA (passing_epa)",
                                    "Rushing EPA (rushing_epa)",
                                    "Passing Touchdowns (passing_tds)",
                                    "Rushing Yards (rushing_yards)",
                                    "Defensive Sacks (def_sacks)",
                                    "Interceptions Thrown (interceptions)",
                                    "Fantasy Points PPR (fantasy_points_ppr)",
                                ],
                                value="Passing EPA (passing_epa)",
                                label="Statistical Metric",
                            )
                            load_stats_btn = gr.Button("📊 Update Stats Chart", variant="secondary")

                        stats_plot = gr.Plot(label="Historical Trend")
                        stats_table = gr.Dataframe(label="Team Stats Raw Data", interactive=False)

                        load_stats_btn.click(
                            fn=load_stats_ui,
                            inputs=[stats_teams, stats_metric],
                            outputs=[stats_plot, stats_table],
                        )

                    # Sub-tab: 2026 Roster
                    with gr.TabItem("👥 2026 Active Rosters"):
                        with gr.Row():
                            roster_team_sel = gr.Dropdown(
                                choices=["ALL"] + [(f"{c} — {TEAM_METADATA[c]['name']}", c) for c in NFC_TEAMS],
                                value="PHI",
                                label="Team Filter",
                            )
                            roster_pos_sel = gr.Dropdown(
                                choices=["ALL", "QB", "RB", "WR", "TE", "OL", "T", "G", "C", "DL", "DT", "DE", "LB", "DB", "CB", "S", "K", "P"],
                                value="ALL",
                                label="Position Filter",
                            )
                            roster_btn = gr.Button("🔍 Search Roster", variant="secondary")

                        roster_table = gr.Dataframe(label="2026 Roster Database", interactive=False)
                        roster_btn.click(
                            fn=load_roster_ui,
                            inputs=[roster_team_sel, roster_pos_sel],
                            outputs=[roster_table],
                        )

                    # Sub-tab: 2026 Transactions
                    with gr.TabItem("🔄 2026 Transactions"):
                        with gr.Row():
                            tx_team_sel = gr.Dropdown(
                                choices=["ALL"] + [(f"{c} — {TEAM_METADATA[c]['name']}", c) for c in NFC_TEAMS],
                                value="ALL",
                                label="Team Filter",
                            )
                            tx_type_sel = gr.Dropdown(
                                choices=["ALL", "SIGNING", "WAIVE_RELEASE", "TRADE"],
                                value="ALL",
                                label="Transaction Type",
                            )
                            tx_kw_input = gr.Textbox(placeholder="Search player or description...", label="Keyword Search")
                            tx_btn = gr.Button("🔍 Filter Transactions", variant="secondary")

                        tx_table = gr.Dataframe(label="Offseason Transactions", interactive=False)
                        tx_btn.click(
                            fn=load_transactions_ui,
                            inputs=[tx_team_sel, tx_type_sel, tx_kw_input],
                            outputs=[tx_table],
                        )

                    # Sub-tab: Historical Champions
                    with gr.TabItem("🏅 Historical NFC Champions (Validation Ground Truth)"):
                        gr.Markdown(
                            "> **Note**: This historical validation dataset (`data/validation/nfc_champions_2006_2025.csv`) is strictly isolated from the predictor subagent during runtime to prevent data leakage."
                        )
                        champs_rows = get_historical_champions()
                        gr.Dataframe(value=pd.DataFrame(champs_rows), interactive=False)

            # ----------------------------------------------------
            # TAB 4: FAISS Vector RAG Search
            # ----------------------------------------------------
            with gr.TabItem("🔍 RAG Search", id="tab_faiss"):
                gr.Markdown(
                    """
                    ### 🧠 Semantic Search over Unstructured Financial Narratives
                    Queries the local FAISS index (`sentence-transformers/all-MiniLM-L6-v2`) built over Sportico franchise valuation profiles,
                    ownership changes, and financial structures.
                    """
                )
                with gr.Row():
                    faiss_query = gr.Textbox(
                        lines=2,
                        value="Seattle Seahawks ownership sale and valuation record",
                        label="Semantic Query",
                        placeholder="e.g. Green Bay non-profit ownership structure, highest franchise valuation...",
                    )
                    faiss_k = gr.Slider(minimum=1, maximum=10, value=4, step=1, label="Top-K Results")
                    faiss_btn = gr.Button("🔎 Query FAISS Index", variant="primary")

                faiss_results_box = gr.Markdown("### Search Results\n*Click 'Query FAISS Index' to search.*")
                faiss_btn.click(
                    fn=load_faiss_search_ui,
                    inputs=[faiss_query, faiss_k],
                    outputs=[faiss_results_box],
                )

            # ----------------------------------------------------
            # TAB 5: Audit Traces & Guardrail Inspector
            # ----------------------------------------------------
            with gr.TabItem("🛡️ Audit", id="tab_audit"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🔒 Data Integrity Verification")
                        integrity_btn = gr.Button("🔍 Verify SHA-256 Manifest Checksums", variant="secondary")
                        regen_manifest_confirm = gr.Checkbox(
                            label="I intentionally re-fetched or changed the source data and want to reset the tamper-detection baseline",
                            value=False,
                        )
                        regen_manifest_btn = gr.Button("⚡ (Re)Generate Baseline Manifest", variant="stop")
                        integrity_output = gr.Markdown("Click 'Verify' to test data files against `.checksums.json`.")

                        integrity_btn.click(fn=run_integrity_ui, outputs=[integrity_output])
                        regen_manifest_btn.click(
                            fn=regenerate_manifest_ui,
                            inputs=[regen_manifest_confirm],
                            outputs=[integrity_output],
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### 📂 Saved Prediction Reports Archive")
                        reports_refresh_btn = gr.Button("🔄 Refresh Saved Reports List", variant="secondary")
                        reports_dropdown = gr.Dropdown(choices=[], label="Select Prediction Report")
                        reports_status = gr.Markdown("Click refresh to scan `outputs/predictions/`.")

                        reports_refresh_btn.click(
                            fn=load_reports_archive_ui,
                            outputs=[reports_dropdown, reports_status],
                        )

                with gr.Row():
                    with gr.Column(scale=2):
                        report_content_view = gr.Markdown("### Report Detail\n*Select a report above to view.*")
                    with gr.Column(scale=1):
                        report_chart_view = gr.Plot(label="Report Odds Visualization")

                reports_dropdown.change(
                    fn=view_report_detail_ui,
                    inputs=[reports_dropdown],
                    outputs=[report_content_view, report_chart_view],
                )

            # ----------------------------------------------------
            # TAB 6: Memory
            # ----------------------------------------------------
            with gr.TabItem("🧠 Memory", id="tab_memory"):
                gr.Markdown(
                    """
                    ### 🧠 Memory Design
                    Champion Predictor AI deliberately has **no long-term, cross-run memory**.
                    Per the checkpoint 2.1 design, every prediction run reasons independently
                    from current data — never from what an earlier run concluded — so that past
                    predictions can't bias later ones.

                    The only memory that exists is the Predictor's **short-term session
                    context**, scoped to a single run: its own ReAct thought/action/observation
                    trace, captured by the `SubagentStop` hook (or written directly by the local
                    simulator) to `outputs/.trace/predictor_latest.json`, and fed back to the
                    `did` guardrail subagent as grounding evidence for its during- and
                    post-generation checks. It is **overwritten on every run**, never accumulated.
                    """
                )
                memory_refresh_btn = gr.Button("🔄 Load Latest Session Trace", variant="secondary")
                memory_trace_view = gr.Markdown(
                    "Click above to load `outputs/.trace/predictor_latest.json`, if a run has produced one."
                )
                memory_refresh_btn.click(fn=load_latest_trace_ui, outputs=[memory_trace_view])

        # Footer
        gr.HTML(
            """
            <div style="text-align: center; color: #64748b; font-size: 13px; margin-top: 30px; padding: 15px; border-top: 1px solid #334155;">
                Champion Predictor AI • NFC Championship Forecasting &amp; What-If Decision Support • Multi-Agent ReAct Architecture with DiD Guardrails
            </div>
            """
        )

    return demo


# Forces the page into Gradio's light theme regardless of the OS/browser
# color-scheme preference, by redirecting to add ?__theme=light on first
# load. Without this, a system in dark mode pulls in the Soft theme's
# default dark tokens (which were never overridden below) instead of this
# app's palette, breaking contrast for anything not covered by CUSTOM_CSS.
FORCE_LIGHT_THEME_HEAD = """
<script>
(function () {
    var params = new URLSearchParams(window.location.search);
    if (params.get("__theme") !== "light") {
        params.set("__theme", "light");
        window.location.replace(
            window.location.pathname + "?" + params.toString() + window.location.hash
        );
    }
})();
</script>
"""

if __name__ == "__main__":
    demo = build_app()
    # Only the LIGHT-mode token slots are overridden here (deliberately no
    # "_dark" counterparts) — the previous version pinned both slots to a
    # dark-navy palette, which broke contrast once ?__theme=light was forced
    # (light mode itself was then defined as dark navy). The header banner,
    # metric cards, and audit badges in CUSTOM_CSS stay intentionally
    # dark-accented; they carry their own explicit (light) text colors, so
    # they stay readable against the light page background.
    #
    # Body font size is bumped by 1px over the Soft theme's default text_md
    # (14px); backgrounds/text below are pushed a shade deeper than the
    # stock Soft theme's very pale slate so panels read as distinct blocks
    # rather than blending into a washed-out white page.
    body_text_size = sizes.Size(
        name="text_md_plus1",
        xxs="12px", xs="13px", sm="15px", md="17px", lg="19px", xl="25px", xxl="29px",
    )
    theme = gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="blue",
        neutral_hue="slate",
        text_size=body_text_size,
    ).set(
        background_fill_primary="#dce4ee",
        body_background_fill="#dce4ee",
        block_background_fill="#eef2f7",
        block_border_color="#94a3b8",
        panel_background_fill="#e4eaf1",
        input_background_fill="#f1f5f9",
        body_text_color="#0f172a",
        block_label_text_color="#1e293b",
        block_title_text_color="#0f172a",
    )
    demo.queue()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=theme,
        css=CUSTOM_CSS,
        head=FORCE_LIGHT_THEME_HEAD,
        share=False,
    )
