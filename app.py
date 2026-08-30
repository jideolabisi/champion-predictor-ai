"""Champion Predictor AI — Interactive Gradio UI.

Provides an interactive dashboard for:
- NFC Championship win-probability predictions & visualizations
- What-If scenario simulation & decision support
- Conference structured data explorer (stats, rosters, transactions)
- FAISS vector RAG semantic search over unstructured narratives
- Audit traces, Defense-in-Depth guardrail inspector, and saved reports
"""

from __future__ import annotations

import functools
import json
import traceback
import pandas as pd
import gradio as gr

from ui.constants import NFC_TEAMS, TEAM_CHOICES, TEAM_METADATA, DIVISIONS
from ui.data_service import (
    check_integrity,
    generate_checksum_manifest,
    get_available_seasons,
    get_historical_champions,
    get_team_roster,
    get_team_stats,
    get_transactions,
    search_unstructured_financial,
)
from ui.charts import (
    create_division_breakdown_chart,
    create_probability_chart,
    create_stats_trend_chart,
    create_whatif_comparison_chart,
)
from ui.prediction_service import (
    compute_baseline_probabilities,
    evaluate_whatif_scenario,
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
    font-size: 26px;
    font-weight: 700;
    color: #38bdf8;
}
.metric-label {
    font-size: 13px;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
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


@_handle_errors
def run_prediction_ui(season_val: int):
    """Execute prediction pipeline and return updated UI components."""
    result = execute_prediction_pipeline(season=int(season_val), mode="predict")
    probs = result["prediction"]["probabilities"]
    sorted_probs = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)

    # Leaderboard table
    table_rows = []
    for idx, (t, p) in enumerate(sorted_probs, 1):
        meta = TEAM_METADATA.get(t, {})
        table_rows.append([
            idx,
            t,
            meta.get("name", t),
            meta.get("division", ""),
            f"{p:.1f}%",
        ])
    df_leaderboard = pd.DataFrame(
        table_rows,
        columns=["Rank", "Code", "Team Name", "Division", "Win Probability"],
    )

    # Charts
    prob_chart = create_probability_chart(probs, f"{season_val} NFC Championship Odds")
    div_chart = create_division_breakdown_chart(probs)

    # Summary metrics
    top1 = f"{sorted_probs[0][0]} ({sorted_probs[0][1]:.1f}%)"
    top2 = f"{sorted_probs[1][0]} ({sorted_probs[1][1]:.1f}%)"
    top3 = f"{sorted_probs[2][0]} ({sorted_probs[2][1]:.1f}%)"

    conf = f"{result['prediction']['confidence']}%"
    critic_rating = f"{result['critic']['rating']}"
    val_status = f"{result['validation']['verdict'].upper()}"

    explanation_md = f"### 📝 Agent Reasoning & Narrative\n{result['prediction']['explanation']}"
    audit_md = (
        f"### 🛡️ Guardrails & Verification (Checkpoint 6.1)\n"
        f"- **Deterministic Validation**: `{result['validation']['verdict'].upper()}` (Sum: `{result['validation']['probability_sum_check']['total']:.1f}%`)\n"
        f"- **Consensus Variance**: `{result['validation']['consensus_variance_check']['overlap']}/3` overlap with Top-3 consensus\n"
        f"- **DiD Pre-gen Prompt**: `{result['did']['pre_generation']['prompt_score']}/100` (`{result['did']['pre_generation']['verdict']}`)\n"
        f"- **DiD During-gen Overconfidence**: `{result['did']['during_generation']['overconfidence_verdict']}` (Suspicion: `{result['did']['during_generation']['suspicion_score']}/100`)\n"
        f"- **DiD Post-gen Groundedness**: `{result['did']['post_generation']['groundedness_verdict']}`\n"
        f"- **Report Saved**: `{result['filename']}`"
    )

    return (
        prob_chart,
        div_chart,
        df_leaderboard,
        top1,
        top2,
        top3,
        conf,
        critic_rating,
        val_status,
        explanation_md,
        audit_md,
    )


@_handle_errors
def run_whatif_ui(target_team: str, scenario_text: str, season_val: int):
    """Execute what-if scenario simulation and return comparison visualizations."""
    if not scenario_text.strip():
        scenario_text = "Trade for an All-Pro Defensive Pass Rusher"

    result = execute_prediction_pipeline(
        season=int(season_val),
        mode="what_if",
        target_team=target_team,
        scenario_description=scenario_text,
    )

    base_probs = result["prediction"]["probabilities"]
    adj_probs = result["prediction"]["adjusted_probabilities"]

    chart = create_whatif_comparison_chart(base_probs, adj_probs, target_team=target_team)

    # Comparison table
    sorted_teams = sorted(base_probs.keys(), key=lambda t: base_probs[t], reverse=True)
    table_rows = []
    for idx, t in enumerate(sorted_teams, 1):
        meta = TEAM_METADATA.get(t, {})
        base_p = base_probs.get(t, 0.0)
        adj_p = adj_probs.get(t, 0.0)
        delta = adj_p - base_p
        delta_str = f"+{delta:.1f}%" if delta > 0 else (f"{delta:.1f}%" if delta < 0 else "0.0%")
        marker = "🎯 TARGET" if t == target_team else ""
        table_rows.append([
            idx,
            t,
            meta.get("name", t),
            f"{base_p:.1f}%",
            f"{adj_p:.1f}%",
            delta_str,
            marker,
        ])

    df_comp = pd.DataFrame(
        table_rows,
        columns=["Rank", "Code", "Team Name", "Baseline", "Scenario Odds", "Net Shift", "Status"],
    )

    target_base = base_probs.get(target_team, 0.0)
    target_adj = adj_probs.get(target_team, 0.0)
    target_delta = target_adj - target_base
    delta_display = f"{'+' if target_delta > 0 else ''}{target_delta:.1f}%"

    delta_md = f"### 🔄 Scenario Delta Analysis\n{result['prediction']['delta_explanation']}"
    report_file_info = f"💾 Report written to: `outputs/predictions/{result['filename']}`"

    return (
        chart,
        df_comp,
        f"{target_base:.1f}%",
        f"{target_adj:.1f}%",
        delta_display,
        delta_md,
        report_file_info,
    )


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
            "Not started. Check the confirmation box above to launch a real "
            "Claude Code agent run (this consumes real usage against your "
            "Claude account, up to the budget cap below).",
            "*Awaiting confirmation.*",
            gr.update(),
        )
        return

    mode = "what_if" if mode_choice == "What-If" else "predict"
    log_lines = ["Starting real predictor run — this can take several minutes...\n"]
    yield "\n".join(log_lines), "*Running...*", gr.update()

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
                yield "\n".join(log_lines), gr.update(), gr.update()
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
                if adj_probs:
                    chart = create_whatif_comparison_chart(probs, adj_probs, target_team=target_team)
                elif probs:
                    chart = create_probability_chart(probs, f"Real Agentic Prediction — {season_val}")
                else:
                    chart = gr.update()
                log_lines.append(f"\nDone. Report saved: {event['report_stem']}{cost_note}")
                yield "\n".join(log_lines), md, chart
            else:
                status = "completed without writing a report" if event["ok"] else "failed"
                explanation = event.get("error") or "No further detail was returned."
                log_lines.append(f"\nRun {status}{cost_note}.")
                yield "\n".join(log_lines), f"### Run {status}\n\n{explanation}", gr.update()
    except Exception as exc:
        traceback.print_exc()
        log_lines.append(f"\nError: {exc}")
        yield "\n".join(log_lines), "", gr.update()
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
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                        <div>
                            <h1 style="margin: 0; font-size: 28px; font-weight: 800; color: #ffffff;">🏈 Champion Predictor AI</h1>
                            <p style="margin: 6px 0 0 0; font-size: 15px; color: #94a3b8;">
                                NFC Championship Win Probability Engine &amp; What-If Decision Support System
                            </p>
                        </div>
                        <div style="text-align: right; margin-top: 10px;">
                            <span class="audit-badge-pass">🛡️ DiD Guardrails Active</span>
                            <span class="audit-badge-pass" style="background:#1e3a8a; color:#93c5fd; margin-left: 6px;">🧠 ReAct Multi-Agent</span>
                        </div>
                    </div>
                </div>
                """
            )

        with gr.Tabs() as main_tabs:
            # ----------------------------------------------------
            # TAB 1: Live Predictions & Rankings
            # ----------------------------------------------------
            with gr.TabItem("🔮 Live Predictions & Rankings", id="tab_predictions"):
                with gr.Row():
                    with gr.Column(scale=1):
                        season_input = gr.Dropdown(
                            choices=SEASON_CHOICES,
                            value=_CURRENT_SEASON,
                            label="Target NFC Season",
                            info=(
                                "Historical stats are restricted to seasons before the one "
                                "selected. Roster/transaction/financial inputs always reflect "
                                f"the current {_CURRENT_SEASON} snapshot — no historical roster "
                                "data exists for earlier seasons."
                            ),
                        )
                        predict_btn = gr.Button("⚡ Generate Championship Predictions", variant="primary", size="lg")

                        gr.Markdown("### 🏆 Top Contenders")
                        with gr.Row():
                            with gr.Column(elem_classes=["metric-card"]):
                                gr.HTML("<div class='metric-label'>#1 Favorite</div>")
                                m_top1 = gr.Markdown("**—**", elem_classes=["metric-val"])
                            with gr.Column(elem_classes=["metric-card"]):
                                gr.HTML("<div class='metric-label'>#2 Contender</div>")
                                m_top2 = gr.Markdown("**—**", elem_classes=["metric-val"])
                            with gr.Column(elem_classes=["metric-card"]):
                                gr.HTML("<div class='metric-label'>#3 Contender</div>")
                                m_top3 = gr.Markdown("**—**", elem_classes=["metric-val"])

                        gr.Markdown("### 🛡️ Guardrail Status")
                        with gr.Row():
                            with gr.Column(elem_classes=["metric-card"]):
                                gr.HTML("<div class='metric-label'>Confidence</div>")
                                m_conf = gr.Markdown("**—**", elem_classes=["metric-val"])
                            with gr.Column(elem_classes=["metric-card"]):
                                gr.HTML("<div class='metric-label'>Critic Rating</div>")
                                m_critic = gr.Markdown("**—**", elem_classes=["metric-val"])
                            with gr.Column(elem_classes=["metric-card"]):
                                gr.HTML("<div class='metric-label'>Validator</div>")
                                m_val = gr.Markdown("**—**", elem_classes=["metric-val"])

                    with gr.Column(scale=2):
                        pred_chart = gr.Plot(label="Championship Win Probability Distribution")

                with gr.Row():
                    with gr.Column(scale=1):
                        div_sunburst = gr.Plot(label="Probability Share by NFC Division")
                    with gr.Column(scale=1):
                        leaderboard_table = gr.Dataframe(
                            headers=["Rank", "Code", "Team Name", "Division", "Win Probability"],
                            label="Conference Leaderboard",
                            interactive=False,
                        )

                with gr.Row():
                    with gr.Column(scale=1):
                        narrative_box = gr.Markdown("### 📝 Agent Reasoning & Narrative\n*Click 'Generate Championship Predictions' to run.*")
                    with gr.Column(scale=1):
                        audit_box = gr.Markdown("### 🛡️ Guardrails & Verification\n*Awaiting execution.*")

                predict_btn.click(
                    fn=run_prediction_ui,
                    inputs=[season_input],
                    outputs=[
                        pred_chart,
                        div_sunburst,
                        leaderboard_table,
                        m_top1,
                        m_top2,
                        m_top3,
                        m_conf,
                        m_critic,
                        m_val,
                        narrative_box,
                        audit_box,
                    ],
                )

            # ----------------------------------------------------
            # TAB 2: What-If Decision Support
            # ----------------------------------------------------
            with gr.TabItem("⚡ What-If Decision Support", id="tab_whatif"):
                gr.Markdown(
                    """
                    ### 🎯 Scenario Impact Simulator
                    Evaluate how a user-named hypothetical change (player trade, free agency signing, key injury, coaching hire)
                    shifts a team's NFC Championship probability and impacts the rest of the conference.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        whatif_team = gr.Dropdown(
                            choices=[(f"{c} — {TEAM_METADATA[c]['name']}", c) for c in NFC_TEAMS],
                            value="PHI",
                            label="Target NFC Franchise",
                        )
                        whatif_preset = gr.Dropdown(
                            choices=[
                                "Trade for an All-Pro Defensive Pass Rusher",
                                "Starting Quarterback Suffers Season-Ending Knee Injury",
                                "Sign Elite Pro-Bowl Wide Receiver in Free Agency",
                                "Hire High-Powered Super Bowl-Winning Offensive Coordinator",
                                "Lose Multiple Starting Offensive Linemen to Injury",
                            ],
                            value="Trade for an All-Pro Defensive Pass Rusher",
                            label="Quick Presets",
                        )
                        whatif_scenario_input = gr.Textbox(
                            lines=3,
                            value="Trade for an All-Pro Defensive Pass Rusher to anchor the defensive front.",
                            label="Hypothetical Scenario Description",
                            placeholder="Describe the trade, signing, coaching change, or injury in detail...",
                        )
                        whatif_preset.change(
                            fn=lambda p: p,
                            inputs=[whatif_preset],
                            outputs=[whatif_scenario_input],
                        )
                        whatif_season = gr.Dropdown(
                            choices=SEASON_CHOICES,
                            value=_CURRENT_SEASON,
                            label="Season Context",
                            info=(
                                "Roster/transaction/financial inputs always reflect the current "
                                f"{_CURRENT_SEASON} snapshot regardless of this selection."
                            ),
                        )
                        whatif_btn = gr.Button("⚡ Simulate What-If Impact", variant="primary", size="lg")

                        with gr.Row():
                            with gr.Column(elem_classes=["metric-card"]):
                                gr.HTML("<div class='metric-label'>Baseline Odds</div>")
                                w_base = gr.Markdown("**—**", elem_classes=["metric-val"])
                            with gr.Column(elem_classes=["metric-card"]):
                                gr.HTML("<div class='metric-label'>Scenario Odds</div>")
                                w_adj = gr.Markdown("**—**", elem_classes=["metric-val"])
                            with gr.Column(elem_classes=["metric-card"]):
                                gr.HTML("<div class='metric-label'>Net Shift</div>")
                                w_delta = gr.Markdown("**—**", elem_classes=["metric-val"])

                    with gr.Column(scale=2):
                        whatif_chart = gr.Plot(label="Baseline vs. Adjusted Scenario Comparison")

                with gr.Row():
                    with gr.Column(scale=1):
                        whatif_delta_box = gr.Markdown("### 🔄 Scenario Delta Analysis\n*Run a scenario to inspect impact.*")
                        whatif_file_box = gr.Markdown("")
                    with gr.Column(scale=1):
                        whatif_table = gr.Dataframe(
                            headers=["Rank", "Code", "Team Name", "Baseline", "Scenario Odds", "Net Shift", "Status"],
                            label="All 16 Teams Adjusted Odds Leaderboard",
                            interactive=False,
                        )

                whatif_btn.click(
                    fn=run_whatif_ui,
                    inputs=[whatif_team, whatif_scenario_input, whatif_season],
                    outputs=[
                        whatif_chart,
                        whatif_table,
                        w_base,
                        w_adj,
                        w_delta,
                        whatif_delta_box,
                        whatif_file_box,
                    ],
                )

            # ----------------------------------------------------
            # TAB 2.5: Real Agentic Predictor (Claude Code CLI)
            # ----------------------------------------------------
            with gr.TabItem("🤖 Real Agentic Predictor", id="tab_real_agent"):
                gr.Markdown(
                    f"""
                    ### 🤖 Run the Real ReAct Predictor Pipeline
                    Everything above uses a fast local heuristic over the CSVs. **This tab launches
                    the actual `champion-predictor` skill in Claude Code** — the `predictor` subagent
                    doing real ReAct tool-calling reasoning over the MCP data tools (and WebSearch),
                    reviewed by the `critic` subagent, checked by the three-stage `did` guardrail, and
                    validated deterministically — exactly as if you'd typed `/champion-predictor` yourself.

                    ⚠️ **This is a real, multi-minute agent run that consumes usage on your Claude
                    account**, not a free local computation. It's hard-capped by the budget below
                    (`--max-budget-usd`), but a run can still take several minutes and multiple
                    subagent invocations. A human-in-the-loop pause the skill would normally ask you
                    about (e.g. a data-integrity mismatch, or a high suspicion score) cannot be
                    answered headlessly — if that happens, the run stops and shows you why instead of
                    guessing on your behalf.
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
                        agent_confirm = gr.Checkbox(
                            label="I understand this launches a real Claude Code agent run using my account's usage, up to the budget cap above",
                            value=False,
                        )
                        agent_run_btn = gr.Button("▶ Run Real Predictor", variant="stop", size="lg")

                    with gr.Column(scale=2):
                        agent_log = gr.Textbox(
                            label="Live Agent Log",
                            lines=20,
                            max_lines=20,
                            interactive=False,
                            autoscroll=True,
                        )

                with gr.Row():
                    with gr.Column(scale=1):
                        agent_report_md = gr.Markdown("*Run the predictor to see the report here.*")
                    with gr.Column(scale=1):
                        agent_chart = gr.Plot(label="Probability Outcome")

                agent_run_btn.click(
                    fn=run_real_predictor_ui,
                    inputs=[agent_mode, agent_season, agent_team, agent_scenario, agent_budget, agent_confirm],
                    outputs=[agent_log, agent_report_md, agent_chart],
                )

            # ----------------------------------------------------
            # TAB 3: Data Explorer
            # ----------------------------------------------------
            with gr.TabItem("📊 Conference Data Explorer", id="tab_data"):
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
            with gr.TabItem("🔍 FAISS Vector RAG Search", id="tab_faiss"):
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
            with gr.TabItem("🛡️ Audit Traces & Integrity", id="tab_audit"):
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

        # Footer
        gr.HTML(
            """
            <div style="text-align: center; color: #64748b; font-size: 13px; margin-top: 30px; padding: 15px; border-top: 1px solid #334155;">
                Champion Predictor AI • NFC Championship Forecasting &amp; What-If Decision Support • Multi-Agent ReAct Architecture with DiD Guardrails
            </div>
            """
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    theme = gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="blue",
        neutral_hue="slate",
    ).set(
        body_background_fill="#0b1120",
        body_text_color="#f1f5f9",
        block_background_fill="#1e293b",
        block_border_color="#334155",
        input_background_fill="#0f172a",
    )
    demo.queue()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=theme,
        css=CUSTOM_CSS,
        share=False,
    )
