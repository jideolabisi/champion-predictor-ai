"""Plotly chart builders for Champion Predictor AI UI."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ui.constants import DIVISIONS, TEAM_METADATA


def create_probability_chart(probabilities: dict[str, float], title: str = "NFC Championship Win Probabilities") -> go.Figure:
    """Create a ranked bar chart of win probabilities with official team colors."""
    if not probabilities:
        fig = go.Figure()
        fig.update_layout(title="No probability data available", template="plotly_dark")
        return fig

    # Sort descending
    sorted_items = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    teams = [item[0] for item in sorted_items]
    probs = [round(float(item[1]), 2) for item in sorted_items]
    team_names = [f"{t} ({TEAM_METADATA.get(t, {}).get('name', t)})" for t in teams]
    colors = [TEAM_METADATA.get(t, {}).get("primary_color", "#004C54") for t in teams]
    text_labels = [f"{p:.1f}%" for p in probs]

    fig = go.Figure(
        go.Bar(
            x=probs,
            y=team_names,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color="#FFFFFF", width=1),
            ),
            text=text_labels,
            textposition="outside",
            hoverinfo="text",
            hovertext=[f"<b>{TEAM_METADATA.get(t, {}).get('name', t)}</b><br>Code: {t}<br>Win Probability: {p:.2f}%<br>Division: {TEAM_METADATA.get(t, {}).get('division', '')}" for t, p in zip(teams, probs)],
        )
    )

    # Invert y axis so highest probability is at top
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=18, color="#FFFFFF")),
        xaxis=dict(
            title="Win Probability (%)",
            range=[0, max(probs + [10]) * 1.15],
            gridcolor="#2D3748",
            zerolinecolor="#4A5568",
        ),
        yaxis=dict(autorange="reversed", gridcolor="#2D3748"),
        plot_bgcolor="#1A202C",
        paper_bgcolor="#111827",
        font=dict(color="#E2E8F0"),
        margin=dict(l=20, r=30, t=50, b=40),
        height=550,
    )
    return fig


def create_whatif_comparison_chart(
    baseline_probs: dict[str, float],
    adjusted_probs: dict[str, float],
    target_team: str | None = None,
) -> go.Figure:
    """Create a side-by-side grouped bar chart comparing Baseline vs Adjusted probabilities."""
    if not baseline_probs or not adjusted_probs:
        fig = go.Figure()
        fig.update_layout(title="No comparison data available", template="plotly_dark")
        return fig

    # Sort by baseline descending
    sorted_teams = sorted(baseline_probs.keys(), key=lambda t: baseline_probs.get(t, 0.0), reverse=True)
    
    baseline_vals = [round(float(baseline_probs.get(t, 0.0)), 2) for t in sorted_teams]
    adjusted_vals = [round(float(adjusted_probs.get(t, 0.0)), 2) for t in sorted_teams]
    deltas = [round(adj - base, 2) for base, adj in zip(baseline_vals, adjusted_vals)]
    team_labels = [f"{t} ({TEAM_METADATA.get(t, {}).get('name', t).split()[-1]})" for t in sorted_teams]

    fig = go.Figure()

    # Baseline trace
    fig.add_trace(
        go.Bar(
            name="Baseline Odds (%)",
            x=team_labels,
            y=baseline_vals,
            marker_color="#4B5563",
            text=[f"{v:.1f}%" for v in baseline_vals],
            textposition="auto",
        )
    )

    # Adjusted trace
    adjusted_colors = []
    for t in sorted_teams:
        if target_team and t == target_team:
            adjusted_colors.append("#10B981")  # Highlight target team
        else:
            adjusted_colors.append(TEAM_METADATA.get(t, {}).get("primary_color", "#3B82F6"))

    fig.add_trace(
        go.Bar(
            name="Scenario Odds (%)",
            x=team_labels,
            y=adjusted_vals,
            marker_color=adjusted_colors,
            text=[f"{adj:.1f}% ({'+' if d > 0 else ''}{d:.1f}%)" if d != 0 else f"{adj:.1f}%" for adj, d in zip(adjusted_vals, deltas)],
            textposition="outside",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b>What-If Scenario Probability Shift</b>{' (Target: ' + target_team + ')' if target_team else ''}",
            font=dict(size=18, color="#FFFFFF"),
        ),
        barmode="group",
        xaxis=dict(title="NFC Teams", gridcolor="#2D3748", tickangle=-35),
        yaxis=dict(
            title="Win Probability (%)",
            range=[0, max(baseline_vals + adjusted_vals + [10]) * 1.2],
            gridcolor="#2D3748",
        ),
        plot_bgcolor="#1A202C",
        paper_bgcolor="#111827",
        font=dict(color="#E2E8F0"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#E2E8F0"),
        ),
        margin=dict(l=20, r=20, t=70, b=70),
        height=520,
    )
    return fig


def create_division_breakdown_chart(probabilities: dict[str, float]) -> go.Figure:
    """Create a sunburst or donut chart summarizing odds by division and team."""
    if not probabilities:
        return go.Figure()

    division_data = []
    for div, teams in DIVISIONS.items():
        div_total = sum(probabilities.get(t, 0.0) for t in teams)
        for t in teams:
            division_data.append({
                "Division": div,
                "Team": t,
                "TeamName": TEAM_METADATA.get(t, {}).get("name", t),
                "Probability": probabilities.get(t, 0.0),
                "DivTotal": div_total,
            })

    df = pd.DataFrame(division_data)

    fig = px.sunburst(
        df,
        path=["Division", "Team"],
        values="Probability",
        color="Division",
        color_discrete_map={
            "NFC East": "#3B82F6",
            "NFC North": "#10B981",
            "NFC South": "#F59E0B",
            "NFC West": "#EF4444",
        },
        title="Win Probability Share by NFC Division",
    )

    fig.update_layout(
        plot_bgcolor="#1A202C",
        paper_bgcolor="#111827",
        font=dict(color="#E2E8F0"),
        margin=dict(l=10, r=10, t=50, b=10),
        height=480,
    )
    return fig


def create_stats_trend_chart(stats_rows: list[dict], metric_col: str, metric_label: str) -> go.Figure:
    """Create a multi-line historical trend chart for selected team stats over 2006-2025."""
    if not stats_rows:
        fig = go.Figure()
        fig.update_layout(title="No stats data available", template="plotly_dark")
        return fig

    df = pd.DataFrame(stats_rows)
    df["Season"] = pd.to_numeric(df["Season"], errors="coerce")
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    df = df.dropna(subset=["Season", metric_col]).sort_values("Season")

    fig = go.Figure()
    teams = df["Team"].unique()

    for team in teams:
        team_df = df[df["Team"] == team]
        team_color = TEAM_METADATA.get(team, {}).get("primary_color", "#3B82F6")
        fig.add_trace(
            go.Scatter(
                x=team_df["Season"],
                y=team_df[metric_col],
                mode="lines+markers",
                name=f"{team} ({TEAM_METADATA.get(team, {}).get('name', team)})",
                line=dict(color=team_color, width=2.5),
                marker=dict(size=6),
            )
        )

    fig.update_layout(
        title=dict(text=f"<b>Historical Trend: {metric_label} (2006–2025)</b>", font=dict(size=16, color="#FFFFFF")),
        xaxis=dict(title="Season", dtick=2, gridcolor="#2D3748"),
        yaxis=dict(title=metric_label, gridcolor="#2D3748"),
        plot_bgcolor="#1A202C",
        paper_bgcolor="#111827",
        font=dict(color="#E2E8F0"),
        legend=dict(font=dict(color="#E2E8F0")),
        margin=dict(l=20, r=20, t=50, b=40),
        height=460,
    )
    return fig
