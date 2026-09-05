#!/usr/bin/env python3
"""Champion Predictor AI — predictor-facing MCP server.

Exposes exact-lookup tools over local structured data (team stats, roster,
transactions, coaching/management changes, injuries) plus a semantic-search
tool over genuinely unstructured text sources (financial narratives, etc.).

IMPORTANT (validation isolation): this file never imports or opens anything
under data/validation/. The historical-champions ground truth there is read
only by scripts/backtest.py, which is not an MCP tool and is never reachable
through any tool call the predictor or critic subagent can make. Do not add
a tool here that touches data/validation/ — that would break the
anti-leakage guarantee the capstone design requires.
"""

import os
from typing import Optional

from mcp.server.fastmcp import FastMCP

from lib.loaders import NFC_TEAMS, filter_rows, load_csv, normalize_team
from lib.retrieval import search as faiss_search

mcp = FastMCP("champion-data")

# Backtest mode: set by scripts/backtest.py so current-season-only sources
# (roster, transactions, coaching, management, injuries) don't silently leak
# present-day data into a historical simulation. Team-season stats is the
# only source with real multi-year depth, so it alone honors AS_OF_SEASON.
_AS_OF_SEASON = os.environ.get("CHAMPION_PREDICTOR_AS_OF_SEASON")
_BACKTEST_MODE = os.environ.get("CHAMPION_PREDICTOR_BACKTEST_MODE") == "1"

_NOT_AVAILABLE_IN_BACKTEST = (
    "Not available for historical evaluation: this source only has "
    "current-season snapshots, no historical depth. See README for the "
    "documented backtest limitation."
)


@mcp.tool()
def list_nfc_teams() -> list[str]:
    """Return the 16 NFC team codes this agent predicts over."""
    return list(NFC_TEAMS)


@mcp.tool()
def get_team_seasonal_stats(team: str, seasons: Optional[list[int]] = None) -> list[dict]:
    """Get a team's regular-season statistical summaries for the given seasons
    (or all available seasons if omitted). team is a code like 'PHI' or 'SF'.
    """
    rows = load_csv("team_stats_2006_2025.csv")
    team_code = normalize_team(team)
    rows = [r for r in rows if normalize_team(r.get("Team")) == team_code]

    if _AS_OF_SEASON is not None:
        cutoff = int(_AS_OF_SEASON)
        rows = [r for r in rows if int(r["Season"]) < cutoff]
    elif seasons:
        wanted = {int(s) for s in seasons}
        rows = [r for r in rows if int(r["Season"]) in wanted]

    return rows


# roster_2026.csv carries ~36 columns per player, most of them cross-platform
# ID/URL fields (espn_id, sportradar_id, yahoo_id, pff_id, headshot_url, ...)
# with no predictive value. Returning them all pushed a single team's ~90
# players (e.g. GB) past the tool output token limit, silently redirecting
# the result to a file the predictor can't read back — wasting the call
# entirely. These are the fields actually useful for assessing roster
# strength; trimming to them keeps every team's roster comfortably under
# the limit without losing anything the predictor uses.
_ROSTER_FIELDS = (
    "team", "jersey_number", "player_name", "position",
    "depth_chart_position", "years_exp", "age", "college", "status",
)


@mcp.tool()
def get_current_roster(team: str, position: Optional[str] = None) -> list[dict] | str:
    """Get the current roster for an NFC team (code like 'PHI' or 'SF'),
    trimmed to the fields relevant for assessing roster strength (name,
    position, depth chart slot, experience, age, college, status). Optionally
    filter to one position group (e.g. 'QB', 'WR', 'OL') to narrow further.
    """
    if _BACKTEST_MODE:
        return _NOT_AVAILABLE_IN_BACKTEST
    rows = load_csv("roster_2026.csv")
    team_code = normalize_team(team)
    rows = [r for r in rows if normalize_team(r.get("team")) == team_code]
    if position:
        rows = filter_rows(rows, position=position.strip().upper())
    return [{k: r.get(k) for k in _ROSTER_FIELDS} for r in rows]


@mcp.tool()
def get_transactions(
    team: Optional[str] = None,
    since_date: Optional[str] = None,
    transaction_type: Optional[str] = None,
) -> list[dict] | str:
    """Get current-season player trades/transactions, optionally filtered by
    team code, a since_date (YYYY-MM-DD, inclusive), and/or transaction_type
    (e.g. 'SIGNING', 'WAIVE_RELEASE', 'TRADE').
    """
    if _BACKTEST_MODE:
        return _NOT_AVAILABLE_IN_BACKTEST
    rows = load_csv("transactions_2026.csv")
    team_code = normalize_team(team) if team else None
    if team_code:
        rows = [r for r in rows if normalize_team(r.get("TEAM")) == team_code]
    if transaction_type:
        rows = filter_rows(rows, TRANSACTION_TYPE=transaction_type)
    if since_date:
        rows = [r for r in rows if r.get("SNAPSHOT_DATE", "") >= since_date]
    return rows


@mcp.tool()
def get_coaching_changes(team: Optional[str] = None) -> list[dict] | str:
    """Get current-season coaching hires/departures/role changes, optionally
    filtered by team code.
    """
    if _BACKTEST_MODE:
        return _NOT_AVAILABLE_IN_BACKTEST
    rows = load_csv("coaching_changes_2026.csv")
    if not rows:
        return "No coaching-changes data source available yet."
    team_code = normalize_team(team) if team else None
    return [r for r in rows if not team_code or normalize_team(r.get("team")) == team_code]


@mcp.tool()
def get_management_changes(team: Optional[str] = None) -> list[dict] | str:
    """Get current-year executive/front-office changes, optionally filtered
    by team code.
    """
    if _BACKTEST_MODE:
        return _NOT_AVAILABLE_IN_BACKTEST
    rows = load_csv("management_changes_2026.csv")
    if not rows:
        return "No management-changes data source available yet."
    team_code = normalize_team(team) if team else None
    return [r for r in rows if not team_code or normalize_team(r.get("team")) == team_code]


@mcp.tool()
def get_injury_report(team: str) -> list[dict] | str:
    """Get the current-season injury report for an NFC team (code like 'PHI')."""
    if _BACKTEST_MODE:
        return _NOT_AVAILABLE_IN_BACKTEST
    rows = load_csv("injury_reports_2026.csv")
    if not rows:
        return "No injury-report data source available yet."
    team_code = normalize_team(team)
    return [r for r in rows if normalize_team(r.get("team")) == team_code]


@mcp.tool()
def search_unstructured(query: str, source: Optional[str] = None, top_k: int = 5) -> list[dict]:
    """Semantic search over unstructured text sources (e.g. financial
    narratives) that don't have a clean structured record format. `source`
    optionally restricts to one category (e.g. 'financial'). Returns the
    top_k most relevant text chunks with their originating file.
    """
    return faiss_search(query, source=source, top_k=top_k)


if __name__ == "__main__":
    mcp.run()
