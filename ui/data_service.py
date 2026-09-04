"""Data service for Champion Predictor AI UI.

Provides helper methods to inspect team statistics, rosters, transactions,
FAISS unstructured search, checksum verification, and historical champions.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACE_PATH = REPO_ROOT / "outputs" / ".trace" / "predictor_latest.json"
sys.path.insert(0, str(REPO_ROOT / "mcp_server"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lib.loaders import NFC_TEAMS, load_csv, normalize_team, filter_rows  # noqa: E402
from lib.retrieval import search as faiss_search  # noqa: E402
import verify_data_integrity  # noqa: E402
from ui.constants import TEAM_METADATA


def get_team_stats(team_code: str | None = None, season: int | None = None) -> list[dict[str, Any]]:
    """Retrieve regular-season stats from team_stats_2006_2025.csv."""
    rows = load_csv("team_stats_2006_2025.csv")
    if team_code:
        norm = normalize_team(team_code)
        rows = [r for r in rows if normalize_team(r.get("Team")) == norm]
    if season:
        rows = [r for r in rows if str(r.get("Season")) == str(season)]
    return rows


def get_team_roster(team_code: str | None = None, position: str | None = None) -> list[dict[str, Any]]:
    """Retrieve roster players for 2026."""
    rows = load_csv("roster_2026.csv")
    if team_code:
        norm = normalize_team(team_code)
        rows = [r for r in rows if normalize_team(r.get("team")) == norm]
    if position and position.strip() and position.strip() != "ALL":
        pos = position.strip().upper()
        rows = [r for r in rows if str(r.get("position", "")).upper() == pos or str(r.get("depth_chart_position", "")).upper() == pos]
    return rows


def get_transactions(
    team_code: str | None = None,
    transaction_type: str | None = None,
    search_keyword: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve 2026 transactions."""
    rows = load_csv("transactions_2026.csv")
    if team_code and team_code != "ALL":
        norm = normalize_team(team_code)
        rows = [r for r in rows if normalize_team(r.get("TEAM")) == norm]
    if transaction_type and transaction_type != "ALL":
        rows = filter_rows(rows, TRANSACTION_TYPE=transaction_type)
    if search_keyword and search_keyword.strip():
        kw = search_keyword.strip().lower()
        rows = [
            r for r in rows
            if kw in str(r.get("DESCRIPTION", "")).lower() or kw in str(r.get("PLAYER", "")).lower()
        ]
    return rows


def search_unstructured_financial(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Perform FAISS semantic search across unstructured documents."""
    return faiss_search(query, source="financial", top_k=top_k)


def get_historical_champions() -> list[dict[str, Any]]:
    """Load the historical NFC champions ground-truth dataset."""
    champions_path = REPO_ROOT / "data" / "validation" / "nfc_champions_2006_2025.csv"
    if not champions_path.exists():
        return []
    with open(champions_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_integrity() -> dict[str, Any]:
    """Verify data checksum manifest."""
    return verify_data_integrity.verify()


def generate_checksum_manifest() -> dict[str, Any]:
    """Generate or update the SHA-256 baseline manifest."""
    return verify_data_integrity.generate()


def get_latest_session_trace() -> dict[str, Any] | None:
    """Load the Predictor's most recent short-term session trace.

    This is the only "memory" the system carries — the ReAct
    thought/action/observation trace for a single run, written by the
    `SubagentStop` hook (or the local simulator) to
    outputs/.trace/predictor_latest.json and overwritten on every run,
    never accumulated across runs.
    """
    if not TRACE_PATH.exists():
        return None
    with open(TRACE_PATH, encoding="utf-8") as f:
        steps = json.load(f)
    mtime = TRACE_PATH.stat().st_mtime
    return {"steps": steps, "mtime": mtime}


def get_available_seasons() -> list[int]:
    """Get list of seasons available in team stats dataset."""
    stats = load_csv("team_stats_2006_2025.csv")
    seasons = sorted({int(r["Season"]) for r in stats if r.get("Season")}, reverse=True)
    return seasons if seasons else list(range(2025, 2005, -1))
