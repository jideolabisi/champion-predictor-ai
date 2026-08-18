"""CSV/JSON loaders shared by the MCP server tools.

All functions here do exact filtering/lookup over local files — no embeddings,
no chunking. This mirrors the checkpoint 3.1 decision that structured records
(team stats, rosters, transactions, coaching/management changes, injuries)
should be served as precise tool results, not split into semantic-search chunks.
"""

from __future__ import annotations

import csv
import os
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"

# Canonical NFC team codes, matching the team-stats source's convention.
NFC_TEAMS = [
    "ARI", "ATL", "CAR", "CHI", "DAL", "DET", "GB", "LA",
    "MIN", "NO", "NYG", "PHI", "SEA", "SF", "TB", "WAS",
]

# Maps alternate abbreviations seen across sources onto the canonical code above.
_TEAM_ALIASES = {
    "LAR": "LA", "STL": "LA",
    "WSH": "WAS", "WFT": "WAS",
}


def normalize_team(team: str | None) -> str | None:
    if team is None:
        return None
    code = team.strip().upper()
    return _TEAM_ALIASES.get(code, code)


# Full franchise name -> canonical code, for reconciling the historical
# validation dataset (which uses full names) against predictor output
# (which uses codes). Covers name variants seen 2006-2025 (e.g. franchise
# relocations). Used only by scripts/backtest.py.
TEAM_NAME_TO_CODE = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Dallas Cowboys": "DAL",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Los Angeles Rams": "LA",
    "St. Louis Rams": "LA",
    "Minnesota Vikings": "MIN",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "Philadelphia Eagles": "PHI",
    "Seattle Seahawks": "SEA",
    "San Francisco 49ers": "SF",
    "Tampa Bay Buccaneers": "TB",
    "Washington Commanders": "WAS",
    "Washington Football Team": "WAS",
    "Washington Redskins": "WAS",
}


def team_name_to_code(name: str) -> str | None:
    return TEAM_NAME_TO_CODE.get(name.strip())


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@lru_cache(maxsize=None)
def _cached_csv(path_str: str) -> tuple[dict, ...]:
    """Cached, read-only load of a CSV as a tuple of dict rows."""
    return tuple(_read_csv(Path(path_str)))


def load_csv(filename: str) -> list[dict]:
    """Load a CSV from data/raw/<filename>, cached for the life of the process."""
    path = RAW_DIR / filename
    if not path.exists():
        return []
    return list(_cached_csv(str(path)))


def filter_rows(rows: list[dict], **filters) -> list[dict]:
    """Exact-match filter; None values in `filters` are ignored (no constraint)."""
    out = rows
    for key, value in filters.items():
        if value is None:
            continue
        out = [r for r in out if r.get(key) == value]
    return out


def data_source_available(filename: str) -> bool:
    return (RAW_DIR / filename).exists()
