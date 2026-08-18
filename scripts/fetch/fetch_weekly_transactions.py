"""Build a weekly-updated 2026 NFC transaction ledger from nflverse roster changes.

run command: uv run --with pandas --with requests fetch_weekly_transactions.py
"""

import os
from datetime import datetime

import pandas as pd
import requests

OUTPUT_FILE = "nfc_weekly_transactions_2026.csv"
ARCHIVE_FILE = "nfc_weekly_transactions_2026_full_history.csv"
LATEST_WEEKS_TO_KEEP = int(os.getenv("LATEST_WEEKS_TO_KEEP", "18"))
TARGET_SEASON = 2026
PAGE_SIZE = 200
ESPN_TRANSACTIONS_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/transactions"
OUTPUT_COLUMNS = [
    "SEASON",
    "SNAPSHOT_DATE",
    "WEEK",
    "TEAM",
    "TEAM_NAME",
    "PLAYER_ID",
    "PLAYER",
    "POSITION",
    "PREV_TEAM",
    "TRANSACTION_TYPE",
    "DESCRIPTION",
    "SOURCE",
]

NFC_MAPPING = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "DAL": "Dallas Cowboys",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "LAR": "Los Angeles Rams",
    "MIN": "Minnesota Vikings",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "PHI": "Philadelphia Eagles",
    "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers",
    "WAS": "Washington Commanders",
}

NFC_TEAMS = set(NFC_MAPPING.keys())


def _parse_week(date_str: str) -> int:
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return int(dt.isocalendar().week)
    except Exception:
        return 0


def fetch_espn_transactions(year: int):
    rows = []
    page = 1

    while True:
        response = requests.get(
            ESPN_TRANSACTIONS_URL,
            params={"year": year, "limit": PAGE_SIZE, "page": page},
            timeout=25,
        )
        response.raise_for_status()
        payload = response.json()

        transactions = payload.get("transactions", [])
        if not transactions:
            break

        for tx in transactions:
            team = tx.get("team", {}) or {}
            team_abbr = team.get("abbreviation")
            if team_abbr not in NFC_TEAMS:
                continue

            date_str = str(tx.get("date", ""))
            if not date_str.startswith(str(year)):
                continue

            description = str(tx.get("description", "")).strip()
            desc_lower = description.lower()
            if "trade" in desc_lower:
                tx_type = "TRADE"
            elif any(word in desc_lower for word in ["waive", "waiver", "released", "release"]):
                tx_type = "WAIVE_RELEASE"
            elif any(word in desc_lower for word in ["signed", "sign", "practice squad"]):
                tx_type = "SIGNING"
            elif "injured reserve" in desc_lower or "reserve" in desc_lower:
                tx_type = "RESERVE_MOVE"
            else:
                tx_type = "OTHER"

            rows.append(
                {
                    "SEASON": year,
                    "SNAPSHOT_DATE": date_str.split("T")[0] if "T" in date_str else date_str,
                    "WEEK": _parse_week(date_str),
                    "TEAM": team_abbr,
                    "TEAM_NAME": NFC_MAPPING.get(team_abbr, team.get("displayName", team_abbr)),
                    "PLAYER_ID": "",
                    "PLAYER": "",
                    "POSITION": "",
                    "PREV_TEAM": "",
                    "TRANSACTION_TYPE": tx_type,
                    "DESCRIPTION": description,
                    "SOURCE": "espn_transactions_api",
                }
            )

        if len(transactions) < PAGE_SIZE:
            break
        page += 1

    return rows


print("Connecting to ESPN NFL transactions API...")

try:
    print("Downloading 2026 transactions...")
    parsed_rows = fetch_espn_transactions(TARGET_SEASON)

    if parsed_rows:
        new_nfc_df = pd.DataFrame(parsed_rows)
        new_nfc_df = new_nfc_df[OUTPUT_COLUMNS]

        source_file = ARCHIVE_FILE if os.path.exists(ARCHIVE_FILE) else OUTPUT_FILE
        if os.path.exists(source_file):
            print(f"Merging fresh trends with data history in '{source_file}'...")
            existing_df = pd.read_csv(source_file)
            for col in OUTPUT_COLUMNS:
                if col not in existing_df.columns:
                    existing_df[col] = pd.NA
            existing_df = existing_df[OUTPUT_COLUMNS]
            combined_df = pd.concat([existing_df, new_nfc_df], ignore_index=True)
        else:
            print("No existing spreadsheet found. Initializing master database...")
            combined_df = new_nfc_df

        # Keep only the target season so this file stays focused on 2026 updates.
        if "SEASON" not in combined_df.columns:
            combined_df["SEASON"] = TARGET_SEASON
        else:
            combined_df["SEASON"] = combined_df["SEASON"].fillna(TARGET_SEASON)
        combined_df = combined_df[combined_df["SEASON"].astype(str) == str(TARGET_SEASON)].copy()

        # Keep only rows generated from the ESPN transaction feed.
        if "SOURCE" not in combined_df.columns:
            combined_df["SOURCE"] = "espn_transactions_api"
        combined_df = combined_df[combined_df["SOURCE"].fillna("") == "espn_transactions_api"].copy()

        combined_df["WEEK"] = pd.to_numeric(combined_df["WEEK"], errors="coerce")

        # Drop malformed rows that cannot identify a team/transaction/description.
        combined_df.dropna(subset=["TEAM", "TRANSACTION_TYPE", "DESCRIPTION"], inplace=True)
        combined_df = combined_df[combined_df["TEAM"].astype(str).str.strip() != ""]
        combined_df = combined_df[combined_df["TRANSACTION_TYPE"].astype(str).str.strip() != ""]
        combined_df = combined_df[combined_df["DESCRIPTION"].astype(str).str.strip() != ""]

        # Keep one row per player/team/type per weekly snapshot so re-runs stay idempotent.
        initial_len = len(combined_df)
        dedupe_keys = [
            "SEASON",
            "SNAPSHOT_DATE",
            "TEAM",
            "TRANSACTION_TYPE",
            "DESCRIPTION",
        ]
        combined_df.drop_duplicates(subset=dedupe_keys, inplace=True)
        combined_df.sort_values(
            by=["WEEK", "TEAM", "TRANSACTION_TYPE", "PLAYER"],
            ascending=[False, True, True, True],
            inplace=True,
        )
        combined_df = combined_df[OUTPUT_COLUMNS]
        final_len = len(combined_df)

        # Write full history archive first.
        combined_df.to_csv(ARCHIVE_FILE, index=False)

        # Keep only the latest N weekly snapshots in the main output.
        latest_df = combined_df.copy()
        if LATEST_WEEKS_TO_KEEP > 0:
            latest_weeks = latest_df["WEEK"].dropna().astype(int).drop_duplicates().head(LATEST_WEEKS_TO_KEEP)
            latest_df = latest_df[latest_df["WEEK"].fillna(-1).astype(int).isin(latest_weeks)].copy()

        latest_df.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ Success! Full-history ledger updated to {final_len} unique active entries.")
        print(f"🗂️ Wrote rolling {LATEST_WEEKS_TO_KEEP}-week view with {len(latest_df)} rows to '{OUTPUT_FILE}'.")
        print(f"🛡️ Safely removed {initial_len - final_len} duplicate log lines.")
    else:
        print("⚠️ No NFC team-change transactions found in the available 2026 roster snapshots yet.")

except Exception as e:
    print(f"❌ Core processing error: {e}")
