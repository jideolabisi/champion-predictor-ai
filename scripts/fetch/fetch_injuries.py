"""Fetch current-season NFC injury reports via nfl_data_py (nflverse).

Requires pyarrow (parquet support): pip install pyarrow

nfl_data_py's weekly injury reports only become available once real games
are played, so this will 404 before the season's first injury report is
published — that's expected, not a bug. Re-run once the season starts.

run command: python fetch_injuries.py
"""

import nfl_data_py as nfl

TARGET_SEASON = 2026
OUTPUT_FILE = "injury_reports_2026.csv"

NFC_TEAMS = {
    "ARI", "ATL", "CAR", "CHI", "DAL", "DET", "GB", "LA", "LAR",
    "MIN", "NO", "NYG", "PHI", "SEA", "SF", "TB", "WAS",
}

print(f"Fetching {TARGET_SEASON} injury reports...")
try:
    df = nfl.import_injuries([TARGET_SEASON])
except Exception as e:
    print(f"Not available yet ({type(e).__name__}: {e}). "
          f"Re-run this script once the {TARGET_SEASON} season has started.")
    raise SystemExit(0)

df = df[df["team"].isin(NFC_TEAMS)]
df.to_csv(OUTPUT_FILE, index=False)
print(f"Wrote {len(df)} rows to {OUTPUT_FILE}")
