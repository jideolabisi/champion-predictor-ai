import nfl_data_py as nfl
import pandas as pd

print("Connecting to the nflverse to fetch active 2026 rosters...")

try:
    # 1. Download full league seasonal rosters for the 2026 season
    # (nfl_data_py supports roster retrieval across years)
    rosters_2026 = nfl.import_seasonal_rosters([2026])
    
    # 2. Define the 16 franchises that belong to the NFC
    # NOTE: nflverse's team-abbreviation convention codes the LA Rams as
    # 'LA', not 'LAR' (confirmed by inspecting import_seasonal_rosters'
    # actual `team` column values) — 'LAR' here silently matched zero rows
    # and dropped the Rams from every fetch.
    nfc_teams = [
        'ARI', 'ATL', 'CAR', 'CHI', 'DAL', 'DET', 'GB', 'LA',
        'MIN', 'NO', 'NYG', 'PHI', 'SEA', 'SF', 'TB', 'WAS'
    ]
    
    # 3. Filter the dataframe to only include the NFC conference teams
    nfc_rosters = rosters_2026[rosters_2026['team'].isin(nfc_teams)].copy()
    
    # Sort nicely by team and player name
    nfc_rosters.sort_values(by=['team', 'player_name'], inplace=True)
    
    # 4. Save to a separate CSV file
    output_filename = "nfl_nfc_roster_2026.csv"
    nfc_rosters.to_csv(output_filename, index=False)
    
    print(f"✅ Success! Compiled {len(nfc_rosters)} active NFC players.")
    print(f"Saved directly to: '{output_filename}'")

except Exception as e:
    print(f"❌ Failed to compile rosters: {e}")
