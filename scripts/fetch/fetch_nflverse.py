import pandas as pd

# Historical archive with completed seasons
ARCHIVE_URL = "https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats.parquet"

# Weekly feed for active/recent season tracking (2025 needs annual roll-up)
WEEKLY_2025_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_2025.parquet"

print("Connecting to nflverse historical + weekly datasets...")

try:
    # Historical archive: use 2006-2024 to avoid any 2025 overlap
    archive_df = pd.read_parquet(ARCHIVE_URL)
    historical_df = archive_df[(archive_df['season'] >= 2006) & (archive_df['season'] <= 2024)]

    # Weekly feed: roll all 2025 weeks into one season total per team
    weekly_2025_df = pd.read_parquet(WEEKLY_2025_URL)
    weekly_2025_df = weekly_2025_df[weekly_2025_df['season'] == 2025]
    if 'recent_team' not in weekly_2025_df.columns and 'team' in weekly_2025_df.columns:
        weekly_2025_df = weekly_2025_df.rename(columns={'team': 'recent_team'})

    # Merge into one 20-year span then summarize team-season totals
    combined_df = pd.concat([historical_df, weekly_2025_df], ignore_index=True, sort=False)
    team_stats = combined_df.groupby(['recent_team', 'season']).sum(numeric_only=True).reset_index()

    # Keep only the requested 2006-2025 window after merge
    team_stats = team_stats[(team_stats['season'] >= 2006) & (team_stats['season'] <= 2025)]

    # Quick validation: ensure each team has one row for every season 2006-2025
    expected_seasons = set(range(2006, 2026))
    missing_pairs = []
    for team, seasons in team_stats.groupby('recent_team')['season']:
        missing_for_team = sorted(expected_seasons - set(seasons.tolist()))
        missing_pairs.extend((team, season) for season in missing_for_team)

    if missing_pairs:
        print("⚠️ Missing team-season pairs detected:")
        for team, season in missing_pairs:
            print(f" - {team} {season}")
    else:
        print("✅ Validation passed: every team has 2006-2025 coverage.")
    
    # Clean up column names for readability
    team_stats.rename(columns={'recent_team': 'Team', 'season': 'Season'}, inplace=True)
    
    # Export cleanly to your project folder
    output_filename = "nflverse_team_stats_2006_2025.csv"
    team_stats.to_csv(output_filename, index=False)
    
    print(f"✅ Success! Compiled {len(team_stats)} rows of team totals (2006-2025).")
    print("Sources merged: archive 2006-2024 + weekly 2025 roll-up")
    print(f"File saved to: '{output_filename}'")

except Exception as e:
    print(f"❌ Failed to parse data: {e}")
