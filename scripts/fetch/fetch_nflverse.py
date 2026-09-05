import pandas as pd

# Uniform per-season weekly player-stats feed, 2006-2025. Replaces the old
# two-source approach (a 2006-2024 "player_stats.parquet" archive + a single
# 2025 "stats_player_week_2025.parquet" file): that archive is offense-only
# (no def_* columns at all) and names interceptions-thrown "interceptions",
# while the weekly feed is offense+defense and names the same stat
# "passing_interceptions". Concatenating the two mismatched schemas and
# summing let pandas silently zero-fill whichever columns a source lacked --
# confirmed empirically: every one of the 15 def_* columns was exactly 0.0
# for every team in every season 2006-2024, and 'interceptions' was 0.0 for
# every team in 2025. nflverse publishes this same rich weekly schema for
# every season back to 2006 (confirmed: stats_player_week_2006.parquet
# through stats_player_week_2025.parquet all resolve), so fetching all 20
# seasons from it avoids the schema mismatch entirely.
WEEKLY_URL_TEMPLATE = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{year}.parquet"
FIRST_SEASON = 2006
LAST_SEASON = 2025

print(f"Connecting to nflverse weekly player-stats feed ({FIRST_SEASON}-{LAST_SEASON})...")

try:
    frames = []
    for year in range(FIRST_SEASON, LAST_SEASON + 1):
        print(f"Fetching season {year}...")
        frames.append(pd.read_parquet(WEEKLY_URL_TEMPLATE.format(year=year)))

    all_weeks = pd.concat(frames, ignore_index=True, sort=False)

    # This tool is documented (README, safe_server.py) as "regular-season
    # performance" -- without this filter, playoff-week stats were silently
    # being summed into the same per-team-season total as the regular season.
    reg_season = all_weeks[all_weeks['season_type'] == 'REG'].copy()

    # Roll weekly player rows up into team-season totals
    team_stats = reg_season.groupby(['team', 'season']).sum(numeric_only=True).reset_index()

    # Keep only the requested 2006-2025 window
    team_stats = team_stats[(team_stats['season'] >= FIRST_SEASON) & (team_stats['season'] <= LAST_SEASON)]

    # Quick validation: ensure each team has one row for every season 2006-2025
    expected_seasons = set(range(FIRST_SEASON, LAST_SEASON + 1))
    missing_pairs = []
    for team, seasons in team_stats.groupby('team')['season']:
        missing_for_team = sorted(expected_seasons - set(seasons.tolist()))
        missing_pairs.extend((team, season) for season in missing_for_team)

    if missing_pairs:
        print("⚠️ Missing team-season pairs detected:")
        for team, season in missing_pairs:
            print(f" - {team} {season}")
    else:
        print("✅ Validation passed: every team has 2006-2025 coverage.")

    # Clean up column names for readability, and keep 'interceptions' as the
    # canonical name for interceptions-thrown (app.py's Data Explorer and
    # this project's docs already reference it under that name).
    team_stats.rename(
        columns={'team': 'Team', 'season': 'Season', 'passing_interceptions': 'interceptions'},
        inplace=True,
    )

    # Export cleanly to your project folder
    output_filename = "nflverse_team_stats_2006_2025.csv"
    team_stats.to_csv(output_filename, index=False)

    print(f"✅ Success! Compiled {len(team_stats)} rows of team totals (2006-2025).")
    print("Source: uniform stats_player_week_<year>.parquet per season, regular season only.")
    print(f"File saved to: '{output_filename}'")

except Exception as e:
    print(f"❌ Failed to parse data: {e}")
