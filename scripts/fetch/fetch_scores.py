import nfl_data_py as nfl

print("Downloading weekly game scores from 2006 to 2025...")

# Fetch full schedule and score history
schedules_df = nfl.import_schedules(list(range(2006, 2026)))

# Keep only the rows for regular season and playoff games that have finished
completed_games = schedules_df[schedules_df['game_type'].notna()]

# Filter for the most important columns to keep your file clean
columns_to_keep = [
    'season', 'week', 'game_type', 
    'home_team', 'away_team', 
    'home_score', 'away_score', 
    'result', 'total'
]
final_scores = completed_games[columns_to_keep]

# Save to a new CSV file
final_scores.to_csv("nfl_weekly_scores_2006_2025.csv", index=False)
print("✅ Success! Weekly scores saved to 'nfl_weekly_scores_2006_2025.csv'")
