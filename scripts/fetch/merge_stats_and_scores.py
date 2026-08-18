import pandas as pd

print("Loading data files...")
# 1. Load your existing data sheets
# (Assumes you ran the team totals aggregation on your player file)
stats_df = pd.read_csv("nflverse_team_stats_2006_2025.csv")
scores_df = pd.read_csv("nfl_weekly_scores_2006_2025.csv")

print("Restructuring the weekly scores data...")
# 2. Extract game info from the Home Team perspective
home_sides = scores_df[[
    'season', 'week', 'game_type', 'home_team', 'home_score', 'away_score'
]].copy()
home_sides.columns = ['Season', 'Week', 'Game_Type', 'Team', 'Points_Scored', 'Points_Allowed']
home_sides['Location'] = 'Home'

# 3. Extract game info from the Away Team perspective
away_sides = scores_df[[
    'season', 'week', 'game_type', 'away_team', 'away_score', 'home_score'
]].copy()
away_sides.columns = ['Season', 'Week', 'Game_Type', 'Team', 'Points_Scored', 'Points_Allowed']
away_sides['Location'] = 'Away'

# 4. Combine them so every team has its own row for every game played
team_games = pd.concat([home_sides, away_sides], ignore_index=True)

# 5. Calculate if the team won, lost, or tied that specific week
def determine_outcome(row):
    if row['Points_Scored'] > row['Points_Allowed']:
        return 'Win'
    elif row['Points_Scored'] < row['Points_Allowed']:
        return 'Loss'
    else:
        return 'Tie'

team_games['Outcome'] = team_games.apply(determine_outcome, axis=1)

print("Merging weekly game outcomes with yearly team stats...")
# 6. Merge the datasets matching on both 'Team' and 'Season'
# This appends the team's full-year stats to every single game they played that year
merged_master = pd.merge(
    team_games, 
    stats_df, 
    on=['Team', 'Season'], 
    how='inner'
)

# Sort chronologically by season, week, and team name
merged_master.sort_values(by=['Season', 'Week', 'Team'], ascending=[False, True, True], inplace=True)

# 7. Save the consolidated analytical master matrix
output_file = "nfl_combined_games_and_stats_2006_2025.csv"
merged_master.to_csv(output_file, index=False)

print(f"✅ Success! Compiled {len(merged_master)} team-game entries.")
print(f"Master file saved to: '{output_file}'")
