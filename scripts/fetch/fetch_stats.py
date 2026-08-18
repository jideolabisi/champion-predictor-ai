import pandas as pd
import requests
import time

all_seasons = []

# Pretend to be a real web browser to avoid the 403 error
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

for year in range(2006, 2026):
    url = f"https://pro-football-reference.com{year}/opp.htm"
    print(f"Fetching data for season: {year}...")
    
    try:
        # Download the HTML code using our custom browser headers
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 403:
            print(f"🛑 Still getting blocked on {year}. The server is forcing a temporary cooling-off period.")
            break
            
        # Parse the downloaded text using pandas
        tables = pd.read_html(response.text)
        
        # Grab the 'Team Total Offense' table
        team_stats_table = tables[0]
        
        # Add the tracking season
        team_stats_table['Season'] = year
        all_seasons.append(team_stats_table)
        
    except Exception as e:
        print(f"Could not fetch year {year}: {e}")
    
    # Crucial: Pause 4 seconds to comply with the 20 requests-per-minute rule
    time.sleep(4)

if all_seasons:
    master_df = pd.concat(all_seasons, ignore_index=True)
    master_df.to_csv("nfl_team_stats_2006_2025.csv", index=False)
    print("Success! File saved as 'nfl_team_stats_2006_2025.csv'")
