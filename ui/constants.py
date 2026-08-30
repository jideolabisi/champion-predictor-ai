"""NFC Teams constants and styling metadata for Champion Predictor AI UI."""

TEAM_METADATA = {
    "ARI": {
        "name": "Arizona Cardinals",
        "division": "NFC West",
        "primary_color": "#97233F",
        "secondary_color": "#FFB612",
        "text_color": "#FFFFFF",
    },
    "ATL": {
        "name": "Atlanta Falcons",
        "division": "NFC South",
        "primary_color": "#A71930",
        "secondary_color": "#000000",
        "text_color": "#FFFFFF",
    },
    "CAR": {
        "name": "Carolina Panthers",
        "division": "NFC South",
        "primary_color": "#0085CA",
        "secondary_color": "#101820",
        "text_color": "#FFFFFF",
    },
    "CHI": {
        "name": "Chicago Bears",
        "division": "NFC North",
        "primary_color": "#0B162A",
        "secondary_color": "#C83803",
        "text_color": "#FFFFFF",
    },
    "DAL": {
        "name": "Dallas Cowboys",
        "division": "NFC East",
        "primary_color": "#003594",
        "secondary_color": "#869397",
        "text_color": "#FFFFFF",
    },
    "DET": {
        "name": "Detroit Lions",
        "division": "NFC North",
        "primary_color": "#0076B6",
        "secondary_color": "#B0B7BC",
        "text_color": "#FFFFFF",
    },
    "GB": {
        "name": "Green Bay Packers",
        "division": "NFC North",
        "primary_color": "#203731",
        "secondary_color": "#FFB612",
        "text_color": "#FFFFFF",
    },
    "LA": {
        "name": "Los Angeles Rams",
        "division": "NFC West",
        "primary_color": "#003594",
        "secondary_color": "#FFA300",
        "text_color": "#FFFFFF",
    },
    "MIN": {
        "name": "Minnesota Vikings",
        "division": "NFC North",
        "primary_color": "#4F2683",
        "secondary_color": "#FFC62F",
        "text_color": "#FFFFFF",
    },
    "NO": {
        "name": "New Orleans Saints",
        "division": "NFC South",
        "primary_color": "#D3BC8D",
        "secondary_color": "#101820",
        "text_color": "#101820",
    },
    "NYG": {
        "name": "New York Giants",
        "division": "NFC East",
        "primary_color": "#0B2265",
        "secondary_color": "#A71930",
        "text_color": "#FFFFFF",
    },
    "PHI": {
        "name": "Philadelphia Eagles",
        "division": "NFC East",
        "primary_color": "#004C54",
        "secondary_color": "#A5ACAF",
        "text_color": "#FFFFFF",
    },
    "SEA": {
        "name": "Seattle Seahawks",
        "division": "NFC West",
        "primary_color": "#002244",
        "secondary_color": "#69BE28",
        "text_color": "#FFFFFF",
    },
    "SF": {
        "name": "San Francisco 49ers",
        "division": "NFC West",
        "primary_color": "#AA0000",
        "secondary_color": "#B3995D",
        "text_color": "#FFFFFF",
    },
    "TB": {
        "name": "Tampa Bay Buccaneers",
        "division": "NFC South",
        "primary_color": "#D50A0A",
        "secondary_color": "#34302B",
        "text_color": "#FFFFFF",
    },
    "WAS": {
        "name": "Washington Commanders",
        "division": "NFC East",
        "primary_color": "#5A1414",
        "secondary_color": "#FFB612",
        "text_color": "#FFFFFF",
    },
}

NFC_TEAMS = list(TEAM_METADATA.keys())

DIVISIONS = {
    "NFC East": ["DAL", "NYG", "PHI", "WAS"],
    "NFC North": ["CHI", "DET", "GB", "MIN"],
    "NFC South": ["ATL", "CAR", "NO", "TB"],
    "NFC West": ["ARI", "LA", "SEA", "SF"],
}

TEAM_CHOICES = [(f"{code} — {TEAM_METADATA[code]['name']} ({TEAM_METADATA[code]['division']})", code) for code in NFC_TEAMS]
