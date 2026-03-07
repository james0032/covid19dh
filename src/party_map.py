import pandas as pd
import plotly.graph_objects as go
import os

PARTY_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "state_party_affiliation_2020_2021.csv")
LEAN_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "pol_lean.csv")
OUT_DIR    = os.path.join(os.path.dirname(__file__), "..", "outputs")

STATE_ABBREV = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY",
}

# Approximate state centroids (lat, lon)
STATE_CENTROIDS = {
    "Alabama": (32.8, -86.8), "Alaska": (64.2, -153.0), "Arizona": (34.3, -111.1),
    "Arkansas": (34.8, -92.2), "California": (37.2, -119.5), "Colorado": (39.0, -105.5),
    "Connecticut": (41.6, -72.7), "Delaware": (39.0, -75.5), "Florida": (28.5, -82.5),
    "Georgia": (32.7, -83.4), "Hawaii": (20.9, -157.0), "Idaho": (44.4, -114.6),
    "Illinois": (40.0, -89.2), "Indiana": (39.9, -86.3), "Iowa": (42.1, -93.5),
    "Kansas": (38.5, -98.4), "Kentucky": (37.5, -85.3), "Louisiana": (31.2, -91.8),
    "Maine": (45.4, -69.2), "Maryland": (39.0, -76.8), "Massachusetts": (42.3, -71.8),
    "Michigan": (44.3, -85.4), "Minnesota": (46.4, -93.1), "Mississippi": (32.7, -89.7),
    "Missouri": (38.4, -92.5), "Montana": (47.0, -110.0), "Nebraska": (41.5, -99.9),
    "Nevada": (39.3, -116.6), "New Hampshire": (43.7, -71.6), "New Jersey": (40.1, -74.5),
    "New Mexico": (34.5, -106.1), "New York": (42.9, -75.5), "North Carolina": (35.5, -79.4),
    "North Dakota": (47.5, -100.5), "Ohio": (40.4, -82.8), "Oklahoma": (35.6, -97.5),
    "Oregon": (43.9, -120.6), "Pennsylvania": (40.9, -77.8), "Rhode Island": (41.7, -71.5),
    "South Carolina": (33.9, -80.9), "South Dakota": (44.4, -100.2), "Tennessee": (35.9, -86.4),
    "Texas": (31.5, -99.3), "Utah": (39.3, -111.1), "Vermont": (44.1, -72.7),
    "Virginia": (37.5, -79.4), "Washington": (47.4, -120.5), "West Virginia": (38.6, -80.6),
    "Wisconsin": (44.6, -89.9), "Wyoming": (43.0, -107.6),
}

# ── Load & join ───────────────────────────────────────────────────────────────
party_df = pd.read_csv(PARTY_PATH)
lean_df  = pd.read_csv(LEAN_PATH)[["State", "Partisan Lean", "Party of Partisan Lean"]]

df = party_df.merge(lean_df, on="State", how="inner")  # drops DC
df["abbrev"] = df["State"].map(STATE_ABBREV)
df["z"]      = df["Party"].map({"Republican": 1, "Democratic": 0})
df["lat"]    = df["State"].map(lambda s: STATE_CENTROIDS[s][0])
df["lon"]    = df["State"].map(lambda s: STATE_CENTROIDS[s][1])

lean_sign = df["Partisan Lean"].apply(lambda v: f"+{v}" if v > 0 else str(v))
df["hover"] = (
    df["State"] + "<br>" +
    "Governor party: " + df["Party"] + "<br>" +
    "Partisan lean: " + lean_sign + " (" + df["Party of Partisan Lean"] + ")"
)

# ── Choropleth — flat party color ─────────────────────────────────────────────
choropleth = go.Choropleth(
    locations=df["abbrev"],
    z=df["z"],
    locationmode="USA-states",
    colorscale=[[0, "#1a5fad"], [1, "#c01a2a"]],   # blue=Dem, red=Rep
    zmin=0, zmax=1,
    showscale=False,
    hovertext=df["hover"],
    hoverinfo="text",
)

# ── Lean score labels in white ─────────────────────────────────────────────────
labels = go.Scattergeo(
    lat=df["lat"],
    lon=df["lon"],
    text=lean_sign,
    mode="text",
    textfont=dict(color="white", size=9, family="Arial Black"),
    hoverinfo="skip",
)

fig = go.Figure(data=[choropleth, labels])
fig.update_layout(
    title=dict(text="US Governor Party Affiliation & Partisan Lean (2020–2021)", x=0.5),
    geo=dict(scope="usa", showlakes=False),
    margin=dict(l=0, r=0, t=50, b=0),
)

out_path = os.path.join(OUT_DIR, "party_map.html")
fig.write_html(out_path)
print(f"Saved: {out_path}")
fig.show()
