from covid19dh import covid19
import plotly.express as px
import os

STATE_ABBREV = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME",
    "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
    "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
    "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
    "Puerto Rico": "PR", "Guam": "GU", "American Samoa": "AS",
    "Virgin Islands": "VI", "Northern Mariana Islands": "MP",
}

x2, _ = covid19("USA", level=2, verbose=False)

counts = (
    x2.groupby("administrative_area_level_2")
    .size()
    .reset_index(name="datapoints")
    .rename(columns={"administrative_area_level_2": "state"})
)
counts["state_code"] = counts["state"].map(STATE_ABBREV)

fig = px.choropleth(
    counts,
    locations="state_code",
    locationmode="USA-states",
    color="datapoints",
    scope="usa",
    hover_name="state",
    hover_data={"state_code": False, "datapoints": True},
    color_continuous_scale="Blues",
    title="COVID-19 Datapoints per State (Level 2 — State)",
    labels={"datapoints": "Datapoints"},
)
fig.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0})

out_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "us_map_l2.html")
fig.write_html(out_path)
print(f"Map saved to: {os.path.normpath(out_path)}")
print("\nDatapoints per state:")
print(counts.sort_values("datapoints", ascending=False).to_string(index=False))

fig.show()
