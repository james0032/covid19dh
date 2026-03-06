from covid19dh import covid19
import pandas as pd

# Fetch US data at country, state, and county levels
x1, _ = covid19("USA", level=1, verbose=False)
x2, _ = covid19("USA", level=2, verbose=False)
x3, _ = covid19("USA", level=3, verbose=False)

def label_geo(df):
    df = df.copy()
    df["geo_country"] = df["administrative_area_level_1"]
    df["geo_state"] = df["administrative_area_level_2"]
    df["geo_county"] = df["administrative_area_level_3"]
    df["geo_level"] = df["administrative_area_level"]
    return df

us = pd.concat([label_geo(x1), label_geo(x2), label_geo(x3)], ignore_index=True)

print("=== US Subset Shape ===")
print(us.shape)

print("\n=== Rows per geo_level ===")
print(us["geo_level"].value_counts().sort_index())

print("\n=== Date Range ===")
print(f"Start: {us['date'].min()}")
print(f"End:   {us['date'].max()}")

print("\n=== Unique States (level 2) ===")
print(sorted(us.loc[us["geo_level"] == 2, "geo_state"].unique()))

print("\n=== Unique Counties (level 3) ===")
print(f"Count: {us.loc[us['geo_level'] == 3, 'geo_county'].nunique()}")

print("\n=== Sample rows (one per level) ===")
for level in [1, 2, 3]:
    row = us[us["geo_level"] == level].iloc[0]
    print(f"\nLevel {level}: country={row['geo_country']}, state={row['geo_state']}, county={row['geo_county']}, lat={row['latitude']}, lon={row['longitude']}")
