from covid19dh import covid19

x, src = covid19()

# General dataset info
print("=== Shape ===")
print(x.shape)

print("\n=== Columns ===")
print(x.columns.tolist())

print("\n=== Data Types ===")
print(x.dtypes)

print("\n=== Date Range ===")
print(f"Start: {x['date'].min()}")
print(f"End:   {x['date'].max()}")

print("\n=== Countries / Regions ===")
print(f"Unique administrative_area_level_1: {x['administrative_area_level_1'].nunique()}")
print(x['administrative_area_level_1'].unique())

print("\n=== Missing Values ===")
print(x.isnull().sum())

print("\n=== Numeric Summary ===")
print(x.describe())

