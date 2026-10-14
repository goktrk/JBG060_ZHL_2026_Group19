import numpy as np
import pandas as pd
from scipy.ndimage import sum_labels

from data_preparation.build_flood_monthly import NO_DATA
from data_preparation.grid import load_config
from data_preparation.loading import (
    load_flood,
    load_permanent_water,
    load_static,
)
from data_preparation.travel import travel_time

# 1. Load Configurations and Pre-built Data Layers
print("Loading grid layers and baseline facility sites...")
cfg = load_config()
static = load_static(cfg)
water = load_permanent_water(cfg)

# Extract 250m arrays
pop_250m = static["population"]
county_id_grid = static["county_id"]

# Load clean facility sites
sites = pd.read_parquet("data/processed/baseline/internal/sites.parquet")
sources = sites[["row", "col"]].to_numpy()

# Load county reference names
counties_df = pd.read_csv("data/interim/counties.csv")

# 2. Define Flood Scenario
SCENARIO_MONTH = "2022-10"
print(f"Simulating flood disruption for scenario: {SCENARIO_MONTH}...")

months, flood = load_flood(SCENARIO_MONTH)
flooded = (flood[0] >= cfg["flood"]["min_days_flooded"]) & (flood[0] != NO_DATA)

# Identify open (unflooded) health facility sites
open_sites_mask = ~flooded[sources[:, 0], sources[:, 1]]

# 3. Travel-Time Surface Solvers
print("Calculating Baseline Travel Times (Normal, Dry Day)...")
res_baseline = travel_time(static["friction_walking"], sources, blocked=water)
baseline_time = res_baseline.minutes

print("Calculating Flooded Travel Times (Disruption Scenario)...")
res_flooded = travel_time(
    static["friction_walking"],
    sources[open_sites_mask],
    blocked=water | flooded,
)
flooded_time = res_flooded.minutes

# 4. Part C: Loss of Access Math
THRESHOLD_MINUTES = 120  # ZOA standard 2-hour walking threshold

print("\n" + "=" * 50)
print(
    f"PART C: MEASURING LOSS OF ACCESS (Threshold: {THRESHOLD_MINUTES} mins)"
)
print("=" * 50)

# True/False Mask: HAD access in baseline, but LOST access during flood
loss_mask = (baseline_time <= THRESHOLD_MINUTES) & (
    (flooded_time > THRESHOLD_MINUTES) | np.isnan(flooded_time)
)

# Total national population cut off
stranded_population = np.sum(pop_250m[loss_mask])
total_population = np.sum(pop_250m)

print(
    f"Total Stranded Population: {int(stranded_population):,} people"
)
print(
    f"Percentage of Total Population: {(stranded_population / total_population) * 100:.2f}%"
)

# 5. Aggregate Cut-off Population by County (Admin-2)
unique_counties = np.unique(county_id_grid)

# Sum stranded people sitting inside each county boundary
stranded_by_county = sum_labels(
    pop_250m * loss_mask, labels=county_id_grid, index=unique_counties
)

# Merge into a summary report table
summary_df = pd.DataFrame(
    {"county_id": unique_counties, "stranded_pop": stranded_by_county}
)

summary_df = summary_df.merge(counties_df, on="county_id", how="left")
summary_df = (
    summary_df[summary_df["county_id"] > 0]
    .sort_values(by="stranded_pop", ascending=False)
    .reset_index(drop=True)
)

# Automatically find whichever name column exists in counties.csv
name_cols = [
    col
    for col in summary_df.columns
    if "name" in col.lower() or "county" in col.lower() and col != "county_id"
]
display_cols = (
    [name_cols[0], "stranded_pop"] if name_cols else ["county_id", "stranded_pop"]
)

print("\nTOP 10 MOST IMPACTED COUNTIES:")
print(summary_df[display_cols].head(10).to_string(index=False))

summary_df.to_csv("loss_of_access_summary.csv", index=False)
print("\nFull county breakdown saved to: loss_of_access_summary.csv")

# Calculate delay array (extra minutes for areas that still have access)
delay_array = np.where(
    np.isfinite(flooded_time),
    flooded_time - baseline_time,
    np.nan,  # Completely cut-off areas marked as NaN
)

# Save 2D spatial arrays for (Part D)
np.save("loss_mask.npy", loss_mask)
np.save("delay_array.npy", delay_array)
np.save("flooded_catchment.npy", res_flooded.nearest)