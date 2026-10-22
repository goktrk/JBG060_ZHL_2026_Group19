"""
Pick each year's largest flood month, to evaluate the analysis over the whole flood record instead of
only the dashboard's showcase scenarios.

For every year flood.first_year..last_year, the month with the largest flooded area inside South Sudan
(cells flooded on at least flood.min_days_flooded days, the same rule as every other step). Months
flagged for data gaps (flood_months.csv) are skipped. Chosen by flood area only, so the choice doesn't
depend on our access model.

Output: evaluation.output_dir/peak_months.csv (year, month, flooded_km2, people_in_flooded_cells).
Then run build_access_loss and build_priorities with --months-file on it, and evaluate_peak_months.

Run from the repository root:  python -m data_preparation.peak_months
"""
import pandas as pd

from data_preparation.build_flood_monthly import NO_DATA
from data_preparation.grid import ROOT, load_config, make_grid
from data_preparation.loading import iter_flood, load_flood_months, load_static


def largest_per_year(months: pd.DataFrame) -> pd.DataFrame:
    """The row with the largest flooded_km2 per year (ties: the earlier month); flagged months skipped."""
    usable = months[~months["flagged"]]
    return (usable.sort_values(["year", "flooded_km2", "month"], ascending=[True, False, True])
            .groupby("year", as_index=False).first())


def main():
    cfg = load_config()
    grid = make_grid(cfg)
    static = load_static(cfg)
    inside, pop = static["county_id"] > 0, static["population"]
    min_days = cfg["flood"]["min_days_flooded"]
    flagged = load_flood_months(cfg).set_index(["year", "month"])["flagged"]
    km2 = (grid.cell_size / 1000) ** 2

    rows = []
    first, last = cfg["flood"]["first_year"], cfg["flood"]["last_year"]
    for month, days in iter_flood(f"{first}-01", f"{last}-12", cfg):
        flooded = (days >= min_days) & (days != NO_DATA) & inside
        rows.append({"year": month.year, "month": month.month, "id": str(month),
                     "flagged": bool(flagged.get((month.year, month.month), False)),
                     "flooded_km2": float(flooded.sum() * km2),
                     "people_in_flooded_cells": float(pop[flooded].sum(dtype="float64"))})
    peaks = largest_per_year(pd.DataFrame(rows))

    out_dir = ROOT / cfg["evaluation"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    peaks.round(1).to_csv(out_dir / "peak_months.csv", index=False)
    print(peaks[["id", "flooded_km2", "people_in_flooded_cells"]].round(0).to_string(index=False))
    print(f"wrote {out_dir / 'peak_months.csv'}")


if __name__ == "__main__":
    main()
