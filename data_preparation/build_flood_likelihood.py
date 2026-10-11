"""
Build the flood likelihood per calendar month: for every cell, the share of years in which it was
flooded in that month (at least flood.min_days_flooded days), from the monthly flood files.

    band m (1-12)  share of years 2003-2025 the cell was flooded in calendar month m, 0-1

Months flagged in flood_months.csv (data gaps, undercounted) are left out, so the share is taken
over the remaining years. NaN means no flood data for the cell (north of 10N), which is not the
same as "never flooded". The routing tool blocks cells above routing.max_flood_likelihood.

Needs build_flood_monthly first.
Run from the repository root:  python -m data_preparation.build_flood_likelihood
"""
from collections.abc import Iterable

import numpy as np
import rasterio

from data_preparation.build_flood_monthly import NO_DATA
from data_preparation.grid import data_path, load_config, make_grid, raster_profile
from data_preparation.loading import iter_flood, load_flood_months

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def likelihood_from_years(days_by_year: Iterable[np.ndarray], min_days: int) -> np.ndarray:
    """
    Share of years a cell was flooded, from one days-flooded array per year (all the same month).

    A year counts as flooded when the cell has at least `min_days` days. Years where the cell is
    NO_DATA count for neither the flooded years nor the total. Cells without any data get NaN.
    """
    flooded = years = None
    for days in days_by_year:
        if flooded is None:
            flooded = np.zeros(days.shape, dtype=np.uint16)
            years = np.zeros(days.shape, dtype=np.uint16)
        has_data = days != NO_DATA
        years += has_data
        flooded += has_data & (days >= min_days)
    if flooded is None:
        raise ValueError("no years given")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(years > 0, flooded / years, np.nan).astype(np.float32)


def main():
    cfg = load_config()
    grid = make_grid(cfg)
    min_days = cfg["flood"]["min_days_flooded"]
    threshold = cfg["routing"]["max_flood_likelihood"]
    first, last = cfg["flood"]["first_year"], cfg["flood"]["last_year"]
    quality = load_flood_months(cfg)
    flagged = set(zip(quality.loc[quality["flagged"], "year"], quality.loc[quality["flagged"], "month"]))

    path = data_path(cfg, "flood_likelihood")
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = raster_profile(grid, 12, "float32", np.nan)
    print(f"flood likelihood {first}-{last}, a year counts as flooded from {min_days} days; "
          f"share of cells with flood data above {threshold:.0%} (blocked when routing):")
    with rasterio.open(path, "w", **profile) as dst:
        for month in range(1, 13):
            years = [y for y in range(first, last + 1) if (y, month) not in flagged]
            days = (next(iter_flood(f"{y}-{month:02d}", cfg=cfg))[1] for y in years)
            share = likelihood_from_years(days, min_days)
            dst.write(share, month)
            dst.set_band_description(month, MONTH_NAMES[month - 1])
            with_data = np.isfinite(share)
            skipped = sorted(set(range(first, last + 1)) - set(years))
            note = f" (left out flagged {skipped})" if skipped else ""
            print(f"  {MONTH_NAMES[month - 1]}: {(share[with_data] > threshold).mean():6.2%}{note}")
        dst.update_tags(min_days_flooded=str(min_days), first_year=str(first), last_year=str(last),
                        flagged_months_left_out=",".join(f"{y}-{m:02d}" for y, m in sorted(flagged)))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
