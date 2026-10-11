"""
Loaders for the prepared files in data/interim (built by the build_* scripts in this folder).

Every array returned here is on the same analysis grid, so they line up cell for cell.

    static = load_static()                          # dict of 2D arrays
    months, flood = load_flood("2020-07", "2020-11")   # 5 months, flood.shape = (5, rows, cols)
    months, unusual = load_flood("2020-07", "2020-11", kind="unusual")
"""
from collections.abc import Iterator

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from data_preparation.build_flood_monthly import BANDS, NO_DATA
from data_preparation.grid import data_path, load_config


def load_static(cfg: dict | None = None) -> dict[str, np.ndarray]:
    """
    Load the static stack as a dictionary of 2D arrays, one per band:

    - population          float32, people per cell
    - friction_walking    float32, minutes per metre
    - friction_motorised  float32, minutes per metre
    - county_id           uint16, id from counties.csv, 0 = outside South Sudan
    """
    cfg = cfg or load_config()
    with rasterio.open(data_path(cfg, "static_stack")) as src:
        layers = {name: src.read(i) for i, name in enumerate(src.descriptions, start=1)}
    layers["county_id"] = layers["county_id"].astype("uint16")
    return layers


def load_permanent_water(cfg: dict | None = None) -> np.ndarray:
    """Boolean 2D array, True where the cell is permanent water (rivers, lakes; see build_permanent_water)."""
    cfg = cfg or load_config()
    with rasterio.open(data_path(cfg, "permanent_water")) as src:
        return src.read(1).astype(bool)


def iter_flood(start, end=None, cfg: dict | None = None,
               kind: str = "all") -> Iterator[tuple[pd.Period, np.ndarray]]:
    """
    Yield (month, days_flooded) for every month from `start` to `end`, both included.

    `start` and `end` can be anything pandas reads as a date, e.g. "2020-07" or "2020-07-15";
    a day inside a month selects that whole month. Leaving `end` out gives the single month `start`.
    Use this instead of load_flood for long periods, so only one month is in memory at a time.

    days_flooded is uint8: the number of days in the month the cell was flagged as flooded (0-31),
    or 255 (NO_DATA) where there is no flood data, which does not mean "not flooded".

    `kind` picks which flooding is counted: "all" (default), "recurring" (inside the area that
    normally floods in that calendar month) or "unusual" (outside it).
    """
    if kind not in BANDS:
        raise ValueError(f"kind must be one of {BANDS}, not {kind!r}")
    cfg = cfg or load_config()
    for month in pd.period_range(start, end or start, freq="M"):
        path = data_path(cfg, "flood_monthly", year=month.year, month=month.month)
        if not path.exists():
            raise FileNotFoundError(f"no flood file for {month}: {path}")
        with rasterio.open(path) as src:
            yield month, src.read(BANDS.index(kind) + 1)


def load_flood(start, end=None, cfg: dict | None = None, kind: str = "all") -> tuple[pd.PeriodIndex, np.ndarray]:
    """
    Load all months from `start` to `end` at once, see iter_flood for the arguments and values.

    Returns the months and one array of shape (n_months, rows, cols); flood[i] belongs to months[i].
    Each month takes about 24 MB, so load a few years at most this way.
    """
    months, arrays = zip(*iter_flood(start, end, cfg, kind))
    return pd.PeriodIndex(months), np.stack(arrays)


def load_flood_months(cfg: dict | None = None) -> pd.DataFrame:
    """
    One row per month (year, month, missing_days, flagged). `flagged` marks months where the flood
    archive has no records on several days (data gaps), so flooding in that month is undercounted.
    """
    cfg = cfg or load_config()
    return pd.read_csv(data_path(cfg, "flood_months"))


def load_facilities(cfg: dict | None = None) -> gpd.GeoDataFrame:
    """Load the facility table: one row per facility with its grid row/col and county."""
    cfg = cfg or load_config()
    return gpd.read_parquet(data_path(cfg, "facilities_table"))


def main():
    """Example usage."""
    static = load_static()
    for name, layer in static.items():
        print(f"{name}: {layer.shape} {layer.dtype}")
    print(f"total population: {static['population'].sum(dtype='float64'):,.0f}")

    min_days = load_config()["flood"]["min_days_flooded"]
    months, flood = load_flood("2020-07", "2020-11")
    print(f"flood: {flood.shape} {flood.dtype}")
    for month, days in zip(months, flood):
        flooded = (days >= min_days) & (days != NO_DATA)
        print(f"{month}: {flooded.sum():,} cells flooded on at least {min_days} days, "
              f"{static['population'][flooded].sum():,.0f} people living in them")

    facilities = load_facilities()
    print(f"facilities: {len(facilities)}")


if __name__ == "__main__":
    main()
