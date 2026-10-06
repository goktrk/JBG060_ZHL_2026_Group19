"""
Build one flood GeoTIFF per month on the analysis grid.

Each file has three bands, each holding a number of days in that month (0-31):

    band 1  all        days the cell was flagged as flooded, recurring or unusual
    band 2  recurring  days flagged as recurring flood (inside the area that normally floods this month)
    band 3  unusual    days flagged as unusual flood (outside that area)

Bands 2 and 3 can add up to more than band 1 when a cell contains two source pixels with different
labels on the same day. The labels are NASA's (see EDA/eda_flood_masks.ipynb). The value 255 means
"no flood data here" (outside the MODIS tiles we have), which is not the same as "not flooded".

It also writes flood_months.csv: per month, the number of days on which the archive has no record
at all (data gaps, see EDA/hydro_flood_eda.ipynb). Months with many missing days undercount
flooding and are flagged.

Run from the repository root:  python -m data_preparation.build_flood_monthly
"""
import numpy as np
import pandas as pd
import rasterio

from data_preparation.grid import (Grid, cell_centres_lonlat, data_path, in_grid, load_config,
                                   lonlat_to_rowcol, make_grid, raster_profile)

NO_DATA = 255
BANDS = ["all", "recurring", "unusual"]


def tile_bounds(tile: str) -> tuple[float, float, float, float]:
    """lon_min, lat_min, lon_max, lat_max of a 10x10 degree MODIS flood tile such as 'h20v08'."""
    h, v = int(tile[1:3]), int(tile[4:6])
    lon_min, lat_max = -180 + 10 * h, 90 - 10 * v
    return lon_min, lat_max - 10, lon_min + 10, lat_max


def coverage_mask(cfg: dict, grid: Grid) -> np.ndarray:
    """True for cells whose centre lies inside one of the flood tiles we have."""
    lon, lat = cell_centres_lonlat(grid)
    covered = np.zeros(grid.shape, dtype=bool)
    for tile in cfg["flood"]["tiles"]:
        lon_min, lat_min, lon_max, lat_max = tile_bounds(tile)
        covered |= (lon >= lon_min) & (lon < lon_max) & (lat >= lat_min) & (lat < lat_max)
    return covered


def load_year(cfg: dict, grid: Grid, year: int) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """
    All flood records of one year as (month, day, cell, unusual), where cell = row * width + col,
    and the days on which the archive has any record at all (anywhere on the tiles).
    """
    parts, seen = [], []
    for kind in ["flood_recurring", "flood_unusual"]:
        for tile in cfg["flood"]["tiles"]:
            path = data_path(cfg, kind) / f"flood_events_{tile}_{year}.parquet"
            df = pd.read_parquet(path, columns=["date", "lat", "lon"])
            dates = pd.to_datetime(df["date"].astype(str))
            seen.append(dates.unique())
            row, col = lonlat_to_rowcol(grid, df["lon"].to_numpy(), df["lat"].to_numpy())
            keep = in_grid(grid, row, col)
            parts.append(pd.DataFrame({
                "month": dates.dt.month.to_numpy()[keep],
                "day": dates.dt.day.to_numpy()[keep],
                "cell": (row * grid.width + col)[keep],
                "unusual": kind == "flood_unusual",
            }))
    return pd.concat(parts, ignore_index=True), pd.DatetimeIndex(np.concatenate(seen)).unique()


def missing_days(year: int, seen: pd.DatetimeIndex) -> pd.DataFrame:
    """Per month of `year`: how many days have no flood record at all (data gaps)."""
    missing = pd.date_range(f"{year}-01-01", f"{year}-12-31").difference(seen)
    per_month = pd.Series(missing.month).value_counts().reindex(range(1, 13), fill_value=0)
    return pd.DataFrame({"year": year, "month": range(1, 13), "missing_days": per_month.to_numpy()})


def days_flooded_per_month(records: pd.DataFrame) -> pd.DataFrame:
    """
    Number of distinct flooded days per (month, cell): in total, as recurring, and as unusual.
    A cell hit twice on one day counts once.
    """
    records = records.drop_duplicates()
    total = records[["month", "day", "cell"]].drop_duplicates().groupby(["month", "cell"]).size().rename("all")
    by_label = (records.groupby(["month", "cell", "unusual"]).size().unstack(fill_value=0)
                .reindex(columns=[False, True], fill_value=0))
    by_label.columns = ["recurring", "unusual"]
    return by_label.join(total).reset_index()


def main():
    cfg = load_config()
    grid = make_grid(cfg)
    covered = coverage_mask(cfg, grid)
    print(f"grid: {grid.width} x {grid.height} cells; {covered.mean():.1%} of cells have flood coverage")
    data_path(cfg, "flood_monthly", year=0, month=0).parent.mkdir(parents=True, exist_ok=True)
    profile = raster_profile(grid, len(BANDS), "uint8", NO_DATA)

    quality = []
    for year in range(cfg["flood"]["first_year"], cfg["flood"]["last_year"] + 1):
        records, seen = load_year(cfg, grid, year)
        quality.append(missing_days(year, seen))
        counts = days_flooded_per_month(records)
        written, skipped = 0, []
        for month in range(1, 13):
            this_month = counts[counts["month"] == month]
            if this_month.empty:
                skipped.append(month)
                continue
            with rasterio.open(data_path(cfg, "flood_monthly", year=year, month=month), "w", **profile) as dst:
                for band, name in enumerate(BANDS, start=1):
                    days = np.zeros(grid.shape, dtype="uint8")
                    days.ravel()[this_month["cell"].to_numpy()] = this_month[name].to_numpy()
                    days[~covered] = NO_DATA
                    dst.write(days, band)
                    dst.set_band_description(band, name)
            written += 1
        note = f"; no records, not written: months {skipped}" if skipped else ""
        print(f"{year}: {len(records):,} records, {len(counts):,} flooded cell-months, {written} files{note}")

    quality = pd.concat(quality, ignore_index=True)
    quality["flagged"] = quality["missing_days"] >= cfg["flood"]["flag_min_missing_days"]
    quality.to_csv(data_path(cfg, "flood_months"), index=False)
    flagged = quality[quality["flagged"]]
    print(f"months with at least {cfg['flood']['flag_min_missing_days']} days without any record (flagged): "
          + ", ".join(f"{r.year}-{r.month:02d} ({r.missing_days} days)" for r in flagged.itertuples()))
    print(f"wrote {data_path(cfg, 'flood_months')}")


if __name__ == "__main__":
    main()
