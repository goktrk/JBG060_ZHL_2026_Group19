"""
Build the permanent-water mask from the monthly flood files.

A cell is permanent water if it counts as flooded (at least `min_days_flooded` days) in more than
`permanent_water_month_share` of all months. Values: 1 = permanent water, 0 = not, 255 = no flood data.

Run after build_flood_monthly:  python -m data_preparation.build_permanent_water
"""
import numpy as np
import rasterio

from data_preparation.build_flood_monthly import NO_DATA
from data_preparation.grid import data_path, load_config, make_grid, raster_profile


def main():
    cfg = load_config()
    grid = make_grid(cfg)
    min_days = cfg["flood"]["min_days_flooded"]
    share = cfg["flood"]["permanent_water_month_share"]

    files = sorted(data_path(cfg, "flood_monthly", year=0, month=0).parent.glob("flood_*.tif"))
    months_flooded = np.zeros(grid.shape, dtype="uint16")
    for path in files:
        with rasterio.open(path) as src:
            days = src.read(1)
        months_flooded += (days >= min_days) & (days != NO_DATA)

    permanent = (months_flooded / len(files) > share).astype("uint8")
    permanent[days == NO_DATA] = NO_DATA    # coverage is the same in every month

    out_path = data_path(cfg, "permanent_water")
    with rasterio.open(out_path, "w", **raster_profile(grid, 1, "uint8", NO_DATA)) as dst:
        dst.write(permanent, 1)
    n = int((permanent == 1).sum())
    print(f"{len(files)} months; {n:,} permanent-water cells "
          f"({n * (grid.cell_size / 1000) ** 2:,.0f} km2) flooded >= {min_days} days "
          f"in > {share:.0%} of months; wrote {out_path}")


if __name__ == "__main__":
    main()
