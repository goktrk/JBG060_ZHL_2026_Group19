"""
Build the permanent-water mask from the JRC Global Surface Water occurrence layer (Pekel et al., 2016).

Occurrence is the share of all satellite observations 1984-2024 in which a 30 m pixel was water.
Neither the flood archive (NASA removes known open water) nor the travel-cost layers (the White
Nile is crossed at almost walking speed) block rivers, so this mask is what makes them barriers.

A 250 m cell is permanent water when:
1. at least one 30 m pixel in it is water in >= `min_occurrence` % of observations. Taking any pixel,
   not the majority, keeps narrow rivers continuous, which matters because they act as barriers;
2. small gaps are closed (if `close_gaps`): a 3x3 closing fills breaks of one or two cells, and
   cells touching only corner to corner are joined, because travel can move diagonally;
3. nobody lives in it (if `keep_populated_cells_open`, applied last). A cell with people has land,
   and riverside towns are where bridges and ferries are; blocking these cells would also leave
   their people unable to reach any facility.


Values: 1 = permanent water, 0 = not. Needs the static stack (population), so run build_static first.

Run from the repository root:  python -m data_preparation.build_permanent_water
"""
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from scipy import ndimage

from data_preparation.grid import Grid, data_path, load_config, make_grid, raster_profile
from data_preparation.loading import load_static


def max_occurrence(cfg: dict, grid: Grid) -> np.ndarray:
    """Highest JRC occurrence (0-100) among the 30 m pixels in each grid cell, over all tiles."""
    out = np.zeros(grid.shape, dtype="uint8")
    tiles = sorted(data_path(cfg, "jrc_water").glob("occurrence_*.tif"))
    if not tiles:
        raise FileNotFoundError(f"no JRC occurrence tiles in {data_path(cfg, 'jrc_water')}")
    for path in tiles:
        tile = np.zeros(grid.shape, dtype="uint8")
        with rasterio.open(path) as src:
            reproject(source=rasterio.band(src, 1), destination=tile, dst_transform=grid.transform,
                      dst_crs=grid.crs, src_nodata=255, dst_nodata=0, init_dest_nodata=False,
                      resampling=Resampling.max)
        np.maximum(out, tile, out=out)
    print(f"read {len(tiles)} JRC tiles")
    return out


def close_diagonal_gaps(water: np.ndarray) -> np.ndarray:
    """Join water cells that touch only at a corner, so a diagonal step cannot slip between them."""
    water = water.copy()
    a, b, c, d = water[:-1, :-1], water[:-1, 1:], water[1:, :-1], water[1:, 1:]
    water[:-1, 1:] |= a & d & ~b & ~c
    water[:-1, :-1] |= b & c & ~a & ~d
    return water


def main():
    cfg = load_config()
    grid = make_grid(cfg)
    wcfg = cfg["permanent_water"]
    static = load_static(cfg)
    inside = static["county_id"] > 0

    occ = max_occurrence(cfg, grid)
    water = (occ >= wcfg["min_occurrence"]) & (occ <= 100)
    print(f"cells with a pixel that is water >= {wcfg['min_occurrence']}% of the time: {water.sum():,}, "
          f"{static['population'][water].sum():,.0f} people live in them")
    if wcfg["close_gaps"]:
        water |= ndimage.binary_closing(water, np.ones((3, 3), bool))
        water = close_diagonal_gaps(water)
    if wcfg["keep_populated_cells_open"]:
        water &= static["population"] <= 0

    n = int(water[inside].sum())
    print(f"permanent water: {water.sum():,} cells, of which {n:,} inside South Sudan "
          f"({n * (grid.cell_size / 1000) ** 2:,.0f} km2, {n / inside.sum():.2%} of the country); "
          f"{static['population'][water].sum():,.0f} people live in blocked cells")

    out_path = data_path(cfg, "permanent_water")
    with rasterio.open(out_path, "w", **raster_profile(grid, 1, "uint8", None)) as dst:
        dst.write(water.astype("uint8"), 1)
        dst.set_band_description(1, "permanent_water")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
