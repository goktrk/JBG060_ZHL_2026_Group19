"""
Build the static stack: one GeoTIFF on the analysis grid with the layers that do not change over time.

    band 1  population          people per cell (WorldPop, summed from ~100 m)
    band 2  friction_walking    minutes per metre (Malaria Atlas Project, ~1 km, nearest neighbour)
    band 3  friction_motorised  minutes per metre, never slower than walking
    band 4  county_id           admin-2 id from counties.csv, 0 = outside South Sudan

Run from the repository root:  python -m data_preparation.build_static
"""
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject

from data_preparation.grid import Grid, data_path, load_config, load_counties, make_grid, raster_profile

BANDS = ["population", "friction_walking", "friction_motorised", "county_id"]


def warp_to_grid(src_path, grid: Grid, resampling: Resampling) -> np.ndarray:
    """Resample band 1 of a raster onto the grid. Cells without source data become NaN."""
    out = np.full(grid.shape, np.nan, dtype="float32")
    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, 1), destination=out,
            dst_transform=grid.transform, dst_crs=grid.crs, dst_nodata=np.nan,
            resampling=resampling,
        )
    return out


def build_population(cfg: dict, grid: Grid) -> np.ndarray:
    """Population per cell. Source cells are summed, then rescaled so the national total is exact."""
    src_path = data_path(cfg, "population", year=cfg["population"]["year"])
    with rasterio.open(src_path) as src:
        source_total = float(src.read(1, masked=True).sum(dtype="float64"))

    pop = np.nan_to_num(warp_to_grid(src_path, grid, Resampling.sum), nan=0.0)
    warped_total = float(pop.sum(dtype="float64"))
    pop *= source_total / warped_total

    print(f"population {cfg['population']['year']}: source total {source_total:,.0f}, "
          f"after resampling {warped_total:,.0f} ({warped_total / source_total - 1:+.3%}), "
          f"rescaled to {float(pop.sum(dtype='float64')):,.0f}")
    return pop


def build_county_ids(cfg: dict, grid: Grid) -> np.ndarray:
    """County id per cell (cell centre inside the county polygon), 0 outside South Sudan."""
    counties = load_counties(cfg)
    out_path = data_path(cfg, "counties_table")
    counties.drop(columns="geometry").to_csv(out_path, index=False)

    shapes = zip(counties.to_crs(grid.crs).geometry, counties["county_id"])
    ids = rasterize(shapes, out_shape=grid.shape, transform=grid.transform, fill=0, dtype="uint16")
    print(f"counties: {len(counties)} in file, {len(np.unique(ids)) - 1} on the grid, "
          f"{(ids > 0).mean():.1%} of cells inside South Sudan; lookup written to {out_path}")
    return ids


def main():
    cfg = load_config()
    grid = make_grid(cfg)
    print(f"grid: {grid.width} x {grid.height} cells of {grid.cell_size} m, {grid.crs}")
    data_path(cfg, "static_stack").parent.mkdir(parents=True, exist_ok=True)

    county_ids = build_county_ids(cfg, grid)
    inside = county_ids > 0
    layers = {"population": build_population(cfg, grid), "county_id": county_ids.astype("float32")}
    for mode in ["walking", "motorised"]:
        friction = warp_to_grid(data_path(cfg, f"friction_{mode}"), grid, Resampling.nearest)
        print(f"friction {mode}: {np.isnan(friction[inside]).mean():.2%} of cells inside South Sudan "
              f"have no value; range {np.nanmin(friction):.4f} to {np.nanmax(friction):.4f} min/m")
        layers[f"friction_{mode}"] = friction

    # Motorised can never be slower than walking (you can always get out and walk); the source has
    # 0.5% of cells where it is, an artefact (see EDA/eda_friction.ipynb), so take the faster of the two.
    slower = layers["friction_motorised"] > layers["friction_walking"]
    layers["friction_motorised"] = np.fmin(layers["friction_motorised"], layers["friction_walking"])
    print(f"friction motorised: {slower[inside].mean():.2%} of cells inside South Sudan were slower than "
          f"walking, set to the walking value")

    out_path = data_path(cfg, "static_stack")
    with rasterio.open(out_path, "w", **raster_profile(grid, len(BANDS), "float32", np.nan)) as dst:
        for i, name in enumerate(BANDS, start=1):
            dst.write(layers[name], i)
            dst.set_band_description(i, name)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
