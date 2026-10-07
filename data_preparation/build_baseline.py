"""
Build the baseline travel cost (no flooding): minutes from every cell to the nearest health
facility, which facility that is, and access statistics per admin unit.

Facilities in the same 250 m cell count as one site, with site_id 0..N-1 in (row, col) order.
Permanent water is blocked, the same as in the flood scenarios. The outputs are described in
data_preparation/README_baseline.md.

Needs build_static, build_permanent_water and build_facilities first.
Run from the repository root:  python -m data_preparation.build_baseline
"""
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version

import geopandas as gpd
import numpy as np
import pandas as pd
import psutil
import rasterio
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.features import rasterize

from data_preparation.grid import ROOT, Grid, data_path, load_config, make_grid, raster_profile
from data_preparation.loading import load_facilities, load_permanent_water, load_static
from data_preparation.travel import travel_time

# picks the representative facility when several share a cell
FACILITY_LEVEL = {
    "Teaching Hospital": 5,
    "State Hospital": 4,
    "County Hospital": 3,
    "Primary Health Care Centre": 2,
    "Primary Health Care Unit": 1,
}
# level: (config section and key of the file, pcode column, name column)
ADMIN_LEVELS = {
    "admin1": (("baseline", "admin1"), "adm1_pcode", "adm1_name"),
    "admin2": (("paths", "admin2"), "adm2_pcode", "adm2_name"),
    "admin3": (("baseline", "admin3"), "adm3_pcode", "adm3_name"),
}
REFERENCE_POPULATION = 10_509_966   # WorldPop 2020 inside counties, from EDA/eda_friction.ipynb


def grid_from_raster(path) -> Grid:
    with rasterio.open(path) as src:
        return Grid(crs=src.crs.to_string(), cell_size=src.res[0], transform=src.transform,
                    width=src.width, height=src.height)


def same_grid(a: Grid, b: Grid) -> bool:
    return (CRS.from_user_input(a.crs) == CRS.from_user_input(b.crs)
            and a.transform == b.transform and a.shape == b.shape)


def build_sites(fac: pd.DataFrame, grid: Grid) -> pd.DataFrame:
    """One row per cell with a facility in it."""
    fac = fac.assign(level=fac["Facility_t"].map(FACILITY_LEVEL))
    if fac["level"].isna().any():
        raise ValueError(f"unknown facility types: {sorted(fac.loc[fac['level'].isna(), 'Facility_t'].unique())}")

    sites = fac[["row", "col"]].drop_duplicates().sort_values(["row", "col"]).reset_index(drop=True)
    sites.insert(0, "site_id", np.arange(len(sites), dtype=np.int32))
    fac = fac.merge(sites, on=["row", "col"])

    # representative: highest level, then lowest FID
    rep = fac.sort_values(["site_id", "level", "FID"], ascending=[True, False, True]).groupby("site_id").first()
    by_site = fac.sort_values("FID").groupby("site_id")

    sites["x"] = grid.transform.c + (sites["col"] + 0.5) * grid.cell_size
    sites["y"] = grid.transform.f - (sites["row"] + 0.5) * grid.cell_size
    to_lonlat = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    sites["lon"], sites["lat"] = to_lonlat.transform(sites["x"].to_numpy(), sites["y"].to_numpy())
    sites["n_facilities"] = by_site.size().to_numpy()
    sites["rep_facility_id"] = rep["FID"].to_numpy()
    sites["rep_facility_name"] = rep["Facility_n"].to_numpy()
    sites["rep_facility_type"] = rep["Facility_t"].to_numpy()
    sites["rep_facility_level"] = rep["level"].astype(int).to_numpy()
    sites["facility_ids"] = by_site["FID"].agg(list).to_numpy()
    sites["has_flood_data"] = by_site["has_flood_data"].all().to_numpy()   # False north of 10N
    return sites


def load_admin(cfg: dict, level: str) -> gpd.GeoDataFrame:
    """Admin units of one level with unit_id 1..N in pcode order (0 = no unit)."""
    (section, key), pcode, name = ADMIN_LEVELS[level]
    units = gpd.read_file(ROOT / cfg[section][key])[[pcode, name, "geometry"]]
    units = units.rename(columns={pcode: "pcode", name: "name"}).sort_values("pcode").reset_index(drop=True)
    units.insert(0, "unit_id", units.index + 1)
    return units


def rasterize_admin(units: gpd.GeoDataFrame, grid: Grid) -> np.ndarray:
    """unit_id of the polygon containing each cell centre, 0 outside."""
    shapes = zip(units.to_crs(grid.crs).geometry, units["unit_id"])
    return rasterize(shapes, out_shape=grid.shape, transform=grid.transform, fill=0, dtype="uint16")


def site_units(units: gpd.GeoDataFrame, ids: np.ndarray, sites: pd.DataFrame, grid: Grid) -> np.ndarray:
    """unit_id of each site's cell; sites just outside every polygon get the nearest one."""
    out = ids[sites["row"], sites["col"]].astype(np.int64)
    outside = np.flatnonzero(out == 0)
    if outside.size:
        points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(sites["x"].iloc[outside], sites["y"].iloc[outside]),
                                  crs=grid.crs)
        near = gpd.sjoin_nearest(points, units[["unit_id", "geometry"]].to_crs(grid.crs))
        out[outside] = near.groupby(level=0)["unit_id"].min().reindex(range(outside.size)).to_numpy()
    return out


def weighted_quantiles(group, values, weights, n_groups, qs) -> dict[float, np.ndarray]:
    """Weighted quantiles per group (NaN for empty groups)."""
    out = {q: np.full(n_groups, np.nan) for q in qs}
    if len(values) == 0:
        return out
    order = np.lexsort((values, group))
    g, v, w = group[order], values[order], weights[order]
    cum = np.cumsum(w, dtype="float64")
    gid = np.arange(n_groups)
    start, end = np.searchsorted(g, gid, "left"), np.searchsorted(g, gid, "right")
    before = np.where(start > 0, cum[np.maximum(start - 1, 0)], 0.0)
    total = np.bincount(g, w, minlength=n_groups)
    has = end > start
    for q in qs:
        idx = np.searchsorted(cum, before + q * total, "left")
        idx = np.clip(idx, start, np.maximum(end - 1, start))
        out[q][has] = v[idx[has]]
    return out


def access_stats(ids, n_units, pop, minutes, thresholds) -> pd.DataFrame:
    """Population-weighted travel times per unit, indexed by unit_id 0..n_units."""
    sel = (pop > 0) & (ids > 0)
    u, w, t = ids[sel].astype(np.int64), pop[sel].astype("float64"), minutes[sel].astype("float64")
    n = n_units + 1
    reached = np.isfinite(t)
    population = np.bincount(u, w, minlength=n)
    pop_reached = np.bincount(u[reached], w[reached], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        df = pd.DataFrame({
            "population": population,
            "mean_min": np.bincount(u[reached], w[reached] * t[reached], minlength=n) / pop_reached,
        })
        q = weighted_quantiles(u[reached], t[reached], w[reached], n, [0.5, 0.9])
        df["median_min"], df["p90_min"] = q[0.5], q[0.9]
        for k in thresholds:
            within = np.where(reached, t, np.inf) <= k
            df[f"pct_within_{k}"] = 100 * np.bincount(u, w * within, minlength=n) / population
    df["unreached_population"] = population - pop_reached
    df["cells"] = np.bincount(ids.ravel(), minlength=n)
    return df


def admin_table(cfg, grid, static, sites, results, thresholds):
    """Access per admin unit and mode, plus each site's admin codes."""
    rows, empty_units, site_codes = [], {}, {}
    pop = static["population"]
    for level in ADMIN_LEVELS:
        units = load_admin(cfg, level)
        ids = rasterize_admin(units, grid)
        if level == "admin2":
            matches_county_id = bool(np.array_equal(ids, static["county_id"]))

        site_unit = site_units(units, ids, sites, grid)
        site_codes[f"{level}_pcode"] = units.set_index("unit_id").loc[site_unit, "pcode"].to_numpy()
        n_sites = np.bincount(site_unit, minlength=len(units) + 1)
        n_fac = np.bincount(site_unit, weights=sites["n_facilities"], minlength=len(units) + 1)

        for mode, res in results.items():
            stats = access_stats(ids, len(units), pop, res.minutes, thresholds).iloc[1:]
            stats = stats.assign(n_sites=n_sites[1:], n_facilities=n_fac[1:].astype(int))
            with np.errstate(divide="ignore", invalid="ignore"):
                stats["population_per_facility"] = stats["population"] / stats["n_facilities"].replace(0, np.nan)
            stats.insert(0, "mode", mode)
            stats.insert(0, "name", units["name"].to_numpy())
            stats.insert(0, "pcode", units["pcode"].to_numpy())
            stats.insert(0, "level", level)
            rows.append(stats)

        empty = units.loc[np.bincount(ids.ravel(), minlength=len(units) + 1)[1:] == 0, ["pcode", "name"]]
        empty_units[level] = empty.to_dict("records")
        print(f"{level}: {len(units)} units, {len(empty)} without cells; "
              f"population assigned {float(pop[ids > 0].sum(dtype='float64')):,.0f}")
    return pd.concat(rows, ignore_index=True), empty_units, site_codes, matches_county_id


def national_summary(pop, minutes, inside, thresholds) -> dict:
    s = access_stats(inside.astype(np.uint16), 1, pop, minutes, thresholds).iloc[1]
    return {k: (round(float(v), 3) if np.isfinite(v) else None) for k, v in s.drop("cells").items()}


def barrier_diagnostic(pop, with_water, without_water) -> dict:
    """What the permanent-water barrier changes in the walking baseline."""
    p = pop.astype("float64")
    both = np.isfinite(with_water.minutes) & np.isfinite(without_water.minutes)
    mean_with = float((p * np.where(both, with_water.minutes, 0)).sum() / p[both].sum())
    mean_without = float((p * np.where(both, without_water.minutes, 0)).sum() / p[both].sum())
    increase = np.where(both, with_water.minutes - without_water.minutes, 0)
    only_with = np.isnan(with_water.minutes) & np.isfinite(without_water.minutes)
    return {
        "population_nearest_site_changes": round(float(p[with_water.nearest != without_water.nearest].sum())),
        "mean_min_with_barrier": round(mean_with, 3),
        "mean_min_without_barrier": round(mean_without, 3),
        "mean_min_change": round(mean_with - mean_without, 3),
        "population_unreachable_only_with_barrier": round(float(p[only_with].sum())),
        "population_time_up_more_than_30_min": round(float(p[increase > 30].sum())),
        "max_increase_min": round(float(increase.max()), 2),
        "note": "means over population reached in both runs; the run with the barrier is the official baseline",
    }


def sanity_checks(static, fac, sites, results, dropped, thresholds, long_trip_min, matches_county_id) -> dict:
    pop = static["population"]
    inside = static["county_id"] > 0
    total = float(pop.sum(dtype="float64"))
    checks = {
        "population_on_grid": round(total),
        "population_inside_counties": round(float(pop[inside].sum(dtype="float64"))),
        "reference_population_2020": REFERENCE_POPULATION,
        "population_on_grid_vs_reference_pct": round(100 * (total / REFERENCE_POPULATION - 1), 3),
        "facilities": len(fac),
        "sites": len(sites),
        "sites_dropped": [int(i) for i in dropped],
        "sites_north_of_10N": int((~sites["has_flood_data"]).sum()),
        "facilities_north_of_10N": int((~fac["has_flood_data"]).sum()),
        "admin2_raster_matches_static_county_id": matches_county_id,
    }
    for mode, res in results.items():
        unreached = np.isnan(res.minutes)
        long = res.minutes > long_trip_min
        checks[mode] = {
            "national": national_summary(pop, res.minutes, inside, thresholds),
            "unreached_cells_inside_south_sudan": int((unreached & inside).sum()),
            "unreached_population": round(float(pop[unreached].sum(dtype="float64"))),
            "cells_over_long_trip_min_inside_south_sudan": int((long & inside).sum()),
            "population_over_long_trip_min": round(float(pop[long].sum(dtype="float64"))),
            "max_min_inside_south_sudan": round(float(np.nanmax(np.where(inside, res.minutes, np.nan))), 1),
        }
    return checks


def write_raster(path, array, grid: Grid, nodata, description: str):
    profile = {**raster_profile(grid, 1, array.dtype.name, nodata), "compress": "lzw"}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)
        dst.set_band_description(1, description)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def peak_memory_bytes() -> int:
    info = psutil.Process().memory_info()
    return getattr(info, "peak_wset", None) or info.rss   # peak_wset only exists on Windows


def main():
    started = time.perf_counter()
    cfg = load_config()
    bcfg, tcfg = cfg["baseline"], cfg["travel"]
    out_dir = ROOT / bcfg["output_dir"]
    (out_dir / "internal").mkdir(parents=True, exist_ok=True)
    thresholds = sorted(set(bcfg["thresholds_min"]) | {tcfg["max_time_min"]})

    grid = grid_from_raster(data_path(cfg, "static_stack"))
    if not same_grid(grid, make_grid(cfg)):
        raise ValueError("static_250m.tif is not on the grid from config.yaml, rebuild it")
    if not same_grid(grid, grid_from_raster(data_path(cfg, "permanent_water"))):
        raise ValueError("permanent_water_250m.tif is not on the same grid as the static stack")
    print(f"grid: {grid.width} x {grid.height} cells of {grid.cell_size:g} m, {grid.crs}")

    static = load_static(cfg)
    water = load_permanent_water(cfg)
    pop = static["population"]
    inside = static["county_id"] > 0
    fac = load_facilities(cfg)
    sites = build_sites(fac, grid)
    sources = sites[["row", "col"]].to_numpy()
    print(f"facilities: {len(fac)} -> sites: {len(sites)} ({len(fac) - len(sites)} share a cell); "
          f"{int((~sites['has_flood_data']).sum())} sites north of 10N")

    results, timings = {}, {}
    for mode in bcfg["modes"]:
        t0 = time.perf_counter()
        res = travel_time(static[f"friction_{mode}"], sources, blocked=water, cell_size_m=grid.cell_size)
        timings[mode] = round(time.perf_counter() - t0, 1)
        results[mode] = res
        print(f"{mode}: solved in {timings[mode]} s, {len(res.dropped)} sites dropped, "
              f"unreached population {float(pop[np.isnan(res.minutes)].sum(dtype='float64')):,.0f}")
    dropped = sorted(set().union(*(res.dropped.tolist() for res in results.values())))

    # walking again without the water barrier, to see what the barrier does
    t0 = time.perf_counter()
    no_water = travel_time(static["friction_walking"], sources, blocked=None, cell_size_m=grid.cell_size)
    timings["walking_without_barrier"] = round(time.perf_counter() - t0, 1)
    diagnostic = barrier_diagnostic(pop, results["walking"], no_water)
    del no_water
    print(f"barrier diagnostic (walking): {diagnostic}")

    for mode, res in results.items():
        write_raster(out_dir / f"travel_time_{mode}_min.tif", res.minutes, grid, np.nan, f"travel_time_{mode}_min")
        write_raster(out_dir / f"nearest_site_{mode}.tif", res.nearest, grid, -1, f"nearest_site_{mode}")

    admin, empty_units, site_codes, matches_county_id = admin_table(cfg, grid, static, sites, results, thresholds)

    for key, codes in site_codes.items():
        sites[key] = codes
    for mode, res in results.items():
        served = res.nearest >= 0
        sites[f"population_served_{mode}"] = np.bincount(
            res.nearest[served], weights=pop[served].astype("float64"), minlength=len(sites))
        cells = np.bincount(res.nearest[served & inside], minlength=len(sites))   # area inside South Sudan only
        sites[f"catchment_km2_{mode}"] = cells * (grid.cell_size / 1000) ** 2
    sites["dropped"] = sites["site_id"].isin(dropped)
    sites.to_parquet(out_dir / "internal" / "sites.parquet", index=False)

    admin["pct_within_max_time"] = admin[f"pct_within_{tcfg['max_time_min']}"]
    admin.round(4).to_csv(out_dir / "baseline_admin.csv", index=False)

    checks = sanity_checks(static, fac, sites, results, dropped, thresholds, bcfg["long_trip_min"],
                           matches_county_id)

    inputs = {key: data_path(cfg, key) for key in ["static_stack", "permanent_water", "facilities_table", "admin2"]}
    inputs |= {level: ROOT / bcfg[level] for level in ["admin1", "admin3"]}
    inputs["config"] = ROOT / "config.yaml"
    packages = ["numpy", "scipy", "scikit-image", "rasterio", "geopandas", "pandas", "pyproj", "shapely"]
    metadata = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "grid": {"crs": grid.crs, "transform": list(grid.transform)[:6], "shape": list(grid.shape),
                 "cell_size_m": grid.cell_size,
                 "convention": "row 0 = north edge; x = transform[2] + (col + 0.5) * cell, y = transform[5] - (row + 0.5) * cell"},
        "units": {"travel_time": "minutes", "nearest_site": "site_id (row index of internal/sites.parquet)",
                  "population": "people (WorldPop, summed to 250 m)", "catchment": "km2 inside South Sudan",
                  "pct_within_*": "percent of unit population (unreached counts as not within)",
                  "mean/median/p90_min": "population-weighted, over reached population only"},
        "nodata": {"travel_time": "NaN", "nearest_site": -1},
        "cost_model": "8-connected, step cost = mean friction of the two cells x step length (sqrt 2 diagonal), "
                      "skimage.graph.MCP_Geometric; permanent water impassable; source cells always passable",
        "inputs": {k: {"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for k, p in inputs.items()},
        "config_used": {"grid": cfg["grid"], "population_year": cfg["population"]["year"],
                        "travel": tcfg, "baseline": bcfg, "permanent_water": cfg["permanent_water"]},
        "versions": {"python": platform.python_version(), **{p: version(p) for p in packages}},
        "barrier_diagnostic_walking": diagnostic,
        "sanity": checks,
        "admin_units_without_cells": empty_units,
        "runtime": {"solve_seconds": timings, "total_seconds": round(time.perf_counter() - started, 1),
                    "peak_memory_gb": round(peak_memory_bytes() / 1e9, 2)},
    }
    with open(out_dir / "baseline_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(json.dumps({"sanity": checks, "runtime": metadata["runtime"]}, indent=2, default=str))
    print(f"wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
