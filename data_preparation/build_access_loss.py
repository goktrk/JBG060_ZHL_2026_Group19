"""
Access loss per flood scenario: who is flood-affected, and who on dry land is cut off from care.

Per scenario (the `scenarios` in config.yaml) and travel mode, using the shared rules in scenarios.py:
cells flooded on at least flood.min_days_flooded days are blocked, sites with a flooded cell within
facilities.flood_buffer_m are closed, and travel times are compared with build_baseline. Everyone is
counted in at most one group:

- living in flooded cells: their own cell is flooded (flood-affected, likely displaced)
- cut off within T min:    on dry land, within T of a facility without floods, beyond T or unable to
                           reach any with floods (T = access_loss.thresholds_min, 60 and 120)
- no facility reachable:   the part of the cut-off people who can't reach any facility at all
- without flood data:      north of 10N, where flooding is unknown (not the same as dry)

Outputs in access_loss.output_dir (data/processed/access_loss/), per scenario id:
    access_loss_<id>.json                     national totals per mode, closed sites, settings
    access_loss_<id>_county.csv               the same per county and mode, with mean/median minutes
    flood_travel_time_<mode>_<id>.tif         minutes to the nearest open site with floods (NaN = none)
    flood_nearest_site_<mode>_<id>.tif        site_id of that site (-1 = none), for pre-positioning
With --all-months, every month of the flood record instead, as two tables (no rasters):
    access_loss_monthly_national.csv, access_loss_monthly_county.csv

Chain: build_baseline -> build_access_loss -> build_priorities. Needs build_baseline first.
Run from the repository root:  python -m data_preparation.build_access_loss [--scenario 2020-11] [--all-months]
    --months-file data/processed/evaluation/peak_months.csv --mode walking   # other months, one mode
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import rasterio

from data_preparation.build_baseline import access_stats
from data_preparation.grid import ROOT, data_path, load_config, make_grid, raster_profile
from data_preparation.loading import (load_baseline, load_flood_months, load_permanent_water, load_sites,
                                      load_static)
from data_preparation.scenarios import closed_sites, lost_cells, scenario_flood
from data_preparation.travel import travel_time


def write_raster(path, array, grid, nodata, description):
    """Lossless GeoTIFF; the floating-point predictor makes travel times about half the size of plain LZW."""
    profile = {**raster_profile(grid, 1, array.dtype.name, nodata), "compress": "deflate",
               "predictor": 3 if array.dtype.kind == "f" else 2}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)
        dst.set_band_description(1, description)


def settings_used(cfg: dict) -> dict:
    """The settings an access-loss run depends on; build_priorities refuses results built with others."""
    return {"min_days_flooded": cfg["flood"]["min_days_flooded"],
            "flood_buffer_m": cfg["facilities"]["flood_buffer_m"],
            "population_year": cfg["population"]["year"],
            "permanent_water": cfg["permanent_water"],
            "thresholds_min": cfg["access_loss"]["thresholds_min"]}


def check_baseline(cfg: dict, meta: dict, n_sites: int):
    """Stop if the baseline was built with other settings than config.yaml has now."""
    used = meta["config_used"]
    differs = [key for key, now in [("grid", cfg["grid"]), ("population_year", cfg["population"]["year"]),
                                    ("permanent_water", cfg["permanent_water"])] if used.get(key) != now]
    if differs or meta["sanity"]["sites"] != n_sites:
        raise SystemExit(f"the baseline was built with other settings ({', '.join(differs) or 'sites'}); "
                         "rerun: python -m data_preparation.build_baseline")


def group_masks(pop, base, flood, flooded, thresholds) -> dict[str, np.ndarray]:
    """One mask per group (see the module docstring)."""
    masks = {f"cut_off_{t}": lost_cells(pop, base, flood, flooded, t) for t in thresholds}
    masks["no_facility_reachable"] = ((pop > 0) & ~flooded & np.isfinite(base) & np.isnan(flood))
    return masks


def county_table(static, counties, sites, closed, flooded, no_data, results, thresholds) -> pd.DataFrame:
    """Per county and mode: the groups, closed facilities and travel times before and after."""
    pop, ids = static["population"], static["county_id"].astype(np.int64)
    n = int(counties["county_id"].max()) + 1

    def per_county(mask):
        return np.bincount(ids.ravel(), weights=(pop * mask).ravel(), minlength=n)[1:]

    closed_by_pcode = sites.loc[closed].groupby("admin2_pcode")["n_facilities"].sum()
    common = counties[["county_id", "adm2_pcode", "adm2_name", "adm1_name"]].copy()
    common["population"] = per_county(np.ones_like(pop, dtype=bool))
    common["living_in_flooded_cells"] = per_county(flooded)
    common["without_flood_data"] = per_county(no_data)
    common["closed_facilities"] = common["adm2_pcode"].map(closed_by_pcode).fillna(0).astype(int)

    rows = []
    for mode, (base, flood) in results.items():
        table = common.copy()
        table.insert(0, "mode", mode)
        for name, mask in group_masks(pop, base, flood, flooded, thresholds).items():
            table[name] = per_county(mask)
        for when, minutes in [("base", base), ("flood", flood)]:
            stats = access_stats(static["county_id"], n - 1, pop, minutes, thresholds).iloc[1:]
            table[f"{when}_mean_min"] = stats["mean_min"].to_numpy()
            table[f"{when}_median_min"] = stats["median_min"].to_numpy()
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def solve_scenario(scenario_id, cfg, grid, static, water, sites, baseline) -> dict:
    """Flooded cells, closed sites and, per mode, the flood-scenario travel time and nearest site_id."""
    flooded, no_data = scenario_flood(scenario_id, cfg)
    sites_rc = sites[["row", "col"]].to_numpy()
    closed = closed_sites(sites_rc, flooded, cfg["facilities"]["flood_buffer_m"], grid.cell_size)
    open_ids = np.flatnonzero(~closed)
    travel = {}
    for mode in cfg["access_loss"]["modes"]:
        res = travel_time(static[f"friction_{mode}"], sites_rc[~closed], blocked=water | flooded,
                          cell_size_m=grid.cell_size)
        nearest = np.where(res.nearest >= 0, open_ids[np.maximum(res.nearest, 0)], -1).astype(np.int32)
        travel[mode] = (baseline[mode], res.minutes, nearest)
    return {"flooded": flooded, "no_data": no_data, "closed": closed, "travel": travel}


def summarise(scenario, solved, cfg, static, sites, counties, seconds) -> tuple[dict, pd.DataFrame]:
    pop, inside = static["population"], static["county_id"] > 0
    thresholds = cfg["access_loss"]["thresholds_min"]
    results = {mode: (base, flood) for mode, (base, flood, _) in solved["travel"].items()}
    county = county_table(static, counties, sites, solved["closed"], solved["flooded"], solved["no_data"],
                          results, thresholds)
    modes = {}
    for mode, (base, flood) in results.items():
        modes[mode] = {name: round(float(pop[mask].sum(dtype="float64")))
                       for name, mask in group_masks(pop, base, flood, solved["flooded"], thresholds).items()}
    summary = {
        "scenario": scenario,
        "population": round(float(pop[inside].sum(dtype="float64"))),
        "living_in_flooded_cells": round(float(pop[solved["flooded"]].sum(dtype="float64"))),
        "without_flood_data": round(float(pop[solved["no_data"] & inside].sum(dtype="float64"))),
        "sites_closed": int(solved["closed"].sum()),
        "facilities_closed": int(sites.loc[solved["closed"], "n_facilities"].sum()),
        "closed_site_ids": np.flatnonzero(solved["closed"]).tolist(),
        "modes": modes,
        "settings": settings_used(cfg),
        "runtime_seconds": round(seconds, 1),
    }
    return summary, county


def months_in_record(cfg: dict) -> list[dict]:
    months = pd.period_range(f"{cfg['flood']['first_year']}-01", f"{cfg['flood']['last_year']}-12", freq="M")
    return [{"id": str(m), "label": m.strftime("%b %Y")} for m in months]


def months_from_file(path, cfg) -> list[dict]:
    """Scenario entries for the months in a csv with an `id` column (e.g. peak_months.csv), leaving out the
    months that are already dashboard scenarios (their results, with every mode, stay as they are)."""
    known = {s["id"] for s in cfg["scenarios"]}
    ids = pd.read_csv(path)["id"].astype(str)
    return [{"id": m, "label": f"{m} (peak month)"} for m in ids if m not in known]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenario", help="only this scenario id (default: all in config.yaml)")
    parser.add_argument("--all-months", action="store_true", help="every month of the flood record, tables only")
    parser.add_argument("--months-file", help="csv with an `id` column: run these months (dashboard scenarios skipped)")
    parser.add_argument("--mode", choices=["walking", "motorised"], help="only this travel mode")
    args = parser.parse_args()

    cfg = load_config()
    if args.mode:
        cfg["access_loss"]["modes"] = [args.mode]
    grid = make_grid(cfg)
    static, water = load_static(cfg), load_permanent_water(cfg)
    sites = load_sites(cfg)
    counties = pd.read_csv(data_path(cfg, "counties_table"))
    baseline = {}
    for mode in cfg["access_loss"]["modes"]:
        baseline[mode], _, meta = load_baseline(mode, cfg)
    check_baseline(cfg, meta, len(sites))
    out_dir = ROOT / cfg["access_loss"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all_months:
        scenarios = months_in_record(cfg)
    elif args.months_file:
        scenarios = months_from_file(ROOT / args.months_file, cfg)
    else:
        scenarios = [s for s in cfg["scenarios"] if args.scenario in (None, s["id"])]
        if not scenarios:
            raise SystemExit(f"scenario {args.scenario} is not in config.yaml")

    flagged = load_flood_months(cfg).set_index(["year", "month"])["flagged"]
    national_rows, county_rows = [], []
    for scenario in scenarios:
        t0 = time.perf_counter()
        solved = solve_scenario(scenario["id"], cfg, grid, static, water, sites, baseline)
        summary, county = summarise(scenario, solved, cfg, static, sites, counties, time.perf_counter() - t0)
        sid = scenario["id"]
        if args.all_months:
            month = pd.Period(sid, freq="M")
            is_flagged = bool(flagged.get((month.year, month.month), False))
            for mode, groups in summary["modes"].items():
                national_rows.append({"month": sid, "flagged": is_flagged, "mode": mode,
                                      "living_in_flooded_cells": summary["living_in_flooded_cells"],
                                      "sites_closed": summary["sites_closed"], **groups})
            county_rows.append(county.assign(month=sid, flagged=is_flagged))
        else:
            with open(out_dir / f"access_loss_{sid}.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            county.round(3).to_csv(out_dir / f"access_loss_{sid}_county.csv", index=False)
            for mode, (_, minutes, nearest) in solved["travel"].items():
                write_raster(out_dir / f"flood_travel_time_{mode}_{sid}.tif", minutes, grid, np.nan,
                             f"flood_travel_time_{mode}_min")
                write_raster(out_dir / f"flood_nearest_site_{mode}_{sid}.tif", nearest, grid, -1,
                             f"flood_nearest_site_{mode}")
        walk = summary["modes"].get("walking", next(iter(summary["modes"].values())))
        print(f"{scenario['label']}: {summary['sites_closed']} sites closed; "
              f"{summary['living_in_flooded_cells']:,} live in flooded cells; walking: "
              + ", ".join(f"{walk[f'cut_off_{t}']:,} cut off within {t} min" for t in cfg["access_loss"]["thresholds_min"])
              + f", {walk['no_facility_reachable']:,} with no facility reachable ({summary['runtime_seconds']} s)")

    if args.all_months:
        pd.DataFrame(national_rows).to_csv(out_dir / "access_loss_monthly_national.csv", index=False)
        pd.concat(county_rows, ignore_index=True).round(3).to_csv(out_dir / "access_loss_monthly_county.csv",
                                                                  index=False)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
