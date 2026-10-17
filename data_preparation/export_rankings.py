"""
Export the facility rankings the dashboard computes (get_priorities, then rank_sites) but never saves.

For each calendar month this makes the same calls as dashboard.py at its default settings: walking,
threshold routing.max_flood_likelihood from config.yaml (0: a cell is flood-prone if it flooded in that
month in any year since 2003). Every flooded site is ranked under both criteria, by calling rank_sites
with top_n = the number of flooded sites. The calculation is described in README_prioritization.md.

Run from the repository root, one process at a time (each walking solve takes about 15 s and a run
peaks at about 3 GB):

    python -m data_preparation.export_rankings --months 10
    python -m data_preparation.export_rankings --months 11 1

Outputs in EDA/outputs/, per month:
    facility_rankings_month{MM}.csv    one row per flooded site (facilities sharing a 250 m cell), both ranks
    facility_rankings_month{MM}.json   settings, timings, peak memory, baseline check, top-5 and top-10 overlap
"""
import argparse
import gc
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import psutil
import rasterio

from data_preparation.build_baseline import build_sites
from data_preparation.grid import ROOT, load_config, make_grid
from data_preparation.loading import load_facilities, load_flood_likelihood, load_permanent_water, load_static
from data_preparation.prioritization import CRITERIA, flooded_sites, rank_sites, site_impacts
from data_preparation.routing import blocked_mask
from data_preparation.travel import travel_time

OUT_DIR = ROOT / "EDA" / "outputs"
OVERLAP_K = (5, 10)
EXPECTED_FLOODED_SITES = {10: 21, 11: 34}   # README_prioritization.md, threshold 0
SORT_COLUMNS = {"most_saved": ["catchment_pop"], "worst_off": ["share_cut_off", "extra_min"]}   # what rank_sites sorts on


def peak_memory_gb() -> dict[str, float]:
    """
    Peak physical memory (working set) and peak committed memory of this process so far, in GB.
    Under memory pressure Windows trims the working set, so the committed peak is the better measure
    of what the run needs.
    """
    info = psutil.Process().memory_info()
    return {"peak_working_set_gb": round(getattr(info, "peak_wset", info.rss) / 1e9, 2),
            "peak_commit_gb": round(getattr(info, "peak_pagefile", 0) / 1e9, 2)}


def check_baseline(base, sites: pd.DataFrame, cfg: dict) -> dict:
    """Compare this run's walking baseline with the files build_baseline saved."""
    out_dir = ROOT / cfg["baseline"]["output_dir"]
    saved_sites = pd.read_parquet(out_dir / "internal" / "sites.parquet", columns=["site_id", "row", "col"])
    same_sites = len(saved_sites) == len(sites) and all(
        np.array_equal(saved_sites[c].to_numpy(), sites[c].to_numpy()) for c in ["site_id", "row", "col"])

    with rasterio.open(out_dir / "travel_time_walking_min.tif") as src:
        saved = src.read(1)
    same_unreached = bool(np.array_equal(np.isnan(saved), np.isnan(base.minutes)))
    both = ~np.isnan(saved) & ~np.isnan(base.minutes)
    max_diff = float(np.abs(saved[both] - base.minutes[both]).max()) if both.any() else 0.0
    minutes_identical = bool(np.array_equal(saved, base.minutes, equal_nan=True))
    del saved, both

    with rasterio.open(out_dir / "nearest_site_walking.tif") as src:
        saved = src.read(1)
    nearest_differ = int((saved != base.nearest).sum())
    del saved

    return {"sites_match_sites_parquet": bool(same_sites),
            "minutes_identical": minutes_identical,
            "minutes_same_unreached_cells": same_unreached,
            "minutes_max_abs_diff": max_diff,
            "nearest_site_cells_differing": nearest_differ,
            "match": bool(same_sites and minutes_identical and nearest_differ == 0)}


def rank_month(month, cfg, grid, static, water, facilities, sites, sites_rc, base) -> pd.DataFrame:
    """One row per flooded site with its impacts and its rank under each criterion (get_priorities + rank_sites)."""
    threshold = cfg["routing"]["max_flood_likelihood"]
    likelihood = load_flood_likelihood(month, cfg)
    flooded = flooded_sites(sites_rc, likelihood, threshold)
    flood = travel_time(static["friction_walking"], sites_rc[~flooded],
                        blocked=blocked_mask(water, likelihood, threshold), cell_size_m=grid.cell_size)
    del likelihood
    impacts = site_impacts(static["population"], base.nearest, base.minutes, flood.minutes, len(sites),
                           cfg["travel"]["max_time_min"])
    del flood
    priorities = sites.assign(flooded=flooded).join(impacts)

    n = int(flooded.sum())
    ranks = {}
    for criterion in CRITERIA:
        ranked = rank_sites(priorities, priorities["flooded"], criterion, top_n=n)   # all flooded sites
        if len(ranked) != n:
            raise RuntimeError(f"rank_sites returned {len(ranked)} of {n} flooded sites for {criterion}")
        ranks[criterion] = pd.Series(np.arange(1, n + 1), index=ranked["site_id"].to_numpy())

    top_n = cfg["priorities"]["top_n"]
    fac = facilities.set_index("FID")
    t = priorities[priorities["flooded"]]
    table = pd.DataFrame({
        "site_id": t["site_id"],
        "facility_id": t["rep_facility_id"],          # representative: highest level in the cell, then lowest FID
        "facility_name": t["rep_facility_name"],
        "facility_type": t["rep_facility_type"],
        "n_facilities_in_cell": t["n_facilities"],
        "all_facility_ids": t["facility_ids"].map(lambda ids: ";".join(map(str, ids))),
        "county": t["rep_facility_id"].map(fac["adm2_name"]),
        "state": t["rep_facility_id"].map(fac["adm1_name"]),
        "lon": t["lon"].round(5),
        "lat": t["lat"].round(5),
        "catchment_pop": t["catchment_pop"],
        "share_cut_off": t["share_cut_off"],
        "extra_min": t["extra_min"],
    })
    for criterion in CRITERIA:
        table[f"rank_{criterion}"] = table["site_id"].map(ranks[criterion])
        table[f"in_top{top_n}_{criterion}"] = table[f"rank_{criterion}"] <= top_n
    return table.sort_values(f"rank_{CRITERIA[0]}").reset_index(drop=True)


def tied_at_cut(table: pd.DataFrame, criterion: str, k: int) -> bool:
    """True when the sites ranked k and k+1 have identical sort values, so the top-k cut depends on tie order."""
    if k >= len(table):
        return False
    by_rank = table.set_index(f"rank_{criterion}")[SORT_COLUMNS[criterion]]
    return bool(by_rank.loc[k].equals(by_rank.loc[k + 1]))


def overlap(table: pd.DataFrame, k: int) -> dict:
    """Top-k under most_saved vs worst_off: how many are in both, and which leave or enter when switching."""
    k = min(k, len(table))
    top = {c: set(table.loc[table[f"rank_{c}"] <= k, "site_id"]) for c in CRITERIA}
    first, second = CRITERIA

    def named(ids, criterion):
        rows = table[table["site_id"].isin(ids)].sort_values(f"rank_{criterion}")
        return [f"{r.facility_name} ({r.county}, site {r.site_id})" for r in rows.itertuples()]

    return {"k": k,
            "in_both": len(top[first] & top[second]),
            f"leave_when_switching_{first}_to_{second}": named(top[first] - top[second], first),
            f"enter_when_switching_{first}_to_{second}": named(top[second] - top[first], second),
            "cut_tied": {c: tied_at_cut(table, c, k) for c in CRITERIA}}


def main():
    parser = argparse.ArgumentParser(description="Export the dashboard's facility rankings, both criteria, per calendar month.")
    parser.add_argument("--months", type=int, nargs="+", default=[10, 11, 1], help="calendar months, run in this order")
    parser.add_argument("--max-peak-gb", type=float, default=3.5,
                        help="stop before the next month when peak memory is above this")
    args = parser.parse_args()

    started = time.perf_counter()
    cfg = load_config()
    timings = {}

    # get_base_pipeline and get_sites in dashboard.py
    t0 = time.perf_counter()
    grid = make_grid(cfg)
    static = load_static(cfg)
    water = load_permanent_water(cfg)
    facilities = load_facilities(cfg)
    sites = build_sites(facilities, grid)
    sites_rc = sites[["row", "col"]].to_numpy()
    timings["load_s"] = round(time.perf_counter() - t0, 1)
    print(f"loaded in {timings['load_s']} s: {len(facilities)} facilities, {len(sites)} sites", flush=True)

    # get_baseline_walking in dashboard.py
    t0 = time.perf_counter()
    base = travel_time(static["friction_walking"], sites_rc, blocked=water, cell_size_m=grid.cell_size)
    timings["baseline_solve_s"] = round(time.perf_counter() - t0, 1)
    print(f"baseline walking solve: {timings['baseline_solve_s']} s, peak {peak_memory_gb()}", flush=True)

    t0 = time.perf_counter()
    check = check_baseline(base, sites, cfg)
    timings["baseline_check_s"] = round(time.perf_counter() - t0, 1)
    print(f"baseline check: {check}", flush=True)
    gc.collect()

    for i, month in enumerate(args.months):
        t0 = time.perf_counter()
        table = rank_month(month, cfg, grid, static, water, facilities, sites, sites_rc, base)
        month_s = round(time.perf_counter() - t0, 1)
        memory = peak_memory_gb()
        n_sites, n_fac = len(table), int(table["n_facilities_in_cell"].sum())
        expected = EXPECTED_FLOODED_SITES.get(month)

        summary = {
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "calendar_month": month,
            "settings": {"mode": "walking", "threshold": cfg["routing"]["max_flood_likelihood"],
                         "max_time_min": cfg["travel"]["max_time_min"], "top_n": cfg["priorities"]["top_n"],
                         "ranked": "all flooded sites (rank_sites with top_n = number of flooded sites)"},
            "flooded_sites": n_sites,
            "flooded_facilities": n_fac,
            "expected_flooded_sites_readme": expected,
            "baseline_check": check,
            "timings_s": {**timings, "month_s": month_s, "total_so_far_s": round(time.perf_counter() - started, 1)},
            "peak_memory_so_far": memory,
            "overlap": {f"top{k}": overlap(table, k) for k in OVERLAP_K},
        }
        stem = OUT_DIR / f"facility_rankings_month{month:02d}"
        table.to_csv(stem.with_suffix(".csv"), index=False)
        stem.with_suffix(".json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        flag = "" if expected in (None, n_sites) else f"  <-- README expects {expected}"
        print(f"month {month}: {n_sites} flooded sites ({n_fac} facilities){flag}, {month_s} s, peak {memory} "
              f"-> {stem.with_suffix('.csv').relative_to(ROOT)}", flush=True)
        del table
        gc.collect()

        if max(memory.values()) > args.max_peak_gb and i < len(args.months) - 1:
            print(f"peak memory {max(memory.values())} GB is above {args.max_peak_gb} GB: "
                  f"stopping before month {args.months[i + 1]}", flush=True)
            break


if __name__ == "__main__":
    main()
