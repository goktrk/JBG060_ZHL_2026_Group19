"""
Precompute which closed facilities to protect first, for every flood scenario in config.yaml.

Per scenario: take the closed sites and the flood-scenario travel times from build_access_loss,
reopen each closed site on its own and count who gets access back (see prioritization.py and
README_prioritization.md). The dashboard only reads the results.

Outputs in priorities.output_dir (data/processed/priorities/), per scenario id:
    priorities_<id>.parquet   one row per closed site: both rankings (greedy order) and what each adds
    priorities_<id>.json      totals (people losing access, flooded in place, not covered) and settings

Chain: build_baseline -> build_access_loss -> build_priorities. Needs both earlier steps.
Run from the repository root:  python -m data_preparation.build_priorities [--scenario 2020-11]
    --months-file data/processed/evaluation/peak_months.csv   # other months (needs build_access_loss on them)
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from data_preparation.build_access_loss import months_from_file, settings_used
from data_preparation.grid import ROOT, data_path, load_config, make_grid
from data_preparation.loading import (load_access_loss, load_baseline, load_permanent_water,
                                      load_scenario_travel, load_sites, load_static)
from data_preparation.prioritization import greedy_order, regained_cells
from data_preparation.scenarios import scenario_flood


def ranking_columns(order: pd.DataFrame, name: str, lost_pop: float) -> pd.DataFrame:
    """The greedy order of one ranking as columns named after it, with the people not yet covered."""
    out = order.set_index("site_id")
    out["not_covered"] = lost_pop - out["cumulative"]
    return out.add_suffix(f"_{name}")


def check_access_loss(summary: dict, cfg: dict, mode: str):
    """Stop if the access loss was built with other settings than config.yaml has now."""
    if summary["settings"] != settings_used(cfg) or mode not in summary["modes"]:
        raise SystemExit(f"access loss for {summary['scenario']['id']} was built with other settings or without "
                         f"mode {mode!r}; rerun: python -m data_preparation.build_access_loss")


def build_scenario(scenario: dict, cfg, grid, static, water, sites, base_minutes) -> tuple[pd.DataFrame, dict]:
    started = time.perf_counter()
    pcfg, tcfg = cfg["priorities"], cfg["travel"]
    mode = pcfg["mode"]
    friction, pop = static[f"friction_{mode}"], static["population"]
    sites_rc = sites[["row", "col"]].to_numpy()

    access, _ = load_access_loss(scenario["id"], cfg)
    check_access_loss(access, cfg, mode)
    flood_minutes, _ = load_scenario_travel(scenario["id"], mode, cfg)
    flooded, _ = scenario_flood(scenario["id"], cfg)
    blocked = water | flooded
    candidates = np.array(access["closed_site_ids"], dtype=np.int64)

    limits = {"most_saved": tcfg["max_time_min"], "worst_off": pcfg["worst_off_min"]}
    table = sites.loc[candidates, ["site_id", "row", "col", "lon", "lat", "n_facilities", "facility_ids",
                                   "rep_facility_id", "rep_facility_name", "rep_facility_type"]].set_index("site_id")
    table["own_cell_flooded"] = flooded[table["row"], table["col"]]
    totals = {}
    for name, limit in limits.items():
        regained, lost = regained_cells(pop, base_minutes, flood_minutes, flooded, friction, blocked,
                                        sites_rc, candidates, limit, grid.cell_size)
        lost_pop = float(pop[lost].sum(dtype="float64"))
        table = table.join(ranking_columns(greedy_order(regained, pop), name, lost_pop))
        totals[name] = {"limit_min": limit, "people_losing_access": round(lost_pop)}

    county = static["county_id"][table["row"], table["col"]]
    names = pd.read_csv(data_path(cfg, "counties_table")).set_index("county_id")
    table["adm2_name"] = names["adm2_name"].reindex(county).to_numpy()
    table["adm1_name"] = names["adm1_name"].reindex(county).to_numpy()

    top_n = pcfg["top_n"]
    for name, info in totals.items():
        top = table.sort_values(f"rank_{name}").head(top_n)
        covered = float(top[f"cumulative_{name}"].max()) if len(top) else 0.0
        info |= {"brought_back_by_top_n": round(covered),
                 "not_covered_by_top_n": round(info["people_losing_access"] - covered)}
    summary = {
        "scenario": scenario,
        "flooded_cells_inside_south_sudan": int((flooded & (static["county_id"] > 0)).sum()),
        "people_living_in_flooded_cells": access["living_in_flooded_cells"],
        "sites": len(sites),
        "sites_closed": access["sites_closed"],
        "sites_without_flood_data": int((~sites["has_flood_data"]).sum()),
        "top_n": top_n,
        "rankings": totals,
        "settings": {**settings_used(cfg), "mode": mode},
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }
    return table.reset_index(), summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenario", help="only this scenario id (default: all in config.yaml)")
    parser.add_argument("--months-file", help="csv with an `id` column: run these months (dashboard scenarios skipped)")
    args = parser.parse_args()

    cfg = load_config()
    grid = make_grid(cfg)
    static, water = load_static(cfg), load_permanent_water(cfg)
    sites = load_sites(cfg)
    base_minutes, _, _ = load_baseline(cfg["priorities"]["mode"], cfg)
    out_dir = ROOT / cfg["priorities"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.months_file:
        scenarios = months_from_file(ROOT / args.months_file, cfg)
    else:
        scenarios = [s for s in cfg["scenarios"] if args.scenario in (None, s["id"])]
    if not scenarios and not args.months_file:
        raise SystemExit(f"scenario {args.scenario} is not in config.yaml")
    for scenario in scenarios:
        table, summary = build_scenario(scenario, cfg, grid, static, water, sites, base_minutes)
        table.to_parquet(out_dir / f"priorities_{scenario['id']}.parquet", index=False)
        with open(out_dir / f"priorities_{scenario['id']}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        r = summary["rankings"]
        print(f"{scenario['label']}: {summary['sites_closed']} sites closed; most saved: "
              f"{r['most_saved']['people_losing_access']:,} lose 2 h access, top {summary['top_n']} bring back "
              f"{r['most_saved']['brought_back_by_top_n']:,}; worst off: {r['worst_off']['people_losing_access']:,} "
              f"pushed beyond 4 h, top {summary['top_n']} bring back {r['worst_off']['brought_back_by_top_n']:,} "
              f"({summary['runtime_seconds']} s)")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
