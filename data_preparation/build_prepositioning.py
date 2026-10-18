"""
Pre-positioning: which facilities receive extra people when a flood closes their neighbours or
blocks the way there, per flood scenario in config.yaml and travel mode (see prepositioning.py and
README_prepositioning.md). People go to the nearest open facility themselves; nobody is moved.

Variants: "as_is" (the flood scenario), and as a side, "top5_<ranking>": the same scenario with the
Top N facilities of build_priorities protected (reopened), for each ranking in
prepositioning.protected_rankings.

Outputs in prepositioning.output_dir (data/processed/prepositioning/), per scenario id:
    prepositioning_<id>_receiving.csv   one row per receiving facility, mode and variant
    prepositioning_<id>_flows.csv       people per (from facility, to facility, reason), mode and variant
    prepositioning_<id>.json            people per group and the settings used

Chain: build_baseline -> build_access_loss -> build_priorities -> build_prepositioning.
Run from the repository root:  python -m data_preparation.build_prepositioning [--scenario 2020-11] [--skip-protected]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from data_preparation.build_access_loss import settings_used
from data_preparation.build_priorities import check_access_loss
from data_preparation.grid import ROOT, data_path, load_config, make_grid
from data_preparation.loading import (load_access_loss, load_baseline, load_permanent_water, load_priorities,
                                      load_scenario_travel, load_sites, load_static)
from data_preparation.prepositioning import check_balance, classify, flow_table, group_totals, receiving_table
from data_preparation.scenarios import scenario_flood
from data_preparation.travel import travel_time


def protected_sites(scenario_id: str, ranking: str, cfg: dict) -> np.ndarray:
    """The Top N sites of one ranking, as the dashboard shows them (greedy order, sites adding nobody skipped)."""
    table, summary = load_priorities(scenario_id, cfg)
    if summary["settings"] != {**settings_used(cfg), "mode": cfg["priorities"]["mode"]}:
        raise SystemExit(f"priorities for {scenario_id} were built with other settings; "
                         "rerun: python -m data_preparation.build_priorities")
    ranked = table.sort_values(f"rank_{ranking}")
    return ranked.loc[ranked[f"added_{ranking}"] > 0, "site_id"].head(cfg["priorities"]["top_n"]).to_numpy()


def reopen(friction, sites_rc, closed, protected, blocked, cell_size_m):
    """Flood-scenario travel time and nearest site_id with the protected sites reopened."""
    open_ids = np.flatnonzero(~closed | np.isin(np.arange(len(closed)), protected))
    res = travel_time(friction, sites_rc[open_ids], blocked=blocked, cell_size_m=cell_size_m)
    nearest = np.where(res.nearest >= 0, open_ids[np.maximum(res.nearest, 0)], -1)
    return res.minutes, nearest


def describe(receiving: pd.DataFrame, sites: pd.DataFrame, county_names: pd.Series) -> pd.DataFrame:
    """Add the facility name, type and county to a receiving table."""
    info = sites.set_index("site_id")[["rep_facility_name", "rep_facility_type", "n_facilities", "admin2_pcode"]]
    out = receiving.join(info, on="site_id")
    out["adm2_name"] = out["admin2_pcode"].map(county_names)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenario", help="only this scenario id (default: all in config.yaml)")
    parser.add_argument("--skip-protected", action="store_true", help="leave out the protected Top N variants")
    args = parser.parse_args()

    cfg = load_config()
    grid = make_grid(cfg)
    pcfg = cfg["prepositioning"]
    limit, small = cfg["travel"]["max_time_min"], pcfg["small_delay_min"]
    static, water = load_static(cfg), load_permanent_water(cfg)
    pop = static["population"]
    sites = load_sites(cfg)
    sites_rc = sites[["row", "col"]].to_numpy()
    county_names = pd.read_csv(data_path(cfg, "counties_table")).set_index("adm2_pcode")["adm2_name"]
    modes = cfg["access_loss"]["modes"]
    baseline = {mode: load_baseline(mode, cfg)[:2] for mode in modes}
    rankings = [] if args.skip_protected else pcfg["protected_rankings"]
    out_dir = ROOT / pcfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [s for s in cfg["scenarios"] if args.scenario in (None, s["id"])]
    if not scenarios:
        raise SystemExit(f"scenario {args.scenario} is not in config.yaml")
    for scenario in scenarios:
        t0 = time.perf_counter()
        sid = scenario["id"]
        access, _ = load_access_loss(sid, cfg)
        flooded, _ = scenario_flood(sid, cfg)
        closed = np.zeros(len(sites), dtype=bool)
        closed[access["closed_site_ids"]] = True
        protected = {ranking: protected_sites(sid, ranking, cfg) for ranking in rankings}

        receiving, flows, groups_out = [], [], {}
        for mode in modes:
            check_access_loss(access, cfg, mode)
            base_minutes, base_nearest = baseline[mode]
            variants = {"as_is": (closed, *load_scenario_travel(sid, mode, cfg))}
            for ranking, top in protected.items():
                minutes, nearest = reopen(static[f"friction_{mode}"], sites_rc, closed, top, water | flooded,
                                          grid.cell_size)
                variants[f"top{cfg['priorities']['top_n']}_{ranking}"] = (closed & ~np.isin(np.arange(len(sites)), top),
                                                                           minutes, nearest)
            for variant, (closed_v, flood_minutes, flood_nearest) in variants.items():
                groups = classify(pop, base_nearest, flood_nearest, flooded, closed_v)
                check_balance(pop, base_nearest, groups, closed_v)
                f = flow_table(pop, base_nearest, base_minutes, flood_nearest, flood_minutes, groups, limit, small)
                r = describe(receiving_table(pop, base_nearest, f, len(sites)), sites, county_names)
                tag = {"mode": mode, "variant": variant}
                receiving.append(r.assign(**tag))
                flows.append(f.assign(**tag))
                totals = {k: round(v) for k, v in group_totals(pop, groups).items()}
                moved = f["people"].sum()
                totals |= {"moved_within_limit": round(float(f["people_within_limit"].sum())),
                           "moved_path_blocked_small_delay": round(float(
                               f.loc[f["reason"] == "path_blocked", "people_small_delay"].sum())),
                           "receiving_facilities": int(len(r)),
                           "moved_total": round(float(moved))}
                groups_out.setdefault(mode, {})[variant] = totals

        pd.concat(receiving, ignore_index=True).round(3).to_csv(out_dir / f"prepositioning_{sid}_receiving.csv",
                                                                index=False)
        pd.concat(flows, ignore_index=True).round(3).to_csv(out_dir / f"prepositioning_{sid}_flows.csv", index=False)
        summary = {"scenario": scenario, "sites_closed": int(closed.sum()),
                   "protected_sites": {k: v.tolist() for k, v in protected.items()},
                   "groups": groups_out,
                   "settings": {**settings_used(cfg), "limit_min": limit, "small_delay_min": small},
                   "runtime_seconds": round(time.perf_counter() - t0, 1)}
        with open(out_dir / f"prepositioning_{sid}.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        w = groups_out[modes[0]]["as_is"]
        print(f"{scenario['label']} ({modes[0]}): {w['moved_total']:,} people move to another facility "
              f"({w['moved_facility_closed']:,} facility closed, {w['moved_path_blocked']:,} path blocked, "
              f"{w['moved_path_blocked_small_delay']:,} of those < {small} min extra); "
              f"{w['moved_within_limit']:,} end up within {limit} min; {w['receiving_facilities']} facilities receive "
              f"people ({summary['runtime_seconds']} s)")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
