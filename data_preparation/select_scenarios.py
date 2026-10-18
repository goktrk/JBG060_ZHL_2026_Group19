"""
Pick the flood scenarios for the dashboard: historical months (YYYY-MM) of observed flooding,
chosen by how many people in the focus areas lose walking access to health care that month.

    step 1  flooding per month in each focus area (flooding_per_month)
    step 2  access loss per month, one travel-time solve each (solve_months)
    step 3  pick the scenarios from those numbers (select_scenarios)

The rule for step 3 is described in data_preparation/README_scenarios.md.

    table = pd.read_csv("EDA/outputs/scenario_candidates.csv")
    selected = select_scenarios(table)    # month, tier, roles and reason of each scenario

Needs build_flood_monthly and build_baseline first. Solves every month once, about 90 min.
Run from the repository root:  python -m data_preparation.select_scenarios
Only redo the picks, from the existing candidates (seconds):  python -m data_preparation.select_scenarios --select-only
"""
import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import rasterio

from data_preparation.build_baseline import grid_from_raster, same_grid, sha256
from data_preparation.build_flood_likelihood import MONTH_NAMES
from data_preparation.build_flood_monthly import NO_DATA
from data_preparation.grid import ROOT, data_path, load_config, make_grid
from data_preparation.loading import (iter_flood, load_facilities, load_flood_months, load_permanent_water,
                                      load_static)
from data_preparation.scenarios import closed_sites, lost_cells
from data_preparation.travel import travel_time

FOCUS_AREAS = {                                                       # county pcodes
    "bor_south": ["SS0303"],                                          # Bor South
    "lol_aweil": ["SS0501", "SS0502", "SS0503", "SS0504", "SS0505"],  # the Aweil counties, for the Lol river
}
AREA_NAMES = {"bor_south": "Bor South", "lol_aweil": "Aweil / Lol"}   # in reasons and labels
# severity: people newly more than this many minutes' walk from care; 120 is the limit of
# build_access_loss.py and travel.max_time_min in config.yaml
LIMIT_MIN = 120
LIMITS_MIN = (60, 120)      # step 2 computes the loss at both, scenario_candidates.csv keeps both
LOSS = f"combined_pop_newly_beyond_{LIMIT_MIN}"
MEDIUM_PERCENTILE = 75      # of the combined loss over the eligible months
MILD_PERCENTILE = 50
MAX_NO_DATA_SHARE = 0.05    # months where more of a focus area has no flood data are not picked
HARDCODED = ["2020-07", "2020-09", "2020-11", "2021-09", "2022-10"]   # the months in dashboard.py
TIER_LABELS = {"severe": "severe", **{f"worst_{area}": f"worst for {name}" for area, name in AREA_NAMES.items()},
               "medium": "medium", "mild": "mild"}
OUT_DIR = ROOT / "EDA" / "outputs"


def area_ids(counties: pd.DataFrame) -> dict[str, np.ndarray]:
    """county_ids of each focus area, of both together ("combined") and of all counties ("national")."""
    ids = {area: counties.loc[counties["adm2_pcode"].isin(pcodes), "county_id"].to_numpy()
           for area, pcodes in FOCUS_AREAS.items()}
    ids["combined"] = np.concatenate(list(ids.values()))
    ids["national"] = counties["county_id"].to_numpy()
    return ids


def per_area(county_id, ids, weights=None) -> dict[str, float]:
    """Sum of `weights` (or the number of cells) per area, added up from the per-county totals."""
    n = int(max(i.max() for i in ids.values())) + 1
    per_county = np.bincount(np.ravel(county_id), None if weights is None else np.ravel(weights), minlength=n)
    return {area: float(per_county[i].sum()) for area, i in ids.items()}


def is_flooded(days, min_days) -> np.ndarray:
    """Cells flooded on at least min_days days; 255 (no data) never counts as flooded."""
    return (days >= min_days) & (days != NO_DATA)


def flood_stats(days, county_id, pop, fac_rc, fac_county, ids, min_days, buffer_m, cell_size_m) -> dict[str, float]:
    """Step 1 for one month, per area: flooded cells and share, people on them, closed facilities
    (a flooded cell within buffer_m, the shared rule in scenarios.py), no-data share."""
    flooded = is_flooded(days, min_days)
    cells = per_area(county_id, ids)
    flooded_cells = per_area(county_id, ids, flooded)
    flooded_pop = per_area(county_id, ids, np.where(flooded, pop, 0))
    flooded_facilities = per_area(fac_county, ids, closed_sites(fac_rc, flooded, buffer_m, cell_size_m))
    no_data = per_area(county_id, ids, days == NO_DATA)
    stats = {}
    for area in ids:
        stats[f"{area}_flooded_cells"] = flooded_cells[area]
        stats[f"{area}_flooded_share"] = flooded_cells[area] / cells[area]
        stats[f"{area}_flooded_pop"] = flooded_pop[area]
        stats[f"{area}_flooded_facilities"] = flooded_facilities[area]
        stats[f"{area}_no_data_share"] = no_data[area] / cells[area]
    return stats


def flood_minutes(friction, sources, water, flooded, buffer_m, cell_size_m=250.0) -> np.ndarray:
    """Minutes to the nearest open site, with flooded cells blocked and sites closed by the shared rule
    in scenarios.py (a flooded cell within buffer_m)."""
    open_sites = ~closed_sites(sources, flooded, buffer_m, cell_size_m)
    return travel_time(friction, sources[open_sites], blocked=water | flooded, cell_size_m=cell_size_m).minutes


def access_loss(base_min, flood_min, flooded, pop, county_id, ids, limits) -> dict[str, float]:
    """Step 2 for one month, per area, with the definitions of build_access_loss.py (scenarios.lost_cells)."""
    loss = {}
    for limit in limits:
        # on dry land, within the limit without floods, but beyond it or unreachable (NaN) with floods;
        # people living in flooded cells are left out (build_access_loss counts them separately)
        newly = lost_cells(pop, base_min, flood_min, flooded, limit)
        for area, people in per_area(county_id, ids, np.where(newly, pop, 0)).items():
            loss[f"{area}_pop_newly_beyond_{limit}"] = people
    reached = np.isfinite(base_min) & np.isfinite(flood_min)
    unreached = per_area(county_id, ids, np.where(np.isnan(flood_min), pop, 0))
    pop_reached = per_area(county_id, ids, np.where(reached, pop, 0))
    extra = per_area(county_id, ids, np.where(reached, pop * (flood_min - base_min), 0))
    for area in ids:
        loss[f"{area}_pop_unreached"] = unreached[area]
        # population-weighted mean of the extra minutes, over the people reached with and without floods
        loss[f"{area}_extra_min"] = extra[area] / pop_reached[area] if pop_reached[area] > 0 else np.nan
    return loss


def largest_first(table, column) -> pd.DataFrame:
    """Rows by `column`, largest first; ties go to the more recent month."""
    return table.sort_values([column, "month"], ascending=False)


def closest_first(table, column, target) -> pd.DataFrame:
    """Rows by how close `column` is to `target`, closest first; ties go to the more recent month."""
    distance = (table[column] - target).abs()
    return table.assign(distance=distance).sort_values(["distance", "month"], ascending=[True, False])


def people(row, column=LOSS) -> str:
    """The loss in `column` as text."""
    return f"{row[column]:,.0f} people newly beyond {LIMIT_MIN} min"


def add_worst(eligible, picks, column, tier, what):
    """Add the month with the largest `column`; if it is already picked, add the tier to that pick instead."""
    order = largest_first(eligible[eligible[column] > 0], column)
    if order.empty:
        return
    worst = order.iloc[0]
    for pick in picks:
        if pick["month"] == worst["month"]:
            pick["roles"].append(tier)
            pick["reason"] += f"; also the {what} ({people(worst, column)})"
            return
    picks.append({"month": worst["month"], "tier": tier, "roles": [tier],
                  "reason": f"{what}: {people(worst, column)}"})


def add_closest(eligible, picks, tier, q):
    """Add the month closest to the q-th percentile of the combined loss, in an unused calendar month if any."""
    target = float(np.percentile(eligible[LOSS], q))
    taken = [pick["month"] for pick in picks]
    free = closest_first(eligible[~eligible["month"].isin(taken)], LOSS, target)
    if free.empty:
        return
    # the dashboard's Top 5 and routes only use the calendar month, so a second October adds little
    distinct = free[~free["calendar_month"].isin([int(month[5:]) for month in taken])]
    row = (distinct if len(distinct) else free).iloc[0]
    note = "" if len(distinct) else "; no unused calendar month left"
    picks.append({"month": row["month"], "tier": tier, "roles": [tier],
                  "reason": f"closest to the {q}th percentile of the non-zero combined loss over {len(eligible)} "
                            f"eligible months ({target:,.0f}): {people(row)}{note}"})


def select_scenarios(table) -> pd.DataFrame:
    """Step 3: the scenario months as rows of month, tier, roles (all tiers the month fills) and reason."""
    ok = table["solved"] & (table[LOSS] > 0) & ~table["gap_flagged"]
    for area in FOCUS_AREAS:
        ok &= table[f"{area}_no_data_share"] <= MAX_NO_DATA_SHARE
    eligible = table[ok]
    picks = []
    if len(eligible):
        add_worst(eligible, picks, LOSS, "severe", "largest combined loss")
        for area, name in AREA_NAMES.items():
            add_worst(eligible, picks, f"{area}_pop_newly_beyond_{LIMIT_MIN}", f"worst_{area}",
                      f"worst month for {name}")
        add_closest(eligible, picks, "medium", MEDIUM_PERCENTILE)
        add_closest(eligible, picks, "mild", MILD_PERCENTILE)
    return pd.DataFrame(picks, columns=["month", "tier", "roles", "reason"])


def hardcoded_ranks(table, selected) -> pd.DataFrame:
    """Where the months hardcoded in dashboard.py rank by combined loss among the solved months (1 = worst).
    Months that are not in the table are left out."""
    solved = table[table["solved"]]
    nonzero = solved.loc[solved[LOSS] > 0, LOSS]
    tiers = dict(zip(selected["month"], selected["tier"]))
    rows = []
    for month in HARDCODED:
        found = solved.loc[solved["month"] == month, LOSS]
        if found.empty:
            continue
        value = float(found.iloc[0])
        percentile = round(float((nonzero <= value).mean() * 100), 1) if value > 0 else None
        rows.append({"month": month, LOSS: value, "rank": int((solved[LOSS] > value).sum()) + 1,
                     "of_solved": len(solved), "percentile_of_nonzero": percentile,
                     "selected": month in tiers, "tier": tiers.get(month)})
    columns = ["month", LOSS, "rank", "of_solved", "percentile_of_nonzero", "selected", "tier"]
    return pd.DataFrame(rows, columns=columns)


def month_label(month) -> str:
    """'2022-10' -> 'Oct 2022'."""
    return f"{MONTH_NAMES[int(month[5:]) - 1]} {month[:4]}"


def scenarios_block(selected, min_days) -> str:
    """The scenarios: block for config.yaml."""
    lines = ["# Flood scenarios for the dashboard, from: python -m data_preparation.select_scenarios",
             f"# flooded = at least flood.min_days_flooded ({min_days}) days that month and not 255 (band \"all\");",
             "# flooded cells are blocked (blocked = water | flooded) and sites on them are closed",
             "scenarios:"]
    for row in selected.itertuples():
        roles = ", ".join(TIER_LABELS[role] for role in row.roles)
        lines += [f'  - id: "{row.month}"',
                  f'    label: "{month_label(row.month)} ({roles})"',
                  f"    tier: {row.tier}"]
    return "\n".join(lines) + "\n"


def check_baseline(cfg) -> dict[str, str]:
    """Stop if the baseline is off the grid or was built from other input files; return the file hashes."""
    base_dir = ROOT / cfg["baseline"]["output_dir"]
    raster = base_dir / "travel_time_walking_min.tif"
    if not same_grid(grid_from_raster(raster), make_grid(cfg)):
        raise ValueError(f"{raster} is not on the grid in config.yaml; rerun build_baseline")
    with open(base_dir / "baseline_metadata.json", encoding="utf-8") as f:
        built_from = json.load(f)["inputs"]
    hashes = {key: sha256(data_path(cfg, key)) for key in ["static_stack", "permanent_water", "facilities_table"]}
    for key, value in hashes.items():
        if value != built_from[key]["sha256"]:
            raise ValueError(f"{key} has changed since the baseline was built; rerun build_baseline")
    hashes["travel_time_walking_min"] = sha256(raster)
    return hashes


def load_inputs(cfg) -> dict:
    """Everything the steps need, all on the analysis grid."""
    base_dir = ROOT / cfg["baseline"]["output_dir"]
    static = load_static(cfg)
    with rasterio.open(base_dir / "travel_time_walking_min.tif") as src:
        base_min = src.read(1)
    facilities = load_facilities(cfg)
    counties = pd.read_csv(data_path(cfg, "counties_table"))
    return {
        "friction": static["friction_walking"], "pop": static["population"], "county_id": static["county_id"],
        "water": load_permanent_water(cfg),
        "base_min": base_min,   # walking minutes without floods
        "sources": pd.read_parquet(base_dir / "internal" / "sites.parquet")[["row", "col"]].to_numpy(),
        "fac_rc": facilities[["row", "col"]].to_numpy(), "fac_county": facilities["county_id"].to_numpy(),
        "ids": area_ids(counties),
    }


def flooding_per_month(cfg, inp) -> pd.DataFrame:
    """Step 1: flood_stats for every month, with the gap flag from flood_months.csv."""
    min_days = cfg["flood"]["min_days_flooded"]
    first, last = cfg["flood"]["first_year"], cfg["flood"]["last_year"]
    buffer_m, cell = cfg["facilities"]["flood_buffer_m"], cfg["grid"]["cell_size_m"]
    rows = [{"month": str(month), **flood_stats(days, inp["county_id"], inp["pop"], inp["fac_rc"],
                                                inp["fac_county"], inp["ids"], min_days, buffer_m, cell)}
            for month, days in iter_flood(f"{first}-01", f"{last}-12", cfg)]
    table = pd.DataFrame(rows)
    table.insert(1, "year", table["month"].str[:4].astype(int))
    table.insert(2, "calendar_month", table["month"].str[5:].astype(int))
    gaps = load_flood_months(cfg).rename(columns={"month": "calendar_month", "flagged": "gap_flagged"})
    table = table.merge(gaps, on=["year", "calendar_month"], how="left")
    if table["gap_flagged"].isna().any():
        raise ValueError("flood_months.csv is missing months; rerun build_flood_monthly")
    table["gap_flagged"] = table["gap_flagged"].astype(bool)
    table["hardcoded"] = table["month"].isin(HARDCODED)
    return table


def solve_month(cfg, inp, month) -> dict[str, float]:
    """Step 2 for one month: block its flooded cells, close the sites on them, compare with the baseline."""
    _, days = next(iter_flood(month, cfg=cfg))
    flooded = is_flooded(days, cfg["flood"]["min_days_flooded"])
    minutes = flood_minutes(inp["friction"], inp["sources"], inp["water"], flooded,
                            cfg["facilities"]["flood_buffer_m"], cfg["grid"]["cell_size_m"])
    return access_loss(inp["base_min"], minutes, flooded, inp["pop"], inp["county_id"], inp["ids"], LIMITS_MIN)


def solve_months(cfg, inp, table) -> tuple[dict, dict]:
    """Step 2 for every month, starting with the most flooded one to time a single solve."""
    timed = largest_first(table, "national_flooded_share")["month"].iloc[0]
    start = time.perf_counter()
    losses = {timed: solve_month(cfg, inp, timed)}
    one_solve = round(time.perf_counter() - start, 1)
    print(f"step 2: one solve took {one_solve} s ({timed}, the most flooded month); solving all {len(table)} months",
          flush=True)
    for i, month in enumerate(table["month"], start=1):
        if month not in losses:
            losses[month] = solve_month(cfg, inp, month)
        if i % 12 == 0 or i == len(table):
            print(f"  {i} of {len(table)} months done, {time.perf_counter() - start:.0f} s", flush=True)
    timing = {"timed_month": timed, "one_solve_seconds": one_solve, "months_solved": len(losses),
              "step2_seconds": round(time.perf_counter() - start, 1)}
    return losses, timing


def candidates_table(table, losses, areas) -> pd.DataFrame:
    """Steps 1 and 2 in one table: one row per month, the columns grouped per area."""
    losses = pd.DataFrame.from_dict(losses, orient="index").rename_axis("month").reset_index()
    table = table.merge(losses, on="month", how="left")
    table["solved"] = table["month"].isin(losses["month"])
    front = ["month", "year", "calendar_month", "missing_days", "gap_flagged", "hardcoded", "solved"]
    by_area = [column for area in areas for column in table.columns if column.startswith(f"{area}_")]
    return table[front + by_area].round(6)


def selected_table(selected, table, min_days, rule) -> pd.DataFrame:
    """The picked months with their focus-area columns, for scenarios_selected.csv."""
    columns = [column for area in [*FOCUS_AREAS, "combined"] for column in table.columns
               if column.startswith(f"{area}_") and not column.endswith("_flooded_cells")]
    out = selected[["month", "tier", "reason"]].merge(table[["month"] + columns], on="month", how="left")
    out = out.rename(columns={"month": "id"})
    out.insert(2, "label", [month_label(month) for month in out["id"]])
    out["min_days_flooded"] = min_days
    out["flooded_rule"] = rule
    return out


def run_metadata(cfg, table, ranks, hashes, timing, rule) -> dict:
    """What the run used and found, for scenarios_metadata.json."""
    used = ["min_days_flooded", "flooded_cell_passable", "first_year", "last_year"]
    counties = pd.read_csv(data_path(cfg, "counties_table"))
    ids, names = area_ids(counties), counties.set_index("county_id")["adm2_name"]
    return {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_used": {**{key: cfg["flood"][key] for key in used}, "population_year": cfg["population"]["year"],
                        "max_time_min": cfg["travel"]["max_time_min"]},
        "flooded_rule": rule, "mode": "walking",
        "focus_areas": {area: {"pcodes": FOCUS_AREAS.get(area), "county_ids": ids[area].tolist(),
                               "names": names[ids[area]].tolist()} for area in [*FOCUS_AREAS, "combined"]},
        "selection": {"metric": LOSS, "max_no_data_share": MAX_NO_DATA_SHARE,
                      "gap_flagged_months_excluded": table.loc[table["gap_flagged"], "month"].tolist()},
        "focus_area_max_no_data_share": table[[f"{area}_no_data_share" for area in FOCUS_AREAS]].max().to_dict(),
        "solve": timing, "hardcoded_months": ranks.to_dict(orient="records"), "inputs_sha256": hashes,
    }


def solve_candidates(cfg) -> tuple[pd.DataFrame, dict, dict]:
    """Steps 1 and 2: check the baseline, solve every month and write scenario_candidates.csv."""
    hashes = check_baseline(cfg)
    inp = load_inputs(cfg)
    table = flooding_per_month(cfg, inp)
    flooded = table["combined_flooded_share"]
    print(f"step 1: {len(table)} months; the focus areas have some flooding in {(flooded > 0).sum()} of them, "
          f"at most {flooded.max():.2%} of their cells")
    losses, timing = solve_months(cfg, inp, table)
    table = candidates_table(table, losses, inp["ids"])
    # written before step 3, so the solves are kept if step 3 fails; step 3 then reads the same rounded numbers
    table.to_csv(OUT_DIR / "scenario_candidates.csv", index=False)
    return table, hashes, timing


def main():
    parser = argparse.ArgumentParser(description="Pick the flood scenario months for the dashboard.")
    parser.add_argument("--select-only", action="store_true",
                        help="redo only the picks, from the existing EDA/outputs/scenario_candidates.csv")
    args = parser.parse_args()
    started = time.perf_counter()
    cfg = load_config()
    min_days = cfg["flood"]["min_days_flooded"]
    rule = f"days >= {min_days} and days != {NO_DATA} (band all); blocked = water | flooded"
    if cfg["flood"]["flooded_cell_passable"]:
        raise ValueError("flood.flooded_cell_passable is true, but this script blocks flooded cells")
    if args.select_only:
        table = pd.read_csv(OUT_DIR / "scenario_candidates.csv")
        with open(OUT_DIR / "scenarios_metadata.json", encoding="utf-8") as f:
            solved = json.load(f)   # the run that solved the months: its hashes and times still describe the table
        hashes, timing, seconds = solved["inputs_sha256"], solved["solve"], solved["total_seconds"]
    else:
        table, hashes, timing = solve_candidates(cfg)
        seconds = round(time.perf_counter() - started, 1)

    selected = select_scenarios(table)
    ranks = hardcoded_ranks(table, selected)
    selected_table(selected, table, min_days, rule).to_csv(OUT_DIR / "scenarios_selected.csv", index=False)
    block = scenarios_block(selected, min_days)
    (OUT_DIR / "scenarios_block.yaml").write_text(block, encoding="utf-8")
    metadata = run_metadata(cfg, table, ranks, hashes, timing, rule)
    metadata["total_seconds"] = seconds
    with open(OUT_DIR / "scenarios_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    with pd.option_context("display.width", 200, "display.max_colwidth", 150):
        print(f"\n{selected.to_string(index=False)}\n\n{ranks.to_string(index=False)}\n\n{block}")
    print(f"wrote the outputs to {OUT_DIR} in {round(time.perf_counter() - started, 1)} s")


if __name__ == "__main__":
    main()
