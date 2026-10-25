"""
Summarise access loss and the two protection rankings over each year's largest flood month
(peak_months.csv), for the poster and the report.

Outputs in evaluation.output_dir (data/processed/evaluation/):
    peak_access_loss.csv   per year: people cut off from care, living in flooded cells, no facility, sites closed
    ranking_overlap.csv    per year: how many of the two Top N lists are the same facilities
    top_frequency.csv      per facility: in how many years it is in each Top N
    coverage.csv           per year, rule and number protected k = 1..10: share of the people losing access
                           that protecting the first k facilities (greedy order) brings back
    receiving_concentration.csv
                           per year and k = 1..20: share of the extra people (walking, still within the
                           2-hour limit) that the k facilities receiving the most of them take in
    top_receiving.csv      per year: the facilities receiving the most extra people (within the limit)
    rule_tradeoff.csv      per year: each ranking's Top N scored on both goals (people brought back within
                           2 h, and people more than 4 h from care brought back within 4 h)
    naive_comparison.csv   per year and goal: the greedy Top N against two simple ways to pick N sites
                           (largest gain on its own; most people usually served), all scored the same way
    evaluation.json        the headline numbers

Needs peak_months, then build_access_loss, build_priorities and build_prepositioning with --months-file.
Run from the repository root:  python -m data_preparation.evaluate_peak_months
"""
import json

import numpy as np
import pandas as pd

from data_preparation.grid import ROOT, load_config, make_grid
from data_preparation.loading import (load_access_loss, load_baseline, load_permanent_water, load_prepositioning,
                                      load_priorities, load_scenario_travel, load_sites, load_static)
from data_preparation.prioritization import regained_cells
from data_preparation.scenarios import scenario_flood

RULES = ("most_saved", "worst_off")
MAX_RECEIVING = 20      # receiving facilities counted in receiving_concentration.csv
TOP_RECEIVING = 10      # busiest receiving facilities per year kept in top_receiving.csv


def top_sites(table: pd.DataFrame, rule: str, n: int) -> list[int]:
    """The Top n sites of one rule as the dashboard shows them: greedy order, sites adding nobody skipped."""
    ranked = table.sort_values(f"rank_{rule}")
    return ranked.loc[ranked[f"added_{rule}"] > 0, "site_id"].head(n).astype(int).tolist()


def overlap(tops: dict[str, list[int]]) -> int:
    """How many facilities the two Top lists share."""
    return len(set(tops[RULES[0]]) & set(tops[RULES[1]]))


def top_frequency(tops_per_year: dict[str, dict[str, list[int]]], info: pd.DataFrame) -> pd.DataFrame:
    """Per facility, in how many years it is in each rule's Top list (facilities never in a Top list left out)."""
    rows = [{"site_id": sid, "rule": rule, "year": year}
            for year, tops in tops_per_year.items() for rule in RULES for sid in tops[rule]]
    if not rows:
        return pd.DataFrame(columns=["site_id", *(f"years_top_{r}" for r in RULES)])
    counts = (pd.DataFrame(rows).groupby(["site_id", "rule"]).size().unstack(fill_value=0)
              .reindex(columns=list(RULES), fill_value=0).add_prefix("years_top_"))
    counts["years_top_either"] = (pd.DataFrame(rows).groupby("site_id")["year"].nunique())
    out = counts.join(info, how="left").reset_index()
    return out.sort_values(["years_top_either", "site_id"], ascending=[False, True], ignore_index=True)


def coverage(table: pd.DataFrame, rule: str, losing: float, max_k: int) -> list[float]:
    """Share of `losing` brought back by protecting the first k = 1..max_k sites of a rule's greedy order.
    With fewer closed sites than k, the share stays at what all of them bring back."""
    cumulative = table.sort_values(f"rank_{rule}")[f"cumulative_{rule}"].to_numpy()
    if losing <= 0:
        return [float("nan")] * max_k
    reached = [cumulative[min(k, len(cumulative)) - 1] if len(cumulative) else 0.0 for k in range(1, max_k + 1)]
    return [float(r) / losing for r in reached]


def concentration(added: pd.Series, max_k: int) -> list[float]:
    """Share of all extra people taken in by the k facilities receiving the most, for k = 1..max_k.
    With fewer receiving facilities than k, the share stays at 1."""
    ordered = added[added > 0].sort_values(ascending=False).to_numpy()
    if not len(ordered):
        return [float("nan")] * max_k
    cumulative = ordered.cumsum() / ordered.sum()
    return [float(cumulative[min(k, len(ordered)) - 1]) for k in range(1, max_k + 1)]


def naive_tops(table: pd.DataFrame, usual: np.ndarray, rule: str, n: int) -> dict[str, list[int]]:
    """
    Two simple ways to pick n closed sites, to compare the greedy Top n against:
    own_gain:         the n sites that bring back the most people on their own (no correction for
                      two sites bringing back the same people);
    usual_population: the n sites with the most people living closest to them before the flood.
    """
    by_gain = table.sort_values([f"gain_{rule}", "site_id"], ascending=[False, True])
    by_size = table.assign(usual=usual[table["site_id"]]).sort_values(["usual", "site_id"], ascending=[False, True])
    return {"own_gain": by_gain["site_id"].head(n).astype(int).tolist(),
            "usual_population": by_size["site_id"].head(n).astype(int).tolist()}


def naive_summary(naive: pd.DataFrame, n: int) -> dict:
    """Per goal, over the years with more than n closed sites (otherwise every way protects all of them):
    the median share brought back by each way of picking, and in how many years greedy beats or ties each."""
    out = {}
    for goal, rows in naive[naive["sites_closed"] > n].groupby("goal"):
        wide = rows.pivot(index="month", columns="picked_by", values="brought_back")
        losing = rows.groupby("month")["people_losing_access"].first()
        out[goal] = {"years": len(wide),
                     "median_share_brought_back": {p: round(float((wide[p] / losing).median()), 3) for p in wide},
                     "years_greedy_at_least": {p: int((wide["greedy"] >= wide[p]).sum()) for p in wide if p != "greedy"},
                     "years_greedy_better": {p: int((wide["greedy"] > wide[p]).sum()) for p in wide if p != "greedy"}}
    return out


def people_brought_back(regained: dict[int, np.ndarray], protected: list[int], pop) -> float:
    """People brought back when all `protected` sites reopen together: everyone in a cell that any of
    them brings back on its own (reopening sites only adds sources, so the gains combine as a union)."""
    cells = [regained[sid] for sid in protected]
    if not cells:
        return 0.0
    return float(np.asarray(pop, dtype="float64").ravel()[np.unique(np.concatenate(cells))].sum())


def main():
    cfg = load_config()
    folder = ROOT / cfg["evaluation"]["output_dir"]
    peaks = pd.read_csv(folder / "peak_months.csv")
    limit, n = cfg["travel"]["max_time_min"], cfg["priorities"]["top_n"]
    scenario_ids = {s["id"] for s in cfg["scenarios"]}
    mode = cfg["priorities"]["mode"]
    goals = {"most_saved": limit, "worst_off": cfg["priorities"]["worst_off_min"]}   # each ranking's own goal
    grid, static, water, sites = make_grid(cfg), load_static(cfg), load_permanent_water(cfg), load_sites(cfg)
    pop, friction = static["population"], static[f"friction_{mode}"]
    sites_rc = sites[["row", "col"]].to_numpy()
    base_minutes, base_nearest, _ = load_baseline(mode, cfg)
    served = base_nearest >= 0
    usual = np.bincount(base_nearest[served], weights=pop[served].astype("float64"), minlength=len(sites))
    tradeoff_rows, naive_rows = [], []

    loss_rows, overlap_rows, coverage_rows, receiving_rows, busiest, tops_per_year, info = [], [], [], [], [], {}, []
    for peak in peaks.itertuples():
        access, _ = load_access_loss(peak.id, cfg)
        walking = access["modes"]["walking"]
        loss_rows.append({"year": peak.year, "month": peak.id, "flooded_km2": peak.flooded_km2,
                          "cut_off": walking[f"cut_off_{limit}"],
                          "living_in_flooded_cells": access["living_in_flooded_cells"],
                          "no_facility_reachable": walking["no_facility_reachable"],
                          "sites_closed": access["sites_closed"], "dashboard_scenario": peak.id in scenario_ids})
        table, ranking_summary = load_priorities(peak.id, cfg)
        tops = {rule: top_sites(table, rule, n) for rule in RULES}
        for rule in RULES:
            losing = ranking_summary["rankings"][rule]["people_losing_access"]
            for k, share in enumerate(coverage(table, rule, losing, 10), start=1):
                coverage_rows.append({"year": peak.year, "month": peak.id, "rule": rule, "k": k,
                                      "people_losing_access": losing, "share_brought_back": share})
        tops_per_year[peak.id] = tops
        flood_minutes, _ = load_scenario_travel(peak.id, mode, cfg)
        flooded, _ = scenario_flood(peak.id, cfg)
        for goal, goal_limit in goals.items():
            naive = naive_tops(table, usual, goal, n)
            needed = sorted({sid for top in [*tops.values(), *naive.values()] for sid in top})
            regained, lost = regained_cells(pop, base_minutes, flood_minutes, flooded, friction, water | flooded,
                                            sites_rc, np.array(needed, dtype=np.int64), goal_limit, grid.cell_size)
            losing = float(pop[lost].sum(dtype="float64"))
            for rule in RULES:
                tradeoff_rows.append({"year": peak.year, "month": peak.id, "goal": goal, "limit_min": goal_limit,
                                      "protected": rule, "people_losing_access": round(losing),
                                      "brought_back": round(people_brought_back(regained, tops[rule], pop))})
            for picked_by, top in {"greedy": tops[goal], **naive}.items():
                naive_rows.append({"year": peak.year, "month": peak.id, "goal": goal, "picked_by": picked_by,
                                   "sites_closed": access["sites_closed"], "people_losing_access": round(losing),
                                   "brought_back": round(people_brought_back(regained, top, pop))})
        receiving = load_prepositioning(peak.id, cfg)[1]
        walking_as_is = receiving[(receiving["mode"] == "walking") & (receiving["variant"] == "as_is")]
        added = walking_as_is["added_within_limit"]
        top = walking_as_is[added > 0].nlargest(TOP_RECEIVING, "added_within_limit")
        busiest.append(top[["site_id", "rep_facility_name", "rep_facility_type", "adm2_name", "usual_population",
                            "added_within_limit", "relative_increase_within_limit"]]
                       .assign(year=peak.year, month=peak.id, cut_off=walking[f"cut_off_{limit}"],
                               rank=range(1, len(top) + 1)))
        for k, share in enumerate(concentration(added, MAX_RECEIVING), start=1):
            receiving_rows.append({"year": peak.year, "month": peak.id, "k": k,
                                   "cut_off": walking[f"cut_off_{limit}"],
                                   "extra_people_within_limit": round(float(added.sum())),
                                   "receiving_facilities": int((added > 0).sum()), "share_taken_in": share})
        overlap_rows.append({"year": peak.year, "month": peak.id, "sites_closed": access["sites_closed"],
                             "shared": overlap(tops), "top_size": min(len(t) for t in tops.values())})
        info.append(table[["site_id", "adm2_name", "adm1_name", "rep_facility_type", "n_facilities"]])

    info = pd.concat(info).drop_duplicates("site_id").set_index("site_id")
    loss, overlaps = pd.DataFrame(loss_rows), pd.DataFrame(overlap_rows)
    frequency = top_frequency(tops_per_year, info)
    loss.round(0).to_csv(folder / "peak_access_loss.csv", index=False)
    overlaps.to_csv(folder / "ranking_overlap.csv", index=False)
    frequency.to_csv(folder / "top_frequency.csv", index=False)
    cover = pd.DataFrame(coverage_rows)
    cover.round(4).to_csv(folder / "coverage.csv", index=False)
    received = pd.DataFrame(receiving_rows)
    received.round(4).to_csv(folder / "receiving_concentration.csv", index=False)
    pd.concat(busiest, ignore_index=True).round(3).to_csv(folder / "top_receiving.csv", index=False)
    pd.DataFrame(tradeoff_rows).to_csv(folder / "rule_tradeoff.csv", index=False)
    naive = pd.DataFrame(naive_rows)
    naive.to_csv(folder / "naive_comparison.csv", index=False)

    full = overlaps[overlaps["top_size"] == n]       # years where both rules have a full Top N
    since = loss[loss["year"] >= 2020]
    summary = {
        "years": len(peaks), "limit_min": limit, "top_n": n,
        "cut_off_range": [int(loss["cut_off"].min()), int(loss["cut_off"].max())],
        "cut_off_min_since_2020": int(since["cut_off"].min()) if len(since) else None,
        "years_with_full_top_n": len(full),
        "mean_shared": round(float(full["shared"].mean()), 2) if len(full) else None,
        "years_sharing_at_most_one": int((full["shared"] <= 1).sum()),
        "facilities_ever_in_a_top_n": len(frequency),
        "median_share_brought_back_by_top_n": {
            rule: round(float(cover[(cover["rule"] == rule) & (cover["k"] == n)]["share_brought_back"].median()), 3)
            for rule in RULES},
        "median_share_taken_in_by_10_receiving": round(float(
            received.loc[received["k"] == 10, "share_taken_in"].median()), 3),
        "naive_comparison": naive_summary(naive, n),
    }
    with open(folder / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(loss[["month", "cut_off", "living_in_flooded_cells", "sites_closed"]].to_string(index=False))
    print(overlaps[["month", "shared", "top_size"]].to_string(index=False))
    print(json.dumps(summary, indent=2))
    print(frequency.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
