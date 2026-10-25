"""
Order-of-magnitude check: facilities our flood rules close, against the facility counts WHO and OCHA
reported during three flood seasons.

The reports count facilities affected by the flooding so far that season, up to a reporting date;
our scenarios are single months. So per report we give a range: the facilities closed in the report
month itself (lower bound), and those closed at least once from the start of the flooding up to the
report month (upper bound). The start is taken from the report where it says so (June for 2025),
otherwise the start of the rainy season in April (WHO 2024: rains "between April and November").
Facilities are closed and own-cell-flooded by the same rules as everywhere else (scenarios.py).
No travel times are needed.

Output in evaluation.output_dir (data/processed/evaluation/):
    external_check.csv   per report: the reported figures, the window used, and our counts

Needs build_flood_monthly and build_baseline (the sites).
Run from the repository root:  python -m data_preparation.external_check
"""
import numpy as np
import pandas as pd

from data_preparation.grid import ROOT, load_config, make_grid
from data_preparation.loading import load_flood_months, load_sites
from data_preparation.scenarios import closed_sites, scenario_flood

# What each source reports, the month its counts run up to, and the month the flooding started
# (checked against the sources, Oct 2026).
RAINY_SEASON_START = 4      # WHO 2024: rains "between April and November"; used where a report gives no start
REPORTS = [
    {"report": "WHO situation report #17", "report_month": "2021-10", "start_month": RAINY_SEASON_START,
     "as_of": "2021-11-03", "reported_affected": 60, "reported_submerged": None,
     "reported_text": "over 60 health facilities cut off, submerged or damaged",
     "url": "https://www.afro.who.int/sites/default/files/2021-11/South%20Sudan%20Humanitarian%20Situation%20Report_%2317%20October%202021.pdf"},
    {"report": "WHO AFRO news release", "report_month": "2024-10", "start_month": RAINY_SEASON_START,
     "as_of": "2024-10-21", "reported_affected": 148, "reported_submerged": 58,
     "reported_text": "58 health facilities submerged, nearly 90 others inaccessible",
     "url": "https://www.afro.who.int/countries/south-sudan/news/severe-flooding-compounds-health-crisis-south-sudan"},
    {"report": "WHO Public Health Situation Analysis", "report_month": "2025-10", "start_month": 6,
     "as_of": "2025-10-24", "reported_affected": 143, "reported_submerged": 44,
     "reported_text": "flooding since June 2025; 143 health facilities affected, 44 fully damaged",
     "url": "https://www.who.int/publications/m/item/public-health-situation-analysis---south-sudan1"},
]


def season_window(report_month: str, start_month: int) -> list[str]:
    """The months from `start_month` of the report's year up to and including the report month."""
    end = pd.Period(report_month, "M")
    start = pd.Period(year=end.year, month=start_month, freq="M")
    return [str(m) for m in pd.period_range(start, end, freq="M")]


def facilities_in(mask: np.ndarray, n_facilities: np.ndarray) -> int:
    """Number of facilities in the sites selected by `mask`."""
    return int(n_facilities[mask].sum())


def main():
    cfg = load_config()
    grid, sites = make_grid(cfg), load_sites(cfg)
    sites_rc, n_fac = sites[["row", "col"]].to_numpy(), sites["n_facilities"].to_numpy()
    flagged = set(load_flood_months(cfg).assign(id=lambda d: [f"{y}-{m:02d}" for y, m in zip(d["year"], d["month"])])
                  .query("flagged")["id"])

    rows = []
    for report in REPORTS:
        window = season_window(report["report_month"], report["start_month"])
        closed_any, own_any = np.zeros(len(sites), bool), np.zeros(len(sites), bool)
        for month in window:
            flooded, _ = scenario_flood(month, cfg)
            closed = closed_sites(sites_rc, flooded, cfg["facilities"]["flood_buffer_m"], grid.cell_size)
            own = flooded[sites_rc[:, 0], sites_rc[:, 1]]
            closed_any |= closed
            own_any |= own
        # closed and own are still those of the last month in the window: the report month
        rows.append({**report, "window": f"{window[0]} to {window[-1]}",
                     "months_with_data_gaps": ";".join(m for m in window if m in flagged),
                     "closed_report_month": facilities_in(closed, n_fac),
                     "closed_any_month": facilities_in(closed_any, n_fac),
                     "own_cell_flooded_report_month": facilities_in(own, n_fac),
                     "own_cell_flooded_any_month": facilities_in(own_any, n_fac)})

    out = pd.DataFrame(rows)
    folder = ROOT / cfg["evaluation"]["output_dir"]
    out.to_csv(folder / "external_check.csv", index=False)
    print(out[["report_month", "window", "reported_affected", "reported_submerged", "closed_report_month",
               "closed_any_month", "own_cell_flooded_report_month", "own_cell_flooded_any_month",
               "months_with_data_gaps"]].to_string(index=False))
    print(f"wrote {folder / 'external_check.csv'}")


if __name__ == "__main__":
    main()
