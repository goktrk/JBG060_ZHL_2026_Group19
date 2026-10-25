"""
Poster figures for the A0 poster (readers: ZOA and humanitarian staff), drawn from existing outputs only.

    poster figure               drawn from
    flood_area_per_year         EDA/eda_flood_masks.ipynb     flood_types_per_year.png
    flood_area_per_month        EDA/eda_flood_masks.ipynb     flood_types_per_month.png
    facilities_near_flooding    EDA/EDA_infrastructure.ipynb  facility_exposure_per_year.png
    peak_access_loss            data/processed/evaluation/peak_access_loss.csv (data_preparation/evaluate_peak_months.py)
    protection_curve            data/processed/evaluation/coverage.csv (same)
    protection_per_year         data/processed/evaluation/coverage.csv (same)
    protection_4h               data/processed/evaluation/rule_tradeoff.csv (same)
    receiving_concentration     data/processed/evaluation/receiving_concentration.csv (same)
    recurring_receivers         data/processed/evaluation/top_receiving.csv (same)

Changes from the EDA versions
* Flooded area is measured inside South Sudan, in km², from the monthly flood files on the 250 m
  analysis grid (data/interim/flood/). The EDA figures count satellite pixels over the whole flood
  tiles (0-10N, 20-40E), which include parts of Sudan, Ethiopia, Uganda, DRC and CAR.
* The recurring/unusual split is dropped. Inside South Sudan at least 88 % of the flooded area is
  'unusual' in every year, so the split adds a line but no message (eda_flood_masks.ipynb also
  advises using total flooded area for trends).
* A cell counts as flooded in a month if the satellite flagged it on at least one day, as in the EDA
  notebook. (The dashboard scenarios use 3 days: config flood.min_days_flooded.)
* Facility counts are read unchanged from EDA/outputs/facility_flood_exposure_per_year.csv.
* Peak access loss covers each year's largest observed flood month (2003-2025, peak_months.py): people
  on dry land cut off from care (beyond a 2-hour walk, or no facility reachable) and people living in
  flooded cells, from build_access_loss. Months that are dashboard scenarios are marked.
* Protection curve: share of the people losing access that protecting the first 1-10 facilities (greedy
  order, build_priorities) brings back, median and range over the large flood years (100,000+ people cut
  off), per rule. Small years are left out of this one: there the top 5 restore almost everything.
* Protection per year: the same share for the top 5, every year, per rule.
* Receiving concentration: share of the extra people (walking, still within 2 hours of care) that the
  1-20 facilities receiving the most of them take in, median and range over the large and the smaller
  flood years. From build_prepositioning on each year's largest flood month.
* Recurring receivers: the facilities most often among each year's 10 busiest receiving facilities
  (most extra people within 2 hours) over the 23 flood months, split into large and smaller flood
  years, with their median extra people and load (usual + extra, over usual) in those years. Only
  facilities with a median of 1,000+ extra people are shown.

Each figure is built at its printed size (see poster_style.py) and saved to figures/poster/ as SVG and
300 dpi PNG. The plotted numbers are saved next to them in figures/poster/data/.

Run from the repository root:
    python poster_figures.py               # reuses figures/poster/data/flood_area_*.csv if present
    python poster_figures.py --recompute   # re-reads the 276 monthly flood files (about a minute)
    python poster_figures.py --only peak_access_loss protection_curve protection_per_year   # just these,
                                                                       # SVG + PNG, data tables not rewritten
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import rasterio
from matplotlib.ticker import FuncFormatter, MultipleLocator

from data_preparation.grid import ROOT, data_path, load_config
from data_preparation.poster_style import (GREY, INK_SOFT, OUT_DIR, TUE_RED, WATER, WATER_LIGHT, apply_poster_style,
                          new_poster_fig, save_poster_fig)

# Printed size of each figure in inches (width, height), from the poster layout. The layouts below are
# designed at these sizes: to change one, change it here, re-run and check the result. Never scale the
# saved files.
SIZES = {
    "flood_area_per_year": (17.5, 10.6),
    "flood_area_per_month": (10.6, 4.1),
    "facilities_near_flooding": (10.6, 4.1),
    "peak_access_loss": (10.6, 4.1),
    "protection_curve": (10.6, 4.1),
    "protection_per_year": (10.6, 4.1),
    "protection_4h": (10.6, 4.1),
    "receiving_concentration": (10.6, 4.1),
    "recurring_receivers": (10.6, 4.1),
}

DATA_DIR = OUT_DIR / "data"
FACILITIES_PER_YEAR = ROOT / "EDA" / "outputs" / "facility_flood_exposure_per_year.csv"
FACILITIES = ROOT / "EDA" / "outputs" / "facility_flood_exposure.csv"
EVALUATION = ROOT / "data" / "processed" / "evaluation"   # evaluation.output_dir in config.yaml
# Stacked bar colours. LIGHT_BLUE is lighter than WATER (OKLCH L 0.63 vs 0.48, colour-blind delta E 15.8)
# but, unlike WATER_LIGHT, still has 3.5:1 contrast on white. Each segment gets a navy outline.
LIGHT_BLUE = "#4e8fc2"
OUTLINE = "#0d2b52"
RULES = {"most_saved": ("Most people regain access", WATER), "worst_off": ("Most isolated first", TUE_RED)}
TOP_K = 5
LARGE_FLOOD = 100_000   # people cut off within 2 h: a "large flood year" in protection_curve
RECEIVER_ROWS = 6       # facilities shown in recurring_receivers
MIN_EXTRA = 1_000       # recurring_receivers: leave out facilities with a median of fewer extra people
SHORT_TYPE = {"Primary Health Care Unit": "health unit", "Primary Health Care Centre": "health centre"}

MIN_DAYS = 1        # days flagged in a month for a cell to count as flooded (as in the EDA notebook)
NO_DATA = 255       # flood files: outside the satellite tiles (north of 10N)
SHIFT_YEAR = 2020   # first year of the wetter period (equatorial lakes rose in 2019-2020)

KM2 = FuncFormatter(lambda v, _: f"{v:,.0f}")
TITLE = dict(x=0.012, ha="left")   # titles and notes line up with the left edge of the figure
NOTE = dict(x=0.012, ha="left", fontsize=20, color=INK_SOFT, linespacing=1.3)
TIMES = {2: "twice", 3: "three times", 4: "four times"}       # whole part of a ratio, for titles
GROWN = {2: "doubled", 3: "tripled", 4: "quadrupled"}


def flood_area_tables(recompute: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Flooded area inside South Sudan per month and per year, in km².

    monthly: year, month, flooded_km2 (cells flooded at least MIN_DAYS that month), flagged (data gap)
    yearly:  year, flooded_km2 (cells flooded in at least one month of the year, each counted once)
    """
    monthly_csv, yearly_csv = DATA_DIR / "flood_area_monthly.csv", DATA_DIR / "flood_area_yearly.csv"
    if monthly_csv.exists() and yearly_csv.exists() and not recompute:
        return pd.read_csv(monthly_csv), pd.read_csv(yearly_csv)

    cfg = load_config()
    cell_km2 = (cfg["grid"]["cell_size_m"] / 1000) ** 2
    with rasterio.open(data_path(cfg, "static_stack")) as src:
        country = src.read(src.descriptions.index("county_id") + 1) > 0

    monthly, yearly = [], []
    for year in range(cfg["flood"]["first_year"], cfg["flood"]["last_year"] + 1):
        flooded_in_year = np.zeros(int(country.sum()), dtype=bool)
        for month in range(1, 13):
            with rasterio.open(data_path(cfg, "flood_monthly", year=year, month=month)) as src:
                days = src.read(src.descriptions.index("all") + 1)[country]
            flooded = (days >= MIN_DAYS) & (days != NO_DATA)
            flooded_in_year |= flooded
            monthly.append({"year": year, "month": month, "flooded_km2": flooded.sum() * cell_km2})
        yearly.append({"year": year, "flooded_km2": flooded_in_year.sum() * cell_km2})
        print(year, end=" ", flush=True)
    print()

    gaps = pd.read_csv(data_path(cfg, "flood_months"))[["year", "month", "flagged"]]
    monthly = pd.DataFrame(monthly).merge(gaps, on=["year", "month"], how="left")
    yearly = pd.DataFrame(yearly)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(monthly_csv, index=False)
    yearly.to_csv(yearly_csv, index=False)
    return monthly, yearly


def plot_flood_area_per_year(yearly: pd.DataFrame, size: tuple[float, float]):
    s = yearly.set_index("year")["flooded_km2"]
    first, last = s.index.min(), s.index.max()
    before, since = s.loc[:SHIFT_YEAR - 1].mean(), s.loc[SHIFT_YEAR:].mean()

    fig, ax = new_poster_fig(*size)
    ax.bar(s.index, s.to_numpy(), width=0.72, color=WATER)
    ax.hlines(before, first - 0.45, SHIFT_YEAR - 0.55, color=GREY, lw=5, zorder=3)
    ax.hlines(since, SHIFT_YEAR - 0.45, last + 0.45, color=TUE_RED, lw=5, zorder=3)
    # grey label sits above the tall 2007 bar, with a leader down the gap between the first two bars
    ax.annotate(f"{first}–{SHIFT_YEAR - 1} average\n{round(before, -2):,.0f} km²",
                xy=(first + 0.5, before), xytext=(first + 0.5, 11_000), ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=GREY, lw=2.5, relpos=(0, 0), shrinkA=4, shrinkB=0))
    ax.text(last + 0.45, 16_600, f"{SHIFT_YEAR}–{last} average\n{round(since, -2):,.0f} km²",
            ha="right", va="bottom", fontweight="bold")
    ax.set_xlim(first - 0.7, last + 0.7)
    ax.set_xticks(range(2004, last + 1, 4))
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 21_000)
    ax.yaxis.set_major_locator(MultipleLocator(5_000))
    ax.yaxis.set_major_formatter(KM2)
    ax.set_xlabel("Year")
    ax.set_ylabel("Flooded land (km²)")
    fig.suptitle(f"More than {TIMES.get(int(since / before), f'{since / before:.1f}×')} as much land "
                 f"floods each year since {SHIFT_YEAR}", **TITLE)
    fig.supxlabel("Land under water at least once in the year, South Sudan south of 10°N. "
                  "Satellites miss flooding under clouds, so real values are higher.", **NOTE)
    return fig


def plot_flood_area_per_month(monthly: pd.DataFrame, size: tuple[float, float]):
    m = monthly.assign(date=pd.to_datetime(dict(year=monthly["year"], month=monthly["month"], day=15)))
    m.loc[m["flagged"], "flooded_km2"] = np.nan        # undercounted months: leave a gap in the line
    lowest = m.loc[m.groupby("year")["flooded_km2"].idxmin()].set_index("year")
    persists = lowest.index[lowest["flooded_km2"] > 1_000].min()    # first year the water stays
    floor = lowest.loc[persists:, "flooded_km2"].min()
    start, end = lowest.loc[persists, "date"], pd.Timestamp(m["year"].max() + 1, 4, 30)

    fig, ax = new_poster_fig(*size)
    ax.fill_between(m["date"], m["flooded_km2"], color=WATER_LIGHT, lw=0)
    ax.plot(m["date"], m["flooded_km2"], color=WATER, lw=3)
    # The level the water has not dropped below since `persists`. The dry-season dips are too narrow
    # at this size for an arrow, so the label sits above the peaks with a leader in the right margin.
    ax.hlines(floor, start, end, color=TUE_RED, lw=3.5, zorder=3)
    ax.plot([end, end], [floor, 10_400], color=TUE_RED, lw=2.5)
    ax.text(end, 10_600, f"From {persists}, never below {np.floor(floor / 100) * 100:,.0f} km²",
            ha="right", va="bottom", fontweight="bold")
    ax.set_xlim(pd.Timestamp(m["year"].min() - 1, 12, 1), end + pd.Timedelta(days=20))
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylim(0, 12_500)
    ax.yaxis.set_major_locator(MultipleLocator(5_000))
    ax.yaxis.set_major_formatter(KM2)
    ax.set_ylabel("Flooded land\n(km²)")
    fig.suptitle(f"Since {persists}, flood water no longer drains away", **TITLE)
    fig.supxlabel("Gaps: no satellite data. Clouds hide some flooding, so real values are higher.", **NOTE)
    return fig


def plot_facilities_near_flooding(per_year: pd.DataFrame, n_facilities: int, size: tuple[float, float]):
    years = per_year["year"]
    near, half = per_year["any flooding within 1 km"], per_year["≥ 50% of 1 km flooded"]
    worst = per_year.loc[half.idxmax()]

    fig, ax = new_poster_fig(*size)
    ax.bar(years, near - half, width=0.72, color=WATER)
    ax.bar(years, half, bottom=near - half, width=0.72, color=TUE_RED, edgecolor="white", linewidth=2)
    # direct labels, each with a short vertical leader to one bar
    ax.text(2007, 128, "Flood water within 1 km", ha="center", va="bottom")
    ax.plot([2007, 2007], [near[years == 2007].item() + 4, 124], color=WATER, lw=2.5)
    ax.text(years.max() + 0.45, 222, "Half or more of the area within 1 km flooded",
            ha="right", va="bottom", fontweight="bold")
    ax.plot([worst["year"]] * 2, [worst["any flooding within 1 km"] + 4, 218], color=TUE_RED, lw=2.5)
    ax.set_xlim(years.min() - 0.7, years.max() + 0.7)
    ax.set_xticks(range(2004, years.max() + 1, 4))
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 260)
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.set_ylabel("Number of\nfacilities")
    ratio = near[years >= SHIFT_YEAR].mean() / near[years < SHIFT_YEAR].mean()
    fig.suptitle(f"Health facilities near floods have more than {GROWN.get(int(ratio), f'grown {ratio:.1f}×')}",
                 **TITLE)
    fig.supxlabel(f"Of {n_facilities:,} health facilities with satellite flood data (south of 10°N).", **NOTE)
    return fig


def plot_peak_access_loss(loss: pd.DataFrame, size: tuple[float, float]):
    """One stacked bar per year: people cut off from care, and people living in flooded areas, in that
    year's largest observed flood month; dashboard scenario months marked with a red triangle."""
    fig, ax = new_poster_fig(*size)
    fig.get_layout_engine().set(h_pad=0.04, w_pad=0.08)
    year = loss["year"].to_numpy()
    cut, flooded = loss["cut_off"] / 1000, loss["living_in_flooded_cells"] / 1000
    ax.bar(year, cut, width=0.72, color=WATER, edgecolor=OUTLINE, linewidth=1, label="Cut off from care")
    ax.bar(year, flooded, bottom=cut, width=0.72, color=LIGHT_BLUE, edgecolor=OUTLINE, linewidth=1,
           label="In flooded areas")
    top = (cut + flooded).to_numpy()
    marked = loss["dashboard_scenario"].to_numpy()
    ax.scatter(year[marked], top[marked] + top.max() * 0.06, marker="v", s=170, color=TUE_RED, zorder=3,
               label="Dashboard month")

    # a figure-level key, so its width doesn't squeeze the plot
    fig.legend(loc="outside upper right", ncol=3, frameon=False, borderpad=0, handlelength=1.1, handleheight=0.9,
               handletextpad=0.5, columnspacing=1.2)   # right-aligned: on the left it runs into the y label
    ax.set_xticks(range(2004, int(year.max()) + 1, 4))
    ax.set_xlim(year.min() - 0.7, year.max() + 0.7)
    ax.set_ylim(0, top.max() * 1.15)
    ax.yaxis.set_major_formatter(KM2)
    ax.set_ylabel("People (thousands)")
    fig.supxlabel("Each year's largest observed flood month. Clouds hide rainy-season floods,\n"
                  "so these are minimums. Cut off = beyond a 2-hour walk to care.", **NOTE)
    return fig


def plot_protection_curve(cover: pd.DataFrame, size: tuple[float, float]):
    """Share of cut-off people brought back by protecting 1-10 facilities: median (line) and range (band)
    over the large flood years, per rule. The same years for both rules."""
    first = cover[(cover["rule"] == "most_saved") & (cover["k"] == 1)]
    large = set(first.loc[first["people_losing_access"] >= LARGE_FLOOD, "month"])
    data = cover[cover["month"].isin(large)]

    fig, ax = new_poster_fig(*size)
    fig.get_layout_engine().set(h_pad=0.04, w_pad=0.08)
    for rule, (name, colour) in RULES.items():
        share = data[data["rule"] == rule].groupby("k")["share_brought_back"]
        k = share.median().index
        ax.fill_between(k, share.min() * 100, share.max() * 100, color=colour, alpha=0.15, lw=0)
        ax.plot(k, share.median() * 100, color=colour, marker="o", label=name)
    fig.legend(loc="outside upper right", ncol=2, frameon=False, borderpad=0, handlelength=1.4,
               handletextpad=0.5, columnspacing=1.2)
    ax.set_xticks(range(1, 11))
    ax.set_xlim(0.7, 10.3)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(MultipleLocator(25))
    ax.set_xlabel("Facilities protected (in ranking order)", labelpad=6)
    ax.set_ylabel("Brought back (%)")
    fig.supxlabel(f"Median (line) and range (band) over the {len(large)} large flood years\n"
                  f"({LARGE_FLOOD:,}+ people cut off within a 2-hour walk).", **NOTE)
    return fig


def plot_protection_per_year(cover: pd.DataFrame, size: tuple[float, float]):
    """Share of cut-off people the top 5 bring back, every year's largest flood month, per rule."""
    top = cover[cover["k"] == TOP_K].pivot(index="year", columns="rule", values="share_brought_back") * 100
    year = top.index.to_numpy()
    width = 0.4

    fig, ax = new_poster_fig(*size)
    fig.get_layout_engine().set(h_pad=0.04, w_pad=0.08)
    for offset, (rule, (name, colour)) in zip((-width / 2, width / 2), RULES.items()):
        ax.bar(year + offset, top[rule], width=width, color=colour, label=name)
    fig.legend(loc="outside upper right", ncol=2, frameon=False, borderpad=0, handlelength=1.1,
               handleheight=0.9, handletextpad=0.5, columnspacing=1.2)
    ax.set_xticks(range(2004, int(year.max()) + 1, 4))
    ax.set_xlim(year.min() - 0.7, year.max() + 0.7)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(MultipleLocator(25))
    ax.set_ylabel("Brought back (%)")
    fig.supxlabel(f"Share of the people cut off that the top {TOP_K} protected facilities bring back,\n"
                  "in each year's largest observed flood month.", **NOTE)
    return fig


def plot_protection_4h(tradeoff: pd.DataFrame, size: tuple[float, float]):
    """People more than a 4-hour walk from care, per year: without protection, and with the top 5 of each
    ranking protected. The 'most isolated first' ranking is built for this goal; the other is not."""
    goal = tradeoff[tradeoff["goal"] == "worst_off"]
    left = goal.pivot(index="year", columns="protected", values="brought_back")
    without = goal.groupby("year")["people_losing_access"].first()
    year = without.index.to_numpy()
    without = without.to_numpy() / 1000
    isolated = without - left["worst_off"].to_numpy() / 1000
    most = without - left["most_saved"].to_numpy() / 1000

    fig, ax = new_poster_fig(*size)
    fig.get_layout_engine().set(h_pad=0.04, w_pad=0.08)
    ax.fill_between(year, isolated, without, color=WATER, alpha=0.12, lw=0)
    ax.plot(year, without, color=TUE_RED, marker="o", label="No protection")
    ax.plot(year, most, color=LIGHT_BLUE, marker="o", ls="--", label="Top 5: most people")
    ax.plot(year, isolated, color=WATER, marker="o", label="Top 5: most isolated")
    fig.legend(loc="outside upper right", ncol=3, frameon=False, borderpad=0, handlelength=1.6,
               handletextpad=0.5, columnspacing=1.0)
    ax.set_xticks(range(2004, int(year.max()) + 1, 4))
    ax.set_xlim(year.min() - 0.5, year.max() + 0.5)
    ax.set_ylim(0, without.max() * 1.1)
    ax.yaxis.set_major_formatter(KM2)
    ax.set_ylabel("People (thousands)")
    fig.supxlabel("People more than 4 hours' walk from care, in each year's largest observed flood\n"
                  f"month, with the top {TOP_K} facilities of each ranking protected.", **NOTE)
    return fig


def plot_receiving_concentration(received: pd.DataFrame, size: tuple[float, float]):
    """Share of the extra people taken in by the 1-20 busiest receiving facilities: median (line) and
    range (band) over the large flood years and over the smaller ones."""
    large = received["cut_off"] >= LARGE_FLOOD
    groups = {"Large floods": (received[large], WATER), "Smaller floods": (received[~large], LIGHT_BLUE)}

    fig, ax = new_poster_fig(*size)
    fig.get_layout_engine().set(h_pad=0.04, w_pad=0.08)
    for name, (data, colour) in groups.items():
        share = data.groupby("k")["share_taken_in"]
        k = share.median().index
        ax.fill_between(k, share.min() * 100, share.max() * 100, color=colour, alpha=0.15, lw=0)
        n_years = data["year"].nunique()
        ax.plot(k, share.median() * 100, color=colour, marker="o", markersize=7, label=f"{name} ({n_years} years)")
    fig.legend(loc="outside upper right", ncol=2, frameon=False, borderpad=0, handlelength=1.4,
               handletextpad=0.5, columnspacing=1.2)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(MultipleLocator(25))
    ax.set_xlabel("Facilities stocked (busiest first)", labelpad=6)
    ax.set_ylabel("Reached (%)")
    ten = received[large & (received["k"] == 10)]["share_taken_in"].median() * 100
    fig.supxlabel(f"Large floods ({LARGE_FLOOD:,}+ cut off): the 10 busiest reach {ten:.0f}% of displaced patients.\n"
                  "Displaced: people whose facility closes who can still walk to another within 2 hours.", **NOTE)
    return fig


def plot_recurring_receivers(top: pd.DataFrame, size: tuple[float, float]):
    """Facilities most often among a year's 10 busiest receiving facilities, years split into large and
    smaller floods, labelled with their median load and extra people in those years."""
    top = top.assign(large=top["cut_off"] >= LARGE_FLOOD, load=1 + top["relative_increase_within_limit"])
    per_site = top.groupby("site_id").agg(
        name=("rep_facility_name", "first"), county=("adm2_name", "first"), years=("year", "nunique"),
        large_years=("large", "sum"), extra=("added_within_limit", "median"), load=("load", "median"))
    shown = (per_site[per_site["extra"] >= MIN_EXTRA].sort_values(["years", "extra"], ascending=False)
             .head(RECEIVER_ROWS))
    label = (shown["name"].str.replace(r"\s*Primary Health Care (Unit|Centre)", "", regex=True)
             .str.replace(" County Hospital", " hospital") + " · " + shown["county"])
    rows = np.arange(len(shown))
    n_years = top["year"].nunique()

    fig, ax = new_poster_fig(*size)
    fig.get_layout_engine().set(h_pad=0.04, w_pad=0.08)
    ax.barh(rows, shown["large_years"], height=0.7, color=WATER, label="Large floods")
    ax.barh(rows, shown["years"] - shown["large_years"], left=shown["large_years"], height=0.7, color=LIGHT_BLUE,
            label="Smaller floods")
    for y, row in zip(rows, shown.itertuples()):
        ax.text(row.years + 0.2, y, f"×{row.load:.1f} · {row.extra / 1000:.0f}k extra", va="center", ha="left",
                fontsize=20, color=INK_SOFT)
    fig.legend(loc="outside upper right", ncol=2, frameon=False, borderpad=0, handlelength=1.1,
               handleheight=0.9, handletextpad=0.5, columnspacing=1.2)
    ax.set_yticks(rows, label.to_list())
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlim(0, n_years)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.set_xlabel(f"Years among the 10 busiest (of {n_years})", labelpad=6)
    fig.supxlabel("×: typical load in those years compared with normal; extra: typical number of\n"
                  "displaced patients (within a 2-hour walk). Facilities with 1,000+ extra people.", **NOTE)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--recompute", action="store_true", help="re-read the monthly flood files")
    parser.add_argument("--only", nargs="+", choices=list(SIZES), metavar="FIGURE",
                        help=f"render only these figures (SVG + PNG), without rewriting the data tables: {', '.join(SIZES)}")
    args = parser.parse_args()
    names = args.only or list(SIZES)
    tables = args.only is None   # the full run also writes the plotted numbers to figures/poster/data/

    apply_poster_style()
    if tables:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    figures = {}
    if {"flood_area_per_year", "flood_area_per_month"} & set(names):
        monthly, yearly = flood_area_tables(args.recompute)
        if "flood_area_per_year" in names:
            figures["flood_area_per_year"] = plot_flood_area_per_year(yearly, SIZES["flood_area_per_year"])
        if "flood_area_per_month" in names:
            figures["flood_area_per_month"] = plot_flood_area_per_month(monthly, SIZES["flood_area_per_month"])
    if "facilities_near_flooding" in names:
        per_year = pd.read_csv(FACILITIES_PER_YEAR)
        n_facilities = int(pd.read_csv(FACILITIES, usecols=["analysable"])["analysable"].sum())
        if tables:
            per_year.to_csv(DATA_DIR / "facilities_near_flooding.csv", index=False)
        figures["facilities_near_flooding"] = plot_facilities_near_flooding(per_year, n_facilities,
                                                                            SIZES["facilities_near_flooding"])
    if "peak_access_loss" in names:
        loss = pd.read_csv(EVALUATION / "peak_access_loss.csv")
        if tables:
            loss.to_csv(DATA_DIR / "peak_access_loss.csv", index=False)
        figures["peak_access_loss"] = plot_peak_access_loss(loss, SIZES["peak_access_loss"])
    if {"protection_curve", "protection_per_year"} & set(names):
        cover = pd.read_csv(EVALUATION / "coverage.csv")
        if tables:
            cover.to_csv(DATA_DIR / "protection_coverage.csv", index=False)
        if "protection_curve" in names:
            figures["protection_curve"] = plot_protection_curve(cover, SIZES["protection_curve"])
        if "protection_per_year" in names:
            figures["protection_per_year"] = plot_protection_per_year(cover, SIZES["protection_per_year"])
    if "protection_4h" in names:
        tradeoff = pd.read_csv(EVALUATION / "rule_tradeoff.csv")
        if tables:
            tradeoff.to_csv(DATA_DIR / "rule_tradeoff.csv", index=False)
        figures["protection_4h"] = plot_protection_4h(tradeoff, SIZES["protection_4h"])
    if "receiving_concentration" in names:
        received = pd.read_csv(EVALUATION / "receiving_concentration.csv")
        if tables:
            received.to_csv(DATA_DIR / "receiving_concentration.csv", index=False)
        figures["receiving_concentration"] = plot_receiving_concentration(received, SIZES["receiving_concentration"])
    if "recurring_receivers" in names:
        top = pd.read_csv(EVALUATION / "top_receiving.csv")
        if tables:
            top.to_csv(DATA_DIR / "recurring_receivers.csv", index=False)
        figures["recurring_receivers"] = plot_recurring_receivers(top, SIZES["recurring_receivers"])

    for name in names:
        fig = figures[name]
        svg, png = save_poster_fig(fig, name, *SIZES[name])
        print(f"{name}: {SIZES[name][0]:g} x {SIZES[name][1]:g} in -> {svg.relative_to(ROOT)}, {png.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
