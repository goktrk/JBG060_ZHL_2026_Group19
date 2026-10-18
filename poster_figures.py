"""
Poster figures for the A0 poster (readers: ZOA and humanitarian staff), drawn from existing outputs only.

    poster figure               drawn from
    flood_area_per_year         EDA/eda_flood_masks.ipynb     flood_types_per_year.png
    flood_area_per_month        EDA/eda_flood_masks.ipynb     flood_types_per_month.png
    facilities_near_flooding    EDA/EDA_infrastructure.ipynb  facility_exposure_per_year.png
    scenario_access_loss        EDA/outputs/scenarios_selected.csv (data_preparation/select_scenarios.py)
    facility_ranking_shift      EDA/outputs/facility_rankings_month10.csv (data_preparation/export_rankings.py)

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
* Scenario access loss is read unchanged from EDA/outputs/scenarios_selected.csv, all four months:
  people within 120 min walk of care without floods, and beyond 120 min or unreachable in that flood
  month, split into Bor South and the five Aweil counties (columns *_pop_newly_beyond_120).
* Facility ranking shift is read unchanged from EDA/outputs/facility_rankings_month10.csv: the top 5
  flooded facilities in October under the most_saved and worst_off criteria. Facility types are
  shortened ('health unit', 'health centre') so two columns fit at 22 pt.

Each figure is built at its printed size (see poster_style.py) and saved to figures/poster/ as SVG and
300 dpi PNG. The plotted numbers are saved next to them in figures/poster/data/.

Run from the repository root:
    python poster_figures.py               # reuses figures/poster/data/flood_area_*.csv if present
    python poster_figures.py --recompute   # re-reads the 276 monthly flood files (about a minute)
    python poster_figures.py --only scenario_access_loss facility_ranking_shift   # just these, SVG + PNG,
                                                                                  # data tables not rewritten
"""
import argparse
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import rasterio
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator

from data_preparation.grid import ROOT, data_path, load_config
from poster_style import (GREY, INK_SOFT, OUT_DIR, TUE_RED, WATER, WATER_LIGHT, apply_poster_style,
                          new_poster_fig, save_poster_fig)

# Printed size of each figure in inches (width, height), from the poster layout. The layouts below are
# designed at these sizes: to change one, change it here, re-run and check the result. Never scale the
# saved files.
SIZES = {
    "flood_area_per_year": (17.5, 10.6),
    "flood_area_per_month": (10.6, 4.1),
    "facilities_near_flooding": (10.6, 4.1),
    "scenario_access_loss": (10.6, 4.1),
    "facility_ranking_shift": (10.6, 4.1),
}

DATA_DIR = OUT_DIR / "data"
FACILITIES_PER_YEAR = ROOT / "EDA" / "outputs" / "facility_flood_exposure_per_year.csv"
FACILITIES = ROOT / "EDA" / "outputs" / "facility_flood_exposure.csv"
SCENARIOS = ROOT / "EDA" / "outputs" / "scenarios_selected.csv"
SCENARIO_LABELS = {   # plain-language row labels, by scenario id
    "2008-11": "Nov 2008 · largest loss",
    "2020-11": "Nov 2020 · worst for Bor South",
    "2025-10": "Oct 2025 · typical",
    "2025-01": "Jan 2025 · small",
}
LOSS = "pop_newly_beyond_120"
# Bar colours for the two areas. BOR_SOUTH_BLUE is lighter than WATER (OKLCH L 0.63 vs 0.48, colour-blind
# delta E 15.8) but, unlike WATER_LIGHT, still has 3.5:1 contrast on white. Each segment gets a navy outline.
BOR_SOUTH_BLUE = "#4e8fc2"
OUTLINE = "#0d2b52"
AREAS = {"lol_aweil": ("Aweil counties", WATER), "bor_south": ("Bor South", BOR_SOUTH_BLUE)}   # stacked left to right

RANKINGS = ROOT / "EDA" / "outputs" / "facility_rankings_month10.csv"
RANKINGS_MONTH = 10
TOP_K = 5
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


def area_sum_mismatch(selected: pd.DataFrame) -> pd.Series:
    """Per scenario id: combined minus the sum of the two areas (the areas don't overlap, so ~0). Warns if not."""
    parts = sum(selected[f"{area}_{LOSS}"] for area in AREAS)
    diff = (selected[f"combined_{LOSS}"] - parts).set_axis(selected["id"])
    off = ~np.isclose(diff, 0, atol=0.01)   # the csv keeps 6 decimals, so allow rounding, not a real gap
    if off.any():
        warnings.warn(f"Bor South + Aweil != combined: {diff[off].to_dict()}")
    return diff


def plot_scenario_access_loss(selected: pd.DataFrame, size: tuple[float, float]):
    s = selected.set_index("id").loc[list(SCENARIO_LABELS)].sort_values(f"combined_{LOSS}", ascending=False)
    total = s[f"combined_{LOSS}"]
    rows = np.arange(len(s))

    fig, ax = new_poster_fig(*size)
    fig.get_layout_engine().set(h_pad=0.04, w_pad=0.08)   # four rows, a key, two 2-line texts: little room to spare
    left = np.zeros(len(s))
    for area, (_, colour) in AREAS.items():
        ax.barh(rows, s[f"{area}_{LOSS}"], left=left, height=0.66, color=colour, edgecolor=OUTLINE, linewidth=1)
        left += s[f"{area}_{LOSS}"].to_numpy()
    for y, v in zip(rows, total):
        ax.text(v + 300, y, f"{float(f'{v:.3g}'):,.0f}", va="center", ha="left")   # 3 significant figures

    # one-line key above the bars, starting at the left edge of the plot
    ax.legend(handles=[Patch(facecolor=colour, edgecolor=OUTLINE, linewidth=1, label=name)
                       for name, colour in AREAS.values()],
              loc="lower left", bbox_to_anchor=(0, 1), ncol=len(AREAS), frameon=False, borderpad=0,
              borderaxespad=0.25, handlelength=1.1, handleheight=0.9, handletextpad=0.5, columnspacing=1.6)

    ax.set_yticks(rows, [SCENARIO_LABELS[k] for k in s.index])
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlim(0, total.max() * 1.22)
    ax.xaxis.set_major_locator(MultipleLocator(5_000))
    ax.xaxis.set_major_formatter(KM2)
    # on one line it is wider than the plot (the row labels take 4.2 in), so two lines, centred under the plot
    ax.set_xlabel("People pushed beyond\na 2-hour walk to care", labelpad=6)
    fig.supxlabel("Aweil counties stand in for the Lol river area.\n"
                  "Numbers are minimums because of cloud cover.", **NOTE)
    return fig


def facility_labels(shown: pd.DataFrame) -> pd.Series:
    """'County · type' per site_id, with (a), (b) ... where two facilities would get the same label."""
    label = shown["county"] + " · " + shown["facility_type"].map(lambda t: SHORT_TYPE.get(t, t.lower()))
    twins = label.duplicated(keep=False)
    order = label[twins].groupby(label[twins]).cumcount()
    label[twins] = label[twins] + order.map(lambda i: f" ({chr(ord('a') + i)})")
    return pd.Series(label.to_numpy(), index=shown["site_id"].to_numpy())


def plot_facility_ranking_shift(ranking: pd.DataFrame, size: tuple[float, float]):
    """
    Top 5 flooded facilities under each criterion, side by side; a grey line joins a facility that is in both.
    Placed in inches (no layout engine), so the columns can be sized to the widest label.
    """
    columns = {"most_saved": "Most people covered", "worst_off": "Most isolated first"}
    top = {c: ranking.nsmallest(TOP_K, f"rank_{c}") for c in columns}
    shown = pd.concat(top.values()).drop_duplicates("site_id")   # left column first, then the ones moving in
    label = facility_labels(shown)
    left, right = (set(t["site_id"]) for t in top.values())
    colour = {sid: GREY if sid in left and sid in right else TUE_RED if sid in left else WATER for sid in label.index}

    w, h = size
    fig, ax = new_poster_fig(w, h)
    fig.set_layout_engine("none")
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_axis_off()
    renderer = fig.canvas.get_renderer()

    def width(text, **kw) -> float:
        t = ax.text(0, 0, text, **kw)
        inches = t.get_window_extent(renderer).width / fig.dpi
        t.remove()
        return inches

    margin, pad = 0.15, 0.12
    left_w = max([width(label[sid]) for sid in top["most_saved"]["site_id"]]
                 + [width(columns["most_saved"], fontweight="bold")])
    right_w = max([width(label[sid]) for sid in top["worst_off"]["site_id"]]
                  + [width(columns["worst_off"], fontweight="bold")])
    x_left, x_right = margin + left_w, w - margin - right_w   # left column ends here, right column starts here

    y_header = h - margin - 0.72
    pitch = (y_header - 0.55 - margin - 0.2) / (TOP_K - 1)
    y = {c: dict(zip(top[c]["site_id"], y_header - 0.55 - pitch * np.arange(TOP_K))) for c in columns}
    # explain each colour once, next to one example, on different rows so the two phrases don't run together
    drops = [sid for sid in top["most_saved"]["site_id"] if sid not in right]
    moves = [sid for sid in top["worst_off"]["site_id"] if sid not in left]
    drop_label = drops[0] if drops else None
    move_label = next((sid for sid in moves if drop_label is None or y["worst_off"][sid] != y["most_saved"][drop_label]),
                      moves[0] if moves else None)
    same_row = drop_label is not None and move_label is not None and \
        y["worst_off"][move_label] == y["most_saved"][drop_label]
    needed = (width("drops out") + width("moves in") + 3 * pad if same_row
              else max(width("drops out"), width("moves in")) + 2 * pad)
    if x_right - x_left < needed:
        raise ValueError(f"facility_ranking_shift: {x_right - x_left:.2f} in between the columns, need {needed:.2f} in")

    n_facilities = int(ranking["n_facilities_in_cell"].sum())
    month = pd.Timestamp(2000, RANKINGS_MONTH, 1).strftime("%B")
    ax.text(margin, h - margin, f"{month} · top {TOP_K} of {n_facilities} flooded facilities in South Sudan",
            ha="left", va="top")
    ax.text(x_left, y_header, columns["most_saved"], ha="right", va="center", fontweight="bold")
    ax.text(x_right, y_header, columns["worst_off"], ha="left", va="center", fontweight="bold")
    ax.text((x_left + x_right) / 2, y_header, "in both", ha="center", va="center", color=GREY)

    for sid, y0 in y["most_saved"].items():
        ax.text(x_left, y0, label[sid], ha="right", va="center", color=colour[sid])
        if sid in right:
            ax.plot([x_left + pad, x_right - pad], [y0, y["worst_off"][sid]], color=GREY, lw=3,
                    solid_capstyle="round")
        elif sid == drop_label:
            ax.text(x_left + pad, y0, "drops out", ha="left", va="center", color=TUE_RED)
    for sid, y0 in y["worst_off"].items():
        ax.text(x_right, y0, label[sid], ha="left", va="center", color=colour[sid])
        if sid == move_label:
            ax.text(x_right - pad, y0, "moves in", ha="right", va="center", color=WATER)
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
    if "scenario_access_loss" in names:
        selected = pd.read_csv(SCENARIOS)
        mismatch = area_sum_mismatch(selected)
        if tables:
            selected[["id", "tier"] + [f"{area}_{LOSS}" for area in AREAS] + [f"combined_{LOSS}"]] \
                .assign(combined_minus_areas=mismatch.to_numpy()) \
                .to_csv(DATA_DIR / "scenario_access_loss.csv", index=False)
        figures["scenario_access_loss"] = plot_scenario_access_loss(selected, SIZES["scenario_access_loss"])
    if "facility_ranking_shift" in names:
        ranking = pd.read_csv(RANKINGS)
        if tables:
            ranking[[c for c in ranking.columns if c not in ("facility_name", "lon", "lat")]] \
                .to_csv(DATA_DIR / "facility_ranking_shift.csv", index=False)
        figures["facility_ranking_shift"] = plot_facility_ranking_shift(ranking, SIZES["facility_ranking_shift"])

    for name in names:
        fig = figures[name]
        svg, png = save_poster_fig(fig, name, *SIZES[name])
        print(f"{name}: {SIZES[name][0]:g} x {SIZES[name][1]:g} in -> {svg.relative_to(ROOT)}, {png.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
