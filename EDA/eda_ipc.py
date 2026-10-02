import re
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "raw_data"
IPC_DIR = RAW / "IPC"
ADMIN2 = RAW / "Administrative boundaries" / "ssd_admin2.geojson"
OUT = Path("eda_out")
FIG = OUT / "figures"

PHASES = ["p1", "p2", "p3", "p4", "p5"]

# IPC names that don't match the admin2 file
NAME_FIXES = {
    "yei county": "Yei",
    "wau (rural only)": "Wau",  # only rural Wau, so population is smaller than the polygon
    "returnees": None,  # not a place
}

COLUMN_MAP = {
    "Est Pop": "est_pop", "From Date": "from_date", "Thru Date": "thru_date",
    "Phase": "phase_class", "Phase 3+": "p3plus", "Phase 3+ %": "p3plus_pct",
}
for i in range(1, 6):
    COLUMN_MAP[f"Phase {i}"] = f"p{i}"
    COLUMN_MAP[f"Phase {i} %"] = f"p{i}_pct"


def norm(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def read_file(path):
    df = pd.read_excel(path)
    df = df[df["Area Name"].notna()]
    area = df["Area Name"].astype(str).str.replace("\xa0", " ")

    # counties are indented under their state
    df["area_name"] = area.str.strip()
    df["level"] = np.where(area.str.startswith(" "), "county", "state")
    df.loc[df["area_name"].str.lower() == "analysis totals", "level"] = "country"

    frames = []
    for period in ["Current", "Projected 1", "Projected 2"]:
        cols = [c for c in df.columns if c.startswith(period + " - ")]
        if not cols:
            continue
        sub = df[cols].rename(columns=lambda c: COLUMN_MAP[c.split(" - ", 1)[1]])
        sub[["area_name", "level"]] = df[["area_name", "level"]]
        sub["period_type"] = period
        sub["source_file"] = path.name
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def load_all():
    files = sorted(IPC_DIR.glob("*.xlsx"))
    print(f"Found {len(files)} IPC files")
    df = pd.concat([read_file(f) for f in files], ignore_index=True)

    df["from_date"] = pd.to_datetime(df["from_date"], errors="coerce").dt.tz_localize(None)
    df["thru_date"] = pd.to_datetime(df["thru_date"], errors="coerce").dt.tz_localize(None)
    num_cols = ["est_pop", "p3plus"] + PHASES + [c for c in df.columns if c.endswith("_pct")]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    df["area_name_ipc"] = df["area_name"]
    for bad, good in NAME_FIXES.items():
        hit = df["area_name"].str.lower() == bad
        if good is None:
            df.loc[hit, "level"] = "special"
        else:
            df.loc[hit, "area_name"] = good

    pct_cols = [c for c in df.columns if c.endswith("_pct")]
    for c in pct_cols:
        if df[c].max() > 1.5:
            df[c] = df[c] / 100

    # population but no phase numbers means the area wasn't analysed, not zero people
    phase_sum = df[PHASES].sum(axis=1)
    df["analysed"] = ~((phase_sum == 0) & (df["est_pop"] > 0))
    df.loc[~df["analysed"], PHASES + ["p3plus"] + pct_cols] = np.nan

    df["p3plus_share"] = df["p3plus"] / df["est_pop"]
    df["period_label"] = df["from_date"].dt.strftime("%Y-%m") + " to " + df["thru_date"].dt.strftime("%Y-%m")
    return df


def run_checks(df):
    print(f"\nDate range: {df['from_date'].min()} to {df['thru_date'].max()}")

    windows = (df.groupby(["source_file", "period_type"])
               .agg(from_date=("from_date", "min"), thru_date=("thru_date", "max"))
               .reset_index().sort_values("from_date"))
    print("\nAnalysis windows:")
    print(windows.to_string(index=False))
    cur = windows[windows.period_type == "Current"]
    gaps = (cur["from_date"].shift(-1) - cur["thru_date"]).dt.days.dropna()
    print("Gaps between Current windows (days):", gaps.tolist())

    not_analysed = df[~df["analysed"]]
    print(f"\n{len(not_analysed)} area-periods not analysed (set to NaN):")
    if len(not_analysed):
        print(not_analysed[["area_name", "period_type", "period_label", "est_pop"]].to_string(index=False))

    ok = df[df["analysed"]]
    rel = (ok[PHASES].sum(axis=1) - ok["est_pop"]).abs() / ok["est_pop"]
    print(f"\nPhase sum vs Est Pop: median diff {rel.median():.2e}, max {rel.max():.2e}, "
          f"{(rel > 0.01).sum()} rows off by >1%")

    gap = (ok["p3plus_pct"] - ok["p3plus_share"]).abs()
    print(f"Reported Phase 3+ % vs computed: median diff {gap.median():.4f}, max {gap.max():.4f}")

    counties = df[df.level == "county"]
    cov = counties.pivot_table(index="area_name", columns="source_file", aggfunc="size", fill_value=0) > 0
    print(f"\nCounties per file: {cov.sum().to_dict()}")
    missing = cov[~cov.all(axis=1)]
    if len(missing):
        print("Counties missing from some rounds:")
        print(missing.astype(int).to_string())

    cur_cty = counties[counties.period_type == "Current"]
    pop = cur_cty.pivot_table(index="area_name", columns="period_label", values="est_pop")
    drift = (pop.max(axis=1) / pop.min(axis=1)).sort_values(ascending=False)
    print("\nEst Pop max/min ratio across rounds (top 10):")
    print(drift.head(10).round(2).to_string())

    adm2 = gpd.read_file(ADMIN2)
    shp = {norm(n): n for n in adm2["adm2_name"]}
    ipc_names = cur_cty["area_name"].unique()
    unmatched = [n for n in ipc_names if norm(n) not in shp]
    no_ipc = [shp[k] for k in shp if k not in {norm(n) for n in ipc_names}]
    print(f"\nIPC counties not in admin2: {unmatched}")
    print(f"Admin2 counties without IPC data: {no_ipc}")


def make_figures(df):
    FIG.mkdir(parents=True, exist_ok=True)

    w = (df.groupby(["source_file", "period_type"])
         .agg(f=("from_date", "min"), t=("thru_date", "max"))
         .reset_index().sort_values("f").reset_index(drop=True))
    colors = {"Current": "tab:blue", "Projected 1": "tab:orange", "Projected 2": "tab:red"}
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(w) + 2))
    for i, r in w.iterrows():
        ax.barh(i, (r["t"] - r["f"]).days, left=r["f"], height=0.6, color=colors[r["period_type"]])
    ax.set_yticks(range(len(w)))
    ax.set_yticklabels([f"{r.source_file[:28]} | {r.period_type}" for r in w.itertuples()], fontsize=7)
    ax.set_title("IPC analysis windows (blue = Current, others = projections)")
    fig.tight_layout()
    fig.savefig(FIG / "ipc_timeline.png", dpi=150)
    plt.close(fig)

    cty = df[(df.level == "county") & (df.period_type == "Current")]
    nat = cty.groupby("period_label")[["est_pop"] + PHASES].sum().sort_index()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bottom = np.zeros(len(nat))
    for p, c in zip(PHASES, ["#c8e6c9", "#fff59d", "#ffb74d", "#e57373", "#7b1fa2"]):
        vals = (nat[p] / nat["est_pop"]).values
        ax.bar(nat.index, vals, bottom=bottom, color=c, label=p.upper())
        bottom += vals
    ax.set_ylabel("share of population")
    ax.set_title("National IPC phase composition (Current periods)")
    ax.legend(ncol=5, fontsize=8)
    plt.xticks(rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "ipc_national_phases.png", dpi=150)
    plt.close(fig)

    piv = cty.pivot_table(index="area_name", columns="from_date", values="p3plus_share")
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(7, 0.18 * len(piv) + 2))
    im = ax.imshow(piv.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels(piv.index, fontsize=6)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([d.strftime("%Y-%m") for d in piv.columns], rotation=45, fontsize=8)
    ax.set_title("Share of population in IPC Phase 3+")
    fig.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    fig.savefig(FIG / "ipc_county_heatmap.png", dpi=150)
    plt.close(fig)

    top = piv.mean(axis=1).head(15)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.index, top.values, color="tab:red")
    ax.set_xlabel("mean share in Phase 3+")
    ax.set_title("Counties with highest average food insecurity")
    fig.tight_layout()
    fig.savefig(FIG / "ipc_top_counties.png", dpi=150)
    plt.close(fig)

    pop = cty.pivot_table(index="area_name", columns="from_date", values="est_pop")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for _, row in pop.iterrows():
        ax.plot(pop.columns, row.values, color="grey", alpha=0.35, lw=0.8)
    ax.plot(pop.columns, pop.mean().values, color="k", lw=2, label="mean county")
    ax.set_ylabel("Est Pop")
    ax.set_title("Estimated population per county across IPC rounds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "ipc_population_drift.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    df = load_all()
    print(df.groupby(["level", "period_type"]).size().to_string())

    df.to_csv(OUT / "ipc_area_period.csv", index=False)
    long = df.melt(id_vars=["area_name", "level", "period_type", "period_label",
                            "from_date", "thru_date", "est_pop", "source_file"],
                   value_vars=PHASES, var_name="phase", value_name="population")
    long["share"] = long["population"] / long["est_pop"]
    long.to_csv(OUT / "ipc_long.csv", index=False)

    run_checks(df)
    make_figures(df)
