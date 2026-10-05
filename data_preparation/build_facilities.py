"""
Build the facility table: cleaned South Sudan health facilities with their grid cell and county.

Facilities stay a table (GeoParquet, one row per facility), not a raster. Cleaning, in order:
1. drop facilities without coordinates;
2. merge duplicate records (same type, close together, near-identical name);
3. attach the county; facilities just outside the border are snapped to the nearest county,
   facilities further out are dropped as coordinate errors.
Every dropped facility is listed with its reason in facilities_removed.csv.

Added columns: county_id / adm2_pcode / adm2_name, admin1_mismatch (the facility's own state
label disagrees with where it lies), row, col (cell on the analysis grid), and
has_flood_data (False where the flood tiles do not reach, i.e. north of 10N).

Run from the repository root:  python -m data_preparation.build_facilities
"""
import re
from difflib import SequenceMatcher

import geopandas as gpd
import numpy as np
import pandas as pd

from data_preparation.build_flood_monthly import tile_bounds
from data_preparation.grid import data_path, in_grid, load_config, load_counties, lonlat_to_rowcol, make_grid

TYPE_SUFFIX = re.compile(r"\s*primary health care (unit|centre|center)\s*$", re.IGNORECASE)


def name_key(name: str) -> str:
    """Facility name without the type suffix, lowercase, letters and digits only."""
    return re.sub(r"[^a-z0-9]", "", TYPE_SUFFIX.sub("", name).lower())


def different_suffix(a: str, b: str) -> bool:
    """True for names like 'Nglimbo A' / 'Ngolimbo B' or 'Kerwa 1' / 'Kerwa 2': numbered sites, not duplicates."""
    last = []
    for name in (a, b):
        words = TYPE_SUFFIX.sub("", name).split()
        last.append(words[-1].lower() if words else "")
    return all(len(t) == 1 for t in last) and last[0] != last[1]


def find_duplicates(fac: gpd.GeoDataFrame, crs: str, max_m: float, min_similarity: float) -> pd.DataFrame:
    """
    Pairs of records that describe the same facility: same type, at most `max_m` apart, and
    names at least `min_similarity` alike. Of each pair the GPS-located record is kept, else the lowest FID.
    """
    fac = fac.sort_values("FID").reset_index(drop=True)
    pts = fac.to_crs(crs).geometry
    x, y = pts.x.to_numpy(), pts.y.to_numpy()
    preferred = (fac["LL_source"] == "GPS").to_numpy()

    pairs, dropped = [], set()
    for i in range(len(fac)):
        dist = np.hypot(x - x[i], y - y[i])
        for j in np.flatnonzero((dist <= max_m) & (np.arange(len(fac)) > i)):
            a, b = fac.loc[i], fac.loc[j]
            if a["Facility_t"] != b["Facility_t"] or different_suffix(a["Facility_n"], b["Facility_n"]):
                continue
            similarity = SequenceMatcher(None, name_key(a["Facility_n"]), name_key(b["Facility_n"])).ratio()
            if similarity < min_similarity:
                continue
            keep, drop = (j, i) if preferred[j] and not preferred[i] else (i, j)   # i has the lower FID
            if drop in dropped:
                continue
            dropped.add(drop)
            pairs.append({
                "FID": fac.loc[drop, "FID"], "Facility_n": fac.loc[drop, "Facility_n"], "reason": "duplicate",
                "kept_FID": fac.loc[keep, "FID"], "kept_name": fac.loc[keep, "Facility_n"],
                "distance_m": round(float(dist[j])), "name_similarity": round(similarity, 2),
            })
    return pd.DataFrame(pairs)


def attach_counties(fac: gpd.GeoDataFrame, counties: gpd.GeoDataFrame, crs: str, snap_m: float):
    """County of each facility. Facilities outside every county are snapped to the nearest one within
    `snap_m`; facilities further out are returned separately as coordinate errors."""
    cols = ["county_id", "adm2_pcode", "adm2_name", "adm1_name", "geometry"]
    fac = gpd.sjoin(fac, counties[cols], how="left", predicate="within").drop(columns="index_right")

    outside = fac.index[fac["county_id"].isna()]
    near = gpd.sjoin_nearest(fac.loc[outside, ["geometry"]].to_crs(crs), counties[cols].to_crs(crs),
                             distance_col="dist_m").drop(columns=["geometry", "index_right"])
    near = near[~near.index.duplicated()]
    snapped = near.index[near["dist_m"] <= snap_m]
    fac.loc[snapped, cols[:-1]] = near.loc[snapped, cols[:-1]].to_numpy()
    for i in outside:
        print(f"    outside every county: {fac.loc[i, 'Facility_n']} ({near.loc[i, 'dist_m']:,.0f} m from "
              f"{near.loc[i, 'adm2_name']}) -> {'snapped' if i in snapped else 'dropped'}")

    too_far = near.index[near["dist_m"] > snap_m]
    removed = pd.DataFrame({"FID": fac.loc[too_far, "FID"], "Facility_n": fac.loc[too_far, "Facility_n"],
                            "reason": [f"{near.loc[i, 'dist_m']:,.0f} m outside South Sudan" for i in too_far]})
    fac = fac.drop(index=too_far)
    fac["county_id"] = fac["county_id"].astype(int)
    return fac, removed


def main():
    cfg = load_config()
    grid = make_grid(cfg)
    fcfg = cfg["facilities"]

    fac = gpd.read_file(data_path(cfg, "facilities"))
    fac = fac[fac["Country"] == fcfg["country"]].reset_index(drop=True)
    print(f"facilities in {fcfg['country']}: {len(fac)}")

    no_loc = fac.geometry.isna() | fac.geometry.is_empty
    removed = [pd.DataFrame({"FID": fac.loc[no_loc, "FID"], "Facility_n": fac.loc[no_loc, "Facility_n"],
                             "reason": "no coordinates"})]
    fac = fac[~no_loc]
    print(f"  without coordinates: {int(no_loc.sum())} dropped")

    dups = find_duplicates(fac, grid.crs, fcfg["duplicate_max_distance_m"], fcfg["duplicate_min_name_similarity"])
    removed.append(dups)
    fac = fac[~fac["FID"].isin(dups["FID"])]
    print(f"  duplicates: {len(dups)} merged into another record")

    counties = load_counties(cfg)
    fac, too_far = attach_counties(fac, counties, grid.crs, fcfg["county_snap_max_m"])
    removed.append(too_far)
    fac["admin1_mismatch"] = fac["Admin1"] != fac["adm1_name"]
    print(f"  own state label disagrees with location (kept, flagged): {int(fac['admin1_mismatch'].sum())}")

    lon, lat = fac.geometry.x.to_numpy(), fac.geometry.y.to_numpy()
    fac["row"], fac["col"] = lonlat_to_rowcol(grid, lon, lat)
    assert in_grid(grid, fac["row"], fac["col"]).all(), "a facility falls outside the grid"

    fac["has_flood_data"] = False
    for tile in cfg["flood"]["tiles"]:
        lon_min, lat_min, lon_max, lat_max = tile_bounds(tile)
        fac["has_flood_data"] |= (lon >= lon_min) & (lon < lon_max) & (lat >= lat_min) & (lat < lat_max)
    print(f"  without flood data (north of 10N): {int((~fac['has_flood_data']).sum())}")
    print(f"  sharing a cell with another facility: {int(fac.duplicated(['row', 'col'], keep=False).sum())}")
    print(f"  in Abyei Region (disputed, counted as South Sudan): {int((fac['adm2_pcode'] == 'SS0001').sum())}")

    out_path = data_path(cfg, "facilities_table")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fac.reset_index(drop=True).to_parquet(out_path)
    removed = pd.concat(removed, ignore_index=True)
    removed[["kept_FID", "distance_m"]] = removed[["kept_FID", "distance_m"]].astype("Int64")
    removed.to_csv(data_path(cfg, "facilities_removed"), index=False)
    print(f"wrote {len(fac)} facilities to {out_path}")
    print(f"wrote {len(removed)} removed facilities with reasons to {data_path(cfg, 'facilities_removed')}")
    print(fac["Facility_t"].value_counts().to_string())


if __name__ == "__main__":
    main()
