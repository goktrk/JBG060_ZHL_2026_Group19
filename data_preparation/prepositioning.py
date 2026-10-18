"""
Where people go when a flood closes their facility or blocks their way there, and which facilities
receive them: candidates for pre-positioning supplies, staff or mobile teams. This does not assume
anyone redistributes patients, only that people go to the nearest open facility themselves.

    groups = classify(pop, base_nearest, flood_nearest, flooded, closed)
    flows = flow_table(pop, base_nearest, base_minutes, flood_nearest, flood_minutes, groups, 120, 15)
    receiving = receiving_table(pop, base_nearest, flows, n_sites)

The calculation is described in README_prepositioning.md.
"""
import numpy as np
import pandas as pd

# what happens to the people of a populated cell (that could reach a facility without floods)
NOT_COUNTED, SAME, MOVED_CLOSED, MOVED_BLOCKED, LOST, FLOODED = -1, 0, 1, 2, 3, 4
GROUP_NAMES = {SAME: "same_facility", MOVED_CLOSED: "moved_facility_closed",
               MOVED_BLOCKED: "moved_path_blocked", LOST: "lost_all_access", FLOODED: "living_in_flooded_cells"}


def classify(pop, base_nearest, flood_nearest, flooded, closed) -> np.ndarray:
    """
    One group per cell (the codes above). Cells without people, or that couldn't reach any facility
    without floods, are NOT_COUNTED.

    - FLOODED:       the cell itself is flooded (counted separately, as in access loss)
    - LOST:          no facility reachable with the flood
    - SAME:          still goes to its usual facility
    - MOVED_CLOSED:  its usual facility is closed, so it goes to another one
    - MOVED_BLOCKED: its usual facility is open, but the flood makes another one closer
    """
    base, flood = np.asarray(base_nearest), np.asarray(flood_nearest)
    closed = np.asarray(closed, dtype=bool)
    groups = np.full(base.shape, NOT_COUNTED, dtype=np.int8)
    counted = (np.asarray(pop) > 0) & (base >= 0)
    usual_closed = np.zeros(base.shape, dtype=bool)
    usual_closed[counted] = closed[base[counted]]

    groups[counted & (flood == base)] = SAME
    groups[counted & (flood >= 0) & (flood != base) & usual_closed] = MOVED_CLOSED
    groups[counted & (flood >= 0) & (flood != base) & ~usual_closed] = MOVED_BLOCKED
    groups[counted & (flood < 0)] = LOST
    groups[counted & np.asarray(flooded, dtype=bool)] = FLOODED     # last: overrides the others
    return groups


def group_totals(pop, groups) -> dict[str, float]:
    """People per group."""
    p = np.asarray(pop, dtype="float64")
    return {name: float(p[groups == code].sum()) for code, name in GROUP_NAMES.items()}


def check_balance(pop, base_nearest, groups, closed):
    """
    Every counted person is in exactly one group, nobody stays at a closed facility, and every
    closed facility's usual population splits into moved + lost + flooded. Raises if not.
    """
    p = np.asarray(pop, dtype="float64")
    base = np.asarray(base_nearest)
    counted = (p > 0) & (base >= 0)
    if (groups[counted] == NOT_COUNTED).any() or (groups[~counted] != NOT_COUNTED).any():
        raise AssertionError("a counted cell has no group, or an uncounted cell has one")
    at_closed = counted & np.isin(base, np.flatnonzero(closed))
    if (groups[at_closed] == SAME).any():
        raise AssertionError("people still assigned to a closed facility")
    if (groups[at_closed] == MOVED_BLOCKED).any():
        raise AssertionError("a closed facility's people counted as 'path blocked'")
    total = sum(group_totals(pop, groups).values())
    if not np.isclose(total, p[counted].sum()):
        raise AssertionError(f"groups add up to {total:,.0f}, not {p[counted].sum():,.0f}")


def flow_table(pop, base_nearest, base_minutes, flood_nearest, flood_minutes, groups, limit_min,
               small_delay_min) -> pd.DataFrame:
    """
    People who moved, per (from_site, to_site, reason): how many, how many are within `limit_min`
    of the new facility, their population-weighted extra minutes, and how many have less than
    `small_delay_min` extra (small flips between two facilities at about the same distance).
    """
    moved = np.isin(groups, [MOVED_CLOSED, MOVED_BLOCKED])
    people = np.asarray(pop, dtype="float64")[moved]
    extra = (np.asarray(flood_minutes, dtype="float64") - np.asarray(base_minutes, dtype="float64"))[moved]
    cells = pd.DataFrame({
        "from_site": np.asarray(base_nearest)[moved],
        "to_site": np.asarray(flood_nearest)[moved],
        "reason": np.where(groups[moved] == MOVED_CLOSED, "facility_closed", "path_blocked"),
        "people": people,
        "people_within_limit": people * (np.asarray(flood_minutes)[moved] <= limit_min),
        "people_small_delay": people * (extra < small_delay_min),
        "extra_people_min": people * extra,
    })
    flows = cells.groupby(["from_site", "to_site", "reason"], as_index=False).sum()
    flows["extra_min"] = flows.pop("extra_people_min") / flows["people"]
    return flows.sort_values("people", ascending=False, ignore_index=True)


def receiving_table(pop, base_nearest, flows, n_sites) -> pd.DataFrame:
    """
    One row per facility that receives people: its usual population, the people added (all, and
    within the limit), where they come from, their extra travel time, and the relative increase
    (added / usual; NaN when the facility usually serves nobody).
    """
    p = np.asarray(pop, dtype="float64")
    base = np.asarray(base_nearest)
    usual = np.bincount(base[(p > 0) & (base >= 0)], weights=p[(p > 0) & (base >= 0)], minlength=n_sites)
    by_to = flows.groupby("to_site")
    table = pd.DataFrame({
        "added": by_to["people"].sum(),
        "added_within_limit": by_to["people_within_limit"].sum(),
        "added_from_closed": flows[flows["reason"] == "facility_closed"].groupby("to_site")["people"].sum(),
        "added_from_blocked_path": flows[flows["reason"] == "path_blocked"].groupby("to_site")["people"].sum(),
        "added_blocked_small_delay": flows[flows["reason"] == "path_blocked"].groupby("to_site")["people_small_delay"].sum(),
        "extra_min": (flows["people"] * flows["extra_min"]).groupby(flows["to_site"]).sum() / by_to["people"].sum(),
        "from_facilities": by_to["from_site"].nunique(),
    }).fillna({"added_from_closed": 0.0, "added_from_blocked_path": 0.0, "added_blocked_small_delay": 0.0})
    sent = flows[flows["reason"] == "path_blocked"].groupby("from_site")["people"].sum()
    table.index.name = "site_id"
    table.insert(0, "usual_population", usual[table.index])
    table["sent_away_path_blocked"] = sent.reindex(table.index).fillna(0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        table["relative_increase"] = np.where(table["usual_population"] > 0,
                                              table["added"] / table["usual_population"], np.nan)
        table["relative_increase_within_limit"] = np.where(
            table["usual_population"] > 0, table["added_within_limit"] / table["usual_population"], np.nan)
    return table.reset_index().sort_values("added_within_limit", ascending=False, ignore_index=True)
