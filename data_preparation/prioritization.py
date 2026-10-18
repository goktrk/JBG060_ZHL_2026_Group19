"""
Which closed facility sites to protect first in a flood scenario.

For every site the flood closes, re-run the scenario with only that site reopened ("protected"),
and count who gets access back. Two rankings, both always shown:

- most_saved: people brought back within `travel.max_time_min` (2 h) of care;
- worst_off:  people who would be more than `priorities.worst_off_min` (4 h) from care without the
              site, brought back within it. This favours remote communities with nowhere else to go.

The Top N is chosen greedily: first the site that brings back the most people, then the site that
brings back the most of the people still left, and so on, so two neighbouring sites that rescue the
same village don't both take a place. The full calculation is in README_prioritization.md.
Which cells are flooded, which sites closed and who loses access come from scenarios.py.

    closed = closed_sites(sites_rc, flooded, buffer_m=1000, cell_size_m=250)     # scenarios.py
    flood = travel_time(friction, sites_rc[~closed], blocked=water | flooded)    # or read from build_access_loss
    regained, lost = regained_cells(pop, base.minutes, flood.minutes, flooded, friction,
                                    water | flooded, sites_rc, np.flatnonzero(closed), limit_min=120)
    order = greedy_order(regained, pop)
"""
import math

import numpy as np
import pandas as pd
from skimage.graph import MCP_Geometric

from data_preparation.scenarios import lost_cells
from data_preparation.travel import cell_costs, source_costs


def reach_radius(friction, max_min, cell_size_m) -> int:
    """
    Cells that can be reached within `max_min` minutes all lie within this many cells of the start:
    no step is cheaper than the lowest friction in the grid, so nothing further away is reachable in time.
    """
    f = np.asarray(friction, dtype="float64")
    fastest = f[np.isfinite(f) & (f > 0)].min()
    return math.ceil(max_min / (fastest * cell_size_m)) + 1


def time_from_site(friction, site_rc, blocked, radius, cell_size_m):
    """
    Minutes from one site to the cells within `radius` cells of it, with the same step costs as
    travel.travel_time (the site's own cell is passable even if blocked). Exact for every cell whose
    time is below the limit `radius` was computed for.

    Returns the window (row slice, col slice) and the minutes in it (NaN = not reached), or
    (None, None, None) if the site's cell has no usable friction.
    """
    r, c = int(site_rc[0]), int(site_rc[1])
    rows = slice(max(r - radius, 0), r + radius + 1)
    cols = slice(max(c - radius, 0), c + radius + 1)
    window_friction = np.asarray(friction)[rows, cols]
    local = np.array([[r - rows.start, c - cols.start]])

    cost, usable = source_costs(window_friction, local, cell_size_m)
    if not usable[0]:
        return None, None, None
    costs = cell_costs(window_friction, None if blocked is None else np.asarray(blocked)[rows, cols], cell_size_m)
    costs[local[0, 0], local[0, 1]] = cost[0]
    cumulative, _ = MCP_Geometric(costs, fully_connected=True).find_costs([tuple(local[0])])
    return rows, cols, np.where(np.isfinite(cumulative), cumulative, np.nan)


def regained_cells(pop, base_minutes, flood_minutes, flooded, friction, blocked, sites_rc, candidates,
                   limit_min, cell_size_m=250.0) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """
    For each candidate site (indices into sites_rc), the cells whose people get access back within
    `limit_min` when only that site is reopened, as flat indices into the grid.

    Reopening one site adds one source, so each cell's new time is the lower of its flood-scenario
    time and its time to that site. Only that second time has to be solved, from the site alone.
    Also returns the `lost_cells` mask the gains are counted against.
    """
    lost = lost_cells(pop, base_minutes, flood_minutes, flooded, limit_min)
    radius = reach_radius(friction, limit_min, cell_size_m)
    width = np.asarray(pop).shape[1]
    regained = {}
    for i in candidates:
        rows, cols, minutes = time_from_site(friction, sites_rc[i], blocked, radius, cell_size_m)
        if minutes is None:
            regained[int(i)] = np.empty(0, dtype=np.int64)
            continue
        with np.errstate(invalid="ignore"):
            hit = lost[rows, cols] & (minutes <= limit_min)
        r, c = np.nonzero(hit)
        regained[int(i)] = (r + rows.start).astype(np.int64) * width + (c + cols.start)
    return regained, lost


def greedy_order(regained: dict[int, np.ndarray], pop) -> pd.DataFrame:
    """
    Order sites greedily: each pick brings back the most people not already brought back by the
    earlier picks. Ties go to the larger gain on its own, then the lower site id.

    Returns one row per site: site_id, gain (people it brings back on its own), rank (1 = protect
    first), added (people this pick adds) and cumulative (people brought back by ranks 1..rank).
    """
    flat_pop = np.asarray(pop, dtype="float64").ravel()
    ids = sorted(regained)
    every = np.unique(np.concatenate([regained[i] for i in ids])) if ids else np.empty(0, np.int64)
    weight = flat_pop[every]
    members = {i: np.searchsorted(every, regained[i]) for i in ids}
    gain = {i: float(weight[members[i]].sum()) for i in ids}

    covered = np.zeros(len(every), dtype=bool)
    rows, total, left = [], 0.0, list(ids)
    while left:
        added = {i: float(weight[members[i]][~covered[members[i]]].sum()) for i in left}
        best = max(left, key=lambda i: (added[i], gain[i], -i))
        covered[members[best]] = True
        total += added[best]
        rows.append({"site_id": best, "gain": gain[best], "rank": len(rows) + 1,
                     "added": added[best], "cumulative": total})
        left.remove(best)
    return pd.DataFrame(rows, columns=["site_id", "gain", "rank", "added", "cumulative"])
