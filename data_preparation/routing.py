"""
Route from one point to the nearest health facility, avoiding cells that are often flooded.

    likelihood = load_flood_likelihood(10)                      # October
    blocked = blocked_mask(water, likelihood, threshold=0.2)    # flooded in > 20% of Octobers
    usable = open_sites(sites_rc, likelihood, threshold=0.2)    # facilities that can still provide care
    res = route(friction, (row, col), sites_rc[usable], blocked=blocked)
    res.path_rc, res.minutes, res.site                          # res.site indexes sites_rc[usable]

Step costs are the same as in travel.travel_time (8-connected, mean friction of the two cells times
the step length), so res.minutes equals travel_time(friction, sites, blocked).minutes at the start.
"""
from dataclasses import dataclass

import numpy as np
from pyproj import Transformer
from skimage.graph import MCP_Geometric

from data_preparation.grid import Grid
from data_preparation.travel import cell_costs, check_sources, source_costs


@dataclass(frozen=True)
class RouteResult:
    path_rc: np.ndarray     # (N, 2) row/col from the start to the facility; empty if unreachable
    minutes: float          # travel time, NaN = no facility can be reached
    length_m: float         # length of the path, NaN = unreachable
    site: int               # index into the sites passed to route(), -1 = unreachable
    through_no_data: bool   # the path crosses cells without flood data (not known to be dry)


def blocked_mask(water, likelihood, threshold) -> np.ndarray:
    """Cells that can't be crossed: permanent water, and cells flooded in more than `threshold` of years.
    Cells without flood data (NaN) are not blocked."""
    blocked = np.asarray(likelihood) > threshold
    if water is not None:
        blocked |= np.asarray(water, dtype=bool)
    return blocked


def open_sites(sites_rc, likelihood, threshold) -> np.ndarray:
    """True for sites that can provide care: their own cell is not flooded in more than `threshold` of years."""
    sites = np.asarray(sites_rc)
    return ~(np.asarray(likelihood)[sites[:, 0], sites[:, 1]] > threshold)


def route(friction, start_rc, sites_rc, blocked=None, cell_size_m=250.0, no_data=None) -> RouteResult:
    """
    Least-cost path from `start_rc` (row, col) to the nearest of `sites_rc` ((N, 2) row/col).

    friction is in minutes per metre (NaN or inf = impassable) and blocked is a boolean mask of cells
    that can't be crossed. The start and site cells are always passable, so someone standing in a
    blocked cell can still leave it; a start without valid friction gets the median of its neighbours.
    no_data, if given, marks cells without flood data so the result can say the path crosses them.
    """
    friction = np.asarray(friction)
    if friction.ndim != 2:
        raise ValueError(f"friction must be 2D, got shape {friction.shape}")
    if blocked is not None:
        blocked = np.asarray(blocked, dtype=bool)
        if blocked.shape != friction.shape:
            raise ValueError(f"blocked has shape {blocked.shape}, friction has {friction.shape}")
    start = check_sources(np.asarray([start_rc]), friction.shape)
    sites = check_sources(sites_rc, friction.shape)
    unreachable = RouteResult(np.empty((0, 2), dtype=np.int64), np.nan, np.nan, -1, False)

    costs = cell_costs(friction, blocked, cell_size_m)
    site_cost, usable = source_costs(friction, sites, cell_size_m)
    costs[sites[usable, 0], sites[usable, 1]] = site_cost[usable]
    start_cost, start_ok = source_costs(friction, start, cell_size_m)
    if not start_ok[0] or not usable.any():
        return unreachable
    costs[start[0, 0], start[0, 1]] = start_cost[0]

    ends = sites[usable]
    mcp = MCP_Geometric(costs, fully_connected=True)
    cumulative, _ = mcp.find_costs([tuple(start[0])], ends=[tuple(rc) for rc in ends.tolist()],
                                   find_all_ends=False)
    end_cost = cumulative[ends[:, 0], ends[:, 1]]
    if not np.isfinite(end_cost).any():
        return unreachable
    k = int(np.argmin(np.where(np.isfinite(end_cost), end_cost, np.inf)))

    path = np.array(mcp.traceback(tuple(ends[k])), dtype=np.int64)
    steps = np.diff(path, axis=0)
    length = float(np.sqrt((steps ** 2).sum(axis=1)).sum() * cell_size_m)
    crosses = bool(no_data is not None and np.asarray(no_data)[path[:, 0], path[:, 1]].any())
    return RouteResult(path, float(end_cost[k]), length, int(np.flatnonzero(usable)[k]), crosses)


def rc_to_lonlat(grid: Grid, rows, cols) -> tuple[np.ndarray, np.ndarray]:
    """Longitude and latitude of the centres of the given cells (the inverse of grid.lonlat_to_rowcol)."""
    x = grid.transform.c + (np.asarray(cols) + 0.5) * grid.cell_size
    y = grid.transform.f - (np.asarray(rows) + 0.5) * grid.cell_size
    return Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True).transform(x, y)
