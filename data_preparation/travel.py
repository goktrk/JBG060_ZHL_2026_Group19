"""
Travel time to the nearest health facility, used by the baseline and the flood scenarios.

    res = travel_time(friction, sources, blocked=water)
    res.minutes   # minutes to the nearest source, NaN = unreachable
    res.nearest   # index into sources of that source, -1 = unreachable
"""
from dataclasses import dataclass

import numpy as np
from skimage.graph import MCP_Geometric


@dataclass(frozen=True)
class TravelResult:
    minutes: np.ndarray   # float32, NaN = unreachable
    nearest: np.ndarray   # int32 index into sources_rc, -1 = unreachable
    dropped: np.ndarray   # indices of sources that could not be used


def travel_time(friction, sources_rc, blocked=None, cell_size_m=250.0) -> TravelResult:
    """
    Minutes from every cell to the nearest source.

    friction is in minutes per metre (NaN or inf = impassable), sources_rc is an (N, 2) array of
    row/col and blocked is a boolean mask of cells that can't be crossed (water, flooded cells).
    Moves are 8-connected; a step costs the mean friction of the two cells times the step length.
    Source cells are always passable. A source without valid friction gets the median of its
    neighbours, or is dropped if it has none.
    """
    friction = np.asarray(friction)
    if friction.ndim != 2:
        raise ValueError(f"friction must be 2D, got shape {friction.shape}")
    if blocked is not None:
        blocked = np.asarray(blocked, dtype=bool)
        if blocked.shape != friction.shape:
            raise ValueError(f"blocked has shape {blocked.shape}, friction has {friction.shape}")
    sources = check_sources(sources_rc, friction.shape)

    costs = cell_costs(friction, blocked, cell_size_m)
    source_cost, usable = source_costs(friction, sources, cell_size_m)
    rows, cols = sources[usable, 0], sources[usable, 1]
    costs[rows, cols] = source_cost[usable]   # sources are passable even when blocked

    if usable.any():
        cumulative, traceback, offsets = solve(costs, sources[usable])
        labels = np.flatnonzero(usable).astype(np.int32)
        nearest = nearest_from_traceback(traceback, offsets, rows * friction.shape[1] + cols, labels)
    else:
        cumulative = np.full(friction.shape, np.inf)
        nearest = np.full(friction.shape, -1, dtype=np.int32)

    reached = np.isfinite(cumulative)
    nearest[~reached] = -1
    minutes = np.where(reached, cumulative, np.nan).astype(np.float32)
    return TravelResult(minutes=minutes, nearest=nearest, dropped=np.flatnonzero(~usable))


def check_sources(sources_rc, shape) -> np.ndarray:
    """Sources as an (N, 2) int64 array; they must be inside the grid and in different cells."""
    src = np.asarray(sources_rc)
    if src.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if src.ndim != 2 or src.shape[1] != 2:
        raise ValueError(f"sources_rc must have shape (N, 2), got {src.shape}")
    if not np.issubdtype(src.dtype, np.integer):
        raise ValueError(f"sources_rc must be integer row/col, got {src.dtype}")
    src = src.astype(np.int64)
    inside = (src[:, 0] >= 0) & (src[:, 0] < shape[0]) & (src[:, 1] >= 0) & (src[:, 1] < shape[1])
    if not inside.all():
        raise ValueError(f"sources outside the grid: {np.flatnonzero(~inside).tolist()}")
    if len(np.unique(src, axis=0)) != len(src):
        raise ValueError("two sources share a cell, merge them into one site first")
    return src


def cell_costs(friction, blocked, cell_size_m) -> np.ndarray:
    """Minutes to cross each cell, inf where it can't be crossed."""
    costs = friction.astype(np.float64) * cell_size_m
    # MCP gives wrong results (or crashes) on NaN, so everything impassable becomes inf
    impassable = ~np.isfinite(costs) | (costs < 0)
    if blocked is not None:
        impassable |= blocked
    costs[impassable] = np.inf
    return costs


def source_costs(friction, sources, cell_size_m):
    """Cost of each source cell, and whether the source can be used.

    Doesn't look at `blocked`, so blocking more cells never changes a source's cost."""
    cost = friction[sources[:, 0], sources[:, 1]].astype(np.float64)
    ok = np.isfinite(cost) & (cost >= 0)
    for i in np.flatnonzero(~ok):
        r, c = sources[i]
        around = friction[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2].astype(np.float64).ravel()
        around = around[np.isfinite(around) & (around >= 0)]
        if around.size:
            cost[i], ok[i] = np.median(around), True
    return cost * cell_size_m, ok


def solve(costs, starts):
    """Run MCP_Geometric from all starts at once."""
    mcp = MCP_Geometric(costs, fully_connected=True)
    cumulative, traceback = mcp.find_costs([tuple(rc) for rc in starts.tolist()])
    offsets = np.array(mcp.offsets, dtype=np.int64)   # copy before mcp goes away
    return cumulative, traceback, offsets


def nearest_from_traceback(traceback, offsets, start_flat, labels) -> np.ndarray:
    """Label each cell with the start its path comes from.

    traceback[cell] = k means the cell was reached from cell - offsets[k] (-1 for starts and
    unreached cells). Following parents with pointer jumping gets every cell to its start."""
    height, width = traceback.shape
    step = (offsets[:, 0] * width + offsets[:, 1]).astype(np.int32)
    tb = traceback.ravel()
    parent = np.arange(tb.size, dtype=np.int32)
    has_parent = tb >= 0
    parent[has_parent] -= step[tb[has_parent]]

    while True:
        jumped = parent[parent]
        if np.array_equal(jumped, parent):
            break
        parent = jumped

    label = np.full(tb.size, -1, dtype=np.int32)
    label[start_flat] = labels
    return label[parent].reshape(height, width)
