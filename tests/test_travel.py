"""
Tests for data_preparation/travel.py. Run from the repository root:  python -m pytest tests
"""
import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from data_preparation.travel import cell_costs, nearest_from_traceback, solve, travel_time

CELL = 250.0
MOVES = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def random_case(rng, shape=(150, 150), n_sources=6, blocked_share=0.15):
    friction = rng.uniform(0.01, 0.2, shape)
    blocked = rng.random(shape) < blocked_share
    flat = rng.choice(shape[0] * shape[1], n_sources, replace=False)
    sources = np.column_stack(np.unravel_index(flat, shape))
    return friction, blocked, sources


def dijkstra_minutes(friction, blocked, sources):
    """Minutes from each source to every cell with scipy, using the same step costs as MCP_Geometric."""
    h, w = friction.shape
    cost = friction * CELL
    passable = ~blocked
    passable[sources[:, 0], sources[:, 1]] = True
    idx = np.arange(h * w).reshape(h, w)
    rows, cols, weights = [], [], []
    for dr, dc in MOVES:
        a = (slice(max(-dr, 0), h - max(dr, 0)), slice(max(-dc, 0), w - max(dc, 0)))
        b = (slice(max(dr, 0), h - max(-dr, 0)), slice(max(dc, 0), w - max(-dc, 0)))
        ok = passable[a] & passable[b]
        rows.append(idx[a][ok])
        cols.append(idx[b][ok])
        weights.append(((cost[a] + cost[b]) / 2 * np.hypot(dr, dc))[ok])
    graph = coo_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(h * w, h * w)).tocsr()
    return dijkstra(graph, directed=True, indices=sources[:, 0] * w + sources[:, 1])


def nan_to_inf(minutes):
    return np.where(np.isnan(minutes), np.inf, minutes)


def test_uniform_straight_and_diagonal():
    friction = np.full((31, 31), 0.012)
    res = travel_time(friction, np.array([[15, 15]]), cell_size_m=CELL)
    assert res.minutes[15, 25] == pytest.approx(10 * 250 * 0.012)   # 30 min
    assert res.minutes[5, 15] == pytest.approx(30.0)
    assert res.minutes[25, 25] == pytest.approx(30.0 * np.sqrt(2), rel=1e-6)
    assert res.minutes.dtype == np.float32 and res.nearest.dtype == np.int32


def test_wall_with_gap_and_enclosed_region():
    friction = np.full((21, 21), 0.012)
    blocked = np.zeros((21, 21), dtype=bool)
    blocked[:, 10] = True
    blocked[0, 10] = False   # gap at the top
    res = travel_time(friction, np.array([[10, 2]]), blocked)
    expected = dijkstra_minutes(friction, blocked, np.array([[10, 2]]))[0].reshape(21, 21)
    assert res.minutes[10, 18] == pytest.approx(expected[10, 18], rel=1e-6)
    assert res.minutes[10, 18] > 16 * 3.0   # longer than going straight
    assert np.isnan(res.minutes[blocked]).all() and (res.nearest[blocked] == -1).all()

    blocked[0, 10] = True   # close the gap
    res = travel_time(friction, np.array([[10, 2]]), blocked)
    assert np.isnan(res.minutes[:, 11:]).all() and (res.nearest[:, 11:] == -1).all()

    blocked = np.zeros((21, 21), dtype=bool)
    blocked[5, 5:12] = blocked[11, 5:12] = blocked[5:12, 5] = blocked[5:12, 11] = True   # closed ring
    res = travel_time(friction, np.array([[0, 0]]), blocked)
    assert np.isnan(res.minutes[6:11, 6:11]).all() and (res.nearest[6:11, 6:11] == -1).all()
    assert np.isfinite(res.minutes[0:5, :]).all()


def test_sources_are_zero_and_label_themselves():
    rng = np.random.default_rng(3)
    friction, blocked, sources = random_case(rng, (60, 60), n_sources=8)
    blocked[sources[0, 0], sources[0, 1]] = True   # a blocked source still works
    res = travel_time(friction, sources, blocked)
    assert (res.minutes[sources[:, 0], sources[:, 1]] == 0).all()
    assert (res.nearest[sources[:, 0], sources[:, 1]] == np.arange(len(sources))).all()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_scipy_dijkstra(seed):
    rng = np.random.default_rng(seed)
    friction, blocked, sources = random_case(rng)
    res = travel_time(friction, sources, blocked)
    per_source = dijkstra_minutes(friction, blocked, sources)
    best = per_source.min(axis=0).reshape(friction.shape)

    assert np.array_equal(np.isfinite(best), np.isfinite(res.minutes))
    reached = np.isfinite(best)
    cells = np.flatnonzero(reached.ravel())
    np.testing.assert_allclose(res.minutes[reached], best[reached], rtol=1e-5)

    # our nearest source must be a closest one; it may only differ from scipy's on a tie
    ours = per_source[res.nearest[reached], cells]
    np.testing.assert_allclose(ours, best[reached], rtol=1e-9)
    theirs = per_source.argmin(axis=0).reshape(friction.shape)[reached]
    differs = theirs != res.nearest[reached]
    gap = np.abs(per_source[theirs, cells] - ours)
    assert (gap[differs] <= 1e-9 * best[reached][differs]).all()


@pytest.mark.parametrize("seed", [10, 11, 12, 13])
def test_more_blocking_or_fewer_sources_is_never_faster(seed):
    rng = np.random.default_rng(seed)
    friction, blocked, sources = random_case(rng, (100, 100), n_sources=10, blocked_share=0.1)
    base = nan_to_inf(travel_time(friction, sources, blocked).minutes)

    more_blocked = blocked | (rng.random(blocked.shape) < 0.1)
    more_blocked[sources[0, 0], sources[0, 1]] = True
    flooded = nan_to_inf(travel_time(friction, sources, more_blocked).minutes)
    assert (flooded >= base - 1e-4).all()

    keep = rng.random(len(sources)) < 0.5
    keep[0] = True
    fewer = nan_to_inf(travel_time(friction, sources[keep], blocked).minutes)
    assert (fewer >= base - 1e-4).all()


def test_nearest_is_where_the_path_ends():
    rng = np.random.default_rng(42)
    friction, blocked, sources = random_case(rng, (40, 40), n_sources=5)
    costs = cell_costs(friction, blocked, CELL)
    costs[sources[:, 0], sources[:, 1]] = friction[sources[:, 0], sources[:, 1]] * CELL
    cumulative, traceback, offsets = solve(costs, sources)
    w = friction.shape[1]
    nearest = nearest_from_traceback(traceback, offsets, sources[:, 0] * w + sources[:, 1],
                                     np.arange(len(sources), dtype=np.int32))

    # walk the traceback by hand from every reached cell
    source_index = {tuple(rc): i for i, rc in enumerate(sources.tolist())}
    for r0, c0 in zip(*np.nonzero(np.isfinite(cumulative))):
        r, c = r0, c0
        while traceback[r, c] >= 0:
            dr, dc = offsets[traceback[r, c]]
            r, c = r - dr, c - dc
        assert (r, c) in source_index
        assert nearest[r0, c0] == source_index[(r, c)]
    assert np.array_equal(nearest, travel_time(friction, sources, blocked).nearest)


def test_source_with_bad_friction_is_filled_or_dropped():
    friction = np.full((9, 9), 0.012)
    friction[2, 2] = np.nan      # gets the median of its neighbours
    friction[6:9, 6:9] = np.nan  # no valid neighbours, so it's dropped
    res = travel_time(friction, np.array([[2, 2], [7, 7]]))
    assert res.dropped.tolist() == [1]
    assert res.minutes[2, 4] == pytest.approx(3.0 + 3.0)
    assert not (res.nearest == 1).any()
    assert np.isnan(res.minutes[7, 7])


def test_rejects_shared_cells_and_outside_sources():
    friction = np.full((5, 5), 0.012)
    with pytest.raises(ValueError):
        travel_time(friction, np.array([[1, 1], [1, 1]]))
    with pytest.raises(ValueError):
        travel_time(friction, np.array([[5, 0]]))


def test_no_sources_reaches_nothing():
    res = travel_time(np.full((5, 5), 0.012), np.empty((0, 2), dtype=int))
    assert np.isnan(res.minutes).all() and (res.nearest == -1).all()
