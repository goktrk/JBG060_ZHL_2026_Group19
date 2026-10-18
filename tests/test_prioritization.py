"""
Tests for data_preparation/prioritization.py. Run from the repository root:  python -m pytest tests
"""
import numpy as np
import pytest

from data_preparation.prioritization import greedy_order, reach_radius, regained_cells, time_from_site
from data_preparation.scenarios import closed_sites, lost_cells
from data_preparation.travel import travel_time

CELL = 250.0   # 0.012 min/m x 250 m = 3 minutes per cell (5 km/h)


def test_closed_sites_uses_a_round_buffer():
    flooded = np.zeros((21, 21), dtype=bool)
    sites = np.array([[10, 10]])
    assert not closed_sites(sites, flooded, 1000, CELL)[0]
    flooded[10, 14] = True                              # 4 cells = 1000 m east: inside
    assert closed_sites(sites, flooded, 1000, CELL)[0]
    assert not closed_sites(sites, flooded, 0, CELL)[0]  # buffer 0: only the own cell counts
    flooded[10, 14] = False
    flooded[14, 14] = True                              # 4 cells diagonally = 1414 m: outside
    assert not closed_sites(sites, flooded, 1000, CELL)[0]
    flooded[10, 10] = True
    assert closed_sites(sites, flooded, 0, CELL)[0]


def test_closed_sites_works_at_the_grid_edge():
    flooded = np.zeros((5, 5), dtype=bool)
    flooded[0, 2] = True
    assert closed_sites(np.array([[0, 0], [4, 4]]), flooded, 500, CELL).tolist() == [True, False]


def random_case(seed=0, shape=(40, 50)):
    rng = np.random.default_rng(seed)
    friction = rng.uniform(0.012, 0.05, shape)
    blocked = rng.random(shape) < 0.15
    sites = np.array([[5, 5], [20, 25], [35, 45], [10, 40]])
    blocked[sites[:, 0], sites[:, 1]] = False
    return friction, blocked, sites


def test_time_from_site_matches_a_full_solve_within_the_limit():
    friction, blocked, sites = random_case()
    limit = 60.0
    radius = reach_radius(friction, limit, CELL)
    rows, cols, window = time_from_site(friction, sites[1], blocked, radius, CELL)
    full = travel_time(friction, sites[[1]], blocked=blocked).minutes[rows, cols]
    within = full <= limit
    assert within.sum() > 20
    np.testing.assert_allclose(window[within], full[within], rtol=1e-5)


def test_reopening_one_site_equals_a_full_rerun():
    """The shortcut: new time = min(flood-scenario time, time to the reopened site)."""
    friction, blocked, sites = random_case(seed=1)
    open_ = np.array([True, False, True, True])           # site 1 closed in the scenario
    limit = 90.0
    flood = travel_time(friction, sites[open_], blocked=blocked).minutes
    rerun = travel_time(friction, sites, blocked=blocked).minutes   # site 1 reopened
    rows, cols, from_site = time_from_site(friction, sites[1], blocked, reach_radius(friction, limit, CELL), CELL)
    shortcut = flood.copy()
    shortcut[rows, cols] = np.fmin(flood[rows, cols], from_site)
    within = rerun <= limit
    np.testing.assert_allclose(shortcut[within], rerun[within], rtol=1e-5)


def corridor():
    """A 1 x 81 corridor, 3 min per cell. Site A at column 0 is closed by the flood, site B at column 80."""
    friction = np.full((1, 81), 0.012)
    sites = np.array([[0, 0], [0, 80]])
    pop = np.zeros((1, 81))
    pop[0, 10] = 100        # 30 min from A, 210 from B
    pop[0, 30] = 50         # 90 min from A, 150 from B
    return friction, sites, pop


def test_regained_counts_only_people_brought_back_within_the_limit():
    friction, sites, pop = corridor()
    flooded = np.zeros_like(pop, dtype=bool)
    base = travel_time(friction, sites).minutes
    flood = travel_time(friction, sites[[1]]).minutes      # A closed
    regained, lost = regained_cells(pop, base, flood, flooded, friction, flooded, sites, [0], limit_min=120)
    assert sorted(regained[0].tolist()) == [10, 30]        # both were > 120 min from B
    regained, _ = regained_cells(pop, base, flood, flooded, friction, flooded, sites, [0], limit_min=60)
    assert regained[0].tolist() == [10]                    # 90 min from A is beyond 60 even with A open
    regained, _ = regained_cells(pop, base, flood, flooded, friction, flooded, sites, [0], limit_min=240)
    assert regained[0].tolist() == []                      # nobody is more than 240 min from B


def test_people_in_flooded_cells_and_already_remote_people_are_not_counted():
    friction, sites, pop = corridor()
    flooded = np.zeros_like(pop, dtype=bool)
    flooded[0, 10] = True                                  # their home is under water
    base = travel_time(friction, sites).minutes
    flood = travel_time(friction, sites[[1]], blocked=flooded).minutes
    regained, lost = regained_cells(pop, base, flood, flooded, friction, flooded, sites, [0], limit_min=120)
    assert not lost[0, 10] and 10 not in regained[0].tolist()
    # someone already more than 120 min from every site before the flood can't "regain" access
    pop2 = np.zeros_like(pop)
    pop2[0, 50] = 10                                       # 150 min from A, 90 from B
    assert not lost_cells(pop2, base, travel_time(friction, sites).minutes, flooded, 60).any()


def test_a_walled_in_site_brings_back_nobody():
    """The old ranking's flaw: a big site cut off by flooded ground ranked high but helped nobody."""
    friction = np.full((15, 15), 0.012)
    sites = np.array([[7, 7], [0, 0]])
    pop = np.full((15, 15), 10.0)
    flooded = np.zeros((15, 15), dtype=bool)
    flooded[5:10, 5:10] = True                             # the ring around site 0, and site 0 itself
    blocked = flooded
    base = travel_time(friction, sites).minutes
    flood = travel_time(friction, sites[[1]], blocked=blocked).minutes
    regained, _ = regained_cells(pop, base, flood, flooded, friction, blocked, sites, [0], limit_min=20)
    assert regained[0].size == 0


def test_greedy_skips_a_site_that_rescues_the_same_people():
    pop = np.zeros((1, 10))
    pop[0, :] = [100, 100, 100, 50, 50, 0, 0, 0, 0, 30]
    regained = {
        3: np.array([0, 1, 2]),        # 300 people
        5: np.array([0, 1]),           # 200, all already rescued by site 3
        7: np.array([3, 4]),           # 100, nobody else reaches them
        8: np.array([9]),              # 30
        9: np.array([], dtype=np.int64),
    }
    order = greedy_order(regained, pop).set_index("site_id")
    assert order["rank"].sort_values().index.tolist() == [3, 7, 8, 5, 9]
    assert order.loc[5, "gain"] == 200 and order.loc[5, "added"] == 0
    assert order.loc[8, "cumulative"] == 430
    assert order["cumulative"].max() == pytest.approx(430)


def test_greedy_breaks_ties_by_own_gain_then_site_id():
    pop = np.ones((1, 6))
    regained = {4: np.array([0, 1]), 2: np.array([2, 3]), 6: np.array([0, 1, 4])}
    order = greedy_order(regained, pop)
    assert order["site_id"].tolist() == [6, 2, 4]          # 6 adds 3; then 2 adds 2, 4 adds only 0
    assert greedy_order({}, pop).empty
