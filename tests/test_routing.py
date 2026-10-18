"""
Tests for data_preparation/routing.py and the flood likelihood. Run from the repository root:  python -m pytest tests
"""
import numpy as np
import pytest

from data_preparation.build_flood_likelihood import likelihood_from_years
from data_preparation.build_flood_monthly import NO_DATA
from data_preparation.routing import blocked_mask, open_sites, route
from data_preparation.travel import travel_time

CELL = 250.0


def random_case(rng, shape=(80, 80), n_sites=6, blocked_share=0.15):
    friction = rng.uniform(0.01, 0.2, shape)
    blocked = rng.random(shape) < blocked_share
    flat = rng.choice(shape[0] * shape[1], n_sites + 1, replace=False)
    cells = np.column_stack(np.unravel_index(flat, shape))
    return friction, blocked, cells[:-1], tuple(cells[-1])


def assert_valid_path(res, blocked, start, sites):
    path = res.path_rc
    assert tuple(path[0]) == tuple(start)
    assert tuple(path[-1]) == tuple(sites[res.site])
    assert (np.abs(np.diff(path, axis=0)).max(axis=1) == 1).all()   # 8-connected, no jumps
    assert not blocked[path[1:-1, 0], path[1:-1, 1]].any()          # only the ends may be blocked


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_matches_travel_time(seed):
    rng = np.random.default_rng(seed)
    friction, blocked, sites, start = random_case(rng)
    blocked[start] = False
    res = route(friction, start, sites, blocked)
    expected = travel_time(friction, sites, blocked)
    if np.isnan(expected.minutes[start]):
        assert res.site == -1 and np.isnan(res.minutes)
        return
    assert res.minutes == pytest.approx(float(expected.minutes[start]), rel=1e-5)
    assert res.site == expected.nearest[start] or res.minutes == pytest.approx(
        float(travel_time(friction, sites[[res.site]], blocked).minutes[start]), rel=1e-5)   # tie
    assert_valid_path(res, blocked, start, sites)


def test_goes_through_the_gap_in_a_wall():
    friction = np.full((21, 21), 0.012)
    blocked = np.zeros((21, 21), dtype=bool)
    blocked[:, 10] = True
    blocked[0, 10] = False
    sites = np.array([[10, 18]])
    res = route(friction, (10, 2), sites, blocked)
    assert_valid_path(res, blocked, (10, 2), sites)
    assert [0, 10] in res.path_rc.tolist()
    assert res.minutes > 16 * 3.0                 # longer than going straight
    assert res.length_m > 16 * CELL

    blocked[0, 10] = True                         # close the gap
    res = route(friction, (10, 2), sites, blocked)
    assert res.site == -1 and np.isnan(res.minutes) and len(res.path_rc) == 0


def test_flood_prone_facility_is_skipped():
    friction = np.full((15, 15), 0.012)
    likelihood = np.zeros((15, 15), dtype=np.float32)
    sites = np.array([[7, 9], [7, 14]])
    likelihood[7, 9] = 0.5                        # nearest facility floods every other year
    usable = open_sites(sites, likelihood, threshold=0.2)
    assert usable.tolist() == [False, True]
    res = route(friction, (7, 2), sites[usable], blocked_mask(None, likelihood, 0.2))
    assert res.site == 0 and tuple(res.path_rc[-1]) == (7, 14)
    assert [7, 9] not in res.path_rc.tolist()      # its cell is blocked too, so the route steps around it
    assert res.minutes == pytest.approx(10 * 3.0 + 2 * 3.0 * np.sqrt(2))


def test_start_in_a_blocked_cell_still_leaves_it():
    friction = np.full((11, 11), 0.012)
    blocked = np.zeros((11, 11), dtype=bool)
    blocked[3:8, 3:6] = True                      # standing at the edge of a flooded area, in (5, 5)
    res = route(friction, (5, 5), np.array([[5, 10]]), blocked)
    assert res.site == 0 and res.minutes == pytest.approx(5 * 3.0)
    assert not blocked[res.path_rc[1:, 0], res.path_rc[1:, 1]].any()


def test_enclosed_start_and_bad_start_are_unreachable():
    friction = np.full((15, 15), 0.012)
    blocked = np.zeros((15, 15), dtype=bool)
    blocked[4, 4:11] = blocked[10, 4:11] = blocked[4:11, 4] = blocked[4:11, 10] = True
    assert route(friction, (7, 7), np.array([[0, 0]]), blocked).site == -1

    friction[0:5, 0:5] = np.nan                   # no valid friction around the start
    assert route(friction, (1, 1), np.array([[12, 12]])).site == -1


def test_start_on_a_facility_is_zero():
    res = route(np.full((5, 5), 0.012), (2, 2), np.array([[0, 0], [2, 2]]))
    assert res.site == 1 and res.minutes == 0 and res.length_m == 0


def test_reports_crossing_no_data():
    friction = np.full((5, 9), 0.012)
    no_data = np.zeros((5, 9), dtype=bool)
    no_data[:, 4] = True
    assert route(friction, (2, 0), np.array([[2, 8]]), no_data=no_data).through_no_data
    assert not route(friction, (2, 0), np.array([[2, 3]]), no_data=no_data).through_no_data


def test_blocked_mask_leaves_no_data_open():
    likelihood = np.array([[0.0, 0.3, np.nan]], dtype=np.float32)
    water = np.array([[True, False, False]])
    assert blocked_mask(water, likelihood, 0.2).tolist() == [[True, True, False]]
    assert blocked_mask(None, likelihood, 0.3).tolist() == [[False, False, False]]


def test_likelihood_from_years():
    years = [np.array([[0, 5, NO_DATA, 3]], dtype=np.uint8),
             np.array([[0, 0, NO_DATA, 2]], dtype=np.uint8),
             np.array([[1, 9, NO_DATA, NO_DATA]], dtype=np.uint8),
             np.array([[2, 0, NO_DATA, 4]], dtype=np.uint8)]
    share = likelihood_from_years(iter(years), min_days=3)
    assert share.dtype == np.float32
    np.testing.assert_allclose(share[0, [0, 1, 3]], [0.0, 0.5, 2 / 3])   # no-data years left out
    assert np.isnan(share[0, 2])


def test_protected_site_stays_open_but_its_surroundings_stay_blocked():
    friction = np.full((15, 15), 0.012)
    likelihood = np.zeros((15, 15), dtype=np.float32)
    sites = np.array([[7, 9], [7, 14]])
    likelihood[7, 9] = 0.5                        # the nearest facility is flood-prone
    blocked = blocked_mask(None, likelihood, 0.0)
    usable = open_sites(sites, likelihood, 0.0)
    assert route(friction, (7, 2), sites[usable], blocked).site == 0   # goes on to (7, 14)

    protected = usable.copy()
    protected[0] = True                           # as the dashboard does for the Top 5
    res = route(friction, (7, 2), sites[protected], blocked)
    assert tuple(sites[protected][res.site]) == (7, 9) and res.minutes == pytest.approx(7 * 3.0)

    likelihood[5:10, 7:12] = 0.5                  # the ground around it floods too
    res = route(friction, (7, 2), sites[protected], blocked_mask(None, likelihood, 0.0))
    assert tuple(sites[protected][res.site]) == (7, 14)
