"""
Tests for data_preparation/prioritization.py. Run from the repository root:  python -m pytest tests
"""
import numpy as np
import pandas as pd
import pytest

from data_preparation.prioritization import flooded_sites, rank_sites, site_impacts
from data_preparation.routing import blocked_mask, open_sites, route
from data_preparation.travel import travel_time


def two_site_case():
    """A 3 x 61 corridor, 3 min per cell: site A at the west end (flood-prone), site B at the east end."""
    friction = np.full((3, 61), 0.012)
    likelihood = np.zeros((3, 61), dtype=np.float32)
    likelihood[1, 0] = 0.5
    sites = np.array([[1, 0], [1, 60]])
    pop = np.zeros((3, 61), dtype=np.float32)
    return friction, likelihood, sites, pop


def impacts_for(friction, likelihood, sites, pop, threshold=0.0):
    flooded = flooded_sites(sites, likelihood, threshold)
    base = travel_time(friction, sites)
    flood = travel_time(friction, sites[~flooded], blocked_mask(None, likelihood, threshold))
    return site_impacts(pop, base.nearest, base.minutes, flood.minutes, len(sites), max_time_min=120)


def test_flooded_sites_flags_above_threshold_and_ignores_no_data():
    likelihood = np.array([[0.0, 0.3, np.nan, 0.1]], dtype=np.float32)
    sites = np.array([[0, 0], [0, 1], [0, 2], [0, 3]])
    assert flooded_sites(sites, likelihood, 0.0).tolist() == [False, True, False, True]
    assert flooded_sites(sites, likelihood, 0.2).tolist() == [False, True, False, False]


def test_losing_a_site_cuts_off_its_catchment():
    friction, likelihood, sites, pop = two_site_case()
    pop[1, 1:5] = 10          # near A: 168-177 min from B once A is lost
    pop[1, 55:60] = 20        # near B
    impacts = impacts_for(friction, likelihood, sites, pop)
    assert impacts["catchment_pop"].tolist() == [40, 100]
    assert impacts["share_cut_off"].tolist() == [1.0, 0.0]
    assert impacts.loc[0, "extra_min"] == pytest.approx(np.mean([180 - 6 * c for c in range(1, 5)]))
    assert impacts.loc[1, "extra_min"] == 0


def test_cut_off_counts_unreachable_and_only_people_beyond_the_limit():
    friction, likelihood, sites, pop = two_site_case()
    pop[1, 1:5] = 10          # beyond 120 min from B
    pop[1, 25:30] = 12        # still nearer A, but within 120 min of B (93-105 min)
    impacts = impacts_for(friction, likelihood, sites, pop)
    assert impacts.loc[0, "catchment_pop"] == 100
    assert impacts.loc[0, "share_cut_off"] == pytest.approx(0.4)
    assert np.isnan(impacts.loc[1, "share_cut_off"])   # nobody lives in B's catchment

    pop[1, 25:30] = 0
    likelihood[:, 5] = 0.5    # a flood-prone column walls in the people near A
    impacts = impacts_for(friction, likelihood, sites, pop)
    assert impacts.loc[0, "share_cut_off"] == 1.0 and np.isnan(impacts.loc[0, "extra_min"])


def test_rank_sites_orders_flooded_sites_by_criterion():
    impacts = pd.DataFrame({"catchment_pop": [500, 100, 900, 300, 0, 700],
                            "share_cut_off": [0.2, 1.0, 1.0, 1.0, np.nan, 0.9],
                            "extra_min": [10, np.nan, 50, 40, np.nan, 30]})
    flooded = np.array([True, True, False, True, True, True])   # site 2 is not flood-prone
    assert rank_sites(impacts, flooded, "most_saved", 3).index.tolist() == [5, 0, 3]
    assert rank_sites(impacts, flooded, "worst_off", 3).index.tolist() == [1, 3, 5]   # nobody reached = worst
    assert rank_sites(impacts, flooded, "worst_off", 10).index.tolist() == [1, 3, 5, 0, 4]
    with pytest.raises(ValueError):
        rank_sites(impacts, flooded, "nearest", 3)


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
