"""
Tests for data_preparation/prepositioning.py. Run from the repository root:  python -m pytest tests
"""
import numpy as np
import pytest

from data_preparation.prepositioning import (FLOODED, LOST, MOVED_BLOCKED, MOVED_CLOSED, NOT_COUNTED, SAME,
                                             check_balance, classify, flow_table, group_totals, receiving_table,
                                             source_table)
from data_preparation.scenarios import closed_sites
from data_preparation.travel import travel_time

CELL = 250.0   # 3 minutes per cell


def three_sites():
    """
    A 3 x 121 corridor; people and sites on the middle row. Sites at columns 0, 60 and 120.
    The flood closes site 1 (a flooded cell beside it) and floods the home at column 100, which
    people can walk around through rows 0 and 2.
    """
    friction = np.full((3, 121), 0.012)
    sites = np.array([[1, 0], [1, 60], [1, 120]])
    pop = np.zeros((3, 121))
    pop[1, 20] = 10      # site 0 (60 min), unaffected
    pop[1, 50] = 30      # site 1 (30 min) -> site 0 (150 min): facility closed
    pop[1, 70] = 40      # site 1 (30 min) -> site 2 (150 min): facility closed
    pop[1, 100] = 5      # lives in the flooded cell
    pop[1, 105] = 50     # site 2 (45 min); its way there is open, so it stays
    flooded = np.zeros((3, 121), dtype=bool)
    flooded[0, 62] = True                                   # 2 cells from site 1: closes it
    flooded[1, 100] = True
    return friction, sites, pop, flooded


def solve(friction, sites, flooded, open_mask=None):
    closed = closed_sites(sites, flooded, 1000, CELL)
    open_ids = np.flatnonzero(~closed if open_mask is None else open_mask)
    base = travel_time(friction, sites)
    flood = travel_time(friction, sites[open_ids], blocked=flooded)
    flood_nearest = np.where(flood.nearest >= 0, open_ids[np.maximum(flood.nearest, 0)], -1)
    return closed, base, flood, flood_nearest


def test_people_of_a_closed_facility_split_into_the_groups():
    friction, sites, pop, flooded = three_sites()
    closed, base, flood, flood_nearest = solve(friction, sites, flooded)
    assert closed.tolist() == [False, True, False]
    groups = classify(pop, base.nearest, flood_nearest, flooded, closed)
    assert groups[1, 20] == SAME and groups[1, 105] == SAME
    assert groups[1, 50] == MOVED_CLOSED and flood_nearest[1, 50] == 0
    assert groups[1, 70] == MOVED_CLOSED and flood_nearest[1, 70] == 2
    assert groups[1, 100] == FLOODED
    assert groups[0, 0] == NOT_COUNTED                      # nobody lives there
    check_balance(pop, base.nearest, groups, closed)
    totals = group_totals(pop, groups)
    assert totals["moved_facility_closed"] == 70 and totals["living_in_flooded_cells"] == 5


def test_path_blocked_switch_is_counted_with_its_own_reason():
    friction, sites, pop, flooded = three_sites()
    flooded[:, 110] = True                                  # the way from column 105 to site 2 floods
    closed, base, flood, flood_nearest = solve(friction, sites, flooded)
    assert not closed[2]                                    # site 2 itself stays open
    groups = classify(pop, base.nearest, flood_nearest, flooded, closed)
    assert groups[1, 105] == MOVED_BLOCKED and flood_nearest[1, 105] == 0
    check_balance(pop, base.nearest, groups, closed)
    flows = flow_table(pop, base.nearest, base.minutes, flood_nearest, flood.minutes, groups, 120, 15)
    blocked = flows[flows["reason"] == "path_blocked"]
    assert blocked["people"].sum() == 50 and set(blocked["from_site"]) == {2}
    receiving = receiving_table(pop, base.nearest, flows, len(sites)).set_index("site_id")
    assert receiving.loc[0, "added_from_blocked_path"] == 50

    flooded[:, 90] = True                                   # walled in between columns 90 and 110
    closed, base, flood, flood_nearest = solve(friction, sites, flooded)
    groups = classify(pop, base.nearest, flood_nearest, flooded, closed)
    assert groups[1, 105] == LOST and groups[1, 100] == FLOODED


def test_flows_and_receiving_table_add_up():
    friction, sites, pop, flooded = three_sites()
    closed, base, flood, flood_nearest = solve(friction, sites, flooded)
    groups = classify(pop, base.nearest, flood_nearest, flooded, closed)
    flows = flow_table(pop, base.nearest, base.minutes, flood_nearest, flood.minutes, groups,
                       limit_min=120, small_delay_min=15)
    assert flows["people"].sum() == 70
    assert set(zip(flows["from_site"], flows["to_site"])) == {(1, 0), (1, 2)}
    assert (flows["people_within_limit"] == 0).all()        # both now 150 min away
    receiving = receiving_table(pop, base.nearest, flows, len(sites)).set_index("site_id")
    assert receiving.loc[0, "added"] == 30 and receiving.loc[2, "added"] == 40
    assert receiving.loc[0, "usual_population"] == 10      # column 20; column 50 used site 1
    assert receiving.loc[0, "relative_increase"] == pytest.approx(3.0)
    # 120 min more, plus going around the flooded home: two diagonal steps instead of two straight ones
    assert receiving.loc[2, "extra_min"] == pytest.approx(120 + 2 * 3 * (np.sqrt(2) - 1), rel=1e-4)
    assert receiving["added_from_closed"].sum() == 70 and receiving["added_from_blocked_path"].sum() == 0


def test_a_facility_that_usually_serves_nobody_gets_no_relative_increase():
    friction, sites, pop, flooded = three_sites()
    pop[1, 20] = 0                                          # site 0 now usually serves nobody
    closed, base, flood, flood_nearest = solve(friction, sites, flooded)
    groups = classify(pop, base.nearest, flood_nearest, flooded, closed)
    flows = flow_table(pop, base.nearest, base.minutes, flood_nearest, flood.minutes, groups, 120, 15)
    receiving = receiving_table(pop, base.nearest, flows, len(sites)).set_index("site_id")
    assert receiving.loc[0, "usual_population"] == 0 and np.isnan(receiving.loc[0, "relative_increase"])


def test_protecting_the_closed_facility_removes_the_load_on_its_neighbours():
    friction, sites, pop, flooded = three_sites()
    closed, base, flood, flood_nearest = solve(friction, sites, flooded, open_mask=np.ones(3, dtype=bool))
    protected_closed = np.zeros(3, dtype=bool)              # site 1 protected: nothing is closed
    groups = classify(pop, base.nearest, flood_nearest, flooded, protected_closed)
    flows = flow_table(pop, base.nearest, base.minutes, flood_nearest, flood.minutes, groups, 120, 15)
    assert flows.empty or flows["people"].sum() == 0
    check_balance(pop, base.nearest, groups, protected_closed)


def test_small_delays_are_flagged():
    pop = np.array([[10.0, 20.0]])
    base_nearest, flood_nearest = np.array([[0, 0]]), np.array([[1, 1]])
    base_min, flood_min = np.array([[30.0, 30.0]]), np.array([[35.0, 80.0]])
    groups = np.array([[MOVED_BLOCKED, MOVED_BLOCKED]], dtype=np.int8)
    flows = flow_table(pop, base_nearest, base_min, flood_nearest, flood_min, groups, 120, 15)
    assert flows.loc[0, "people_small_delay"] == 10 and flows.loc[0, "people_within_limit"] == 30


def test_source_table_splits_each_closed_facility():
    friction, sites, pop, flooded = three_sites()
    flooded[1, 58] = True                                   # a home near site 1 floods too
    pop[1, 58] = 7
    closed, base, flood, flood_nearest = solve(friction, sites, flooded)
    groups = classify(pop, base.nearest, flood_nearest, flooded, closed)
    sources = source_table(pop, base.nearest, groups, closed).set_index("site_id")
    assert sources.index.tolist() == [1]                    # only the closed site
    row = sources.loc[1]
    assert row["moved"] == 70 and row["living_in_flooded_cells"] == 7 and row["lost_all_access"] == 0
    assert row["moved"] + row["lost_all_access"] + row["living_in_flooded_cells"] == row["usual_population"]
