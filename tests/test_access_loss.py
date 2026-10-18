"""
Tests for data_preparation/build_access_loss.py and scenarios.py. Run from the repository root:  python -m pytest tests
"""
import numpy as np
import pandas as pd

from data_preparation.build_access_loss import county_table, group_masks
from data_preparation.scenarios import closed_sites
from data_preparation.travel import travel_time

CELL = 250.0   # 0.012 min/m x 250 m = 3 minutes per cell


def corridor_scenario():
    """
    A 1 x 101 corridor, 3 min per cell. Sites at columns 0 and 100.
    Column 40 floods: it blocks the corridor and, with a 1 km buffer, closes neither site.
    """
    friction = np.full((1, 101), 0.012)
    sites = np.array([[0, 0], [0, 100]])
    pop = np.zeros((1, 101))
    pop[0, 30] = 10    # west of the flood: 90 min from site 0 before and after
    pop[0, 40] = 20    # lives in the flooded cell
    pop[0, 45] = 30    # east of the flood: was 135 min (west), now 165 min (east)
    pop[0, 70] = 40    # 90 min from site 1 before and after
    flooded = np.zeros((1, 101), dtype=bool)
    flooded[0, 40] = True
    return friction, sites, pop, flooded


def solve(friction, sites, flooded, buffer_m):
    closed = closed_sites(sites, flooded, buffer_m, CELL)
    base = travel_time(friction, sites).minutes
    flood = travel_time(friction, sites[~closed], blocked=flooded).minutes
    return closed, base, flood


def test_people_in_flooded_cells_are_a_separate_group():
    friction, sites, pop, flooded = corridor_scenario()
    closed, base, flood = solve(friction, sites, flooded, 1000)
    assert not closed.any()
    masks = group_masks(pop, base, flood, flooded, [60, 120])
    for mask in masks.values():
        assert not mask[0, 40]                              # never counted as cut off
    assert not masks["cut_off_120"][0, 30] and not masks["cut_off_120"][0, 70]   # unaffected
    assert not masks["cut_off_120"][0, 45]                  # 135 min before: wasn't within 2 h
    assert not masks["cut_off_60"].any()


def test_a_closed_facility_cuts_people_off_and_some_can_reach_nothing():
    """A 3 x 101 corridor; people and sites on the middle row."""
    friction = np.full((3, 101), 0.012)
    sites = np.array([[1, 0], [1, 100]])
    pop = np.zeros((3, 101))
    pop[1, 30], pop[1, 45], pop[1, 70] = 10, 30, 40
    flooded = np.zeros((3, 101), dtype=bool)
    flooded[0, 3] = True                                    # beside the path, ~790 m from site 0: closes it
    flooded[:, 60] = True                                   # and the corridor is blocked further east
    closed, base, flood = solve(friction, sites, flooded, 1000)
    assert closed.tolist() == [True, False]
    masks = group_masks(pop, base, flood, flooded, [60, 120])
    assert masks["cut_off_120"][1, 30]                      # was 90 min from site 0, now nothing reachable
    assert masks["no_facility_reachable"][1, 30] and masks["no_facility_reachable"][1, 45]
    assert not masks["cut_off_120"][1, 70]                  # still 90 min from site 1
    assert not masks["cut_off_60"][1, 30]                   # 90 min before: wasn't within 1 h
    # with the old own-cell rule site 0 stays open, so the people west of the flood still reach it
    _, base0, flood0 = solve(friction, sites, flooded, 0)
    assert not group_masks(pop, base0, flood0, flooded, [120])["cut_off_120"][1, 30]


def test_county_table_adds_up_to_the_national_totals():
    friction, sites, pop, flooded = corridor_scenario()
    flooded[0, 3] = True
    closed, base, flood = solve(friction, sites, flooded, 1000)
    county_id = np.zeros((1, 101), dtype=np.uint16)
    county_id[0, :50], county_id[0, 50:] = 1, 2
    static = {"population": pop, "county_id": county_id}
    counties = pd.DataFrame({"county_id": [1, 2], "adm2_pcode": ["SS01", "SS02"],
                             "adm2_name": ["West", "East"], "adm1_name": ["A", "A"]})
    sites_table = pd.DataFrame({"admin2_pcode": ["SS01", "SS02"], "n_facilities": [2, 1]})
    no_data = np.zeros_like(flooded)
    table = county_table(static, counties, sites_table, closed, flooded, no_data,
                         {"walking": (base, flood)}, [60, 120])
    masks = group_masks(pop, base, flood, flooded, [60, 120])
    for name, mask in masks.items():
        assert table[name].sum() == pop[mask].sum()
    assert table["living_in_flooded_cells"].sum() == pop[flooded].sum()
    assert table["population"].sum() == pop.sum()
    assert table.set_index("adm2_name").loc["West", "closed_facilities"] == 2
