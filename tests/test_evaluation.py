"""
Tests for data_preparation/peak_months.py and evaluate_peak_months.py. Run from the repository root:  python -m pytest tests
"""
import pandas as pd

import numpy as np

from data_preparation.evaluate_peak_months import (concentration, coverage, naive_summary, naive_tops, overlap,
                                               people_brought_back, top_frequency, top_sites)
from data_preparation.external_check import season_window
from data_preparation.peak_months import largest_per_year


def test_largest_month_per_year_skips_flagged_months():
    months = pd.DataFrame({
        "year":        [2006, 2006, 2006, 2007, 2007],
        "month":       [4, 10, 11, 1, 2],
        "flagged":     [True, False, False, False, False],
        "flooded_km2": [9000.0, 500.0, 800.0, 300.0, 300.0],
    })
    peaks = largest_per_year(months)
    assert peaks[["year", "month"]].values.tolist() == [[2006, 11], [2007, 1]]   # April 2006 is a data gap; ties -> earlier


def test_top_sites_follow_the_greedy_rank_and_skip_sites_adding_nobody():
    table = pd.DataFrame({"site_id": [10, 11, 12, 13],
                          "rank_most_saved": [2, 1, 3, 4], "added_most_saved": [50, 90, 0, 5]})
    assert top_sites(table, "most_saved", 5) == [11, 10, 13]
    assert top_sites(table, "most_saved", 1) == [11]


def test_season_window_runs_from_season_start_to_report_month():
    assert season_window("2024-10", 7) == ["2024-07", "2024-08", "2024-09", "2024-10"]
    assert season_window("2024-07", 7) == ["2024-07"]


def test_naive_tops_pick_by_own_gain_and_by_usual_population():
    table = pd.DataFrame({"site_id": [0, 1, 2, 3], "gain_most_saved": [10.0, 40.0, 40.0, 5.0]})
    usual = np.array([900.0, 100.0, 50.0, 500.0])
    tops = naive_tops(table, usual, "most_saved", 2)
    assert tops["own_gain"] == [1, 2]               # ties -> lower site id
    assert tops["usual_population"] == [0, 3]


def test_naive_summary_skips_years_with_few_closed_sites():
    rows = [(m, closed, p, b) for m, closed, picks in [("2020-12", 8, {"greedy": 90, "own_gain": 70, "usual_population": 90}),
                                                        ("2021-01", 3, {"greedy": 50, "own_gain": 50, "usual_population": 50})]
            for p, b in picks.items()]
    naive = pd.DataFrame(rows, columns=["month", "sites_closed", "picked_by", "brought_back"]).assign(
        goal="most_saved", people_losing_access=100)
    out = naive_summary(naive, 5)["most_saved"]
    assert out["years"] == 1
    assert out["median_share_brought_back"] == {"greedy": 0.9, "own_gain": 0.7, "usual_population": 0.9}
    assert out["years_greedy_better"] == {"own_gain": 1, "usual_population": 0}


def test_overlap_and_frequency():
    tops = {"2020-12": {"most_saved": [1, 2, 3], "worst_off": [3, 4, 5]},
            "2021-01": {"most_saved": [1, 2, 6], "worst_off": [1, 7, 8]}}
    assert overlap(tops["2020-12"]) == 1 and overlap(tops["2021-01"]) == 1
    info = pd.DataFrame({"adm2_name": list("abcdefgh")}, index=range(1, 9))
    freq = top_frequency(tops, info).set_index("site_id")
    assert freq.loc[1, "years_top_most_saved"] == 2 and freq.loc[1, "years_top_worst_off"] == 1
    assert freq.loc[1, "years_top_either"] == 2 and freq.loc[3, "years_top_either"] == 1
    assert freq.index[0] == 1                       # most often in a Top list comes first


def test_coverage_follows_the_greedy_cumulative_and_plateaus():
    table = pd.DataFrame({"rank_most_saved": [2, 1, 3], "cumulative_most_saved": [150.0, 100.0, 160.0]})
    shares = coverage(table, "most_saved", 200.0, 5)
    assert shares == [0.5, 0.75, 0.8, 0.8, 0.8]          # only 3 closed sites: stays at 160 / 200
    assert all(s != s for s in coverage(table, "most_saved", 0.0, 2))   # nobody losing access: NaN


def test_concentration_takes_the_busiest_receivers_first():
    added = pd.Series([10.0, 0.0, 60.0, 30.0])
    assert concentration(added, 4) == [0.6, 0.9, 1.0, 1.0]       # zero-receiving facility ignored, plateaus at 1
    assert all(s != s for s in concentration(pd.Series([0.0]), 2))   # nobody moved: NaN


def test_people_brought_back_counts_shared_cells_once():
    pop = np.array([[10.0, 20.0], [30.0, 40.0]])
    regained = {1: np.array([0, 1]), 2: np.array([1, 3]), 3: np.array([], dtype=np.int64)}
    assert people_brought_back(regained, [1, 2], pop) == 70.0          # cell 1 counted once
    assert people_brought_back(regained, [3], pop) == 0.0
    assert people_brought_back(regained, [], pop) == 0.0
