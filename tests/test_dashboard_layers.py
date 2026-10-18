"""
Tests for data_preparation/build_dashboard_layers.py. Run from the repository root:  python -m pytest tests
"""
import numpy as np

from data_preparation.build_dashboard_layers import (EXTRA_CLASSES, PEOPLE_CLASSES, TIME_CLASSES, block_mean,
                                                     block_sum, classify, layer_codes)


def test_block_sum_and_mean_work_on_4x4_blocks():
    values = np.arange(32, dtype=float).reshape(4, 8)
    assert block_sum(values).tolist() == [[np.arange(32).reshape(4, 8)[:, :4].sum(), np.arange(32).reshape(4, 8)[:, 4:].sum()]]
    valid = np.zeros((4, 8), dtype=bool)
    valid[0, 0] = valid[1, 1] = True
    mean, any_valid = block_mean(values, valid)
    assert mean[0, 0] == (0 + 9) / 2 and any_valid.tolist() == [[True, False]]


def test_classify_uses_lower_bounds_and_leaves_nan_transparent():
    codes = classify(np.array([0.0, 1, 10, 999, 5000, np.nan]), PEOPLE_CLASSES)
    assert codes.tolist() == [0, 1, 2, 4, 5, 0]
    assert classify(np.array([0.0, 45, 300]), TIME_CLASSES).tolist() == [1, 2, 5]
    assert classify(np.array([0.1, 20]), EXTRA_CLASSES).tolist() == [0, 2]


def test_a_walled_in_block_is_marked_unreachable():
    pop = np.full((4, 8), 5.0)
    inside = np.ones((4, 8), dtype=bool)
    flooded = np.zeros((4, 8), dtype=bool)
    base = np.full((4, 8), 30.0)
    flood = base.copy()
    flood[:, 4:] = np.nan                                  # the right-hand block can reach nothing now
    codes = layer_codes("travel_time", pop, inside, base, flood, flooded, 120)
    assert codes[0, 0] == 2 and codes[0, 1] == len(TIME_CLASSES) + 1
    people = layer_codes("no_facility", pop, inside, base, flood, flooded, 120)
    assert people[0, 0] == 0 and people[0, 1] == 3         # 16 cells x 5 people = 80 people: class 50-200
