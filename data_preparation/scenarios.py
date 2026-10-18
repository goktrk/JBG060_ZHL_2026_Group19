"""
The flood-scenario rules every analysis shares, so access loss, protection priorities and scenario
selection always agree on what is flooded, which facilities are closed and who loses access.

    flooded, no_data = scenario_flood("2020-11", cfg)
    closed = closed_sites(sites_rc, flooded, cfg["facilities"]["flood_buffer_m"], cell_size_m=250)
    cut_off = lost_cells(pop, base_minutes, flood_minutes, flooded, limit_min=120)
"""
import numpy as np

from data_preparation.build_flood_monthly import NO_DATA
from data_preparation.loading import load_flood


def scenario_flood(scenario_id: str, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Cells flooded on at least flood.min_days_flooded days in the scenario month (band "all"), and
    cells without flood data (north of 10N), which are not the same as dry.
    """
    _, days = load_flood(scenario_id, cfg=cfg)
    return (days[0] >= cfg["flood"]["min_days_flooded"]) & (days[0] != NO_DATA), days[0] == NO_DATA


def closed_sites(sites_rc, flooded, buffer_m, cell_size_m) -> np.ndarray:
    """
    True for sites with a flooded cell within `buffer_m` of their own cell (centre to centre),
    so buffer_m = 0 means only the site's own cell counts. Cells without flood data count as dry.
    """
    sites = np.asarray(sites_rc)
    k = int(buffer_m // cell_size_m)
    dy, dx = np.mgrid[-k:k + 1, -k:k + 1]
    disk = (dy ** 2 + dx ** 2) * cell_size_m ** 2 <= buffer_m ** 2
    padded = np.pad(np.asarray(flooded, dtype=bool), k)
    return np.array([(padded[r:r + 2 * k + 1, c:c + 2 * k + 1] & disk).any() for r, c in sites], dtype=bool)


def lost_cells(pop, base_minutes, flood_minutes, flooded, limit_min) -> np.ndarray:
    """
    Cells whose people are cut off from care on dry land: within `limit_min` of a facility without
    floods, beyond it or unable to reach any (NaN) with floods. People living in a flooded cell are
    left out: they are flood-affected themselves and counted separately.
    """
    with np.errstate(invalid="ignore"):
        return ((np.asarray(pop) > 0) & ~np.asarray(flooded, dtype=bool)
                & (np.asarray(base_minutes) <= limit_min) & ~(np.asarray(flood_minutes) <= limit_min))
