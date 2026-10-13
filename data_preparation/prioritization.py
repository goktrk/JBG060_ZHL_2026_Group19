"""
Which facility sites are flood-prone, and which of them to protect first.

    likelihood = load_flood_likelihood(10)                           # October
    flooded = flooded_sites(sites_rc, likelihood, threshold=0.0)     # the site's own cell is flood-prone
    base = travel_time(friction, sites_rc, blocked=water)            # no floods
    flood = travel_time(friction, sites_rc[~flooded], blocked=blocked_mask(water, likelihood, 0.0))
    impacts = site_impacts(pop, base.nearest, base.minutes, flood.minutes, len(sites_rc), max_time_min=120)
    top = rank_sites(impacts, flooded, "most_saved", top_n=5)       # rows of impacts, by site_id

The calculation is described in data_preparation/README_prioritization.md.
"""
import numpy as np
import pandas as pd

from data_preparation.routing import open_sites

CRITERIA = ("most_saved", "worst_off")


def flooded_sites(sites_rc, likelihood, threshold) -> np.ndarray:
    """True for sites whose own cell is flooded in more than `threshold` of years (the opposite of
    routing.open_sites). Sites without flood data (NaN) are not flagged."""
    return ~open_sites(sites_rc, likelihood, threshold)


def site_impacts(pop, base_nearest, base_minutes, flood_minutes, n_sites, max_time_min) -> pd.DataFrame:
    """
    What losing each site costs the people in its catchment, indexed by site_id 0..n_sites-1.

    base_nearest and base_minutes come from travel_time without floods, flood_minutes from
    travel_time with the flood-prone cells blocked and the flooded sites removed (NaN = unreachable).

    - catchment_pop: people whose nearest site without floods is this one
    - share_cut_off: share of them who are unreachable or more than max_time_min away in the flood
      scenario (NaN when nobody lives in the catchment)
    - extra_min: population-weighted mean of flood minus baseline minutes over the people still
      reached (NaN when nobody is reached)
    """
    served = np.asarray(base_nearest) >= 0
    site = np.asarray(base_nearest)[served]
    people = np.asarray(pop)[served].astype("float64")
    base = np.asarray(base_minutes)[served].astype("float64")
    flood = np.asarray(flood_minutes)[served].astype("float64")

    reached = np.isfinite(flood)
    cut_off = ~reached | (flood > max_time_min)
    catchment = np.bincount(site, people, minlength=n_sites)
    pop_reached = np.bincount(site[reached], people[reached], minlength=n_sites)
    extra = np.bincount(site[reached], people[reached] * (flood[reached] - base[reached]), minlength=n_sites)
    with np.errstate(invalid="ignore", divide="ignore"):
        return pd.DataFrame({
            "catchment_pop": catchment,
            "share_cut_off": np.bincount(site, people * cut_off, minlength=n_sites) / catchment,
            "extra_min": extra / pop_reached,
        })


def rank_sites(impacts, flooded, criterion, top_n) -> pd.DataFrame:
    """
    The flooded sites, the one to protect first on top, at most top_n rows of `impacts`.

    most_saved: largest catchment population first. worst_off: largest share cut off first, then
    most extra minutes; a catchment where nobody is reached counts as the most extra minutes, and
    one where nobody lives comes last.
    """
    candidates = impacts[np.asarray(flooded, dtype=bool)]
    if criterion == "most_saved":
        ranked = candidates.sort_values("catchment_pop", ascending=False, kind="stable")
    elif criterion == "worst_off":
        nan_as = {"share_cut_off": -np.inf, "extra_min": np.inf}
        ranked = candidates.sort_values(["share_cut_off", "extra_min"], ascending=False,
                                        key=lambda col: col.fillna(nan_as[col.name]))
    else:
        raise ValueError(f"criterion must be one of {CRITERIA}, not {criterion!r}")
    return ranked.head(top_n)
