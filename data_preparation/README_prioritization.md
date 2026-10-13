# Flood-prone facilities and which to protect first

The unit is a **site**: the facilities sharing a 250 m cell, from `build_sites` in
`build_baseline.py`. Protecting a site protects every facility in that cell. The functions are in
`prioritization.py`; the dashboard runs them in `get_priorities`.

1. **Flooded (submerged / inaccessible).** A site is flooded when `likelihood[its cell] > threshold`
   (`flooded_sites`, the same rule as `open_sites` in `routing.py`).
   - `likelihood` is for the selected month's calendar month (`load_flood_likelihood`): the share of
     years 2003-2025 the cell was flooded on 3+ days in that month.
   - `threshold` is the "Block Cells Flooded in More Than (% of Years)" slider (default 0, i.e.
     flooded in that month in any year since 2003).
   - Sites without flood data (NaN, north of 10°N) are not flagged.
2. **Baseline catchments, no floods.** `travel_time(friction_walking, all sites, blocked=water)`, the
   same solve as `build_baseline`. `nearest` gives each cell's usual site and `minutes` its usual
   travel time.
3. **Flood scenario.** `travel_time(friction_walking, open sites only, blocked=water | likelihood > threshold)`:
   every flooded site is lost and flood-prone cells are impassable. People who live in blocked cells,
   or are walled in by them, are unreachable (NaN).
4. **Per flooded site** (`site_impacts`), over the people in its baseline catchment:
   - **Most Saved:** `catchment_pop`, the population whose nearest site without floods is this one.
   - **Worst-Off First:** `share_cut_off`, the part of that population whose flood-scenario time is
     NaN or above `travel.max_time_min` (120 min). Ties are broken by `extra_min`, the
     population-weighted mean of (flood minutes − baseline minutes) over the people still reached; a
     catchment where nobody is reached counts as the largest.
5. **Top 5** (`rank_sites`): the flooded sites sorted by the selected criterion, descending, first
   `priorities.top_n` (5).
6. **Protection in Route to Care.** With "Assume Top 5 Are Protected" on (the default), the Top 5
   sites count as open destinations (`usable[protected] = True`). Their own cells are passable as
   route endpoints, but the flood-prone cells around them stay blocked: sandbagging protects the
   facility, not its access.

Steps 2-4 use walking whatever the route's travel mode, since most people walk. Each solve takes
about 15 s on the national grid. The baseline is cached once and the flood scenario per (calendar
month, threshold), so the first view takes about 30 s and later views are instant.

| Dashboard | Comes from |
|---|---|
| Submerged / Inaccessible, red markers | facilities on flooded sites (step 1) |
| Blue markers "PROTECTED (TOP 5)" | the Top 5, when protection is on |
| Top 5 table | step 5: Catchment Population, or Cut Off (%) with Extra Time (min) |
| County colours, Flooded In-Situ Population | the selected month only (Min Days slider), not the likelihood |

At threshold 0, 21 sites are flooded in October and 34 in November.
