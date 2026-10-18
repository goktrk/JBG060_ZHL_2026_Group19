# Access loss per flood scenario

Who is flood-affected, and who on dry land is cut off from health care, for each flood scenario in
`config.yaml`. Replaces the old `measure_access_loss.py`.

## Running it

The analysis steps form a chain; each reads the previous one's output instead of recomputing it:

```
python -m data_preparation.build_baseline        # ~1.5 min: travel times without floods
python -m data_preparation.build_access_loss     # ~4 min: the 4 scenarios, walking and motorised
python -m data_preparation.build_priorities      # ~15 s: protection ranking (README_prioritization.md)
```

Options: `--scenario 2020-11` for one scenario; `--all-months` for every month of the flood record
(2003-2025) as tables only, about 3 hours, for the recurrence and lag analyses.

Each step stops with a clear message if the step before it is missing or was built with other
settings (population year, permanent water, flood days, buffer).

## The rules

Shared with the protection ranking and scenario selection, in `scenarios.py`:

- **Flooded cell:** flagged on at least `flood.min_days_flooded` (3) days that month. Blocked for travel,
  like permanent water.
- **Closed facility:** a flooded cell within `facilities.flood_buffer_m` (1000 m). This also covers
  facilities cut off by water around them. Only this case is reported (flooded facilities closed),
  not a best/worst range.
- **Travel time:** walking and motorised (`access_loss.modes`). The slow traveller is walking minutes
  × `travel.slow_traveller_factor` (1.5).

## The groups

Everyone is counted in at most one group:

| Group | Who |
|---|---|
| `living_in_flooded_cells` | people whose own 250 m cell is flooded: flood-affected, likely displaced |
| `cut_off_60`, `cut_off_120` | on dry land, within 1 h / 2 h of a facility without floods, beyond it or unable to reach any with floods |
| `no_facility_reachable` | the part of the cut-off people who can't reach any facility at all |
| `without_flood_data` | people north of 10°N, where flooding is unknown (not the same as dry) |

`cut_off_60` can be larger than `cut_off_120`: they are different groups of people (who was within 1 h
before, and who was within 2 h), not one inside the other.

## Outputs (`data/processed/access_loss/`, not in git)

| File | Content |
|---|---|
| `access_loss_<id>.json` | national totals per mode, closed sites and facilities, `closed_site_ids`, settings used |
| `access_loss_<id>_county.csv` | per county and mode: the groups, closed facilities, mean/median minutes before and after |
| `flood_travel_time_<mode>_<id>.tif` | minutes to the nearest open facility with floods, NaN = none reachable |
| `flood_nearest_site_<mode>_<id>.tif` | `site_id` of that facility, -1 = none (for pre-positioning) |
| `access_loss_monthly_*.csv` | with `--all-months`: national and county tables for every month, `flagged` marks months with data gaps |

`site_id` is the row of `data/processed/baseline/internal/sites.parquet`. Load the results with
`load_access_loss(id)` and `load_scenario_travel(id, mode)` from `loading.py`.

## Current results (walking)

| Scenario | Facilities closed | Living in flooded cells | Cut off, 1 h | Cut off, 2 h | No facility at all |
|---|---|---|---|---|---|
| Nov 2008 | 43 | 57,691 | 201,865 | 188,470 | 6,440 |
| Nov 2020 | 78 | 103,845 | 307,496 | 279,079 | 1,447 |
| Oct 2025 | 24 | 22,090 | 86,812 | 86,341 | 1,095 |
| Jan 2025 | 45 | 117,750 | 192,814 | 211,304 | 16,435 |

Motorised (where roads exist; walking elsewhere): cut off within 2 h is 10,368 (Nov 2008), 32,648
(Nov 2020), 5,778 (Oct 2025) and 60,000 (Jan 2025). People in flooded cells and with no facility at
all are the same as walking, since those don't depend on speed.

The walking 2 h numbers are exactly the "people losing access" of the protection ranking. About
400,000 people live north of 10°N, without flood data, in every scenario.

## Limits

- Observed MODIS floods are a lower bound (clouds hide flooding); nothing is known north of 10°N.
- One facility case only (flooded facilities closed); all facilities count as equal, without capacity.
- Travel times come from 1 km travel-cost data; read results per county, not per cell.
- A screening tool for humanitarian staff to verify locally, not an instruction.
