# Pre-positioning: facilities likely to receive extra demand

When a flood closes facilities, or blocks the way to them, people go to the nearest facility that is
still open. This step finds which facilities receive them, so supplies, staff or mobile teams can be
sent there in advance. It does **not** assume anyone redistributes patients; it follows where people
go on their own. No routes are drawn: only facilities, numbers of people and travel times.

## Running it

The last step of the chain:

```
python -m data_preparation.build_baseline
python -m data_preparation.build_access_loss
python -m data_preparation.build_priorities        # needed for the protected-Top-5 side variant
python -m data_preparation.build_prepositioning    # all scenarios; --scenario 2020-11, --skip-protected
```

The main result needs no new travel-time calculations: it compares the nearest facility of every
cell before the flood (`build_baseline`) and during it (`build_access_loss`), about 12 s per scenario.
The side variant solves the flood scenario again with the Top 5 reopened, once per mode and ranking,
which makes a full run about 18 minutes; `--skip-protected` leaves it out.
Settings: `prepositioning:` in `config.yaml`; the time limit is `travel.max_time_min` (2 h).

## Where people go

Every person who could reach a facility before the flood is in exactly one group (`classify`):

| Group | Meaning |
|---|---|
| `same_facility` | still goes to their usual facility |
| `moved_facility_closed` | their usual facility is closed (flooded within 1 km), so they go to another one |
| `moved_path_blocked` | their usual facility is open, but the flood makes another one closer |
| `lost_all_access` | no facility reachable |
| `living_in_flooded_cells` | their home is flooded; not followed, as in the access loss |

`check_balance` stops the run if the groups don't add up, or if anyone is still assigned to a closed facility.

"Path blocked" switches count as added load (those people do arrive at the other facility), labelled
separately. Some of them are small flips between two facilities at about the same distance, where in
reality people would likely keep their usual facility; `moved_path_blocked_small_delay` counts the ones
whose trip got less than `prepositioning.small_delay_min` (15) minutes longer.

## Outputs (`data/processed/prepositioning/`)

Every table has a `mode` (walking, motorised) and a `variant` column: `as_is` is the flood scenario,
`top5_most_saved` / `top5_worst_off` the same with that ranking's Top 5 protected.

**`prepositioning_<id>_receiving.csv`**, one row per facility that receives people (the
pre-positioning list):

| Column | Meaning |
|---|---|
| `site_id`, `rep_facility_name`, `rep_facility_type`, `n_facilities`, `adm2_name` | the facility (site) |
| `usual_population` | people for whom it is the nearest facility without floods |
| `added` | people who move to it |
| `added_within_limit` | of those, people within 2 h of it: the more realistic demand |
| `relative_increase`, `relative_increase_within_limit` | added / usual population (2.5 = +250%); empty if it usually serves nobody |
| `added_from_closed`, `added_from_blocked_path` | where the added people come from |
| `added_blocked_small_delay` | path-blocked arrivals with less than 15 min extra |
| `extra_min` | population-weighted extra travel time of the people who move to it |
| `from_facilities` | how many facilities they come from |
| `sent_away_path_blocked` | its own usual people who now go elsewhere because their way floods |

The "all" and "within 2 h" columns side by side give the range of added load.

**`prepositioning_<id>_flows.csv`**: people per `from_site` → `to_site` and `reason`, with
`people_within_limit`, `people_small_delay` and `extra_min`. Shows where each closed facility's people go.

**`prepositioning_<id>.json`**: people per group, per mode and variant, the protected sites, settings.

## Current results (walking, as is)

| Scenario | Move to another facility | of which within 2 h of it | Path blocked (< 15 min extra) | Facilities receiving people |
|---|---|---|---|---|
| Nov 2008 | 378,120 | 167,040 | 9,742 (4,850) | 164 |
| Nov 2020 | 848,917 | 391,512 | 8,544 (3,757) | 229 |
| Oct 2025 | 216,507 | 92,734 | 6,484 (2,951) | 91 |
| Jan 2025 | 391,266 | 135,792 | 12,279 (5,199) | 154 |

Almost all moves are because a facility closed; path-blocked switches are 1-3%, and about half of
those are small flips. Example, Nov 2020: Langbaar Primary Health Care Unit (Bor South) usually serves
about 34,000 people and receives about 76,600 more, all within 2 h (+224%), because the facilities in
Bor town close. With the Top 5 (most saved) protected, 711,000 instead of 849,000 people have to move.

## Limits

- **Nearest by travel time is a simplification**: facility type, cost, trust and security also decide
  where people go, and the data has none of that.
- **Load is people in the area served, not patients**; there is no utilisation rate.
- **No capacity data**: facilities count as equal, so the relative increase is the main measure.
- **Displaced people are not followed**: people from flooded cells add load somewhere (often towns and
  camps), probably the biggest blind spot.
- One facility case (flooded facilities closed); observed floods are a lower bound; nothing is known
  north of 10°N.
- A screening tool for humanitarian staff to verify locally, not an instruction.
