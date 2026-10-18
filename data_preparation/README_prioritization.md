# Which closed facilities to protect first

For each flood scenario, this ranks the facility sites the flood closes by how many people get
access back if that one site is protected (e.g. sandbagged), following the "dry-out" step in the
project PDF and the protection ranking in `CLAUDE.md`. The unit is a **site**: the facilities
sharing a 250 m cell (`build_sites` in `build_baseline.py`); protecting a site protects all of them.

The functions are in `prioritization.py`, the run is `build_priorities.py`, and the dashboard only
reads the results.

## Running it

```
python -m data_preparation.build_baseline                       # once
python -m data_preparation.build_access_loss                    # closed sites and flood travel times
python -m data_preparation.build_priorities                     # all scenarios in config.yaml, ~15 s
python -m data_preparation.build_priorities --scenario 2020-11  # one scenario
python -m pytest tests                                          # includes the tests for this
```

It reads the sites and baseline from `build_baseline`, and the closed sites and flood-scenario travel
times from `build_access_loss` (`README_access_loss.md`), so nothing is computed twice and "people
losing access" is the same number in both. It stops if those were built with other settings.
Settings: `scenarios`, `priorities`, `facilities.flood_buffer_m` and `travel.max_time_min` in `config.yaml`.

## How it works

1. **Scenario.** One observed month from the `scenarios` list (picked by `select_scenarios.py`). A
   cell is flooded if it was flagged on at least `flood.min_days_flooded` (3) days that month.
   Permanent water and flooded cells are blocked.
2. **Closed sites.** A site is closed if a flooded cell lies within `facilities.flood_buffer_m`
   (1000 m) of it (`closed_sites` in `scenarios.py`). The site's own cell rarely floods, so without the buffer almost
   no site would ever close. Sites north of 10°N have no flood data and are never closed.
3. **Baseline and flood scenario.** Walking times to the nearest site without floods (all sites,
   water blocked; from `build_baseline`), and with the flood (closed sites removed, flooded cells
   blocked; from `build_access_loss`).
4. **Reopen each closed site on its own.** Reopening one site adds one destination, so every cell's
   new time is the lower of its flood-scenario time and its time to that site. Only the second needs
   solving, from the site alone, in a window around it. The window is large enough that nothing
   reachable within the limit is cut off, so this gives exactly the same result as re-running the
   whole country. The tests check this, and on the real data the Top 5 together matched a full rerun.
5. **Who gets access back**, per closed site, against two limits:
   - **Most saved:** people within `travel.max_time_min` (2 h) of care without floods, beyond it
     with the flood, and back within it when this site is protected.
   - **Worst-off first:** the same with `priorities.worst_off_min` (4 h). These are people who would
     be more than 4 hours from care without this site, often remote communities with nowhere else to go.

   People living in a flooded cell are not counted: protecting a facility doesn't make their homes
   passable. They are reported separately (`people_living_in_flooded_cells`).
6. **Greedy order** (`greedy_order`), per ranking: first the site that brings back the most people,
   then the site that brings back the most people *not already brought back*, and so on. Two
   neighbouring sites that rescue the same village don't both take a top place: the second one adds
   nobody and drops down.
7. **Not covered.** For the Top N, the people who lose access and are not brought back by protecting
   those N together (ethics requirement: every ranking comes with the population it doesn't cover).

A site surrounded by flooded ground brings nobody back when reopened, so it ranks low. This fixes
the earlier ranking by catchment size, where 3 of the Top 5 were walled in and protecting them helped
nobody.

## Outputs (`data/processed/priorities/`, not in git)

`priorities_<scenario>.parquet`, one row per closed site:

| Column | Meaning |
|---|---|
| `site_id`, `row`, `col`, `lon`, `lat` | the site (same ids as the baseline) |
| `rep_facility_*`, `n_facilities`, `facility_ids` | the facilities in it |
| `adm2_name`, `adm1_name` | county and state |
| `own_cell_flooded` | the site's own cell is flooded (otherwise it is closed by flooding within the buffer) |
| `gain_<ranking>` | people this site brings back on its own |
| `rank_<ranking>` | greedy rank, 1 = protect first |
| `added_<ranking>` | people this pick adds on top of the sites ranked above it |
| `cumulative_<ranking>` | people brought back by ranks 1..rank together |
| `not_covered_<ranking>` | people losing access who are still not brought back by ranks 1..rank |

`<ranking>` is `most_saved` or `worst_off`. `priorities_<scenario>.json` has the totals (people
losing access, people in flooded cells, closed sites, brought back and not covered by the Top N) and
the settings used.

## In the dashboard

- The scenario picker lists the `scenarios` from `config.yaml`; the table shows the Top N for the
  chosen criterion, with "People Brought Back" (what each pick adds) and "If Protected Alone".
- Below the table: how many people lose access, how many the Top N bring back, and how many are not
  covered.
- Facility status (red dots) is "closed in this scenario": a flooded cell within 1 km.
- "Assume Top 5 Are Protected" passes the Top N to Route to Care as open destinations. Route to Care
  itself still decides which other facilities are open from the calendar-month flood likelihood, so
  its picture can differ from the scenario's until routing is aligned.

## Current results (walking, 1 km buffer)

| Scenario | Sites closed | Lose 2 h access | Top 5 bring back | Pushed beyond 4 h | Top 5 bring back |
|---|---|---|---|---|---|
| Nov 2008 | 42 | 188,470 | 105,258 | 18,795 | 12,966 |
| Nov 2020 | 77 | 279,079 | 149,121 | 114,579 | 89,244 |
| Oct 2025 | 24 | 86,341 | 66,627 | 18,526 | 16,249 |
| Jan 2025 | 43 | 211,304 | 126,754 | 79,086 | 42,210 |

These are national numbers. The scenario tiers (severe, medium, mild) were picked on Bor South and
Aweil only, so a "mild" month can still be severe elsewhere (Jan 2025).

## Limits

- Walking only; the slow traveller and the best/worst facility cases are not ranked yet.
- Protection reopens the facility, not the flooded ground around it, and assumes it then works.
- All facilities count as equal; there is no capacity yet.
- Observed MODIS floods are a lower bound (clouds), and nothing is known north of 10°N.
- A screening tool for humanitarian staff to verify locally, not an instruction.
