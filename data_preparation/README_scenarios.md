# Flood scenarios for the dashboard

Picks the months the dashboard offers as flood scenarios: months of observed flooding (2003-2025),
ranked by how many people in the focus areas lose walking access to health care that month.

## Running it

```
# from the repository root, after build_flood_monthly and build_baseline
python -m data_preparation.select_scenarios                 # about 90 min: one ~18 s solve per month
python -m data_preparation.select_scenarios --select-only   # seconds: only redo the picks from the csv
```

## Outputs (in `EDA/outputs/`)

- `scenario_candidates.csv`: every month, flooding and access loss (at 120 and also at 60 min) for Bor South, Aweil, both, the country.
- `scenarios_selected.csv`: the picked months with tier, reason and focus-area numbers.
- `scenarios_block.yaml`: a `scenarios:` block to paste into `config.yaml`.
- `scenarios_metadata.json`: settings, run times, input file hashes, ranks of the old dashboard months.

## How a month is picked

- Severity: people on dry land newly more than 120 min from care, or cut off, in Bor South plus Aweil
  (the rules of `build_access_loss.py`, from `scenarios.py`: facilities closed by a flooded cell within
  1 km, people living in flooded cells counted separately). **The current picks were made with the
  older rules** (only a facility's own cell, flooded people counted as cut off); rerun to update them.
- Left out: months with no loss, months where over 5% of a focus area has no flood data, and the
  months with gaps in the flood record (2006-04, 2006-05, 2007-11).
- Always in: the worst month overall and the worst month of each focus area, even if they share a
  calendar month. A month that is more than one of these is listed once, with all its roles.
- Medium and mild: closest to the 75th percentile and the median of the eligible months. These two
  prefer a calendar month not used yet. Ties go to the later year.

Current picks (113 eligible months): Nov 2008 severe and worst for Aweil (15,684 people), Nov 2020 worst
for Bor South (2,582), Oct 2025 medium (478), Jan 2025 mild (84).

Things to know:

- Scenarios are observed MODIS months, not forecasts. Clouds hide flooding, so the losses are lower bounds.
- Flooded means 3 or more days that month (`flood.min_days_flooded`). Flooded cells are blocked and
  facilities on them closed, so people living in a flooded cell count as cut off.
- The Lol river has no geometry in the repository; the Aweil counties (county_id 35-39) stand in for it.
- The Top 5 is precomputed per scenario month (`build_priorities`, see `README_prioritization.md`).
  Routes still only use the calendar month, so they are the same for Nov 2008 and Nov 2020.
