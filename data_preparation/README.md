# Data preparation: how to use the prepared data

Everything the model needs is put on **one grid**, so that position `[row, col]` means the same
patch of ground in every layer. This page explains how to build the files, load them, and combine
them into the inputs for the travel-time calculations.

## The grid

| | |
|---|---|
| Cell size | 250 m x 250 m |
| Shape | 4500 rows x 5308 columns |
| CRS | Africa Albers Equal Area (`ESRI:102022`), units are metres |
| Extent | South Sudan plus a 50 km buffer |

All of this comes from the `grid` section of `config.yaml`. The arrays you load are plain NumPy
arrays with no coordinates attached; location is implied by position. `make_grid(cfg)` gives you
the grid definition when you need real-world coordinates:

- Row 0 is the **north** edge; rows increase southwards. Column 0 is the west edge.
- Top-left corner of cell `[row, col]`: `x = grid.transform.c + col * 250`, `y = grid.transform.f - row * 250`.
- `lonlat_to_rowcol(grid, lon, lat)` converts longitude/latitude to a cell.
- `cell_centres_lonlat(grid)` gives the longitude/latitude of every cell centre.

## Building the files

Run from the repository root. Outputs go to `data/interim/` (not in git, so everyone builds them once).

```
python -m data_preparation.build_static            # ~30 s
python -m data_preparation.build_permanent_water   # ~15 s, needs build_static first
python -m data_preparation.build_facilities        # ~5 s
python -m data_preparation.build_flood_monthly     # ~6 min
```

| File | Content |
|---|---|
| `static_250m.tif` | 4 bands: population, walking travel cost, motorised travel cost, county id |
| `permanent_water_250m.tif` | 1 where the cell is permanent water (rivers, lakes), from JRC Global Surface Water |
| `counties.csv` | `county_id` to county name and pcode |
| `facilities.parquet` | One row per health facility (after cleaning), with its `row`, `col` and county |
| `facilities_removed.csv` | Every facility dropped during cleaning, with the reason |
| `flood/flood_YYYY_MM.tif` | One file per month, 2003-01 to 2025-12: days flagged as flooded, in total and split into recurring and unusual (3 bands) |
| `flood_months.csv` | Per month: days without any flood record (data gaps), and whether the month is flagged |

The `.tif` files open directly in QGIS if you want to look at them.

## Cleaning applied

The decisions come from the EDA notebooks in `EDA/`; the thresholds are in `config.yaml`.

**Facilities** (`build_facilities.py`): 1,747 records in South Sudan, 1,715 kept.

| Step | Removed | Rule |
|---|---|---|
| No coordinates | 13 | Latitude and longitude are 0 and the geometry is empty. |
| Duplicates | 18 | Same facility type, at most 50 m apart, names at least 80% alike. Names ending in different single letters or numbers (e.g. "Nglimbo A" / "Ngolimbo B") are never merged. Of each pair the GPS-located record is kept, else the lower `FID`. |
| Outside South Sudan | 1 | Facilities outside every county are snapped to the nearest county if it is within 2 km (2 facilities); further out is a coordinate error (Rupaker, 25.6 km). |

Kept but flagged: 4 facilities whose own state label (`Admin1`) disagrees with where they lie
(`admin1_mismatch = True`; the location wins), and 347 located from a place name instead of GPS
(`LL_source = "Geonames"`). The EDA found no difference in flood exposure between GPS and Geonames
locations, so they are not treated differently. `facilities_removed.csv` lists every dropped record.

**Travel cost** (`build_static.py`): in 0.5% of cells the motorised surface is slower than walking,
which cannot happen in reality. There the walking value is used.

**Population, counties, flood:** nothing removed. Population totals are preserved exactly; 2000-2002
are excluded from the flood record (Terra satellite only, heavily undercounted).

## Loading

```python
import numpy as np
from data_preparation.grid import load_config, make_grid
from data_preparation.loading import (load_static, load_permanent_water, load_flood, iter_flood,
                                     load_flood_months, load_facilities)
from data_preparation.build_flood_monthly import NO_DATA

cfg = load_config()
grid = make_grid(cfg)
static = load_static(cfg)          # dict of 2D arrays
water = load_permanent_water(cfg)  # 2D boolean array, True = permanent water
facilities = load_facilities(cfg)  # GeoDataFrame
```

| `static[...]` | Type | Meaning |
|---|---|---|
| `"population"` | float32 | People living in the cell (WorldPop 2020). 0 in the buffer outside South Sudan. |
| `"friction_walking"` | float32 | Minutes needed to travel one metre on foot. |
| `"friction_motorised"` | float32 | Minutes per metre using motorised transport where roads exist; never slower than walking. |
| `"county_id"` | uint16 | County of the cell, see `counties.csv`. 0 = outside South Sudan. |

Flood data is loaded by date range (both ends included):

```python
months, flood = load_flood("2020-07", "2020-11")   # flood.shape = (5, 4500, 5308)
months, flood = load_flood("2020-11")              # a single month, flood.shape = (1, 4500, 5308)
months, unusual = load_flood("2020-11", kind="unusual")   # only unusual flooding

for month, days in iter_flood("2003-01", "2025-12"):   # one month at a time, for long periods
    ...
```

Each flood array holds the **number of days in that month the cell was flagged as flooded** (0-31).
`kind` picks which flooding is counted, using NASA's labels:

| `kind` | Counts | Meaning |
|---|---|---|
| `"all"` (default) | every flooded day | |
| `"recurring"` | days labelled recurring | the cell is inside the area that normally floods in this calendar month |
| `"unusual"` | days labelled unusual | the cell is outside that area |

The label describes the place and month, not the size of the flood, so recurring flooding can't
grow over the years and all growth after 2020 shows up as unusual. Use `"all"` for trends; see
`EDA/eda_flood_masks.ipynb`. Recurring and unusual days can add up to more than the total when a
cell contains two source pixels with different labels on the same day.
The value 255 (`NO_DATA`) means there is no flood data for that cell. That is not the same as
"not flooded", so always handle it separately.

`load_flood_months()` returns one row per month with `missing_days` and `flagged`. Flagged months
have several days without any record in the archive, so their flooding is undercounted; mark them
in any monthly time series.

## Combining the layers

**Which cells are flooded in a month.** The threshold lives in `config.yaml`, not in the files.

```python
days = flood[0]
flooded = (days >= cfg["flood"]["min_days_flooded"]) & (days != NO_DATA)
no_data = days == NO_DATA
```

**Travel cost for a flood scenario.** Copy the static layer and block permanent water and the
flooded cells. Multiply by the cell size to get the minutes needed to cross a cell.

```python
friction = static["friction_walking"].copy()
friction[water] = np.inf                    # rivers and lakes cannot be crossed, in every scenario
friction[flooded] = np.inf                  # flooded cells cannot be crossed
minutes_per_cell = friction * grid.cell_size   # 3 to 60 minutes per 250 m when walking
```

The baseline (no flood) is the same without the `flooded` line; permanent water is blocked in the
baseline too, so it never counts as flood impact. The slow-traveller variant multiplies
the walking friction by `cfg["travel"]["slow_traveller_factor"]`.

**Facilities on the grid.** The facility table already has the cell of each facility.

```python
rows, cols = facilities["row"], facilities["col"]
facility_flooded = flooded[rows, cols]      # True/False per facility
```

**Totals per county.** `county_id` works as a group label for any other layer.

```python
n = static["county_id"].max() + 1
pop_per_county = np.bincount(static["county_id"].ravel(),
                             weights=static["population"].ravel(), minlength=n)
pop_flooded_per_county = np.bincount(static["county_id"].ravel(),
                                     weights=(static["population"] * flooded).ravel(), minlength=n)
# index i is county_id i; index 0 is "outside South Sudan"
```

## Known limitations

- **No flood data north of 10N.** About 6% of South Sudan (Renk, Manyo and parts of 7 other
  counties) has the value 255 in every flood file. 49 facilities are affected; they have
  `has_flood_data = False`. Report these as "no data", never as "not flooded".
- **Permanent water blocks most, not all, of the rivers.** Neither the flood files (NASA removes known
  open water) nor the travel-cost layers (see `EDA/eda_friction.ipynb`) block rivers, so
  `permanent_water_250m.tif` does, from JRC Global Surface Water (v1.5, 1984-2024). A cell is blocked
  if any 30 m pixel in it is water in at least 75% of observations, small gaps are closed, and cells
  where people live are always left open (they have land; riverside towns are where bridges and
  ferries are; 0 people end up in blocked cells). Checked along the White Nile, the detour needed to
  cross from 2 km on one bank to 2 km on the other:

  | Juba | Mangalla | Bor | Shambe | Adok (Sudd) | Melut | Renk |
  |---|---|---|---|---|---|---|
  | open (town) | open (town) | 5 km | 18 km | not blocked | 11 km | 85 km |

  So the river is a barrier for anyone within a 2-hour walk at Shambe, Melut and Renk, but near Bor
  the braided channel leaks, and in the Sudd JRC barely sees the channel under the vegetation.
  Bridges and ferries outside towns are not represented. The settings are under `permanent_water`
  in `config.yaml`.
- **Recurring vs unusual is NASA's fixed mask.** It was computed once over 2003-2024 and is the same
  every year, so it describes where flooding normally happens, not how unusual a given year is.
  Our recalculation with NASA's rule agrees with it for 97-99% of pixel-months
  (`EDA/eda_flood_masks.ipynb`).
- **Gaps in the flood record.** Some days have no records at all. Two longer gaps fall in the period we
  use: 15 April to 19 May 2006 and 18 to 29 November 2007, so April 2006, May 2006 and November 2007
  are flagged in `flood_months.csv` (3 or more missing days, `flag_min_missing_days` in the config).
  Every other month misses at most 2 days. See the missing-days check in `EDA/hydro_flood_eda.ipynb`.
- **Flood counts are a lower bound.** Cloudy days are missing from the source data, so a cell can be
  flooded on more days than the files show.
- **Only the flood layer is truly 250 m.** Travel cost is 1 km data repeated into 250 m cells, and
  many facility coordinates are less precise than that. Read results per county, not per cell.
- **Facilities:** see "Cleaning applied" above. 80 of the kept facilities share a 250 m cell with
  another; the travel-time engine should treat each shared cell as one site.
- **Abyei Region is counted as part of South Sudan.** County `SS0001` ("Abyei Region") is an area
  disputed between South Sudan and Sudan. We include it as a county because the boundary file does.
  The facility dataset has no facilities there, so its population can only reach facilities in
  neighbouring counties. Mention this wherever county results are reported.
