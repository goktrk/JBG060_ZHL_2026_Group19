# Baseline travel cost

Travel time from every cell to the nearest health facility without flooding, which facility that
is (the catchment), and access per admin unit. The flood steps use the same function,
`travel_time` in `data_preparation/travel.py`, so their results compare directly with the baseline.

## Running it

From the repository root, after the steps in `data_preparation/README.md`:

```
python -m data_preparation.build_static
python -m data_preparation.build_permanent_water
python -m data_preparation.build_facilities
python -m data_preparation.build_baseline    # about a minute
python -m pytest tests                        # tests for the solver
```

Needs `scikit-image`. Settings are under `baseline:` in `config.yaml`.

## Outputs

Everything goes to `data/processed/baseline/` (not in git):

| File | Content |
|---|---|
| `travel_time_{walking,motorised}_min.tif` | minutes to the nearest site, NaN = unreachable |
| `nearest_site_{walking,motorised}.tif` | `site_id` of the nearest site, -1 = unreachable |
| `baseline_admin.csv` | per admin1/2/3 unit and mode: population, mean/median/p90 minutes, % within 30/60/120 min, unreached population, number of sites and facilities |
| `baseline_metadata.json` | grid, units, input hashes, config, sanity checks, effect of the water barrier |
| `internal/sites.parquet` | one row per site with its facilities, admin codes, population served and catchment area. Contains facility locations, so keep it out of the dashboard. |

The rasters are on the same grid as `static_250m.tif` and the flood files.

Facilities in the same 250 m cell are one site. `site_id` goes 0..N-1 in (row, col) order. All
facilities count as open and equal, since the data has nothing on status or capacity. The slow
traveller isn't solved separately: it's walking minutes x `travel.slow_traveller_factor`.

## Using `travel_time` for a flood scenario

```python
import numpy as np
import pandas as pd
from data_preparation.grid import load_config
from data_preparation.loading import load_static, load_permanent_water, load_flood
from data_preparation.build_flood_monthly import NO_DATA
from data_preparation.travel import travel_time

cfg = load_config()
static = load_static(cfg)
water = load_permanent_water(cfg)
sites = pd.read_parquet("data/processed/baseline/internal/sites.parquet")
sources = sites[["row", "col"]].to_numpy()   # row i is site_id i

_, flood = load_flood("2022-10")
flooded = (flood[0] >= cfg["flood"]["min_days_flooded"]) & (flood[0] != NO_DATA)

open_sites = ~flooded[sources[:, 0], sources[:, 1]]   # e.g. close sites whose cell is flooded
res = travel_time(static["friction_walking"], sources[open_sites], blocked=water | flooded)
site_id = np.where(res.nearest >= 0, np.flatnonzero(open_sites)[res.nearest], -1)
```

Things to know:

- `res.nearest` indexes into the sources you passed, so map it back to `site_id` like above.
- Always include `water` in `blocked`, otherwise rivers show up as flood impact.
- Moves are 8-connected (diagonals count sqrt 2). Source cells are always passable, even if blocked.
- Blocking more cells or removing sources never makes any cell faster to reach (tested).
