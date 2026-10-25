# Analysis update: fixes, protection ranking, access loss, pre-positioning

A summary of what changed in the analysis steps. The details are in the READMEs in `data_preparation/`.

## 1. The analysis is now a chain

Each step reads the previous step's output instead of recomputing it:

```
build_baseline -> build_access_loss -> build_priorities -> build_prepositioning
```

The flood rules all steps share are in one file, `data_preparation/scenarios.py`, so the numbers can't drift apart again:

- **Flooded cell:** flagged on at least 3 days that month (`flood.min_days_flooded`). Blocked for travel.
- **Closed facility:** a flooded cell within **1 km** (`facilities.flood_buffer_m`). Before, only the facility's own cell counted, which closed almost no facility in a typical month.
- **Cut off:** on dry land, within the time limit before the flood, beyond it (or unable to reach any facility) with the flood. People living in flooded cells are a separate group.

Each step stops with a clear message if the previous one is missing or was built with other settings.

## 2. Scenarios

The four observed months from `select_scenarios.py` are now in `config.yaml` (`scenarios:`), and every
step runs on them: Nov 2008, Nov 2020, Oct 2025, Jan 2025. The protection ranking no longer uses
"flooded in any year since 2003", which was about three times the worst real flood.

Note: these months were picked under the old rules. `select_scenarios.py` now uses the shared rules;
rerunning it (~90 min) may pick different months.

## 3. Protection & prioritisation (`build_priorities.py`)

Reworked to follow the PDF's "dry-out" step: each closed facility is **reopened on its own** and we count
who gets access back. Before, it ranked by normal catchment size, and 3 of the Top 5 were surrounded by
flooded ground, so protecting them helped nobody.

- **Most saved:** people brought back within 2 h.
- **Worst-off first:** people who would be more than 4 h from care without the facility, brought back within 4 h (`priorities.worst_off_min`).
- **Greedy Top 5:** each pick adds the most people not already brought back, so neighbouring clinics that rescue the same village don't both make the list.
- Every ranking comes with the **people not covered** by the Top 5.

The dashboard reads the precomputed results (`get_priorities`).

## 4. Access loss (`build_access_loss.py`, replaces `measure_access_loss.py`)

Per scenario, walking and motorised, nationally and per county, people are split into:

| Group | Who |
|---|---|
| Living in flooded cells | their home cell is flooded |
| Cut off within 1 h / 2 h | on dry land, lost access within that limit |
| No facility reachable | the part of the cut-off people who can't reach any facility |
| Without flood data | north of 10°N, unknown |

Outputs also include flood travel-time and nearest-facility maps, used by the next steps.

## 5. Pre-positioning / load shift (`build_prepositioning.py`)

Which facilities receive extra people when neighbours close or the way there floods, so supplies, staff
or mobile teams can be sent ahead. It doesn't assume patients are redistributed: people go to the
nearest open facility themselves.

- **Receiving list:** usual population, people added (all and within 2 h), relative increase (added / usual), where they come from, extra travel time.
- **Flows:** from facility → to facility → people.
- **Side variant:** the same with the Top 5 protected.

Example, Nov 2020: Langbaar PHCU (Bor South) usually serves ~34,000 people and receives ~76,600 more (+224%), because Bor town's facilities close.

## 6. Current results (walking)

| Scenario | Facilities closed | In flooded cells | Cut off, 2 h | Top 5 bring back (2 h) | Move to another facility |
|---|---|---|---|---|---|
| Nov 2008 | 43 | 57,691 | 188,470 | 105,258 | 378,120 |
| Nov 2020 | 78 | 103,845 | 279,079 | 149,121 | 848,917 |
| Oct 2025 | 24 | 22,090 | 86,341 | 66,627 | 216,507 |
| Jan 2025 | 45 | 117,750 | 211,304 | 126,754 | 391,266 |

## 7. Running everything

```
pip install -r requirements.txt
python -m data_preparation.build_static
python -m data_preparation.build_permanent_water
python -m data_preparation.build_facilities
python -m data_preparation.build_flood_monthly
python -m data_preparation.build_baseline
python -m data_preparation.build_access_loss
python -m data_preparation.build_priorities
python -m data_preparation.build_prepositioning     # ~18 min; --skip-protected for ~1 min
python -m data_preparation.build_dashboard_layers   # ~1.5 min, map layers for the dashboard
python -m pytest tests                              # 45 tests
$env:PYTHONPATH = "."; streamlit run dashboard.py
```

About 40 minutes from scratch. Always run as `python -m ...` from the repository root.

## 7b. Evaluation over the whole flood record

The four dashboard scenarios were picked as showcases, so the poster's charts use **each year's largest
observed flood month** (2003-2025, by flooded area; `peak_months.py`). Access loss and the protection
rankings are run on those 23 months (walking) and summarised by `evaluate_peak_months.py` into
`data/processed/evaluation/`. Note: the largest observed months are mostly December/January, because
clouds hide rainy-season flooding. The old `export_rankings.py` (old ranking method) is removed.

Evaluation over the whole flood record (each year's largest observed flood month), for the poster:

```
python -m data_preparation.peak_months                                                       # ~1.5 min
python -m data_preparation.build_access_loss --months-file data/processed/evaluation/peak_months.csv --mode walking   # ~10 min
python -m data_preparation.build_priorities --months-file data/processed/evaluation/peak_months.csv                  # ~1 min
python -m data_preparation.build_prepositioning --months-file data/processed/evaluation/peak_months.csv --mode walking --skip-protected   # ~1 min
python -m data_preparation.evaluate_peak_months                                              # seconds
python poster_figures.py --only peak_access_loss protection_curve protection_per_year protection_4h receiving_concentration recurring_receivers
```

## 8. Still open

- **Dashboard:** show the access-loss groups and the pre-positioning list (both precomputed and ready).
- **Route to Care** now follows the selected scenario (same closed facilities and flooded cells as everything else); it still solves the route line live on click. The sidebar slider "Block Cells Flooded in More Than (% of Years)" is no longer used (shown in red) and can be removed.
- **Optional:** rerun `select_scenarios`; `build_access_loss --all-months` (~3 h) for recurrence/lag.
- **Not started:** recurrence (P2), lag analysis (P2), temporary facility siting (P3).
- Large maps in `data/processed/baseline` and `access_loss` are gitignored; `priorities` and `prepositioning` are small and committed.
