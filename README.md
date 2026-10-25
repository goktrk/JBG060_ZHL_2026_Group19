# Flood impact on healthcare access in South Sudan (JBG060 Capstone, group 19)

[![Original Brief](https://img.shields.io/badge/Challenge-Project_Brief-blue?style=flat-square)](Project_README.md)
[![Python](https://img.shields.io/badge/Python-3.12-yellow?style=flat-square)](https://www.python.org/)

**Developed by:** Göktürk Kalender, Nikola Čupić, Petko Tepov, Vedat Yalçin and Povilas Masiulionis
**Stakeholders:** ZOA International and the Zero Hunger Lab (Tilburg University)
**Course:** JBG060 Capstone Data Challenge (2026), TU/e and TiU (JADS)

## 1. Introduction and overview

Floods in South Sudan do not only flood health facilities: far more often they cut people off from
care while their own homes stay dry. ZOA wants to act *ahead* of floods (anticipatory action) by
protecting facilities with temporary flood defences and by pre-positioning supplies and staff. This
project builds the **impact layer** for that: it turns a flood map into who loses access to health
care, which facilities to protect first, and where extra demand will arrive.

The pipeline:

1. puts all data on one 250 m grid (Africa Albers Equal Area);
2. computes walking and motorised travel time from every cell to the nearest health facility over a
   friction surface (least-cost path), with rivers and lakes as barriers;
3. applies observed flood months (NASA MODIS, 2003-2025): flooded cells become impassable and a
   facility closes when flooding lies within 1 km of it; everyone is placed in one group (living in
   a flooded cell, cut off from care within 2 hours, no facility reachable, unaffected);
4. ranks the closed facilities by how much access protecting each one restores, under two rules
   that are always shown together: **most saved** (people brought back within 2 h) and
   **worst-off first** (people more than 4 h from care brought back within 4 h);
5. finds the open facilities that receive the people whose facility closed (pre-positioning);
6. evaluates the results over the largest flood month of every year and against WHO reports;
7. shows everything in a Streamlit dashboard for humanitarian staff.

We do not forecast floods: any flood map, including a forecast one, can be run through the same steps.

## 2. Installation

**Requirements:** Python 3.12 (developed and tested with 3.12.4 on Windows 11; the code is plain
Python and should also run on macOS/Linux), about 10 GB free disk space for the data and outputs.

1. Get the code (clone the repository or unzip the submission) and open a terminal in the project folder.
2. Create a virtual environment and install all packages at once from `requirements.txt` (exact versions):

   ```
   python -m venv .venv
   .venv\Scripts\activate          # Windows (macOS/Linux: source .venv/bin/activate)
   pip install -r requirements.txt
   ```

3. Get the data (below) and put it in `raw_data/`.

### Getting the data

The data is not in the Git repository (about 6.6 GB). Two sources:

**Provided by the course** (SURFdrive; link and password in the assignment description on Canvas).
Unzip into `raw_data/`. We use:

| Dataset | Path in `raw_data/` |
|---|---|
| NASA MODIS NRT flood product, recurring and unusual floods, 2000-2025 | `flood_masks/compact_recurring/`, `flood_masks/compact_unusual/` |
| WorldPop constrained population, 100 m (we use 2020) | `worldpop/ssd_pop_2020_CN_100m_R2025A_v1.tif` |
| Public-sector health facilities of sub-Saharan Africa (Maina et al., 2019) | `health facilities/Sub-Saharan_public_health_facilities.geojson` |
| OCHA administrative boundaries | `Administrative boundaries/ssd_admin1.geojson`, `ssd_admin2.geojson`, `ssd_admin3.geojson` |

**Collected by us** (open data). These two datasets are **included in the submitted zip** under
`raw_data/`; if they are missing, download them and place them as shown:

| Dataset | Where to download | Path in `raw_data/` |
|---|---|---|
| Malaria Atlas Project global friction surfaces, walking-only and motorised, ~1 km (Weiss et al., 2020) | [data.malariaatlas.org](https://data.malariaatlas.org/) (accessibility layers) | `map_friction_maps/2020_walking_only_friction_surface.geotiff`, `map_friction_maps/2020_motorized_friction_surface.geotiff` (renamed to these names) |
| JRC Global Surface Water, occurrence v1.5 (1984-2024), tiles 20E_10N, 20E_20N, 30E_10N, 30E_20N | [global-surface-water.appspot.com/download](https://global-surface-water.appspot.com/download) | `JRC_surface_water_data/occurrence_<tile>_v1_5_2024.tif` |

All paths are set in `config.yaml` (`paths:`); change them there if your files are elsewhere.
The first cell of the notebook checks that every input is in place.

## 3. Usage

### Run everything: `pipeline.ipynb`

Open **`pipeline.ipynb`** from the project folder (VS Code, or `pip install notebook` and
`jupyter notebook`), select the `.venv` environment as kernel and run all cells. It runs every step
in order and ends with all tables and figures of the report and poster.

- **Runtime:** about 45 minutes. By default (`REBUILD = True` in the setup cell) every step is rerun
  from the raw data. Set `REBUILD = False` to skip the steps whose outputs already exist (included
  in the zip); a rerun then takes about 5 minutes and needs no raw data.
- Every step is a module in `data_preparation/` that can also be run on its own from the project
  folder, for example `python -m data_preparation.build_baseline`. The order and the options of each
  step are in `data_preparation/README.md`.

| Step (module in `data_preparation/`) | Output | Time |
|---|---|---|
| `build_static`, `build_permanent_water`, `build_facilities`, `build_flood_monthly` | aligned grids, cleaned facilities, monthly flood files in `data/interim/` | ~7 min |
| `build_baseline` | travel time without floods, `data/processed/baseline/` | ~1.5 min |
| `build_access_loss` | who is cut off per scenario, `data/processed/access_loss/` | ~4 min |
| `build_priorities` | the two protection rankings, `data/processed/priorities/` | ~15 s |
| `build_prepositioning` (`--skip-protected` for ~1 min) | facilities receiving extra people, `data/processed/prepositioning/` | ~18 min |
| `build_dashboard_layers` | map layers for the dashboard, `data/processed/dashboard/` | ~1.5 min |
| `peak_months`, then the three steps above with `--months-file data/processed/evaluation/peak_months.csv` | the largest flood month of every year | ~13 min |
| `evaluate_peak_months` | all evaluation measures, `data/processed/evaluation/` | ~2 min |
| `external_check` | our closed facilities against WHO reports, `data/processed/evaluation/` | ~10 s |
| `poster_figures` | figures in `figures/poster/` | ~1.5 min |

**Parameters** (flood days per month, 1 km closure buffer, 2 h and 4 h limits, Top 5, population year,
permanent-water threshold, ...) are all in `config.yaml`. Every output records the settings it was
built with, and a step stops with a clear message if the step before it used other settings.

### Where the results are

| Result | File |
|---|---|
| Report and poster figures | `figures/poster/*.png` / `*.svg` (data behind them in `figures/poster/data/`); pipeline diagram `figures/report/pipeline_figure.tex` |
| Access loss per year, rankings, simple-rule comparison, trade-off, pre-positioning concentration | `data/processed/evaluation/*.csv`, headline numbers in `evaluation.json` and `large_years_summary.csv` |
| Comparison with WHO reports | `data/processed/evaluation/external_check.csv`, `external_check_monthly.csv` |
| Per scenario: access loss per county, rankings, receiving facilities | `data/processed/access_loss/`, `priorities/`, `prepositioning/` |

Example (Jan 2025, walking): 45 facilities closed, 117,750 people living in flooded cells and 211,304
cut off from care within 2 hours; protecting the Top 5 of the *most saved* ranking brings 126,754 of
them back.

### Dashboard

After the pipeline has run, start the dashboard from the project folder:

```
$env:PYTHONPATH = "."; streamlit run dashboard.py     # Windows PowerShell
PYTHONPATH=. streamlit run dashboard.py               # macOS / Linux
```

It only reads the precomputed outputs. The first load takes about 30-45 s; later views are cached.
Choose a flood scenario and travel mode, switch between the two rankings, and see which facilities
are closed, which to protect, and which receive extra people.

### Tests

`python -m pytest tests` runs 58 unit tests on small synthetic grids (also run at the end of the
notebook): the travel-time engine against SciPy's Dijkstra, no extra blocking ever makes travel
faster, reopening one facility equals a full rerun, and population is never lost or double counted.

### Repository structure

```text
README.md                  this file
pipeline.ipynb             runs the whole analysis and shows the results
config.yaml                every threshold, parameter and path
requirements.txt           Python packages with exact versions
dashboard.py               Streamlit dashboard (reads data/processed/ only)
data_preparation/          one module per pipeline step, each with a docstring; README*.md explain
                           every step in detail (README.md: order, inputs, outputs)
tests/                     pytest unit tests
data/interim/              aligned inputs built by the pipeline (not in Git)
data/processed/            all outputs: baseline/, access_loss/, priorities/, prepositioning/,
                           dashboard/, evaluation/
figures/poster/            poster and report figures;  figures/report/  pipeline diagram (LaTeX)
EDA/                       exploratory notebooks behind our data choices, and their outputs
processing_data/           data loaders provided by the course team (used by the EDA notebooks)
literature/                papers and data manuals we used
raw_data/                  the downloaded data (not in Git, see "Getting the data")
Project_README.md          original course README
```

## 4. Troubleshooting and FAQ

- **`ModuleNotFoundError: No module named 'data_preparation'`**: run from the project folder, as
  `python -m data_preparation.<step>` (not `python data_preparation/<step>.py`). For the dashboard,
  set `PYTHONPATH` as shown above.
- **`FileNotFoundError` for a raw file**: check the paths in "Getting the data" and in `config.yaml`;
  the first notebook cell lists what is missing.
- **"... was built with other settings; rerun ..."**: you changed `config.yaml`; rerun the steps
  from the one named in the message onwards (or the notebook with `REBUILD = True`).
- **`python` not found / wrong packages**: activate `.venv` first, and pick it as the notebook kernel.
- **Long runtimes or memory errors**: the grid has 24 million cells; close other programs. The
  slowest steps are `build_prepositioning` (use `--skip-protected`) and the 23 peak months.
- **Strange characters in the output on Windows**: set `PYTHONIOENCODING=utf-8` (the notebook does this).
- **No flooding north of 10°N**: not a bug. The flood data stops at 10°N; about 400,000 people
  there are reported as "without flood data", never as unaffected.
- **Flooding looks small in the rainy season**: the satellite cannot see through clouds, so
  observed floods are a lower bound; the largest observed months are usually December/January.

## 5. Credits and acknowledgements

**Data:** NASA MODIS/VIIRS NRT Global Flood Product (Slayback, 2025;
[user guide](literature/MCDWD_VCDWD_UserGuide_RevF.pdf)); WorldPop (www.worldpop.org); Maina et al. (2019),
*A spatial database of health facilities managed by the public health sector in sub-Saharan Africa*,
Scientific Data 6:134; OCHA/HDX administrative boundaries; Weiss et al. (2020), *Global maps of
travel time to healthcare facilities*, Nature Medicine 26 (Malaria Atlas Project friction surfaces);
Pekel et al. (2016), *High-resolution mapping of global surface water and its long-term changes*,
Nature 540 (JRC Global Surface Water); WHO situation reports for the plausibility check (sources in
`data_preparation/external_check.py`). The course team provided the data download and the loaders in
`processing_data/`.

**Method:** the before/after travel-time comparison follows Petricola et al. (2022), *Assessing road
criticality and loss of healthcare accessibility during floods: the case of Cyclone Idai*,
International Journal of Health Geographics 21:14. Other literature is in [`literature/`](literature/).

**Software:** least-cost travel time with `skimage.graph.MCP_Geometric`
([scikit-image](https://scikit-image.org/)); rasters with [rasterio](https://rasterio.readthedocs.io/);
vectors with [GeoPandas](https://geopandas.org/); dashboard with [Streamlit](https://streamlit.io/)
and [Folium](https://python-visualization.github.io/folium/); also NumPy, pandas, SciPy, Matplotlib
and pytest (full list with versions in `requirements.txt`).

**AI use:** parts of the code and documentation were written with the help of an AI assistant
(Claude, Anthropic) and reviewed, tested and adapted by the team.

## 6. Legal and ethical considerations

- **A screening tool, not instructions.** All outputs are meant for humanitarian staff to check and
  verify locally. They are not validated for operational use and must never be passed on as
  instructions to the population.
- **Ranking is a value choice.** Ranking facilities by "most people served" favours densely
  populated areas; remote villages rarely come out on top. We therefore always show both rankings
  (*most saved* and *worst-off first*) together with the people each one does not cover, and leave
  the choice to ZOA staff. The worst-off rule uses one partial proxy (being more than 4 hours from
  care); it measures loss of access, not need, and ignores food insecurity, displacement and the
  capacity to cope.
- **Who the model does not see.** People already beyond the time limit before a flood and people
  living in flooded cells are not counted in the rankings (the latter are reported separately);
  nothing is known north of 10°N; private and informal providers are missing; every facility not
  closed by flooding counts as open and equal, without data on staffing, stock or capacity.
- **One traveller.** Travel times are for an able-bodied walker. A 2-hour walk takes much longer for
  older or disabled people, children or a woman in labour, so their real loss of access is larger.
- **Paths are not routes.** Travel paths on the dashboard follow a raster model across open ground,
  not real roads or safe crossings; they show what the model assumes, not a journey to take.
- **Floods are a lower bound.** Clouds hide much of the flooding in the rainy season (in 2024 and
  2025 the model closes far fewer facilities than WHO reported), and the data has no water depth, so
  a defended facility is not guaranteed to stay open.
- **Data and privacy.** The project uses no personal data, only open, aggregated datasets. Facility
  locations come from a published public-sector database. Respect each data provider's licence and
  citation requirements; the course data may only be used under the course's terms.
