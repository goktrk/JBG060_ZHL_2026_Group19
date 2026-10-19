# JBG060 ZHL 2026 Capstone Data Challenge (Group 19)

[![Original Brief](https://img.shields.io/badge/Challenge-Project_Brief-blue?style=flat-square)](Project_README.md)
[![Python](https://img.shields.io/badge/Python-3.12+-yellow?style=flat-square)](https://www.python.org/)

**Developed by**: Göktürk Kalender, Nikola Čupić, Petko Tepov, Vedat Yalçin, and Povilas Masiulionis  
**Collaborator:** Zero Hunger Lab (ZHL)    
**Course:** JBG060 Capstone Data Challenge (2026) – TU/e & TiU (JADS)

> For the original course prompt (introduction) and challenge setup, see [Project Brief](Project_README.md).

## Introduction & Project Overview
This project supports the 2026 JBG060 course collaboration with Zero Hunger Lab to investigate flood dynamics in South Sudan. The system integrates hydrometeorological hazard data with exposure and impact data to model how riverine flooding disrupts healthcare access.
The analysis operates as an end-to-end chain where each step reads the output of the previous step: 
```text
build_baseline -> build_access_loss -> build_priorities -> build_prepositioning
```

### Analysis Methadology
The workflow shares a centralized set of flood rules located in data_preparation/scenarios.py to ensure consistency across all outputs:
- **Flooded cell:** A cell is flagged as flooded if it is under water for at least 3 days in a given month. This status blocks travel through the cell.
- **Closed facility:** A facility is considered closed if there is a flooded cell within a 1 km radius.
- **Cut-off populations:** People are categorized as cut off if they remain on dry land but lose their route to a facility due to floodwaters. People living directly in flooded cells are tracked as a separate group.

The pipeline evaluates four core observed months: November 2008, November 2020, October 2025, and January 2025. It produces three main deliverables:

* **Access Loss:** Calculates populations living in flooded cells, populations cut off within 1 or 2 hours of travel, and populations left with no reachable facility at all. This analysis handles walking and motorized travel both nationally and per county.
* **Protection & Prioritisation:** Simulates a "dry-out" step where each closed facility is individually reopened to evaluate how many people regain access within 2 to 4 hours. The model generates a greedy Top 5 list of facilities to protect, ensuring neighboring clinics do not redundantly rescue the same populations. It also reports the population that remains uncovered by these Top 5 facilities.
* **Pre-positioning (Load Shift):** Calculates how much extra patient load an open facility will receive when neighboring clinics close or routes flood. This tracks the usual population, added people, relative increase, their origin, and the extra travel time required, allowing resources to be sent ahead.

### Repository structure

```text
JBG060-2026/
|-- data_preparation/              # Analysis grid, flood, facilities, travel time, routing, prioritization
|-- EDA/                           # Exploratory notebooks and figures
|-- literature/                    # Supporting papers and data documentation
|-- processing_data/
|   |-- loading.py                 # Hydrometeorological data loaders
|   `-- loading_impact_data.py     # Exposure and impact data loaders
|-- raw_data/                      # External download; ignored by Git
|-- tests/                         # pytest tests (python -m pytest tests)
|-- .gitignore
|-- config.yaml                    # Every threshold and parameter
|-- dashboard.py                   # Streamlit dashboard
|-- Project_README.md              # Official README file created by Course Team
|-- README.md                      # README file for project description
`-- requirements.txt               # Pinned Python dependencies

```
### Running everything
About 40 minutes from scratch. Always run as `python -m ...` from the repository root.
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

### Dashboard

Build the prepared data once (`data_preparation/README.md`, "Building the files"), then run from the
repository root:

```
$env:PYTHONPATH = "."; streamlit run dashboard.py    # PowerShell
PYTHONPATH=. streamlit run dashboard.py              # macOS / Linux
```

The first load takes about 30-45 s (facility ranking); later views are cached. How each part is calculated:

| Part | README |
|---|---|
| Prepared data: grid, flood months, facilities | `data_preparation/README.md` |
| Travel time without floods | `data_preparation/README_baseline.md` |
| Flood-prone facilities, Top 5 ranking, protection | `data_preparation/README_prioritization.md` |
| Route to Care | `data_preparation/README_routing.md` |
| Flood scenario months for the dashboard | `data_preparation/README_scenarios.md` |

---

## Current Results
The current impact metrics for the walking scenario across the four evaluated months are as follows:
| Scenario | Facilities closed | In flooded cells | Cut off, 2 h | Top 5 bring back (2 h) | Move to another facility |
|---|---|---|---|---|---|
| Nov 2008 | 43 | 57,691 | 188,470 | 105,258 | 378,120 |
| Nov 2020 | 78 | 103,845 | 279,079 | 149,121 | 848,917 |
| Oct 2025 | 24 | 22,090 | 86,341 | 66,627 | 216,507 |
| Jan 2025 | 45 | 117,750 | 211,304 | 126,754 | 391,266 |

---

### Credits and acknowledgements

Dataset descriptions, file overviews, provenance, and original provider information are documented in `Data_overview.xlsx`. 
Consult the original providers for licenses, citation instructions, and usage restrictions.

### Literature

- Alfieri, L., Libertino, A., Campo, L., Dottori, F., Gabellani, S., Ghizzoni, T., Masoero, A., Rossi, L., Rudari, R.,
  Testa, N., Trasforini, E., Amdihun, A., Ouma, J., Rossi, L., Tramblay, Y., Wu, H., & Massabò, M. (2024).
  Impact-based flood forecasting in the Greater Horn of Africa. *Natural Hazards and Earth System Sciences, 24*,
  199–224. [PDF](<literature/Impact-based flood forecasting in the Greater Horn of Africa.pdf>)
- Slayback, D. (2025). *MODIS/VIIRS NRT global flood products: User guide* (Rev. F). NASA Goddard Space Flight
  Center. [PDF](literature/MCDWD_VCDWD_UserGuide_RevF.pdf) | [NASA Earthdata](https://www.earthdata.nasa.gov/data/instruments/viirs/near-real-time-data/nrt-global-flood-products)
- Petricola, S., Reinmuth, M., Lautenbach, S., Hatfield, C., & Zipf, A. (2022). Assessing road criticality and loss
  of healthcare accessibility during floods: The case of Cyclone Idai, Mozambique 2019. *International Journal of
  Health Geographics, 21*, Article 14. [PDF](<literature/Assessing road crticality and loss of healthcare acessibility during floods.pdf>) |
  [DOI](https://doi.org/10.1186/s12942-022-00315-2)
- Pacetti, T., Caporali, E., & Rulli, M. C. (2017). Floods and food security: A method to estimate the effect of
  inundation on crops availability. *Advances in Water Resources, 110*, 494–504.
  [PDF](<literature/Floods and food insecurity - A method to estimate the effect of inundation on crops availability.pdf>) |
  [DOI](https://doi.org/10.1016/j.advwatres.2017.06.019)

Supporting papers and data manuals are retained in [`literature/`](literature/). The project relies on pandas, NumPy, Xarray,
Dask, GeoPandas, Shapely, Rasterio, rioxarray, NetworkX, OSMnx, and Matplotlib; see [`requirements.txt`](requirements.txt)
for the complete version-pinned environment.

## Legal and ethical considerations
The analytical tools and maps generated by this repository are intended for educational and research use. They have not been validated for operational emergency response or humanitarian resource allocation. Additionally, we acknowledge several critical limitations and ethical dimensions in our methodology:

**Assumptions About the Data:**
- Our models assume all unflooded public-sector facilities are open, staffed, and adequately supplied, which may not reflect reality during a crisis.
- Satellite flood masks indicate extent, not depth. As such, protection recommendations (like sandbags or trenches) cannot guarantee a facility will actually remain functional.
- Travel paths generated by the friction map represent algorithmic movement across open ground, not necessarily navigable, real-world roads.

**Assumptions About People:**
- Our baseline travel estimates rely on a standardized "average traveller," failing to account for disabled individuals, the elderly, children, or pregnant women—the exact populations made most vulnerable by flooding.
- Simulations that reroute cut-off populations to open facilities assume perfect compliance and do not account for the potential of severe crowding or supply depletion at the receiving sites.

**Equity in Prioritization:**
- Ranking facilities for protection based on "most people served" contains an inherent bias toward densely populated urban centers. This utilitarian approach systematically disadvantages remote communities, making scattered populations functionally invisible to the optimizer.
- To mitigate this, our prioritization tools make this value judgment explicitly visible on the dashboard and specifically report the isolated populations that are not covered by top-ranking protections.
