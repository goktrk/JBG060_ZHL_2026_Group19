# JBG060 ZHL 2026 Capstone Data Challenge (Group 19)

[![Original Brief](https://img.shields.io/badge/Challenge-Project_Brief-blue?style=flat-square)](Project_README.md)
[![Python](https://img.shields.io/badge/Python-3.12+-yellow?style=flat-square)](https://www.python.org/)

**Developed by**: Göktürk Kalender, Nikola Čupić, Petko Tepov, Vedat Yalçin, and Povilas Masiulionis  
**Collaborator:** Zero Hunger Lab (ZHL)    
**Course:** JBG060 Capstone Data Challenge (2026) – TU/e & TiU (JADS)

## Project Overview

### Introduction & Overview

- 1...
- 2...
- 3...

> For the original course prompt (introduction) and challenge setup, see [Project Brief](Project_README.md).

### Repository structure

```text
JBG060-2026/
|-- literature/                    # Supporting papers and data documentation
|-- processing_data/
|   |-- loading.py                 # Hydrometeorological data loaders
|   `-- loading_impact_data.py     # Exposure and impact data loaders
|-- raw_data/                      # External download; ignored by Git
|-- .gitignore
|-- Project_README.md              # Official README file created by Course Team
|-- README.md                      # README file for project description
`-- requirements.txt               # Pinned Python dependencies

```

## Credits and acknowledgements

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
...

to be edited...
