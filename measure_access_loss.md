the script (measure_access_loss.py) calculates the total population in south Sudan that loses access to operational healthcare facilities during a specified flood scenario

### DESCRIPTION:
The script compares a dry day "baseline" travel-time surface against a "flooded" travel-time surface. Using a standart 120 minute walking threshold, it identifies exact 250m raster grid cells where communities previously had access to care but were cut off due to submerged clinics or impassable floodwaters.

It then overlays this cut-off area onto the WorldPop 2020 population grid and aggregates the stranded population by Admin-2 county boundaries.

### PREQUISITES:
Before running this script, the data environment must be built by executing the previous pipeline steps:
 .The raw datasets (map_friction_maps, JRC_surface_water_data, etc.) must be in raw_data/.

 .scikit-image must be installed (pip install scikit-image).

 .All data_preparation/build_*.py scripts must have been successfully executed to populate the data/interim/ and data/processed/ folders.

 ### USAGE:
 Run the script directly from the root repository folder using your environment:

 Note: The script is currently configured to analyze the peak flood scenario of October 2022 (SCENARIO_MONTH = "2022-10").

 ### OUTPUTS:
 [loss_of_access_summary.csv:] A ranked, county-by-county table summarizing the stranded population.

 [loss_mask.npy:] (a boolean mask) (True/False) highlighting exact pixels where the population became stranded (>120 mins).

[delay_array.npy:] A float array calculating the extra travel time (in minutes) for populations that still have access but are forced to take detours.

[flooded_catchment.npy:] An integer array mapping every grid cell to the ID of its new closest surviving clinic under flood conditions.