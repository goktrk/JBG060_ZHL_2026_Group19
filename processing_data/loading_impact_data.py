import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
import rasterio 
import rioxarray as rxr
from geopandas.geodataframe import GeoDataFrame
from networkx.classes import MultiDiGraph
from rasterio.windows import from_bounds

## for open streetmap
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import os
from shapely.geometry import Point


#### NOTE: FIRST DOWNLOAD THE RAW DATA FROM SURFDRIVE

def load_admin_boundaries() -> tuple[GeoDataFrame, GeoDataFrame]:
    """
    Load the administrative boundaries of South Sudan at admin levels 1 and 2.

    The data was downloaded from
    https://data.humdata.org/dataset/south-sudan-administrative-boundaries.
    """
    data_dir = Path('./raw_data/Administrative boundaries')
    admin_boundaries_1 = gpd.read_file(data_dir / 'ssd_admin1.geojson')
    admin_boundaries_2 = gpd.read_file(data_dir / 'ssd_admin2.geojson')

    return admin_boundaries_1, admin_boundaries_2


admin_boundaries_1, admin_boundaries_2 = load_admin_boundaries()


def locate_coordinate(long: float, lat: float) -> tuple[str, str]:
    """
    Return the admin-level 1 and 2 names containing a longitude/latitude pair.
    """
    point = Point(long, lat)
    admin1_hits = admin_boundaries_1[admin_boundaries_1.contains(point)]
    admin2_hits = admin_boundaries_2[admin_boundaries_2.contains(point)]

    admin1_name = "None"
    admin2_name = "None"

    if not admin1_hits.empty:
        admin1_name = admin1_hits.iloc[0]['adm1_name']
    else:
        print("Warning! did not find an admin1 area for this coordinate")

    if not admin2_hits.empty:
        admin2_name = admin2_hits.iloc[0]['adm2_name']
    else:
        print("Warning! did not find an admin2 area for this coordinate")

    return admin1_name, admin2_name


def load_worldpop_coordinate(longitude: float, latitude: float) -> dict[int, pd.DataFrame] | float:
    """
    Constrained estimates of the total number of people per grid square at a resolution of 3 arc
    (approximately 100m at the equator) R2025A version v1.
    Unit: number of people per pixel. 
    
    This function extracts the population count at the given coordinate, at each year between 2015-2025.

    Notice that due to the small spatial grid size of this data, lots of grid locations will not have people.
    """
    years = np.arange(2015, 2026)  # 2015-2025
    data_dir = './raw_data/worldpop'

    # Extract population count per year in a specific location
    pop_data = {}
    for year in years:
        fname = data_dir + f"/ssd_pop_{year}_CN_100m_R2025A_v1.tif"

        with rasterio.open(fname) as src:
            bounds = src.bounds  
            # Check if the provided coordinates are within South Sudan
            if not (bounds.left <= longitude <= bounds.right and
                    bounds.bottom <= latitude <= bounds.top):
                print(f"  WARNING: ({longitude}, {latitude}) is outside raster bounds {bounds}")
                return np.nan

            # Convert lon/lat to (row, col) using the inverse affine transform
            row, col = src.index(longitude, latitude)   # src.index takes (x, y) = (lon, lat)
            value = src.read(1)[row, col]

            if src.nodata is not None and value == src.nodata:
                pop_year = np.nan
            else:
                pop_year = value

        pop_data[year] = pop_year

    return pop_data

    
def load_worldpop_area(bbox: dict) -> dict[int, pd.DataFrame]:
    """
    Constrained estimates of the total number of people per grid square at a resolution of 3 arc
    (approximately 100m at the equator) R2025A version v1.
    Unit: number of people per pixel. 
    
    This function extracts the total population count in the provided bounding box, at each year between 2015-2025.    
    """
    years = np.arange(2015, 2026)
    data_dir = './raw_data/worldpop'

    # Extract population count per year within a bounding box
    pop_total = {}
    for year in years:
        fname = data_dir + f"/ssd_pop_{year}_CN_100m_R2025A_v1.tif"
        with rasterio.open(fname) as src:
            # Extract the correct pixels in the .tif file using the rasterio function
            window = from_bounds(
                bbox['lon_min'], bbox['lat_min'],
                bbox['lon_max'], bbox['lat_max'],
                src.transform
            )
            
            # Load the data in this window
            arr = src.read(window=window).astype(np.float32)

            # Mask no data
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan

            total = float(np.nansum(arr))
            pop_total[year] = total

    return pop_total


def download_OSM_network(name: str) -> MultiDiGraph:
    """
    OpenStreetMap (OSM) has a built-in Python library, with which road networks can be downloaded.

    Here, we provide an example how to do so for the city Malakal, extracting the "drive" network.
    """
    output_dir = Path(f'./raw_data/OSM/{name}')
    output_dir.mkdir(parents=True, exist_ok=True)

    G = ox.graph_from_place(name, network_type='drive')

    # Convert networkx graph
    nodes, edges = ox.graph_to_gdfs(G)

    # Save file
    nodes.to_file(output_dir / 'nodes.shp')
    edges.to_file(output_dir / 'edges.shp')
    print(f"Graph files saved in {output_dir}")

    ox.plot_graph(G)

    return G


def plot_network(name: str) -> tuple[MultiDiGraph, GeoDataFrame, GeoDataFrame]:
    """
    Loads a previously saved/downloaded OSM graph from shapefiles.
    """
    output_dir = Path(f'./raw_data/OSM/{name}')

    nodes = gpd.read_file(output_dir / 'nodes.shp')
    edges = gpd.read_file(output_dir / 'edges.shp')

    # Reconstruct the networkx graph from the GeoDataFrames
    G = ox.graph_from_gdfs(nodes.set_index('osmid'), edges.set_index(['u', 'v', 'key']))
    ox.plot_graph(G)

    print(f"Loaded graph: {len(nodes)} nodes, {len(edges)} edges")
    return G, nodes, edges


def load_health_facilities() -> GeoDataFrame:
    """"
    Load the healthcare facility data, downloaded from
    https://www.africageoportal.com/datasets/NRF-SAEON::sub-saharan-public-health-facilities/about.

    Paper published about this dataset:

            Maina, J., Ouma, P.O., Macharia, P.M. et al.
            A spatial database of health facilities managed by the public health sector in sub-Saharan Africa.
            Sci Data 6, 134 (2019). https://doi.org/10.1038/s41597-019-0142-2

    It contains the location and type of all mapped health facilities in Sub-Saharan Africa,
    from which we extract those in South Sudan, which serves as a master list of the national health facility list.

    It includes the facility types
        - Primary Health Care Unit (PHCU),
        - Primary Health Care Centre (PHCC),
        - State Hospital,
        - Teaching Hospital,
        - County Hospital,

    where
    `` PHCUs are the first level of primary care and provide basic preventive, promotive and curative services and
    expected to serve a population of 15,000. PHCCs, aimed at serving a population of 50,000, are the immediate
    reference facilities  for the PHCUs, providing all the services provided by a PHCU but in theory additional
    services covering diagnostic laboratory, maternity and inpatient care."

        Macharia PM, Ouma PO, Gogo EG, Snow RW, Noor AM. 
        Spatial accessibility to basic public health services in South Sudan. 
        Geospatial Health. 2017 May;12(1):510. DOI: 10.4081/gh.2017.510. PMID: 28555479; PMCID: PMC5483170.
    """
    # Load health facilities file and filter country by South Sudan
    fname = './raw_data/health facilities/Sub-Saharan_public_health_facilities.geojson'
    health_facilities = gpd.read_file(fname)
    health_facilities_SS = health_facilities[health_facilities["Country"]=="South Sudan"]

    return health_facilities_SS


def load_cattle() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    This data contains the Cattle population count at pixel level in the Greater Horn of Africa.

    The data is from Harvard dataverse 2010, uploaded by ICPAC in October 2018.

    Downloaded from: https://geoportal.icpac.net/layers/geonode:cattle_gha/metadata_detail

    This function returns a dataframe with index name 'x', representing the longitudes, and column name 'y',
    with each column being a latitude.
    It returns the dataframe, as well as the longitudes and latitudes.
    """
    fname = './raw_data/farmland/geonode__cattle_gha.tif'
    data = rxr.open_rasterio(fname)
    nan_value = data.rio.nodata

    # Convert to dataframe and create longitudes and latitudes arrays
    df = data[0].to_pandas()
    longitudes = df.columns.values
    latitudes = df.index.values

    # Fill in empty values with np.nan
    df = df.replace(nan_value, np.nan)

    return df, longitudes, latitudes


def load_farmland_mask(bbox: dict, mask_type: str) -> xr.DataArray:
    """
    This dataset contains crop or rangeland masks at 0.004464285715 degree resolution (about 1/4 square kilometer).
    Each pixel represents the area fraction of the specific cover (i.e. percentage of the pixel with crops).
    Data ranges between 1 and 100 showing the % value. 

    This is static data, last updated on December 1st, 2023.
    
    Downloaded from: https://agricultural-production-hotspots.ec.europa.eu/download.php
    """
    # Load either crop or rangeland mask
    if mask_type not in ("crop", "rangeland"):
        raise ValueError("mask_type must be 'crop' or 'rangeland'")

    fname = f"./raw_data/farmland/asap_mask_{mask_type}_v04.tif"
    with rasterio.open(fname) as src:
        # Load the window based on our bounding box
        window = from_bounds(
            bbox['lon_min'], bbox['lat_min'],
            bbox['lon_max'], bbox['lat_max'],
            src.transform
        )
        arr = src.read(1, window=window).astype(np.float32)

        # Extract the transformation
        win_transform = rasterio.windows.transform(window, src.transform)

        # Initialize the longitude (column) and latitude (row) center coordinates
        rows, cols = arr.shape
        col_indexes = np.arange(cols)
        row_indexes = np.arange(rows)

        # Define the spatial coordinates corresponding to the center of each pixel
        longitudes, _ = rasterio.transform.xy(win_transform, np.zeros_like(col_indexes), col_indexes, offset="center")
        _, latitudes = rasterio.transform.xy(win_transform, row_indexes, np.zeros_like(row_indexes), offset="center")

    # Wrap into an Xarray DataArray
    da = xr.DataArray(
        data=arr,
        coords={"latitude": latitudes, "longitude": longitudes},
        dims=["latitude", "longitude"],
        name="crop_mask"
    )

    return da


def load_GDP() -> dict:
    """
    Loads the data of World Bank containing several World Development Indicators.

    Extracts the GDP of South Sudan at which this data is known (2008-2015), and returns this as a dictionary.

    GDP is expressed in USD.

    Downloaded from: https://data.worldbank.org/country/south-sudan
    """
    # Filter on one of the GDP indicators
    data = pd.read_csv("./raw_data/GDP/API_SSD_DS2_en_csv_v2_2529.csv", skiprows=4)
    gdp = data[data["Indicator Name"] == "GDP (current US$)"]
    gdp = gdp.dropna(axis=1)
    print(gdp)

    # Filter on available date range
    GDP_dict = {}
    for year in np.arange(2008, 2016):
        GDP_dict[year] = gdp[str(year)].values[0]

    return GDP_dict


def load_ipc_data() -> pd.DataFrame:
    """
    We load the IPC data on county-level for South Sudan.

    We have downloaded the data available between 2022-2025, and extract the
    number of people classified to be in phase IPC3+ per county.

    A person's household with status classified as IPC3+ means it is classified as crisis, emergency or
    catastrophic/famine, meaning that urgent action is required to reduce the acute food insecurity.

    This data is available in three-month windows, not spanning the entire year.
    Data for different years is available on the website.

    Data downloaded from: https://www.ipcinfo.org/ipc-country-analysis/en/
    """
    folder_name = './raw_data/IPC/'

    # Create list with all files in the IPC folder
    fnames = [f for f in os.listdir(folder_name) if os.path.isfile(os.path.join(folder_name, f))]

    # Get county data
    all_counties = []
    for filename in fnames:
        filepath = os.path.join(folder_name, filename)
        df = pd.read_excel(filepath)

        # Identify county rows from states based on hierarchical spacing, then remove the whitespace
        df["is_county"] = df["Area Name"].str.startswith("  ")
        df["Area Name"] = df["Area Name"].str.strip()

        # Keep county rows and rename Area Name
        counties = df[df["is_county"]].copy()
        counties = counties.rename(columns={"Area Name": "County"})

        # Convert dates
        counties["Current - From Date"] = pd.to_datetime(counties["Current - From Date"], unit ='D')
        counties["Current - Thru Date"] = pd.to_datetime(counties["Current - Thru Date"], unit ='D')

        # Keep relevant columns
        all_counties.append(counties[["Current - From Date", "Current - Thru Date", "County", "Current - Phase 3+"]])

    # Concatenate all counties and rename columns
    combined = pd.concat(all_counties, ignore_index=True)
    combined = combined.rename(columns={
        "Current - From Date": "Start Date",
        "Current - Thru Date": "End Date",
        "Current - Phase 3+": "Phase 3+ Pop"
    })

    # Pivot so each county is a column, value = Phase 3+ population
    result = combined.pivot_table(
        index=["Start Date", "End Date"],
        columns="County",
        values="Phase 3+ Pop"
    ).reset_index()

    # Sort dataframe by start date and reset index
    result = result.sort_values("Start Date").reset_index(drop=True)

    return result



def main():
    """Example usage"""

    ## 1) Loading the administrative boundaries
    admin1, admin2 = load_admin_boundaries()
    print(admin1.columns.tolist())
    print(admin1.head())

    ## 2) Check the administrative area for a given coordinate

    # Roughly Aweil
    long, lat = 27.386, 8.771
    admin1_aweil, admin2_aweil = locate_coordinate(long, lat)
    print(
        "Administrative area for coordinate (lon, lat) = ({}, {}) is: "
        "admin1 = {}, admin2 = {}".format(
            long, lat, admin1_aweil, admin2_aweil
        )
    )

    # Not in South Sudan
    long, lat = 26.6, 10.87
    admin1, admin2 = locate_coordinate(long, lat)
    print(
        "For coordinate (lon, lat) = ({}, {}), NOT in South Sudan".format(
            long, lat
        )
    )
    print("Administrative area is: {}, {}".format(admin1, admin2))

    ### 3) Population data

    ## Loading population data at a single location
    # Where there are people
    pop_loc = load_worldpop_coordinate(31.046249993815, 9.473750002105)
    print(f"Location with population : lon = 31.046249993815, lat = 9.473750002105" )
    for year, count in pop_loc.items():
        print(f"Year {year} has total population count of {count}")
    print("")

    # Where there aren't people
    pop_loc = load_worldpop_coordinate(31, 9)
    print("pop_loc : ", pop_loc)
    print(f"Location with population : lon = 31, lat = 9" )
    for year, count in pop_loc.items():
        print(f"Year {year} has total population count of {count}")
    print("")
    
    ## Loading the data for an area
    bbox_ex = dict()
    bbox_ex['lon_min'] = 29.5
    bbox_ex['lat_min'] = 8.5
    bbox_ex['lon_max'] = 32.5
    bbox_ex['lat_max'] = 10

    pop_area = load_worldpop_area(bbox_ex)
    print("Area : ", bbox_ex)
    for year, count in pop_area.items():
        print(f"Year {year} has total population count of {count}")

    ### 4) Road network from Open Streetmap
    city_name = "Malakal, South Sudan"

    ## Download network once:
    download_OSM_network(city_name)

    ## Then plotting the network
    G = plot_network(f"{city_name}")

    ### 5) Loading the health facility data
    hf = load_health_facilities()
    print(f"Loaded {len(hf)} health facilities in South Sudan")

    ### 6) Grazing cattle
    df, longitudes, latitudes = load_cattle()
    lon_value = longitudes[15]
    grazing_lats = latitudes[np.nonzero(df[lon_value].notna())]
    print(f"at longitude {lon_value} there are cattle grazing at latitudes {grazing_lats}"  )
    lat_value = grazing_lats[-1]

    print(f"at (long, lat) = ({lon_value}, {lat_value}), there are {df.loc[lat_value, lon_value]} cattle grazing")

    ## 7) Cropland mask
    bbox_ssd = dict()
    bbox_ssd['lon_min'] = 24
    bbox_ssd['lat_min'] = 3
    bbox_ssd['lon_max'] = 36
    bbox_ssd['lat_max'] = 13

    cropmask_da = load_farmland_mask(bbox_ssd, "crop")
    target_lat, target_lon = 9.475, 30.725
    crop_val = cropmask_da.sel(latitude=target_lat, longitude=target_lon, method="nearest")
    print(f"At (lat, long) = ({target_lat}, {target_lon}) , a % of {crop_val} is cropland")

    cropmask_da.plot(cmap="YlGn", vmin=0, vmax=100)
    plt.title("Crop Mask (area percentage)")
    plt.show()

    ### 8) Rangeland mask
    rangeland_da = load_farmland_mask(bbox_ssd, "rangeland")
    rangeland_val = cropmask_da.sel(latitude=target_lat, longitude=target_lon, method="nearest")
    print(f"At (lat, long) = ({target_lat}, {target_lon}) , a % of {rangeland_val} is rangeland")

    rangeland_da.plot(cmap="YlGn", vmin=0, vmax=100)
    plt.title("Rangeland Mask (area percentage)")
    plt.show()

    ### 9) Load GDP data
    GDP_dict = load_GDP()
    print(GDP_dict)

    ### 10) Load IPC data
    phase3plus = load_ipc_data()
    print(phase3plus)


if __name__ == "__main__":
    main()


