"""
The one analysis grid that every layer is put on, plus the shared config and county helpers.

The grid is defined by config.yaml: a projected CRS with square cells, covering South Sudan
plus a buffer, with its edges snapped to multiples of `snap_m` so a 1 km grid is exact 4x4 blocks.
"""
from dataclasses import dataclass
from pathlib import Path
import math

import geopandas as gpd
import numpy as np
import yaml
from affine import Affine
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    """Read config.yaml from the repository root."""
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def data_path(cfg: dict, key: str, **fmt) -> Path:
    """Full path for an entry under `paths` in the config, filling in e.g. {year} and {month}."""
    return ROOT / cfg["paths"][key].format(**fmt)


@dataclass(frozen=True)
class Grid:
    crs: str
    cell_size: float
    transform: Affine   # maps (col, row) to the (x, y) of a cell's top-left corner
    width: int          # number of columns
    height: int         # number of rows

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)


def make_grid(cfg: dict) -> Grid:
    """Build the analysis grid: South Sudan's bounding box plus the buffer, snapped outwards."""
    crs = cfg["grid"]["crs"]
    cell = cfg["grid"]["cell_size_m"]
    snap = cfg["grid"]["snap_m"]
    buffer = cfg["grid"]["buffer_km"] * 1000

    admin2 = gpd.read_file(data_path(cfg, "admin2")).to_crs(crs)
    xmin, ymin, xmax, ymax = admin2.total_bounds
    xmin = math.floor((xmin - buffer) / snap) * snap
    ymin = math.floor((ymin - buffer) / snap) * snap
    xmax = math.ceil((xmax + buffer) / snap) * snap
    ymax = math.ceil((ymax + buffer) / snap) * snap

    return Grid(
        crs=crs,
        cell_size=cell,
        transform=Affine(cell, 0, xmin, 0, -cell, ymax),
        width=round((xmax - xmin) / cell),
        height=round((ymax - ymin) / cell),
    )


def lonlat_to_rowcol(grid: Grid, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row and column of the grid cell containing each lon/lat point (may fall outside the grid)."""
    to_grid = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    x, y = to_grid.transform(lon, lat)
    col = np.floor((x - grid.transform.c) / grid.cell_size).astype(np.int64)
    row = np.floor((grid.transform.f - y) / grid.cell_size).astype(np.int64)
    return row, col


def in_grid(grid: Grid, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    """True where a row/column pair lies inside the grid."""
    return (row >= 0) & (row < grid.height) & (col >= 0) & (col < grid.width)


def cell_centres_lonlat(grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Longitude and latitude of every cell centre, as two arrays of the grid's shape."""
    cols, rows = np.meshgrid(np.arange(grid.width), np.arange(grid.height))
    x = grid.transform.c + (cols + 0.5) * grid.cell_size
    y = grid.transform.f - (rows + 0.5) * grid.cell_size
    to_lonlat = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    return to_lonlat.transform(x, y)


def raster_profile(grid: Grid, count: int, dtype: str, nodata) -> dict:
    """Settings for writing a compressed GeoTIFF on the grid."""
    return dict(
        driver="GTiff", crs=grid.crs, transform=grid.transform,
        width=grid.width, height=grid.height,
        count=count, dtype=dtype, nodata=nodata,
        compress="deflate", tiled=True, blockxsize=512, blockysize=512,
    )


def load_counties(cfg: dict) -> gpd.GeoDataFrame:
    """Admin-2 counties with a `county_id` (1..N in pcode order; 0 is kept for 'outside')."""
    counties = gpd.read_file(data_path(cfg, "admin2"))[["adm2_pcode", "adm2_name", "adm1_name", "geometry"]]
    counties = counties.sort_values("adm2_pcode").reset_index(drop=True)
    counties.insert(0, "county_id", counties.index + 1)
    return counties
