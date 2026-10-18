"""
Pre-render the access-loss grid layers for the dashboard's overview map, so the app only places
finished images. Per flood scenario in config.yaml and travel mode, at 1 km (4 x 4 grid cells):

    cut_off        people cut off from care (within travel.max_time_min) living in each pixel
    no_facility    people with no facility reachable living in each pixel
    travel_time    minutes to the nearest open facility during the flood (mean over the pixel's populated cells)
    extra_time     flood minus normal travel time (same), or "now unreachable"

The two time layers only show pixels where people live; uninhabited land is left transparent.

The groups use the same rules as build_access_loss (scenarios.py). 1 km because the travel-cost data
is 1 km, and finer pixels would suggest a precision the data doesn't have.

Outputs in dashboard.output_dir (data/processed/dashboard/):
    layer_<id>_<mode>_<layer>.png   the image, in Web Mercator, transparent where there is nothing to show
    layers.json                     per image its map bounds, and per layer its colour legend

Chain: build_baseline -> build_access_loss -> build_dashboard_layers.
Run from the repository root:  python -m data_preparation.build_dashboard_layers
"""
import json

import numpy as np
from PIL import Image
from pyproj import Transformer
from rasterio.transform import Affine, array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject

from data_preparation.build_access_loss import group_masks
from data_preparation.grid import ROOT, load_config, make_grid
from data_preparation.loading import load_baseline, load_scenario_travel, load_static
from data_preparation.scenarios import scenario_flood

BLOCK = 4   # 4 x 4 cells of 250 m = 1 km

# per layer: (lower bound of each class, colour, label); class 0 is transparent
PEOPLE_CLASSES = [(1, "#fcc5c0", "1–10 people"), (10, "#fa9fb5", "10–50"), (50, "#f768a1", "50–200"),
                  (200, "#c51b8a", "200–1,000"), (1000, "#7a0177", "more than 1,000")]
TIME_CLASSES = [(0, "#ffffcc", "under 30 min"), (30, "#fed976", "30–60 min"), (60, "#fd8d3c", "1–2 h"),
                (120, "#e31a1c", "2–4 h"), (240, "#800026", "more than 4 h")]
EXTRA_CLASSES = [(0.5, "#fee6ce", "under 15 min"), (15, "#fdae6b", "15–60 min"), (60, "#f16913", "1–2 h"),
                 (120, "#a63603", "more than 2 h")]
UNREACHABLE = ("#252525", "no facility reachable")
LAYERS = {
    "cut_off": ("People cut off from care (per km²)", PEOPLE_CLASSES, False),
    "no_facility": ("People with no facility reachable (per km²)", PEOPLE_CLASSES, False),
    "travel_time": ("Travel time to care during the flood", TIME_CLASSES, True),
    "extra_time": ("Extra travel time due to the flood", EXTRA_CLASSES, True),
}
ALPHA = 210


def block_sum(values) -> np.ndarray:
    """Sum over 4 x 4 blocks (the grid's size is a multiple of 4: its edges are snapped to 1 km)."""
    h, w = values.shape
    return values.reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK).sum(axis=(1, 3))


def block_mean(values, valid) -> tuple[np.ndarray, np.ndarray]:
    """Mean of `values` over the valid cells of each block, and whether the block has any valid cell."""
    count = block_sum(valid.astype("float64"))
    total = block_sum(np.where(valid, values, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return total / count, count > 0


def classify(values, classes) -> np.ndarray:
    """Class codes 1..len(classes) by lower bound; 0 (transparent) below the first bound or NaN."""
    codes = np.zeros(values.shape, dtype=np.uint8)
    for code, (lower, _, _) in enumerate(classes, start=1):
        codes[np.nan_to_num(values, nan=-np.inf) >= lower] = code
    return codes


def layer_codes(name, pop, inside, base, flood, flooded, limit) -> np.ndarray:
    """Class codes at 1 km for one layer; the code after the last class means 'no facility reachable'."""
    masks = group_masks(pop, base, flood, flooded, [limit])
    if name in ("cut_off", "no_facility"):
        mask = masks[f"cut_off_{limit}"] if name == "cut_off" else masks["no_facility_reachable"]
        return classify(block_sum(np.where(mask, pop, 0.0)), PEOPLE_CLASSES)
    # populated cells that could reach care before the flood: empty land would fill the map with "> 4 h"
    land = inside & (pop > 0) & ~flooded & np.isfinite(base)
    if name == "travel_time":
        minutes, any_reached = block_mean(flood, land & np.isfinite(flood))
        codes = classify(minutes, TIME_CLASSES)
    else:
        extra = np.where(np.isfinite(flood), flood - base, np.nan)
        minutes, any_reached = block_mean(extra, land & np.isfinite(flood))
        codes = classify(minutes, EXTRA_CLASSES)
    _, any_land = block_mean(np.zeros_like(base), land)
    codes[any_land & ~any_reached] = len(LAYERS[name][1]) + 1   # nothing in the block can reach a facility
    return codes


def to_png(codes, grid, classes, with_unreachable, path) -> list:
    """Reproject the class codes to Web Mercator, colour them and save a PNG; returns lat/lon bounds."""
    transform = grid.transform * Affine.scale(BLOCK)
    h, w = codes.shape
    dst_transform, dst_w, dst_h = calculate_default_transform(
        grid.crs, "EPSG:3857", w, h, *array_bounds(h, w, transform), resolution=1000)
    out = np.zeros((dst_h, dst_w), dtype=np.uint8)
    reproject(codes, out, src_transform=transform, src_crs=grid.crs, dst_transform=dst_transform,
              dst_crs="EPSG:3857", resampling=Resampling.nearest, src_nodata=0, dst_nodata=0)
    palette = [c for _, c, _ in classes] + ([UNREACHABLE[0]] if with_unreachable else [])
    rgba = np.zeros(out.shape + (4,), dtype=np.uint8)
    for code, colour in enumerate(palette, start=1):
        rgba[out == code] = [int(colour[i:i + 2], 16) for i in (1, 3, 5)] + [ALPHA]
    Image.fromarray(rgba, "RGBA").save(path, optimize=True)
    west, south, east, north = array_bounds(dst_h, dst_w, dst_transform)
    (lon_w, lon_e), (lat_s, lat_n) = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform(
        [west, east], [south, north])
    return [[lat_s, lon_w], [lat_n, lon_e]]


def main():
    cfg = load_config()
    grid = make_grid(cfg)
    static = load_static(cfg)
    pop, inside = static["population"], static["county_id"] > 0
    limit = cfg["travel"]["max_time_min"]
    out_dir = ROOT / cfg["dashboard"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    index = {"legends": {name: {"title": title,
                                "classes": [[colour, label] for _, colour, label in classes]
                                + ([list(UNREACHABLE)] if unreachable else [])}
                         for name, (title, classes, unreachable) in LAYERS.items()},
             "images": {}}
    for mode in cfg["access_loss"]["modes"]:
        base, _, _ = load_baseline(mode, cfg)
        for scenario in cfg["scenarios"]:
            sid = scenario["id"]
            flood, _ = load_scenario_travel(sid, mode, cfg)
            flooded, _ = scenario_flood(sid, cfg)
            for name, (_, classes, unreachable) in LAYERS.items():
                codes = layer_codes(name, pop, inside, base, flood, flooded, limit)
                file = f"layer_{sid}_{mode}_{name}.png"
                index["images"][f"{sid}|{mode}|{name}"] = {"file": file,
                                                           "bounds": to_png(codes, grid, classes, unreachable,
                                                                            out_dir / file)}
            print(f"{scenario['label']}, {mode}: {len(LAYERS)} layers")
    with open(out_dir / "layers.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
