from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds, transform as window_transform

RAW = Path(__file__).resolve().parent.parent / "raw_data"
FARM = RAW / "farmland"
ADMIN2 = RAW / "Administrative boundaries" / "ssd_admin2.geojson"
OUT = Path("eda_out")
FIG = OUT / "figures"

# South Sudan bbox, the ASAP masks are global so we only read this window
BBOX = (23.4, 3.4, 36.0, 12.3)


def read_window(path):
    with rasterio.open(path) as src:
        win = from_bounds(*BBOX, src.transform)
        raw = src.read(1, window=win)
        # check nodata before casting, float64 nodata overflows in float32
        bad = raw == src.nodata if src.nodata is not None else np.zeros(raw.shape, bool)
        with np.errstate(over="ignore"):
            arr = raw.astype(np.float32)
        arr[bad | ~np.isfinite(arr)] = np.nan
        tf = window_transform(win, src.transform)
        print(f"\n{path.name}: crs={src.crs} res={src.res} dtype={src.dtypes[0]} "
              f"nodata={src.nodata} window={arr.shape}")
    return arr, tf, src.crs


def summarize(arr, max_value=None):
    if max_value is not None:
        too_high = arr > max_value
        print(f"  values above {max_value}: {too_high.sum():,}")
        arr = np.where(too_high, np.nan, arr)
    v = arr[np.isfinite(arr)]
    print(f"  valid px: {v.size:,} ({v.size / arr.size:.1%})  "
          f"min={v.min():.3g} max={v.max():.3g} mean={v.mean():.3g}  zeros={(v == 0).mean():.1%}")
    return arr


def pixel_area_km2(shape, tf):
    # pixels in degrees get smaller away from the equator
    lats = tf.f + (np.arange(shape[0]) + 0.5) * tf.e
    h = 110.574 * abs(tf.e)
    w = 111.320 * np.cos(np.deg2rad(lats)) * abs(tf.a)
    return np.repeat((h * w)[:, None], shape[1], axis=1)


def county_labels(adm2, shape, tf, crs):
    shapes = ((geom, i + 1) for i, geom in enumerate(adm2.to_crs(crs).geometry))
    return rasterize(shapes, out_shape=shape, transform=tf, fill=0, dtype="int32")


def zonal_sum(values, labels, n):
    return np.bincount(labels.ravel(), weights=values.ravel(), minlength=n + 1)[1:]


def extent(arr, tf):
    return (tf.c, tf.c + arr.shape[1] * tf.a, tf.f + arr.shape[0] * tf.e, tf.f)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)

    crop, crop_tf, crop_crs = read_window(FARM / "asap_mask_crop_v04.tif")
    crop = summarize(crop, max_value=100)
    rng, rng_tf, _ = read_window(FARM / "asap_mask_rangeland_v04.tif")
    rng = summarize(rng, max_value=100)
    cattle, cat_tf, cat_crs = read_window(FARM / "geonode__cattle_gha.tif")
    cattle = summarize(cattle)

    print(f"\ncrop and rangeland on same grid: {crop.shape == rng.shape and crop_tf == rng_tf}")
    print(f"cattle pixel size: ~{abs(cat_tf.a) * 111:.1f} km")

    px_km2 = pixel_area_km2(crop.shape, crop_tf)
    crop_km2 = np.nan_to_num(crop) / 100 * px_km2
    rng_km2 = np.nan_to_num(rng) / 100 * px_km2
    print(f"cropland: {crop_km2.sum():,.0f} km2, rangeland: {rng_km2.sum():,.0f} km2")
    overlap = (np.nan_to_num(crop) + np.nan_to_num(rng) > 100).sum()
    print(f"pixels where crop + rangeland > 100%: {overlap:,}")

    for arr, name, cmap in [(crop, "crop", "YlGn"), (rng, "rangeland", "YlOrBr")]:
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(arr, extent=extent(arr, crop_tf), cmap=cmap, vmin=0, vmax=100)
        ax.set_title(f"ASAP {name} mask (% cover)")
        fig.colorbar(im, ax=ax, shrink=0.7, label="% of cell")
        save(fig, f"farmland_{name}_map.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(np.log10(np.where(cattle > 0, cattle, np.nan)), extent=extent(cattle, cat_tf), cmap="magma")
    ax.set_title("Cattle (log10 head per cell)")
    fig.colorbar(im, ax=ax, shrink=0.7)
    save(fig, "farmland_cattle_map.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, arr, name, color in [(axes[0], crop, "crop", "tab:green"), (axes[1], rng, "rangeland", "tab:orange")]:
        ax.hist(arr[arr > 0], bins=50, color=color)
        ax.set_title(f"{name}: non-zero % cover")
        ax.set_xlabel("% cover")
    save(fig, "farmland_histograms.png")

    # aggregate to counties
    adm2 = gpd.read_file(ADMIN2)
    n = len(adm2)
    lab = county_labels(adm2, crop.shape, crop_tf, crop_crs)
    lab_cat = county_labels(adm2, cattle.shape, cat_tf, cat_crs)

    res = pd.DataFrame({
        "adm2_name": adm2["adm2_name"],
        "adm2_pcode": adm2["adm2_pcode"],
        "adm1_name": adm2["adm1_name"],
        "area_sqkm_official": adm2["area_sqkm"],
        "area_sqkm_raster": zonal_sum(np.where(np.isfinite(crop), px_km2, 0), lab, n),
        "crop_km2": zonal_sum(crop_km2, lab, n),
        "rangeland_km2": zonal_sum(rng_km2, lab, n),
        "cattle_head": zonal_sum(np.nan_to_num(cattle), lab_cat, n),
        "cattle_px": np.bincount(lab_cat.ravel(), minlength=n + 1)[1:],
    })
    res["crop_share"] = res["crop_km2"] / res["area_sqkm_raster"]
    res["rangeland_share"] = res["rangeland_km2"] / res["area_sqkm_raster"]
    res["cattle_per_km2"] = res["cattle_head"] / res["area_sqkm_raster"]
    res = res.sort_values("crop_km2", ascending=False)
    res.to_csv(OUT / "exposure_by_county.csv", index=False)

    print("\nTop 10 counties by cropland:")
    print(res[["adm2_name", "adm1_name", "crop_km2", "crop_share", "rangeland_share", "cattle_head"]]
          .head(10).round(3).to_string(index=False))
    ratio = res["area_sqkm_raster"] / res["area_sqkm_official"]
    print(f"\nraster area / official area: median {ratio.median():.3f}, "
          f"min {ratio.min():.3f}, max {ratio.max():.3f}")
    print(f"counties with <5 cattle pixels: {res.loc[res['cattle_px'] < 5, 'adm2_name'].tolist()}")

    top = res.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top["adm2_name"], top["crop_km2"], color="tab:green")
    ax.set_xlabel("cropland (km2)")
    ax.set_title("Counties by cropland area")
    save(fig, "farmland_top_counties.png")

    fig, ax = plt.subplots(figsize=(6.5, 6))
    sc = ax.scatter(res["crop_share"], res["rangeland_share"],
                    s=np.sqrt(res["cattle_head"].clip(lower=0)) / 3 + 5,
                    c=res["cattle_per_km2"], cmap="magma", alpha=0.8)
    ax.set_xlabel("cropland share")
    ax.set_ylabel("rangeland share")
    ax.set_title("County profile (size/colour = cattle)")
    fig.colorbar(sc, ax=ax, label="cattle per km2")
    save(fig, "farmland_county_profile.png")

    merged = adm2.merge(res, on="adm2_pcode", suffixes=("", "_r"))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, col, cmap in zip(axes, ["crop_share", "rangeland_share", "cattle_per_km2"],
                             ["YlGn", "YlOrBr", "magma"]):
        merged.plot(column=col, cmap=cmap, legend=True, ax=ax, edgecolor="white", lw=0.3)
        ax.set_title(col)
        ax.set_axis_off()
    save(fig, "farmland_choropleths.png")
