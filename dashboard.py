# Use: PYTHONPATH=. streamlit run dashboard.py

####################### Pre-Aggregating and Caching the Heavy Data #######################
import base64
import io
import json
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pyproj import Transformer
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
from folium.template import Template

from data_preparation.grid import ROOT, in_grid, load_config, lonlat_to_rowcol, make_grid
from data_preparation.loading import (
    load_static,
    load_permanent_water,
    load_flood,
    load_facilities,
    load_priorities,
    load_access_loss,
    load_prepositioning,
)
from data_preparation.build_baseline import build_sites
from data_preparation.build_flood_monthly import NO_DATA
from data_preparation.routing import rc_to_lonlat, route
from data_preparation.scenarios import scenario_flood

MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August",
               "September", "October", "November", "December"]
FLOOD_LAYER = "Flooded Cells (selected month)"
WATER_LAYER = "Permanent Water (rivers, lakes)"
FLOOD_RGBA, WATER_RGBA = (0, 188, 212, 150), (8, 48, 107, 190)   # light cyan / dark blue, see the map legends

# The route map's layer menu folds into a ☰ button
LAYER_MENU_CSS = """<style>
.leaflet-control-layers .leaflet-control-layers-toggle {
    background-image: none !important; font-size: 22px; color: #333; text-decoration: none;
}
.leaflet-control-layers:not(.leaflet-control-layers-expanded) .leaflet-control-layers-toggle {
    display: flex; align-items: center; justify-content: center;
}
.leaflet-control-layers-toggle::before { content: "☰"; }
</style>"""


@st.cache_resource
def get_base_pipeline():
    cfg = load_config()
    grid = make_grid(cfg)
    static = load_static(cfg)
    water = load_permanent_water(cfg)
    facilities = load_facilities(cfg)
    counties_meta = pd.read_csv("data/interim/counties.csv")
    if "adm2_pcode" not in counties_meta.columns and "county_pcode" in counties_meta.columns:
        counties_meta["adm2_pcode"] = counties_meta["county_pcode"]
    facilities["Facility_n"] = (
        facilities["adm2_name"].fillna(facilities["Admin1"])
        + " · "
        + facilities["Facility_t"].fillna("Clinic")
    )
    return cfg, grid, static, water, facilities, counties_meta


@st.cache_data
def compute_flood_scenario(month_str, min_days):
    cfg, grid, static, water, facilities, counties_meta = get_base_pipeline()

    months, flood = load_flood(month_str)
    days = flood[0]
    flooded = (days >= min_days) & (days != NO_DATA)

    #Facilities (their flood status comes from get_priorities: closed in this scenario or not)
    fac_df = facilities.copy()

    county_ids = static["county_id"].ravel()
    pop = static["population"].ravel()
    n_counties = int(county_ids.max() + 1)

    total_pop = np.bincount(county_ids, weights=pop, minlength=n_counties)
    flooded_pop = np.bincount(county_ids, weights=(pop * flooded.ravel()), minlength=n_counties)

    counties_df = counties_meta.copy()
    counties_df["total_pop"] = counties_df["county_id"].map(lambda cid: total_pop[cid] if cid < n_counties else 0)
    counties_df["flooded_pop"] = counties_df["county_id"].map(lambda cid: flooded_pop[cid] if cid < n_counties else 0)
    counties_df["pct_pop_affected"] = (counties_df["flooded_pop"] / np.maximum(counties_df["total_pop"], 1)) * 100

    return fac_df, counties_df


@st.cache_resource
def get_sites():
    cfg, grid, static, water, facilities, _ = get_base_pipeline()
    sites = build_sites(facilities, grid)
    return sites, sites[["row", "col"]].to_numpy()


@st.cache_resource(max_entries=4)
def get_route_inputs(scenario_id):
    """Blocked cells, open sites and cells without flood data in one scenario, with the same rules as
    build_access_loss: permanent water and the month's flooded cells are blocked, and the sites it closes
    (a flooded cell within facilities.flood_buffer_m) can't be destinations."""
    cfg, grid, static, water, facilities, _ = get_base_pipeline()
    sites, _ = get_sites()
    flooded, no_data = scenario_flood(scenario_id, cfg)
    usable = np.ones(len(sites), dtype=bool)
    usable[get_access_loss(scenario_id)[0]["closed_site_ids"]] = False
    return water | flooded, usable, no_data


def output_stamp(section):
    """Latest modification time of a build step's output folder. Passing it to the cached loaders below
    makes Streamlit reload the files after the step is rerun, instead of serving the old cached result."""
    folder = ROOT / load_config()[section]["output_dir"]
    return max((f.stat().st_mtime for f in folder.glob("*")), default=0.0)


@st.cache_data
def cached_priorities(scenario_id, stamp):
    return load_priorities(scenario_id, load_config())


@st.cache_data
def cached_access_loss(scenario_id, stamp):
    return load_access_loss(scenario_id, load_config())


@st.cache_data
def cached_prepositioning(scenario_id, stamp):
    return load_prepositioning(scenario_id, load_config())


def get_priorities(scenario_id):
    """Precomputed protection priorities for one scenario, from data_preparation.build_priorities:
    one row per closed site with both rankings, and the totals. See README_prioritization.md."""
    return cached_priorities(scenario_id, output_stamp("priorities"))


def get_access_loss(scenario_id):
    """Precomputed access loss for one scenario (build_access_loss): national summary and county table."""
    return cached_access_loss(scenario_id, output_stamp("access_loss"))


def get_prepositioning(scenario_id):
    """Precomputed pre-positioning for one scenario (build_prepositioning): summary, receiving, flows, sources."""
    return cached_prepositioning(scenario_id, output_stamp("prepositioning"))


def get_dashboard_layers():
    return cached_dashboard_layers(output_stamp("dashboard"))


@st.cache_data
def cached_dashboard_layers(stamp):
    """The pre-rendered grid layers (build_dashboard_layers): legends, and per image a data URL and bounds."""
    folder = ROOT / load_config()["dashboard"]["output_dir"]
    with open(folder / "layers.json", encoding="utf-8") as f:
        index = json.load(f)
    for info in index["images"].values():
        info["url"] = "data:image/png;base64," + base64.b64encode((folder / info["file"]).read_bytes()).decode()
    return index


def legend_html(classes):
    """A colour legend line: [[colour, label], ...]."""
    return " &nbsp; ".join(f"<span style='color:{colour}; font-size:18px;'>■</span> {label}"
                           for colour, label in classes)


# hover text on the county map: GeoJSON property -> (label, format)
COUNTY_TOOLTIP = {
    "pct_pop_affected": ("Population Flooded:", lambda v: f"{v:.1f}%"),
    "cut_off_120": ("Cut Off From Care (2 h):", lambda v: f"{v:,.0f}"),
    "pct_cut_off_120": ("Share Cut Off (2 h):", lambda v: f"{v:.1f}%"),
    "no_facility_reachable": ("No Facility Reachable:", lambda v: f"{v:,.0f}"),
    "closed_facilities": ("Facilities Closed:", lambda v: f"{v:,.0f}"),
}


def county_measure_options(cfg):
    """Choices for colouring the county map: label -> (column, legend)."""
    t = cfg["travel"]["max_time_min"]
    options = {"% Population Flooded (Min Days slider)": ("pct_pop_affected", "% Population Flooded per County")}
    for limit in sorted(cfg["access_loss"]["thresholds_min"], reverse=True):
        options[f"People Cut Off From Care (within {limit / 60:g} h)"] = (
            f"cut_off_{limit}", f"People cut off from care, within {limit / 60:g} h")
    options[f"% of County Cut Off From Care (within {t / 60:g} h)"] = (
        f"pct_cut_off_{t}", f"% of county population cut off, within {t / 60:g} h")
    options["People With No Facility Reachable"] = ("no_facility_reachable", "People with no facility reachable")
    options["Facilities Closed"] = ("closed_facilities", "Facilities closed (flood within 1 km)")
    return options


def county_measures(county_stats, access_county, mode, cfg):
    """The county table of the overview map, with the access-loss columns for one travel mode added."""
    cols = ["adm2_pcode", "population", "no_facility_reachable", "closed_facilities"]
    cols += [f"cut_off_{limit}" for limit in cfg["access_loss"]["thresholds_min"]]
    access = access_county.loc[access_county["mode"] == mode, cols]
    out = county_stats.merge(access, on="adm2_pcode", how="left")
    t = cfg["travel"]["max_time_min"]
    out[f"pct_cut_off_{t}"] = 100 * out[f"cut_off_{t}"] / out["population"].clip(lower=1)
    return out


def county_bins(values):
    """Colour classes for a county measure: 0, then quartiles of the non-zero counties, up to the maximum."""
    v = np.asarray(values, dtype="float64")
    v = v[np.isfinite(v)]
    top = float(v.max()) if v.size else 0.0
    if top <= 0:
        return [0.0, 1.0]
    inner = np.quantile(v[v > 0], [0.25, 0.5, 0.75, 0.9])
    edges = sorted({0.0, *(float(round(q, 1)) for q in inner if 0 < q < top)})
    return edges + [float(np.ceil(top * 1.001 + 0.01))]


# relative increase classes on the pre-positioning map: (upper bound, colour, label)
INCREASE_CLASSES = [(0.25, "#cbc9e2", "up to +25%"), (1.0, "#9e9ac8", "+25% to +100%"),
                    (np.inf, "#54278f", "more than +100%")]
NO_USUAL_COLOUR = "#999999"   # facility that usually serves nobody
# WorldPop spreads people thinly, so a facility can "receive" a fraction of a person; below this it rounds
# to 0 people and is left off the pre-positioning maps and tables
MIN_PEOPLE_SHOWN = 0.5


def increase_colour(rel):
    if not np.isfinite(rel):
        return NO_USUAL_COLOUR
    return next(colour for bound, colour, _ in INCREASE_CLASSES if rel <= bound)


def format_increase(rel):
    return "usually serves nobody" if not np.isfinite(rel) else f"+{100 * rel:,.0f}%"


@st.cache_resource(max_entries=16)
def compute_route(lat, lon, scenario_id, mode, protected=()):
    """Flood-aware route from a clicked point, plus the same route ignoring floods for comparison."""
    cfg, grid, static, water, facilities, _ = get_base_pipeline()
    sites, sites_rc = get_sites()
    row, col = lonlat_to_rowcol(grid, np.array([lon]), np.array([lat]))
    if not in_grid(grid, row, col)[0] or static["county_id"][row[0], col[0]] == 0:
        return None
    start = (int(row[0]), int(col[0]))
    friction = static[f"friction_{mode}"]
    blocked, usable, no_data = get_route_inputs(scenario_id)
    if protected:
        usable = usable.copy()
        usable[list(protected)] = True

    flood_aware = route(friction, start, sites_rc[usable], blocked, grid.cell_size, no_data)
    site = sites.iloc[np.flatnonzero(usable)[flood_aware.site]] if flood_aware.site >= 0 else None
    no_flood = route(friction, start, sites_rc, water, grid.cell_size)
    return dict(start=start, flood_aware=flood_aware, site=site, no_flood=no_flood, blocked=blocked)


def mask_to_png(mask, transform, crs, resolution_m, resampling, rgba=(33, 102, 172, 150)):
    """A boolean mask as a PNG data URL plus lat/lon bounds, reprojected to Web Mercator."""
    h, w = mask.shape
    dst_transform, dst_w, dst_h = calculate_default_transform(
        crs, "EPSG:3857", w, h, *array_bounds(h, w, transform), resolution=resolution_m)
    out = np.zeros((dst_h, dst_w), dtype=np.uint8)
    reproject(mask.astype(np.uint8), out, src_transform=transform, src_crs=crs,
              dst_transform=dst_transform, dst_crs="EPSG:3857", resampling=resampling)
    image = np.zeros(out.shape + (4,), dtype=np.uint8)
    image[out > 0] = rgba
    png = io.BytesIO()
    Image.fromarray(image, "RGBA").save(png, format="PNG", optimize=True)
    west, south, east, north = array_bounds(dst_h, dst_w, dst_transform)
    (lon_w, lon_e), (lat_s, lat_n) = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform(
        [west, east], [south, north])
    return ("data:image/png;base64," + base64.b64encode(png.getvalue()).decode(),
            [[lat_s, lon_w], [lat_n, lon_e]])


@st.cache_data(max_entries=8, show_spinner="Drawing flooded cells and permanent water...")
def scenario_layers(scenario_id):
    """The month's flooded cells and the permanent water as map images at 500 m (a 500 m pixel is
    drawn if any 250 m cell in it is flooded / water). Cached per scenario."""
    cfg, grid, static, water, facilities, _ = get_base_pipeline()
    flooded, _ = scenario_flood(scenario_id, cfg)
    return {WATER_LAYER: mask_to_png(water, grid.transform, grid.crs, 500, Resampling.max, WATER_RGBA),
            FLOOD_LAYER: mask_to_png(flooded, grid.transform, grid.crs, 500, Resampling.max, FLOOD_RGBA)}


def add_scenario_layers(m, layers):
    """Permanent water and flooded cells, both shown; each can be switched off in the layer menu."""
    for name, (url, bounds) in layers.items():
        folium.raster_layers.ImageOverlay(image=url, bounds=bounds, name=name).add_to(m)


@st.cache_data
def boundary_geojson():
    folder = "raw_data/Administrative boundaries/"
    with open(folder + "ssd_admin2.geojson") as f2, open(folder + "ssd_admin0.geojson") as f0:
        return json.load(f2), json.load(f0)


def add_boundaries(m):
    """County lines and the country outline, as lines only (no fill), so clicks go through to the map."""
    counties, country = boundary_geojson()
    folium.GeoJson(counties, name="County Boundaries", control=False,
                   style_function=lambda _: {"fill": False, "color": "#555555", "weight": 0.8, "opacity": 0.8}
                   ).add_to(m)
    folium.GeoJson(country, name="Country Boundary", control=False,
                   style_function=lambda _: {"fill": False, "color": "#222222", "weight": 2, "opacity": 0.9}
                   ).add_to(m)


def layer_legend():
    """One line under a map explaining the flood and water colours."""
    flood, water = (f"rgb{c[:3]}" for c in (FLOOD_RGBA, WATER_RGBA))
    return (f"<span style='color:{flood}; font-size:18px;'>■</span> flooded cells in the selected month &nbsp; "
            f"<span style='color:{water}; font-size:18px;'>■</span> permanent water")


def add_facilities(m, facilities, top_n):
    """Every facility, drawn above the flood layers and below the routes."""
    protected, flooded = facilities["is_protected"], facilities["is_flooded"]
    points = gpd.GeoDataFrame(
        {"name": facilities["Facility_n"], "type": facilities["Facility_t"],
         "status": np.select([protected, flooded], [f"PROTECTED (TOP {top_n})", "CLOSED (FLOOD NEARBY)"], "OPERATIONAL"),
         "color": np.select([protected, flooded], ["#1E6FD9", "#FF3344"], "#00E676")},
        geometry=gpd.points_from_xy(facilities["Long"], facilities["Lat"]), crs="EPSG:4326")
    points = points.iloc[np.argsort(flooded.to_numpy(int) + protected.to_numpy(int), kind="stable")]
    folium.GeoJson(
        points.to_json(), name="Health facilities",
        marker=folium.CircleMarker(radius=4.5, color="#ffffff", weight=0.6, fill=True, fill_opacity=0.9),
        style_function=lambda f: {"fillColor": f["properties"]["color"]},
        tooltip=folium.GeoJsonTooltip(fields=["name", "type", "status"], aliases=["Facility:", "Type:", "Status:"]),
    ).add_to(m)


class RememberLayers(folium.MacroElement):
    """Saves layer toggle states across script reruns."""
    _template = Template("""
        {% macro script(this, kwargs) %}
        (function () {
            var map = {{ this._parent.get_name() }}, control = {{ this.control.get_name() }};
            var key = {{ this.key|tojson }}, skip = {{ this.skip|tojson }}, saved = {}, leaving = false;
            try { saved = JSON.parse(localStorage.getItem(key)) || {}; } catch (e) {}
            var bases = control._layers.filter(function (l) { return !l.overlay; });
            var want = bases.filter(function (l) { return saved[l.name]; })[0]
                || bases.filter(function (l) { return map.hasLayer(l.layer); }).pop();
            bases.forEach(function (l) { if (l !== want) map.removeLayer(l.layer); });
            if (want) map.addLayer(want.layer);
            control._layers.forEach(function (l) {
                if (!l.overlay || !(l.name in saved) || skip.indexOf(l.name) >= 0) return;
                if (saved[l.name] !== map.hasLayer(l.layer)) saved[l.name] ? map.addLayer(l.layer) : map.removeLayer(l.layer);
            });
            map.on("unload", function () { leaving = true; });
            map.on("overlayadd overlayremove baselayerchange", function () {
                if (leaving) return;
                control._layers.forEach(function (l) {
                    if (skip.indexOf(l.name) < 0) saved[l.name] = map.hasLayer(l.layer);
                });
                try { localStorage.setItem(key, JSON.stringify(saved)); } catch (e) {}
            });
        })();
        {% endmacro %}
    """)

    def __init__(self, control, key, skip=()):
        super().__init__()
        self._name = "RememberLayers"
        self.control, self.key, self.skip = control, key, list(skip)


def add_layer_menu(m):
    control = folium.LayerControl(position="topright", collapsed=True).add_to(m)
    m.get_root().header.add_child(folium.Element(LAYER_MENU_CSS))
    RememberLayers(control, "route_map_layers").add_to(m)


def build_route_map(grid, result, clicked, facilities, top_n, layers):
    m = folium.Map(location=[7.5, 30.5], zoom_start=6.2, tiles="OpenStreetMap", control_scale=True)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery',
        name='Satellite Imagery'
    ).add_to(m)
    add_scenario_layers(m, layers)
    add_boundaries(m)
    if result is None:
        add_facilities(m, facilities, top_n)
        if clicked:
            folium.Marker(clicked, icon=folium.Icon(color="gray", icon="question-sign")).add_to(m)
        add_layer_menu(m)
        return m

    paths = [result["flood_aware"].path_rc, result["no_flood"].path_rc]
    add_facilities(m, facilities, top_n)

    no_flood = result["no_flood"].path_rc
    if len(no_flood):
        lon, lat = rc_to_lonlat(grid, no_flood[:, 0], no_flood[:, 1])
        folium.PolyLine(list(zip(lat, lon)), color="#666666", weight=3, dash_array="6 6",
                        tooltip="Route if there were no floods").add_to(m)

    flood_aware = result["flood_aware"]
    if len(flood_aware.path_rc):
        lon, lat = rc_to_lonlat(grid, flood_aware.path_rc[:, 0], flood_aware.path_rc[:, 1])
        folium.PolyLine(list(zip(lat, lon)), color="#d7301f", weight=4,
                        tooltip=f"Flood-aware route: {flood_aware.minutes:.0f} min").add_to(m)
        site = result["site"]
        folium.Marker([site["lat"], site["lon"]], icon=folium.Icon(color="green", icon="plus-sign"),
                      popup=f"{site['rep_facility_name']}").add_to(m)

    folium.Marker(clicked, icon=folium.Icon(color="red", icon="user"), tooltip="Start").add_to(m)
    bounds = np.vstack([p for p in paths if len(p)] + [np.array([result["start"]])])
    lon, lat = rc_to_lonlat(grid, bounds[:, 0], bounds[:, 1])
    m.fit_bounds([[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]], padding=(30, 30))
    add_layer_menu(m)
    return m


def format_minutes(minutes):
    return f"{minutes:.0f} min" if minutes < 60 else f"{int(minutes // 60)} h {minutes % 60:02.0f} min"


def render_route_section(scenario_id, scenario_label, mode, protected, facilities_data, top_n):
    cfg, grid, static, water, facilities, _ = get_base_pipeline()
    st.subheader("🧭 Route to Care")
    caption = (f"Click anywhere in South Sudan. The route avoids permanent water and the cells flooded in "
               f"{scenario_label}, and goes to the nearest facility that is open in this scenario (no flooded "
               f"cell within {cfg['facilities']['flood_buffer_m'] / 1000:g} km). The dashed grey line is the "
               f"route without floods. Facilities: red = closed, green = open, blue = protected.")
    if protected:
        caption += (f" The Top {len(protected)} facilities are assumed protected, so they also count as "
                    f"open; the flooded ground around them stays blocked.")
    st.caption(caption)
    layers = scenario_layers(scenario_id)

    clicked = st.session_state.get("route_start")
    result = compute_route(clicked[0], clicked[1], scenario_id, mode, protected) if clicked else None
    if clicked and result is None:
        st.warning("That point is outside South Sudan. Click inside the country.")
    out = st_folium(build_route_map(grid, result, clicked, facilities_data, top_n, layers), key="route_map",
                    width=1000, height=550, returned_objects=["last_clicked"])
    st.markdown(f"<small>{layer_legend()}</small>", unsafe_allow_html=True)
    last = (out or {}).get("last_clicked")
    if last and (last["lat"], last["lng"]) != clicked:
        st.session_state["route_start"] = (last["lat"], last["lng"])
        st.rerun()

    if result is None:
        return
    flood_aware, no_flood, site = result["flood_aware"], result["no_flood"], result["site"]
    if flood_aware.site < 0:
        st.error("No open facility can be reached from here without crossing flooded cells or permanent "
                 "water in this scenario.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Travel Time", format_minutes(flood_aware.minutes))
    c2.metric("Route Length", f"{flood_aware.length_m / 1000:.1f} km")
    if no_flood.site >= 0:
        c3.metric("Extra Time vs. No Floods", f"+{format_minutes(max(flood_aware.minutes - no_flood.minutes, 0.0))}")
    others = f" (+{site['n_facilities'] - 1} more in the same cell)" if site["n_facilities"] > 1 else ""
    c4.metric("Destination", site["rep_facility_type"], help=f"{site['rep_facility_name']}{others}")
    protected_note = " (protected)" if site["site_id"] in protected else ""
    st.write(f"**Destination:** {site['rep_facility_name']}{others}{protected_note}")
    if flood_aware.through_no_data:
        st.warning("Part of this route lies north of 10°N, where there is no flood data. Those cells are "
                   "treated as passable, but they are not known to be dry.")


####################### Map Generator Function #######################

def build_advanced_map(counties_meta, facilities_data, top_n, selected_pcode=None, measure=None, grid=None):
    initial_location = [7.5, 30.5]
    initial_zoom = 6.2

    with open("raw_data/Administrative boundaries/ssd_admin2.geojson") as f:
        admin2_geojson = json.load(f)

    selected_feature = None
    if selected_pcode:
        for feat in admin2_geojson["features"]:
            props = feat.get("properties", {})
            if (props.get("adm2_pcode") == selected_pcode or props.get("county_pcode") == selected_pcode):
                selected_feature = feat
                c_lat = props.get("center_lat") or 7.5
                c_lon = props.get("center_lon") or 30.5
                initial_location = [c_lat, c_lon]
                initial_zoom = 8.5
                break

    m = folium.Map(
        location=initial_location,
        zoom_start=initial_zoom,
        tiles="OpenStreetMap",
        name="Open Street Map",
        control_scale=True
    )

    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery',
        name='Satellite Imagery'
    ).add_to(m)

    if grid is not None:   # a pre-rendered grid layer instead of the county colours
        url, bounds, name = grid
        folium.raster_layers.ImageOverlay(image=url, bounds=bounds, name=name).add_to(m)
        add_boundaries(m)
    else:
        column, legend_name = measure or ("pct_pop_affected", "% Population Flooded per County")
        if column == "pct_pop_affected":
            max_val = max(float(counties_meta["pct_pop_affected"].max()), 15.0)
            bin_edges = [0.0, 1.0, 3.0, 7.0, 15.0, round(max_val + 0.5, 1)]
        else:
            bin_edges = county_bins(counties_meta[column])

        # the county numbers go into the GeoJSON properties, which is where folium's tooltip reads them
        by_pcode = counties_meta.set_index("adm2_pcode")
        shown = {key: spec for key, spec in COUNTY_TOOLTIP.items() if key in by_pcode.columns}
        for feat in admin2_geojson["features"]:
            props = feat["properties"]
            for key, (_, fmt) in shown.items():
                value = by_pcode[key].get(props.get("adm2_pcode"), np.nan)
                props[key] = fmt(value) if pd.notna(value) else "n/a"

        choropleth = folium.Choropleth(
            geo_data=admin2_geojson,
            name="County Vulnerability Risk",
            data=counties_meta,
            columns=["adm2_pcode", column],
            key_on="feature.properties.adm2_pcode",
            bins=bin_edges,
            fill_color="RdPu",
            fill_opacity=0.6,
            nan_fill_color="#cccccc",
            line_color="#333333",
            line_weight=0.7,
            line_opacity=0.8,
            legend_name=legend_name,
        ).add_to(m)

        folium.GeoJsonTooltip(
            fields=["adm2_name", "adm1_name", *shown],
            aliases=["County:", "State:", *(label for label, _ in shown.values())],
            style="background-color: #1e1e1e; color: #ffffff; font-family: sans-serif; font-size: 12px; padding: 6px; border-radius: 4px;"
        ).add_to(choropleth.geojson)

    if selected_feature:
        folium.GeoJson(
            selected_feature,
            name="Focused County Boundary",
            style_function=lambda x: {
                "fillColor": "none",
                "color": "#0055FF",
                "weight": 3.5,
                "opacity": 1.0,
            }
        ).add_to(m)

    facility_cluster = MarkerCluster(
        name="Health Facilities (Clustered)",
        overlay=True,
        control=True,
        options={"maxClusterRadius": 40, "disableClusteringAtZoom": 9}
    ).add_to(m)

    protected_group = folium.FeatureGroup(name=f"Protected Facilities (Top {top_n})")
    if facilities_data["is_protected"].any():
        protected_group.add_to(m)

    for _, fac in facilities_data.iterrows():
        is_flooded = fac["is_flooded"]
        color = "#FF3344" if is_flooded else "#00E676"
        heading_color = "#d9534f" if is_flooded else "#2e7d32"
        status_label = "CLOSED (FLOOD NEARBY)" if is_flooded else "OPERATIONAL"
        if fac["is_protected"]:
            color = heading_color = "#1E6FD9"
            status_label = f"PROTECTED (TOP {top_n})"

        popup_html = f"""
        <div style="font-family: Arial; font-size: 12px; width: 180px;">
            <h4 style="margin: 0 0 5px 0; color: {heading_color};">
                {fac['Facility_n']}
            </h4>
            <b>Type:</b> {fac.get('Facility_t', 'Clinic')}<br>
            <b>State:</b> {fac.get('Admin1', 'N/A')}<br>
            <b>Status:</b> <b>{status_label}</b>
        </div>
        """

        folium.CircleMarker(
            location=[fac["Lat"], fac["Long"]],
            radius=4.5,
            color="#ffffff",
            weight=0.6,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=250)
        ).add_to(protected_group if fac["is_protected"] else facility_cluster)

    folium.LayerControl(position="topright", collapsed=False).add_to(m)
    return m


def build_prepositioning_map(receiving, sites, closed_ids, protected_ids, limit_text, layers, focus_id=None):
    """Receiving facilities as circles (size = people added within the limit, colour = relative
    increase), closed facilities as small red dots and protected ones as blue dots."""
    m = folium.Map(location=[7.5, 30.5], zoom_start=6.2, tiles="OpenStreetMap", control_scale=True)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery', name='Satellite Imagery'
    ).add_to(m)
    add_scenario_layers(m, layers)
    add_boundaries(m)
    where = sites.set_index("site_id")[["lat", "lon"]]

    closed_group = folium.FeatureGroup(name="Closed Facilities (flood within 1 km)").add_to(m)
    for sid in closed_ids:
        protected = sid in protected_ids
        selected = sid == focus_id
        folium.CircleMarker(
            location=where.loc[sid].tolist(), radius=9 if selected else 3.5, color="#222222" if selected else "#ffffff",
            weight=2 if selected else 0.6, fill=True,
            fill_color="#1E6FD9" if protected else "#FF3344", fill_opacity=0.9,
            tooltip=("Selected closed facility" if selected else
                     "Protected (stays open)" if protected else "Closed (flood within 1 km)"),
        ).add_to(closed_group)

    receiving_group = folium.FeatureGroup(name="Facilities Receiving People").add_to(m)
    biggest = max(float(receiving["added_within_limit"].max()), 1.0) if len(receiving) else 1.0
    from_text = "From the selected facility: " if focus_id is not None else "Extra "
    for _, r in receiving.sort_values("added_within_limit").iterrows():   # largest drawn on top
        popup_html = f"""
        <div style="font-family: Arial; font-size: 12px; width: 220px;">
            <h4 style="margin: 0 0 5px 0; color: #54278f;">{r['adm2_name']} · {r['rep_facility_type']}</h4>
            <b>Usually serves:</b> {r['usual_population']:,.0f} people<br>
            <b>{from_text}people within {limit_text}:</b> {r['added_within_limit']:,.0f}<br>
            <b>{from_text}people (all):</b> {r['added']:,.0f}<br>
            <b>Increase within {limit_text}:</b> {format_increase(r['relative_increase_within_limit'])}<br>
            <b>From:</b> {int(r['from_facilities'])} closed or cut-off facilities<br>
            <b>Extra travel time:</b> {r['extra_min']:,.0f} min on average
        </div>"""
        folium.CircleMarker(
            location=where.loc[int(r["site_id"])].tolist(),
            radius=4 + 18 * np.sqrt(max(r["added_within_limit"], 0) / biggest),
            color="#ffffff", weight=1, fill=True,
            fill_color=increase_colour(r["relative_increase_within_limit"]), fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=260),
        ).add_to(receiving_group)

    folium.LayerControl(position="topright", collapsed=False).add_to(m)
    return m


def render_prepositioning_section(cfg, selected_month, criterion, protect_top, protected, top_n):
    """Facilities likely to receive extra people (2a map, 2c list), read from build_prepositioning."""
    st.subheader("📦 Facilities Likely to Receive Extra People")
    summary, receiving, flows, sources = get_prepositioning(selected_month)
    limit = cfg["travel"]["max_time_min"]
    limit_text = f"{limit / 60:g} h"
    variant = f"top{top_n}_{criterion}" if protect_top else "as_is"
    if variant not in set(receiving["variant"]):
        variant = "as_is"
    mode = st.radio("Travel Mode", ("walking", "motorised"), horizontal=True, key="prep_mode",
                    format_func=str.capitalize)
    st.caption(
        "When a flood closes facilities or cuts off the way to them, people go to the nearest facility that is "
        "still open. These are the facilities that receive them: places to send supplies, staff or mobile teams "
        "in advance. Nobody is moved; this follows where people go on their own. "
        + (f"The Top {top_n} facilities are assumed protected (sidebar), so they stay open."
           if variant != "as_is" else "No facilities are assumed protected."))

    groups = summary["groups"][mode][variant]
    c1, c2, c3 = st.columns(3)
    c1.metric("People Going to Another Facility", f"{groups['moved_total']:,}",
              help=f"{groups['moved_facility_closed']:,} because their facility closed, "
                   f"{groups['moved_path_blocked']:,} because their way there floods "
                   f"({groups['moved_path_blocked_small_delay']:,} of those with less than "
                   f"{cfg['prepositioning']['small_delay_min']} min extra).")
    c2.metric(f"Of Whom Within {limit_text} of It", f"{groups['moved_within_limit']:,}",
              help="The more realistic extra demand: people far from any open facility may not make the trip.")
    r = receiving[(receiving["mode"] == mode) & (receiving["variant"] == variant)
                  & (receiving["added"] >= MIN_PEOPLE_SHOWN)].copy()
    c3.metric("Facilities Receiving People", f"{len(r):,}")

    states = get_base_pipeline()[5].set_index("adm2_pcode")["adm1_name"]
    r["State"] = r["admin2_pcode"].map(states)

    focus = choose_focus(sources, mode, selected_month, protected if variant != "as_is" else ())
    if focus is not None:
        render_focus(focus, r, flows, sources, mode, variant, selected_month, limit_text)
        return

    left, right = st.columns(2)
    sort_by = left.radio("Sort By", (f"Most Extra People (within {limit_text})",
                                     f"Largest Relative Increase (within {limit_text})"),
                         horizontal=True, key="prep_sort")
    how_many = right.radio("Show", ("Top 25", "All"), horizontal=True, key="prep_show")
    key = "added_within_limit" if sort_by.startswith("Most") else "relative_increase_within_limit"
    r = r.sort_values(key, ascending=False, na_position="last")
    shown = r.head(25) if how_many == "Top 25" else r

    sites, _ = get_sites()
    closed_ids = get_priorities(selected_month)[0]["site_id"].tolist()
    st_folium(build_prepositioning_map(shown, sites, closed_ids, set(protected), limit_text,
                                       scenario_layers(selected_month)),
              key="prep_map", width=1000, height=550, returned_objects=[])
    legend = " &nbsp; ".join(f'<span style="color:{colour}; font-size:18px;">●</span> {label}'
                             for _, colour, label in INCREASE_CLASSES)
    st.markdown(
        f"<small>Circle size: extra people within {limit_text}. Colour: increase on the facility's usual "
        f"population: {legend} &nbsp; <span style='color:{NO_USUAL_COLOUR}; font-size:18px;'>●</span> usually "
        f"serves nobody. Small dots: <span style='color:#FF3344;'>●</span> closed, "
        f"<span style='color:#1E6FD9;'>●</span> protected. {layer_legend()}</small>", unsafe_allow_html=True)

    others = shown["n_facilities"] - 1
    table = pd.DataFrame({
        "County": shown["adm2_name"],
        "State": shown["State"],
        "Facility Type": shown["rep_facility_type"] + others.map(lambda n: f" (+{n} more in cell)" if n else ""),
        "Usually Serves": shown["usual_population"].round().astype(int),
        f"Extra People (within {limit_text})": shown["added_within_limit"].round().astype(int),
        "Extra People (all)": shown["added"].round().astype(int),
        f"Increase (within {limit_text})": shown["relative_increase_within_limit"].map(format_increase),
        "Extra Travel Time (min)": shown["extra_min"].round().astype(int),
        "From Facilities": shown["from_facilities"].astype(int),
    })
    table.index = pd.RangeIndex(1, len(table) + 1, name="#")
    st.dataframe(table, use_container_width=True)
    st.download_button(
        "Download the full list (CSV)", r.drop(columns=["variant"]).to_csv(index=False).encode("utf-8"),
        file_name=f"prepositioning_{selected_month}_{mode}_{variant}.csv", mime="text/csv")
    st.caption("People in the area a facility usually serves, not patients; facilities count as equal (no "
               "capacity data). Displaced people from flooded homes are not followed. Screening tool for local "
               "verification, not an instruction.")


def choose_focus(sources, mode, selected_month, protected):
    """The "Focus on a closed facility" dropdown; returns a site_id or None."""
    closed = (sources[(sources["mode"] == mode) & (sources["variant"] == "as_is")]
              .sort_values("usual_population", ascending=False))
    labels = {None: "None (show all receiving facilities)"}
    for row in closed.itertuples():
        note = ", protected" if row.site_id in protected else ""
        labels[int(row.site_id)] = (f"{row.adm2_name} · {row.rep_facility_type} "
                                    f"(usually serves {row.usual_population:,.0f}{note})")
    return st.selectbox("Focus on a Closed Facility: Where Do Its People Go?", list(labels), format_func=labels.get,
                        key=f"prep_focus_{selected_month}_{mode}",
                        help="Pick a facility the flood closes to see which facilities its people go to instead.")


def render_focus(focus, receiving, flows, sources, mode, variant, selected_month, limit_text):
    """Map and summary for one closed facility: where its usual people go."""
    src = sources[(sources["mode"] == mode) & (sources["variant"] == variant) & (sources["site_id"] == focus)]
    sites, _ = get_sites()
    layers = scenario_layers(selected_month)
    if src.empty:   # protected in this variant: not closed
        top = variant.split("_")[0].removeprefix("top")
        st.info(f"This facility is one of the protected Top {top}: it stays open, so its people keep going there. "
                f"Switch off \"Assume Top {top} Are Protected\" in the sidebar to see where they would go.")
        return
    src = src.iloc[0]
    f = flows[(flows["mode"] == mode) & (flows["variant"] == variant) & (flows["from_site"] == focus)]
    to = f.groupby("to_site").agg(added=("people", "sum"), added_within_limit=("people_within_limit", "sum"))
    to["extra_min"] = (f["people"] * f["extra_min"]).groupby(f["to_site"]).sum() / to["added"]
    info = receiving.set_index("site_id")[["usual_population", "relative_increase_within_limit", "adm2_name",
                                           "State", "rep_facility_type", "n_facilities"]]
    to = to.join(info).reset_index().rename(columns={"to_site": "site_id"})
    to["from_facilities"] = 1
    to = to[to["added"] >= MIN_PEOPLE_SHOWN].sort_values("added", ascending=False)

    st_folium(build_prepositioning_map(to, sites, [focus], set(), limit_text, layers, focus_id=focus),
              key="prep_focus_map", width=1000, height=550, returned_objects=[])
    st.markdown(f"<small>Large dark-edged dot: the selected facility. Circles: where its people go, sized by "
                f"how many of them arrive within {limit_text}; colour = that facility's total increase. "
                f"{layer_legend()}</small>", unsafe_allow_html=True)

    parts = [f"**{row.added:,.0f}** go to {row.adm2_name} · {row.rep_facility_type} "
             f"({row.added_within_limit:,.0f} within {limit_text}, +{row.extra_min:,.0f} min on average)"
             for row in to.head(5).itertuples()]
    more = f"; {len(to) - 5} more facilities receive the rest" if len(to) > 5 else ""
    moving = "; ".join(parts) + more if parts else "none go to another facility"
    st.markdown(f"Of its **{src['usual_population']:,.0f}** usual people: " + moving
                + f". **{src['lost_all_access']:,.0f}** can't reach any facility, and "
                  f"**{src['living_in_flooded_cells']:,.0f}** live in flooded cells.")
    table = pd.DataFrame({
        "County": to["adm2_name"], "State": to["State"], "Facility Type": to["rep_facility_type"],
        f"People From This Facility (within {limit_text})": to["added_within_limit"].round().astype(int),
        "People From This Facility (all)": to["added"].round().astype(int),
        "Extra Travel Time (min)": to["extra_min"].round().astype(int),
        f"That Facility's Total Increase (within {limit_text})": to["relative_increase_within_limit"].map(format_increase),
    })
    table.index = pd.RangeIndex(1, len(table) + 1, name="#")
    st.dataframe(table, use_container_width=True)


####################### User Interface (Folium + Streamlit) #######################

def main():
    st.set_page_config(page_title="ZOA Healthcare Access Tool", layout="wide")

    cfg = load_config()
    st.sidebar.title("Scenario & Policy Parameters (Monthly, Yearly Flood Selection)")
    scenario_labels = {s["id"]: s["label"] for s in cfg["scenarios"]}
    selected_month = st.sidebar.selectbox(
        "Flood Scenario (Observed Month)", list(scenario_labels), format_func=scenario_labels.get,
        help="Observed flood months picked by data_preparation/select_scenarios.py (the `scenarios` list in "
             "config.yaml). The protection ranking is precomputed for each of them."
    )

    min_days = st.sidebar.slider(
        "Min Days Flooded to Block Access",
        min_value=3, max_value=14, value=cfg["flood"]["min_days_flooded"],
        help="Used for the county map. The protection ranking is precomputed with flood.min_days_flooded "
             f"from config.yaml ({cfg['flood']['min_days_flooded']} days)."
    )

    route_cfg, top_n = cfg["routing"], cfg["priorities"]["top_n"]
    st.sidebar.slider(   # no longer used: Route to Care follows the selected scenario
        ":red[Block Cells Flooded in More Than (% of Years)]",
        min_value=0, max_value=50, step=5, value=round(route_cfg["max_flood_likelihood"] * 100),
        help="NO LONGER USED: Route to Care now follows the selected flood scenario, like the rest of the "
             "dashboard. Kept (in red) only so the team can decide to remove it."
    )

    priority_mode = st.sidebar.radio(
        "Anticipatory Defense Prioritization Criterion",
        ("Population Density (Utilitarian: Most Saved)",
         "Isolation Risk (Egalitarian: Worst-Off First)"),
        help="Switch between protecting high-density urban facilities vs remote lone facilities."
    )

    show_only_flooded = st.sidebar.checkbox("Show Only Inaccessible / Flooded Facilities", value=False)

    st.sidebar.subheader("Route to Care")
    route_mode = st.sidebar.radio("Travel Mode", ("walking", "motorised"),
                                  index=("walking", "motorised").index(route_cfg["default_mode"]),
                                  format_func=str.capitalize)
    protect_top = st.sidebar.checkbox(
        f"Assume Top {top_n} Are Protected", value=True,
        help=f"Treats the Top {top_n} facilities in the table as protected (e.g. sandbagged): they stay open and "
             "routes can end at them. The flood-prone ground around them stays blocked."
    )
    st.session_state["scenario_id"] = selected_month
    facilities_data, county_stats = compute_flood_scenario(selected_month, min_days)

    priorities, priority_summary = get_priorities(selected_month)
    criterion = "worst_off" if "Worst-Off" in priority_mode else "most_saved"
    ranked = priorities.sort_values(f"rank_{criterion}")
    top_sites = ranked[ranked[f"added_{criterion}"] > 0].head(top_n)   # greedy order; skip sites adding nobody
    protected = tuple(top_sites["site_id"].tolist()) if protect_top else ()

    # facility status: closed in this scenario (a flooded cell within facilities.flood_buffer_m)
    sites, _ = get_sites()
    status = sites[["row", "col", "site_id"]].assign(flooded=sites["site_id"].isin(priorities["site_id"]))
    facilities_data = facilities_data.merge(status, on=["row", "col"])
    facilities_data["is_flooded"] = facilities_data.pop("flooded")
    facilities_data["is_protected"] = facilities_data["site_id"].isin(protected)


    county_options = {"All Counties (National View)": None}
    for _, row in county_stats.sort_values("adm2_name").iterrows():
        county_options[f"{row['adm2_name']} ({row['adm1_name']})"] = row["adm2_pcode"]

    selected_label = st.sidebar.selectbox("Filter / Zoom to County", list(county_options.keys()))
    selected_pcode = county_options[selected_label]

    st.title("Anticipating Healthcare Access Loss · South Sudan")
    st.caption("Decision Support Platform for Anticipatory Flood Actions (Zero Hunger Lab & ZOA)")

    display_facilities = facilities_data.copy()
    if show_only_flooded:
        display_facilities = display_facilities[display_facilities["is_flooded"] | display_facilities["is_protected"]]

    total_facilities = len(facilities_data)
    flooded_facilities = int(facilities_data["is_flooded"].sum())
    access_summary, access_county = get_access_loss(selected_month)
    max_time = cfg["travel"]["max_time_min"]

    metric_label = "Worst-Off First" if criterion == "worst_off" else "Most Saved"
    limit_min = cfg["priorities"]["worst_off_min"] if criterion == "worst_off" else cfg["travel"]["max_time_min"]
    limit_text = f"{limit_min / 60:g} hours"
    if criterion == "worst_off":
        metric_help = (f"Ranks closed facilities by how many people would be more than {limit_text} from care "
                       f"without them, and are brought back within {limit_text} if that facility is protected "
                       f"(walking). Favours remote communities with nowhere else to go.")
    else:
        metric_help = (f"Ranks closed facilities by how many people get access back within {limit_text} of care "
                       f"if that facility is protected (walking).")
    metric_help += (" Greedy order: each next facility adds the most people not already brought back by the "
                    "ones above it.")

    walking, motorised = access_summary["modes"]["walking"], access_summary["modes"]["motorised"]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Facilities", f"{total_facilities:,}")
    col2.metric(f"Closed (Flood Within {cfg['facilities']['flood_buffer_m'] / 1000:g} km)", f"{flooded_facilities:,}", delta=f"{flooded_facilities/total_facilities*100:.1f}%", delta_color="inverse")
    col3.metric("Living in Flooded Areas", f"{access_summary['living_in_flooded_cells']:,}",
                help=f"People whose home cell is flooded in this month (at least "
                     f"{access_summary['settings']['min_days_flooded']} days, precomputed). They are counted "
                     f"separately from the people cut off from care.")
    col4.metric(f"Cut Off From Care ({max_time / 60:g} h Walk)", f"{walking[f'cut_off_{max_time}']:,}",
                help=f"People on dry land who were within {max_time / 60:g} h of a facility (walking) before the "
                     f"flood and are now beyond it, or can't reach any. {walking['no_facility_reachable']:,} of "
                     f"them can't reach any facility. With motorised transport: "
                     f"{motorised[f'cut_off_{max_time}']:,}.")
    col5.metric("Active Decision Metric", metric_label, help=metric_help)

    section = st.segmented_control(
        "Section", ["🗺️ Overview Map", "🛡️ Protection (Top 5)", "📦 Pre-positioning", "🧭 Route to Care",
                    "ℹ️ Limitations"],
        default="🗺️ Overview Map", key="section", label_visibility="collapsed") or "🗺️ Overview Map"
    st.divider()

    if section == "🗺️ Overview Map":
        render_overview_section(cfg, county_stats, access_county, display_facilities, top_n, selected_pcode)
    elif section == "🛡️ Protection (Top 5)":
        render_protection_section(top_n, metric_label, facilities_data, top_sites, criterion, limit_text,
                                  priority_summary)
    elif section == "📦 Pre-positioning":
        render_prepositioning_section(cfg, selected_month, criterion, protect_top, protected, top_n)
    elif section == "🧭 Route to Care":
        render_route_section(selected_month, scenario_labels[selected_month], route_mode, protected,
                             facilities_data, top_n)
    else:
        render_limitations()


def render_overview_section(cfg, county_stats, access_county, display_facilities, top_n, selected_pcode):
    """County map coloured by the chosen measure, or a 1 km grid layer, with the facilities on top."""
    options = county_measure_options(cfg)
    layers = get_dashboard_layers()
    grid_options = {legend["title"]: name for name, legend in layers["legends"].items()}
    show_as, middle, right = st.columns([1, 3, 1])
    view = show_as.radio("Show As", ("Counties", "Grid"), horizontal=True, key="overview_view",
                         help="Grid: the access loss on a 1 km grid, where it happens within counties.")
    if view == "Counties":
        choice = middle.selectbox("Colour Counties By", list(options), key="county_measure")
    else:
        choice = middle.selectbox("Grid Layer", list(grid_options), key="grid_layer")
    mode = right.radio("Travel Mode", ("walking", "motorised"), horizontal=True, key="county_mode",
                       format_func=str.capitalize, help="For the access-loss measures.")
    counties = county_measures(county_stats, access_county, mode, cfg)
    max_time = cfg["travel"]["max_time_min"]

    if selected_pcode:
        selected_row = counties[counties["adm2_pcode"] == selected_pcode].iloc[0]
        st.subheader(f"📍 County Focus: {selected_row['adm2_name']} ({selected_row['adm1_name']})")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("County Population", f"{int(selected_row['total_pop']):,}")
        c2.metric("Flooded Population", f"{int(selected_row['flooded_pop']):,}")
        c3.metric("Population Impact Rate", f"{selected_row['pct_pop_affected']:.1f}%")
        c4.metric(f"Cut Off From Care ({max_time / 60:g} h)", f"{selected_row[f'cut_off_{max_time}']:,.0f}",
                  help=f"{mode.capitalize()}. {selected_row['no_facility_reachable']:,.0f} can't reach any facility; "
                       f"{int(selected_row['closed_facilities'])} facilities closed in this county.")
    else:
        st.subheader("🇺🇳 National Overview (All 79 Counties)")

    grid = None
    if view == "Grid":
        layer = grid_options[choice]
        image = layers["images"][f"{st.session_state['scenario_id']}|{mode}|{layer}"]
        grid = (image["url"], image["bounds"], choice)
    scenario_map = build_advanced_map(counties, display_facilities, top_n, selected_pcode=selected_pcode,
                                      measure=None if grid else options[choice], grid=grid)
    st_folium(scenario_map, key="scenario_map", width=1000, height=600, returned_objects=[])
    if grid:
        st.markdown(f"<small><b>{choice}</b> (1 km pixels; read per area, not per pixel): "
                    f"{legend_html(layers['legends'][grid_options[choice]]['classes'])}</small>",
                    unsafe_allow_html=True)


def render_protection_section(top_n, metric_label, facilities_data, top_sites, criterion, limit_text,
                              priority_summary):
    """The Top N table and who it does not cover."""
    st.write(f"### Top {top_n} Recommended Facilities for Temporary Defences ({metric_label})")
    fid_to_county = dict(zip(facilities_data["FID"], facilities_data["adm2_name"]))
    fid_to_admin1 = dict(zip(facilities_data["FID"], facilities_data["Admin1"]))

    if not top_sites.empty:
        others = top_sites["n_facilities"] - 1
        county_col = top_sites["rep_facility_id"].map(fid_to_county)
        table = pd.DataFrame({
            "County": county_col,
            "Facility Type": top_sites["rep_facility_type"]
                             + others.map(lambda n: f" (+{n} more in cell)" if n else ""),
            "State": top_sites["rep_facility_id"].map(fid_to_admin1),
        })
        table["County"] = top_sites["adm2_name"]
        table[f"People Brought Back (within {limit_text})"] = top_sites[f"added_{criterion}"].round().astype(int)
        table["If Protected Alone"] = top_sites[f"gain_{criterion}"].round().astype(int)
        table.index = pd.RangeIndex(1, len(table) + 1, name="Rank")

        st.dataframe(table, use_container_width=True)
    else:
        st.info("No closed facility brings anyone back within the limit in this scenario.")

    ranking = priority_summary["rankings"][criterion]
    covered = float(top_sites[f"cumulative_{criterion}"].max()) if len(top_sites) else 0.0
    losing = ranking["people_losing_access"]
    losing_text = (f"are pushed beyond {limit_text} from care" if criterion == "worst_off"
                   else f"lose access within {limit_text}")
    st.caption(
        f"In this scenario {losing:,} people {losing_text}. Protecting these {len(top_sites)} facilities brings back "
        f"{covered:,.0f}; **{losing - covered:,.0f} people are not covered**. "
        f"{priority_summary['people_living_in_flooded_cells']:,} people live in flooded cells and are not counted: "
        f"protecting a facility doesn't make their homes passable. Screening tool for local verification, "
        f"not an instruction.")


def render_limitations():
    with st.expander("Methodological Assumptions & SLE Limitations", expanded=True):
        st.markdown("""
- **Friction vs. Reality:** Friction surfaces represent mathematical resistance over land, not confirmed physical roads.
- **Operational Integrity:** Facility status is derived from spatial coordinates; unflooded facilities may still lack staff, medicines, or operational power.
- **Equity Alert (Protection Bias):** Utilitarian optimization biases sandbagging and flood defenses toward densely populated towns, systematically under-ranking remote communities.
- **The 'Average' Traveller:** Routing assumes an average, able-bodied traveller. A standard two-hour walk is significantly longer for a disabled person, elderly patient, or someone in labour.
- **Crowding & Capacity Limitations:** Rerouting catchments to the nearest open facility assumes infinite capacity and perfect compliance, which could overwhelm receiving clinics in reality.
- **No Data North of 10°N:** Approximately 6% of the country (e.g., Renk, Manyo) lacks satellite flood data. These areas are treated as passable in routing grids but are not confirmed to be dry.
- **River Crossability:** The permanent water mask blocks most, but not all, major rivers. Seasonal khors (e.g., Nyanding, Nyirol) are typically unblocked and modeled as passable terrain.
- **Abyei Region:** Included in the population modeling metrics but has no recorded health facilities within this dataset.
- **Cloud Cover Underestimation:** Satellite flood counts function strictly as a lower bound, as dense cloud cover frequently obscures flooding detection during the wet season.
        """)
if __name__ == "__main__":
    main()