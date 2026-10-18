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
from affine import Affine
from PIL import Image
from pyproj import Transformer
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
from folium.template import Template

from data_preparation.grid import in_grid, load_config, lonlat_to_rowcol, make_grid
from data_preparation.loading import (
    load_static,
    load_permanent_water,
    load_flood,
    load_flood_likelihood,
    load_facilities,
    load_priorities,
)
from data_preparation.build_baseline import build_sites
from data_preparation.build_flood_monthly import NO_DATA
from data_preparation.routing import blocked_mask, open_sites, rc_to_lonlat, route

MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August",
               "September", "October", "November", "December"]
NATIONAL_LAYER = "All flood-prone cells and permanent water (500 m)"

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
def get_route_inputs(calendar_month, threshold):
    """Blocked cells and the sites that can still provide care, for one calendar month and threshold."""
    cfg, grid, static, water, facilities, _ = get_base_pipeline()
    sites, sites_rc = get_sites()
    likelihood = load_flood_likelihood(calendar_month, cfg)
    blocked = blocked_mask(water, likelihood, threshold)
    usable = open_sites(sites_rc, likelihood, threshold)
    return blocked, usable, np.isnan(likelihood)


@st.cache_data
def get_priorities(scenario_id):
    """Precomputed protection priorities for one scenario, from data_preparation.build_priorities:
    one row per closed site with both rankings, and the totals. See README_prioritization.md."""
    return load_priorities(scenario_id, load_config())


@st.cache_resource(max_entries=16)
def compute_route(lat, lon, calendar_month, threshold, mode, protected=()):
    """Flood-aware route from a clicked point, plus the same route ignoring floods for comparison."""
    cfg, grid, static, water, facilities, _ = get_base_pipeline()
    sites, sites_rc = get_sites()
    row, col = lonlat_to_rowcol(grid, np.array([lon]), np.array([lat]))
    if not in_grid(grid, row, col)[0] or static["county_id"][row[0], col[0]] == 0:
        return None
    start = (int(row[0]), int(col[0]))
    friction = static[f"friction_{mode}"]
    blocked, usable, no_data = get_route_inputs(calendar_month, threshold)
    if protected:
        usable = usable.copy()
        usable[list(protected)] = True

    flood_aware = route(friction, start, sites_rc[usable], blocked, grid.cell_size, no_data)
    site = sites.iloc[np.flatnonzero(usable)[flood_aware.site]] if flood_aware.site >= 0 else None
    no_flood = route(friction, start, sites_rc, water, grid.cell_size)
    return dict(start=start, flood_aware=flood_aware, site=site, no_flood=no_flood, blocked=blocked)


def mask_to_png(mask, transform, crs, resolution_m, resampling):
    """A boolean mask as a PNG data URL plus lat/lon bounds, reprojected to Web Mercator."""
    h, w = mask.shape
    dst_transform, dst_w, dst_h = calculate_default_transform(
        crs, "EPSG:3857", w, h, *array_bounds(h, w, transform), resolution=resolution_m)
    out = np.zeros((dst_h, dst_w), dtype=np.uint8)
    reproject(mask.astype(np.uint8), out, src_transform=transform, src_crs=crs,
              dst_transform=dst_transform, dst_crs="EPSG:3857", resampling=resampling)
    rgba = np.zeros(out.shape + (4,), dtype=np.uint8)
    rgba[out > 0] = (33, 102, 172, 150)
    png = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(png, format="PNG", optimize=True)
    west, south, east, north = array_bounds(dst_h, dst_w, dst_transform)
    (lon_w, lon_e), (lat_s, lat_n) = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform(
        [west, east], [south, north])
    return ("data:image/png;base64," + base64.b64encode(png.getvalue()).decode(),
            [[lat_s, lon_w], [lat_n, lon_e]])


@st.cache_data(max_entries=8, show_spinner="Drawing all flood-prone cells...")
def national_overlay(calendar_month, threshold):
    """Every blocked cell in the country at 500 m."""
    cfg, grid, static, water, facilities, _ = get_base_pipeline()
    blocked, _, _ = get_route_inputs(calendar_month, threshold)
    return mask_to_png(blocked, grid.transform, grid.crs, 500, Resampling.max)


def blocked_overlay(grid, blocked, path_rcs, margin=40):
    """Blocked cells at 250 m in a window around the routes (40 cells = 10 km on each side)."""
    cells = np.vstack([p for p in path_rcs if len(p)])
    r0, c0 = np.maximum(cells.min(axis=0) - margin, 0)
    r1, c1 = np.minimum(cells.max(axis=0) + margin + 1, grid.shape)
    window_transform = grid.transform * Affine.translation(int(c0), int(r0))
    url, bounds = mask_to_png(blocked[r0:r1, c0:c1], window_transform, grid.crs, grid.cell_size,
                              Resampling.nearest)
    return folium.raster_layers.ImageOverlay(image=url, bounds=bounds,
                                             name="Blocked near the route (250 m)")


def add_facilities(m, facilities, top_n):
    """Every facility, drawn above the flood layers and below the routes."""
    protected, flooded = facilities["is_protected"], facilities["is_flooded"]
    points = gpd.GeoDataFrame(
        {"name": facilities["Facility_n"], "type": facilities["Facility_t"],
         "status": np.select([protected, flooded], [f"PROTECTED (TOP {top_n})", "CLOSED (FLOOD NEARBY)"], "OPERATIONAL"),
         "color": np.select([protected, flooded], ["#FF9800", "#FF3344"], "#00E676")},
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
    RememberLayers(control, "route_map_layers", skip=[NATIONAL_LAYER]).add_to(m)


def build_route_map(grid, result, clicked, facilities, top_n, national=None):
    m = folium.Map(location=[7.5, 30.5], zoom_start=6.2, tiles="OpenStreetMap", control_scale=True)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery',
        name='Satellite Imagery'
    ).add_to(m)
    if national is not None:
        url, bounds = national
        folium.raster_layers.ImageOverlay(image=url, bounds=bounds, name=NATIONAL_LAYER).add_to(m)
    if result is None:
        add_facilities(m, facilities, top_n)
        if clicked:
            folium.Marker(clicked, icon=folium.Icon(color="gray", icon="question-sign")).add_to(m)
        add_layer_menu(m)
        return m

    paths = [result["flood_aware"].path_rc, result["no_flood"].path_rc]
    if any(len(p) for p in paths):
        blocked_overlay(grid, result["blocked"], paths).add_to(m)
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


def render_route_section(calendar_month, threshold, mode, protected, facilities_data, top_n):
    cfg, grid, static, water, facilities, _ = get_base_pipeline()
    st.subheader("🧭 Route to Care")
    month_name = MONTH_NAMES[calendar_month - 1]
    flood_rule = (f"cells flooded in any {month_name} since 2003" if threshold == 0 else
                  f"cells flooded in more than {threshold:.0%} of {month_name}s (2003-2025)")
    caption = (f"Click anywhere in South Sudan. The route avoids permanent water and {flood_rule}, and goes to "
               f"the nearest facility whose own cell is not flood-prone. The dashed grey line is the route "
               f"without floods. Facilities: red = flood-prone, green = open, blue = protected.")
    if protected:
        caption += (f" The Top {len(protected)} facilities above are assumed protected, so they also count as "
                    f"open; the flood-prone ground around them stays blocked.")
    st.caption(caption)

    show_all = st.toggle("Show all flood-prone cells (nationwide)", key="route_show_all",
                         help="Without this, blocked cells are only drawn within 10 km of the route. Either way "
                              "the route avoids every flood-prone cell in the country.")
    national = national_overlay(calendar_month, threshold) if show_all else None

    clicked = st.session_state.get("route_start")
    result = compute_route(clicked[0], clicked[1], calendar_month, threshold, mode, protected) if clicked else None
    if clicked and result is None:
        st.warning("That point is outside South Sudan. Click inside the country.")
    out = st_folium(build_route_map(grid, result, clicked, facilities_data, top_n, national), key="route_map",
                    width=1000, height=550, returned_objects=["last_clicked"])
    last = (out or {}).get("last_clicked")
    if last and (last["lat"], last["lng"]) != clicked:
        st.session_state["route_start"] = (last["lat"], last["lng"])
        st.rerun()

    if result is None:
        return
    flood_aware, no_flood, site = result["flood_aware"], result["no_flood"], result["site"]
    if flood_aware.site < 0:
        st.error("No facility that can provide care is reachable without crossing flood-prone cells. "
                 "Try a higher flood threshold.")
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

def build_advanced_map(counties_meta, facilities_data, top_n, selected_pcode=None):
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

    max_val = max(float(counties_meta["pct_pop_affected"].max()), 15.0)
    bin_edges = [0.0, 1.0, 3.0, 7.0, 15.0, round(max_val + 0.5, 1)]
    
    choropleth = folium.Choropleth(
        geo_data="raw_data/Administrative boundaries/ssd_admin2.geojson",
        name="County Vulnerability Risk",
        data=counties_meta,
        columns=["adm2_pcode", "pct_pop_affected"],
        key_on="feature.properties.adm2_pcode",
        bins=bin_edges,
        fill_color="RdPu",
        fill_opacity=0.6,
        line_color="#333333",
        line_weight=0.7,
        line_opacity=0.8,
        legend_name="% Population Flooded per County",
    ).add_to(m)

    folium.GeoJsonTooltip(
        fields=["adm2_name", "adm1_name"],
        aliases=["County:", "State:"],
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
    route_threshold = st.sidebar.slider(
        "Block Cells Flooded in More Than (% of Years)",
        min_value=0, max_value=50, step=5, value=round(route_cfg["max_flood_likelihood"] * 100),
        help="Blocks cells whose flood likelihood for this calendar month is above this value (2003-2025)."
    ) / 100

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
    facilities_data, county_stats = compute_flood_scenario(selected_month, min_days)

    calendar_month = pd.Period(selected_month, freq="M").month
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

    fid_to_admin1 = dict(zip(facilities_data["FID"], facilities_data["Admin1"]))

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
    total_affected_people = int(county_stats["flooded_pop"].sum())

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

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Facilities", f"{total_facilities:,}")
    col2.metric(f"Closed (Flood Within {cfg['facilities']['flood_buffer_m'] / 1000:g} km)", f"{flooded_facilities:,}", delta=f"{flooded_facilities/total_facilities*100:.1f}%", delta_color="inverse")
    col3.metric("Flooded In-Situ Population", f"{total_affected_people:,}")
    col4.metric("Active Decision Metric", metric_label, help=metric_help)

    st.divider()

    if selected_pcode:
        selected_row = county_stats[county_stats["adm2_pcode"] == selected_pcode].iloc[0]
        st.subheader(f"📍 County Focus: {selected_row['adm2_name']} ({selected_row['adm1_name']})")
    
        c1, c2, c3 = st.columns(3)
        c1.metric("County Population", f"{int(selected_row['total_pop']):,}")
        c2.metric("Flooded Population", f"{int(selected_row['flooded_pop']):,}")
        c3.metric("Population Impact Rate", f"{selected_row['pct_pop_affected']:.1f}%")
    else:
        st.subheader("🇺🇳 National Overview (All 79 Counties)")

    scenario_map = build_advanced_map(county_stats, display_facilities, top_n, selected_pcode=selected_pcode)
    st_folium(scenario_map, key="scenario_map", width=1000, height=600, returned_objects=[])

    st.divider()

    ####################### Top-N Prioritization Table #######################

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

    st.divider()

    render_route_section(calendar_month, route_threshold, route_mode, protected, facilities_data, top_n)

    st.divider()

    with st.expander("Methodological Assumptions & SLE Limitations"):
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