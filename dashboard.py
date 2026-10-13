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

from data_preparation.grid import ROOT, data_path, in_grid, load_config, lonlat_to_rowcol, make_grid
from data_preparation.loading import (
    load_static,
    load_permanent_water,
    load_flood,
    load_flood_likelihood,
    load_facilities
)
from data_preparation.build_baseline import build_sites
from data_preparation.build_flood_monthly import NO_DATA
from data_preparation.prioritization import flooded_sites, rank_sites, site_impacts
from data_preparation.routing import blocked_mask, open_sites, rc_to_lonlat, route
from data_preparation.travel import travel_time

MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August",
               "September", "October", "November", "December"]


@st.cache_resource
def get_base_pipeline():
    cfg = load_config()
    grid = make_grid(cfg)
    static = load_static(cfg)
    water = load_permanent_water(cfg)
    facilities = load_facilities(cfg)
    return cfg, grid, static, water, facilities


@st.cache_data
def compute_flood_scenario(month_str, min_days):
    cfg, grid, static, water, facilities = get_base_pipeline()

    #Flood Raster
    months, flood = load_flood(month_str)
    days = flood[0]
    flooded = (days >= min_days) & (days != NO_DATA)

    #Facilities (their flood status comes from get_priorities)
    fac_df = facilities.copy()

    # Population aggregation
    county_ids = static["county_id"].ravel()
    pop = static["population"].ravel()
    n_counties = int(county_ids.max() + 1)

    total_pop = np.bincount(county_ids, weights=pop, minlength=n_counties)
    flooded_pop = np.bincount(county_ids, weights=(pop * flooded.ravel()), minlength=n_counties)

    counties_meta = pd.read_csv("data/interim/counties.csv")

    if ("adm2_pcode" not in counties_meta.columns and "county_pcode" in counties_meta.columns):
        counties_meta["adm2_pcode"] = counties_meta["county_pcode"]

    counties_meta["total_pop"] = counties_meta["county_id"].map(lambda cid: total_pop[cid] if cid < n_counties else 0)
    counties_meta["flooded_pop"] = counties_meta["county_id"].map(lambda cid: flooded_pop[cid] if cid < n_counties else 0)
    counties_meta["pct_pop_affected"] = (counties_meta["flooded_pop"] / np.maximum(counties_meta["total_pop"], 1)) * 100

    return fac_df, counties_meta


@st.cache_resource
def get_sites():
    cfg, grid, static, water, facilities = get_base_pipeline()
    sites = build_sites(facilities, grid)   # facilities sharing a 250 m cell are one site
    return sites, sites[["row", "col"]].to_numpy()


@st.cache_resource(max_entries=4)
def get_route_inputs(calendar_month, threshold):
    """Blocked cells and the sites that can still provide care, for one calendar month and threshold."""
    cfg, grid, static, water, facilities = get_base_pipeline()
    sites, sites_rc = get_sites()
    likelihood = load_flood_likelihood(calendar_month, cfg)
    blocked = blocked_mask(water, likelihood, threshold)
    usable = open_sites(sites_rc, likelihood, threshold)
    return blocked, usable, np.isnan(likelihood)


@st.cache_resource(show_spinner="Computing walking times to care without floods...")
def get_baseline_walking():
    """Walking time to the nearest site without floods, the same solve as build_baseline."""
    cfg, grid, static, water, facilities = get_base_pipeline()
    sites, sites_rc = get_sites()
    return travel_time(static["friction_walking"], sites_rc, blocked=water, cell_size_m=grid.cell_size)


@st.cache_resource(max_entries=4, show_spinner="Finding and ranking flood-prone facilities (up to 30 s)...")
def get_priorities(calendar_month, threshold):
    """Sites with their flood status and what losing each one costs its catchment (walking).
    See data_preparation/README_prioritization.md."""
    cfg, grid, static, water, facilities = get_base_pipeline()
    sites, sites_rc = get_sites()
    likelihood = load_flood_likelihood(calendar_month, cfg)
    flooded = flooded_sites(sites_rc, likelihood, threshold)
    base = get_baseline_walking()
    flood = travel_time(static["friction_walking"], sites_rc[~flooded],
                        blocked=blocked_mask(water, likelihood, threshold), cell_size_m=grid.cell_size)
    impacts = site_impacts(static["population"], base.nearest, base.minutes, flood.minutes, len(sites),
                           cfg["travel"]["max_time_min"])
    return sites.assign(flooded=flooded).join(impacts)


@st.cache_resource(max_entries=16)   # each route takes a few seconds; other widgets rerun the script
def compute_route(lat, lon, calendar_month, threshold, mode, protected=()):
    """Flood-aware route from a clicked point, plus the same route ignoring floods for comparison.
    Sites in `protected` (site ids) count as open even when their cell is flood-prone."""
    cfg, grid, static, water, facilities = get_base_pipeline()
    sites, sites_rc = get_sites()
    row, col = lonlat_to_rowcol(grid, np.array([lon]), np.array([lat]))
    if not in_grid(grid, row, col)[0] or static["county_id"][row[0], col[0]] == 0:
        return None
    start = (int(row[0]), int(col[0]))
    friction = static[f"friction_{mode}"]
    blocked, usable, no_data = get_route_inputs(calendar_month, threshold)
    if protected:
        usable = usable.copy()   # the cached array is shared
        usable[list(protected)] = True

    flood_aware = route(friction, start, sites_rc[usable], blocked, grid.cell_size, no_data)
    site = sites.iloc[np.flatnonzero(usable)[flood_aware.site]] if flood_aware.site >= 0 else None
    no_flood = route(friction, start, sites_rc, water, grid.cell_size)
    return dict(start=start, flood_aware=flood_aware, site=site, no_flood=no_flood, blocked=blocked)


def mask_to_png(mask, transform, crs, resolution_m, resampling):
    """A boolean mask as a PNG data URL plus lat/lon bounds, reprojected to Web Mercator so Leaflet
    places it exactly (an image is stretched linearly in Web Mercator, not in our Albers grid)."""
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
    """Every blocked cell in the country at 500 m (a 500 m pixel is shown if any 250 m cell in it is
    blocked). Cached, so switching it on again for the same month and threshold is instant."""
    cfg, grid, static, water, facilities = get_base_pipeline()
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


@st.cache_data
def boundaries_geojson(level):
    """County (2) or state (1) outlines as GeoJSON, simplified to about 110 m so the map stays light."""
    cfg = load_config()
    path = data_path(cfg, "admin2") if level == 2 else ROOT / cfg["baseline"]["admin1"]
    cols = ["adm2_name", "adm1_name"] if level == 2 else ["adm1_name"]
    units = gpd.read_file(path)[cols + ["geometry"]]
    return units.assign(geometry=units.simplify(0.001, preserve_topology=True)).to_json()


def add_boundaries(m):
    """County and state outlines, drawn above the flood layers and below the routes."""
    folium.GeoJson(
        boundaries_geojson(2), name="County boundaries",
        style_function=lambda f: {"fill": False, "color": "#555555", "weight": 1, "opacity": 0.8},
        tooltip=folium.GeoJsonTooltip(fields=["adm2_name", "adm1_name"], aliases=["County:", "State:"]),
    ).add_to(m)
    folium.GeoJson(
        boundaries_geojson(1), name="State boundaries",
        style_function=lambda f: {"fill": False, "color": "#222222", "weight": 2.5, "opacity": 0.9},
    ).add_to(m)


def build_route_map(grid, result, clicked, national=None):
    m = folium.Map(location=[7.5, 30.5], zoom_start=6.2, tiles="OpenStreetMap", control_scale=True)
    if national is not None:
        url, bounds = national
        folium.raster_layers.ImageOverlay(image=url, bounds=bounds,
                                          name="All flood-prone cells and permanent water (500 m)").add_to(m)
    if result is None:
        add_boundaries(m)
        if clicked:
            folium.Marker(clicked, icon=folium.Icon(color="gray", icon="question-sign")).add_to(m)
        folium.LayerControl(position="topright", collapsed=False).add_to(m)
        return m

    paths = [result["flood_aware"].path_rc, result["no_flood"].path_rc]
    if any(len(p) for p in paths):
        blocked_overlay(grid, result["blocked"], paths).add_to(m)
    add_boundaries(m)

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
                      popup=f"{site['rep_facility_name']} ({site['rep_facility_type']})").add_to(m)

    folium.Marker(clicked, icon=folium.Icon(color="red", icon="user"), tooltip="Start").add_to(m)
    bounds = np.vstack([p for p in paths if len(p)] + [np.array([result["start"]])])
    lon, lat = rc_to_lonlat(grid, bounds[:, 0], bounds[:, 1])
    m.fit_bounds([[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]], padding=(30, 30))
    folium.LayerControl(position="topright", collapsed=False).add_to(m)
    return m


def format_minutes(minutes):
    return f"{minutes:.0f} min" if minutes < 60 else f"{int(minutes // 60)} h {minutes % 60:02.0f} min"


def render_route_section(calendar_month, threshold, mode, protected):
    cfg, grid, static, water, facilities = get_base_pipeline()
    st.subheader("🧭 Route to Care")
    month_name = MONTH_NAMES[calendar_month - 1]
    flood_rule = (f"cells flooded in any {month_name} since 2003" if threshold == 0 else
                  f"cells flooded in more than {threshold:.0%} of {month_name}s (2003-2025)")
    caption = (f"Click anywhere in South Sudan. The route avoids permanent water and {flood_rule}, and goes to "
               f"the nearest facility whose own cell is not flood-prone. The dashed grey line is the route "
               f"without floods.")
    if protected:
        caption += (f" The Top {len(protected)} facilities above are assumed protected, so they also count as "
                    f"open; the flood-prone ground around them stays blocked.")
    st.caption(caption)

    show_all = st.toggle("Show all flood-prone cells (nationwide)", key="route_show_all",
                         help="Without this, blocked cells are only drawn within 10 km of the route. Either way "
                              "the route avoids every flood-prone cell in the country.")
    national = national_overlay(calendar_month, threshold) if show_all else None

    clicked = st.session_state.get("route_start")
    result = compute_route(*clicked, calendar_month, threshold, mode, protected) if clicked else None
    if clicked and result is None:
        st.warning("That point is outside South Sudan. Click inside the country.")

    # only a click reruns the script; zooming and panning stay in the browser
    out = st_folium(build_route_map(grid, result, clicked, national), key="route_map", width=1000, height=550,
                    returned_objects=["last_clicked"])
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
                initial_location = [props.get("center_lat", 7.5), props.get("center_lon", 30.5),]
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
        protected_group.add_to(m)   # outside the cluster, so they stay visible at every zoom

    for _, fac in facilities_data.iterrows():
        is_flooded = fac["is_flooded"]
        color = "#FF3344" if is_flooded else "#00E676"
        heading_color = "#d9534f" if is_flooded else "#2e7d32"
        status_label = "SUBMERGED / CUT OFF" if is_flooded else "OPERATIONAL"
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

    st.sidebar.title("Scenario & Policy Parameters (Monthly, Yearly Flood Selection)")
    selected_month = st.sidebar.selectbox(
        "Historical / Predicted Scenario Month",
        ["2020-07", "2020-09", "2020-11", "2021-09", "2022-10"],
        index=2
    )

    min_days = st.sidebar.slider(
        "Min Days Flooded to Block Access",
        min_value=3, max_value=14, value=3
    )

    cfg = load_config()
    route_cfg, top_n = cfg["routing"], cfg["priorities"]["top_n"]
    route_threshold = st.sidebar.slider(
        "Block Cells Flooded in More Than (% of Years)",
        min_value=0, max_value=50, step=5, value=round(route_cfg["max_flood_likelihood"] * 100),
        help="Blocks cells whose flood likelihood for this calendar month is above this value. The likelihood is "
             "the share of years (2003-2025) a cell was flooded on 3+ days in that month, so it moves in steps of "
             "one year: about 4.3% with 23 years (4.5% for April, May and November, which have 22). "
             "0% = flooded in at least 1 year · 5% = 2 years · 10% = 3 · 15% = 4 · 20% = 5 · 25% = 6. "
             "Values in between give the same result as the step below (e.g. 2% = 0%)."
    ) / 100

    priority_mode = st.sidebar.radio(
        "Anticipatory Defense Prioritization Criterion",
        ("Population Density (Utilitarian: Most Saved)", 
         "Isolation Risk (Egalitarian: Worst-Off First)"),
         help="Explicitly switch between protecting high-density urban facilities vs remote lone facilities."
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
    priorities = get_priorities(calendar_month, route_threshold)
    criterion = "worst_off" if "Worst-Off" in priority_mode else "most_saved"
    top_sites = rank_sites(priorities, priorities["flooded"], criterion, top_n)
    protected = tuple(top_sites["site_id"].tolist()) if protect_top else ()

    # facility status from its site's flood likelihood, the same rule as Route to Care
    facilities_data = facilities_data.merge(priorities[["row", "col", "site_id", "flooded"]], on=["row", "col"])
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
        display_facilities = display_facilities[display_facilities["is_flooded"]]

    total_facilities = len(facilities_data)
    flooded_facilities = int(facilities_data["is_flooded"].sum())
    total_affected_people = int(county_stats["flooded_pop"].sum())

    metric_label = "Worst-Off First" if criterion == "worst_off" else "Most Saved"
    if criterion == "worst_off":
        metric_help = (f"Ranks flood-prone facilities by the share of their catchment population left more than "
                       f"{cfg['travel']['max_time_min']} min from care, or unable to reach any, once flood-prone "
                       f"cells and facilities are lost (walking). Ties go to the larger extra travel time.")
    else:
        metric_help = ("Ranks flood-prone facilities by catchment population: the people for whom it is the "
                       "nearest facility when there are no floods (walking).")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Facilities", f"{total_facilities:,}")
    col2.metric("Submerged / Inaccessible", f"{flooded_facilities:,}", delta=f"{flooded_facilities/total_facilities*100:.1f}%", delta_color="inverse")
    col3.metric("Flooded In-Situ Population", f"{total_affected_people:,}")
    
    col4.metric("Active Decision Metric", metric_label,  help=metric_help)

    st.divider()

    #

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
    # returned_objects=[]: zooming or panning doesn't rerun the script (which redrew the map grey)
    st_folium(scenario_map, key="scenario_map", width=1000, height=600, returned_objects=[])

    st.divider()

    ####################### Additional Info #######################

    st.write(
        f"### Top {top_n} Recommended Facilities for Temporary Defences"
        f" ({metric_label})")
    if not top_sites.empty:
        others = top_sites["n_facilities"] - 1
        table = pd.DataFrame({
            "Facility Name": top_sites["rep_facility_name"]
                             + others.map(lambda n: f" (+{n} more in the same cell)" if n else ""),
            "Type": top_sites["rep_facility_type"],
            "State": top_sites["rep_facility_id"].map(facilities_data.set_index("FID")["Admin1"]),
        })
        if criterion == "worst_off":
            table["Cut Off (%)"] = (100 * top_sites["share_cut_off"]).round(1)
            table["Extra Time (min)"] = top_sites["extra_min"].round()   # the tie-break; empty = nobody reached
        else:
            table["Catchment Population"] = top_sites["catchment_pop"].round().astype(int)
        table.index = pd.RangeIndex(1, len(table) + 1, name="Rank")

        st.dataframe(table, use_container_width=True)
    else:
        st.info("No facilities submerged under this specific scenario threshold.")

    st.divider()

    render_route_section(calendar_month, route_threshold, route_mode, protected)

    st.divider()

    with st.expander("Methodological Assumptions & SLE Limitations"):
        st.markdown("""
        - **Friction vs. Reality:** Friction surfaces represent mathematical resistance over land, not confirmed physical roads.
        - **Operational Integrity:** Facility status is derived from spatial coordinates; unflooded facilities may still lack staff, medicines, or operational power.
        - **Equity Alert:** Utilitarian optimization biases sandbagging and flood defenses toward densely populated towns, systematically under-ranking remote communities.
        """)

if __name__ == "__main__":
    main()