# Use: PYTHONPATH=. streamlit run dashboard.py

####################### Pre-Aggregating and Caching the Heavy Data #######################
import json
import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster

from data_preparation.grid import load_config, make_grid
from data_preparation.loading import (
    load_static, 
    load_permanent_water, 
    load_flood, 
    load_facilities
)
from data_preparation.build_flood_monthly import NO_DATA


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

    #Facilities
    fac_df = facilities.copy()
    rows = fac_df["row"].values
    cols = fac_df["col"].values
    fac_df["is_flooded"] = flooded[rows, cols]
    fac_df["cell_pop"] = static["population"][rows, cols]

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
    
    risk_dict = dict(zip(counties_meta["county_id"], counties_meta["pct_pop_affected"]))
    if "county_id" in fac_df.columns:
        fac_df["county_pct_affected"] = fac_df["county_id"].map(risk_dict).fillna(0)
    else:
        fac_df["county_pct_affected"] = 0

    return fac_df, counties_meta

####################### Map Generator Function #######################

def build_advanced_map(counties_meta, facilities_data, selected_pcode=None):

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

    
    for _, fac in facilities_data.iterrows():
        is_flooded = fac["is_flooded"]
        color = "#FF3344" if is_flooded else "#00E676"
        status_label = "SUBMERGED / CUT OFF" if is_flooded else "OPERATIONAL"

        popup_html = f"""
        <div style="font-family: Arial; font-size: 12px; width: 180px;">
            <h4 style="margin: 0 0 5px 0; color: {'#d9534f' if is_flooded else '#2e7d32'};">
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
        ).add_to(facility_cluster)

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

    priority_mode = st.sidebar.radio(
        "Anticipatory Defense Prioritization Criterion",
        ("Population Density (Utilitarian: Most Saved)", 
         "Isolation Risk (Egalitarian: Worst-Off First)"),
         help="Explicitly switch between protecting high-density urban facilities vs remote lone facilities."
    )

    show_only_flooded = st.sidebar.checkbox("Show Only Inaccessible / Flooded Facilities", value=False)

    facilities_data, county_stats = compute_flood_scenario(selected_month, min_days)

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

    metric_label = "Worst-Off First" if "Worst-Off" in priority_mode else "Most Saved"
    submerged_facs = facilities_data[facilities_data["is_flooded"]].copy()

    if "county_pct_affected" not in submerged_facs.columns:
        submerged_facs["county_pct_affected"] = 0

    if "cell_pop" not in submerged_facs.columns:
        submerged_facs["cell_pop"] = 0

    if "Worst-Off" in priority_mode:
        ranked_facilities = submerged_facs.sort_values(by="county_pct_affected", ascending=False)
        metric_help = ("Prioritizing remote clinics whose loss leaves zero alternative care nearby.")
    
    else:
        ranked_facilities = submerged_facs.sort_values(by="cell_pop", ascending=False)
        metric_help = ("Prioritizing facilities in high-density areas to preserve care for the most people.")

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

    scenario_map = build_advanced_map(county_stats, display_facilities, selected_pcode=selected_pcode)
    st_folium(scenario_map, width=1000, height=600)

    st.divider()

    ####################### Additional Info #######################

    st.write(
        "### Top 5 Recommended Facilities for Temporary Defences"
        f" ({metric_label})")
    if not ranked_facilities.empty:
        cols_to_show = ["Facility_n", "Facility_t", "Admin1"]
        rename_dict = {"Facility_n": "Facility Name", "Facility_t": "Type", "Admin1": "State"}
      
        if ("Worst-Off" in priority_mode and "county_pct_affected" in ranked_facilities.columns):
                cols_to_show.append("county_pct_affected")
                rename_dict["county_pct_affected"] = "County Risk (%)"

        elif "cell_pop" in ranked_facilities.columns:
                cols_to_show.append("cell_pop")
                rename_dict["cell_pop"] = "Local Population"

        st.dataframe(
            ranked_facilities[cols_to_show].head(5).rename(columns=rename_dict),
            use_container_width=True,
        )
    else:
        st.info("No facilities submerged under this specific scenario threshold.")

    with st.expander("Methodological Assumptions & SLE Limitations"):
        st.markdown("""
        - **Friction vs. Reality:** Friction surfaces represent mathematical resistance over land, not confirmed physical roads.
        - **Operational Integrity:** Facility status is derived from spatial coordinates; unflooded facilities may still lack staff, medicines, or operational power.
        - **Equity Alert:** Utilitarian optimization biases sandbagging and flood defenses toward densely populated towns, systematically under-ranking remote communities.
        """)

if __name__ == "__main__":
    main()