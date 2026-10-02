# Use: streamlit run src/dashboard.py
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

APP_TITLE = "Flood Prediction and Healthcare Access for South Sudan"
APP_SUB_TITLE = "Dataset Sources: floodobservatory.colorado.edu, dahiti.dgfi.tum.de, hydroweb.next.theia-land.fr, data.worldbank.org, www.worldpop.org, www.ipcinfo.org, www.earthdata.nasa.gov, geoportal.icpac.net, agricultural-production-hotspots.ec.europa.eu, cds.climate.copernicus.eu, www.africageoportal.com, data.humdata.org"

def display_map():
    """
    Creates the map necessary to use for streamlit dashboard, by utilizing folium library.
    """
    map = folium.Map(location=[6.877, 31.307], zoom_start=6)
    map_2 = folium.Map(location=[6.877, 31.307], zoom_start=6)

    choropleth_1 = folium.Choropleth(
        geo_data="raw_data/Administrative boundaries/ssd_admin3.geojson",
        fill_color="YlGn",
        fill_opacity=0.3,
        line_weight=1
        )
    choropleth_1.add_to(map)

    choropleth_2 = folium.Choropleth(
            geo_data="raw_data/Administrative boundaries/ssd_adminlines.geojson",
            fill_color="YlGn",
            fill_opacity=0.3,
            line_weight=1
            )
    choropleth_2.add_to(map_2)

    st.map = st_folium(map, width=700, height=450)
    st.map = st_folium(map_2, width=700, height=450)

def main():
    """
    Main function to organize and assemble the streamlit dashboard.
    """
    st.set_page_config(APP_TITLE)
    st.title(APP_TITLE)
    st.caption(APP_SUB_TITLE)

    #LOAD DATA
    #df = pd.read_csv("")

    #DISPLAY FILTERS AND MAP
    display_map()

    #DISPLAY METRICS

if __name__ == "__main__":
    main()
