import streamlit as st

APP_TITLE = "Flood Prediction and Healthcare Access for South Sudan"

APP_SUB_TITLE = "Dataset Sources: floodobservatory.colorado.edu, dahiti.dgfi.tum.de, hydroweb.next.theia-land.fr, data.worldbank.org, www.worldpop.org, www.ipcinfo.org, www.earthdata.nasa.gov, geoportal.icpac.net, agricultural-production-hotspots.ec.europa.eu, cds.climate.copernicus.eu, www.africageoportal.com, data.humdata.org"

def main():
    st.set_page_config(APP_TITLE)
    st.title(APP_TITLE)


if __name__ == "__main__":
    main()
